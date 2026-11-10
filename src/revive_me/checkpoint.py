"""Reusable, explicit PyTorch checkpointing with committed-boundary semantics."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ._version import __version__
from .errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointRecoveryWarning,
    CheckpointWriteWarning,
    IncompatibleCheckpointError,
    RecoveryError,
)
from .sampler import ResumableBatchSampler

try:  # NumPy is optional; only its RNG is affected when absent.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only in no-numpy installs
    _np = None

__all__ = ["Checkpointer", "Progress", "ResumeInfo"]

logger = logging.getLogger("revive_me")

_FORMAT_VERSION = 1
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)\.pt$")
_TEMP_MARKER = ".checkpoint-"
_REQUIRED_TOP_LEVEL = {
    "format_version",
    "created_at",
    "package_version",
    "progress",
    "sampler",
    "models",
    "optimizers",
    "schedulers",
    "scalers",
    "components",
    "rng",
    "metadata",
    "compatibility",
}


@dataclass(frozen=True)
class Progress:
    """Committed progress.  ``batch`` counts consumed batches in ``epoch``."""

    epoch: int
    batch: int
    total_batches: int | None
    batches_seen: int


@dataclass(frozen=True)
class ResumeInfo:
    """Information returned by :meth:`Checkpointer.restore`."""

    progress: Progress
    generation: int
    metadata: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    recovered_from_previous: bool

    @property
    def epoch(self) -> int:
        return self.progress.epoch

    @property
    def batch(self) -> int:
        return self.progress.batch


class Checkpointer:
    """Save and restore a single-process training run.

    The training loop remains yours.  Register live objects once, call
    :meth:`restore` before creating a DataLoader iterator, call :meth:`commit`
    only after a completed accumulation window (after optimizer/scaler/scheduler
    handling and gradient clearing), and call :meth:`finish_epoch` at epoch end.
    Saves happen synchronously at those committed boundaries when the interval
    has elapsed.  A failed write warns and leaves training running.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        model: Any | Mapping[str, Any],
        optimizer: Any | Mapping[str, Any] | None = None,
        scheduler: Any | Mapping[str, Any] | None = None,
        scaler: Any | Mapping[str, Any] | None = None,
        sampler: ResumableBatchSampler | None = None,
        components: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        compatibility: Mapping[str, Any] | None = None,
        interval_seconds: float = 300.0,
    ) -> None:
        if isinstance(interval_seconds, bool) or not isinstance(
            interval_seconds, (int, float)
        ):
            raise TypeError("interval_seconds must be a number")
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be >= 0")

        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.complete_marker = self.run_dir / "COMPLETE"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._models = _normalize(model, "model", required=True)
        self._optimizers = _normalize(optimizer, "optimizer")
        self._schedulers = _normalize(scheduler, "scheduler")
        self._scalers = _normalize(scaler, "scaler")
        self._components = _normalize(components, "component")
        if sampler is not None and not isinstance(sampler, ResumableBatchSampler):
            raise TypeError(
                "sampler must be a ResumableBatchSampler; arbitrary custom "
                "samplers are not supported for within-epoch resume"
            )
        self._sampler = sampler
        self._metadata = _as_mapping(metadata, "metadata")
        self._compatibility = _as_mapping(compatibility, "compatibility")
        self._interval = float(interval_seconds)
        self._progress = Progress(
            epoch=0,
            batch=0,
            total_batches=sampler.total_batches if sampler is not None else None,
            batches_seen=0,
        )
        self._last_save = time.monotonic()

    # ------------------------------------------------------------------ state
    @property
    def progress(self) -> Progress:
        return self._progress

    @property
    def epoch(self) -> int:
        return self._progress.epoch

    @property
    def batch(self) -> int:
        return self._progress.batch

    @property
    def sampler(self) -> ResumableBatchSampler | None:
        return self._sampler

    def is_complete(self) -> bool:
        """Whether the durable completion marker exists."""
        return self.complete_marker.is_file()

    # --------------------------------------------------------------- progress
    def commit(self, batches: int = 1) -> None:
        """Commit consumed batches at the end of a completed accumulation window.

        Call this only after the optimizer/scaler/scheduler work and gradient
        clearing for that window are finished.  A shorter final window passes
        its actual batch count.  An AMP-skipped optimizer update still consumed
        batches, so it is still committed here.  The library cannot inspect your
        loop; calling this mid-window would misrepresent the cursor.
        """
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
            raise ValueError("batches must be a positive int")
        if self._sampler is not None:
            self._require_sampler_epoch()
            total = self._sampler.total_batches
            if self._progress.batch + batches > total:
                raise ValueError(
                    f"committing {batches} batches would exceed total_batches "
                    f"{total} (already committed {self._progress.batch})"
                )
        self._progress = replace(
            self._progress,
            batch=self._progress.batch + batches,
            batches_seen=self._progress.batches_seen + batches,
        )
        self._maybe_save()

    def finish_epoch(self) -> None:
        """Mark the current epoch finished and advance to the next one."""
        if self._sampler is not None:
            self._require_sampler_epoch()
        next_epoch = self._progress.epoch + 1
        if self._sampler is not None:
            self._sampler.set_position(next_epoch, 0)
            total: int | None = self._sampler.total_batches
        else:
            total = self._progress.total_batches
        self._progress = Progress(
            epoch=next_epoch,
            batch=0,
            total_batches=total,
            batches_seen=self._progress.batches_seen,
        )
        self._maybe_save()

    def _require_sampler_epoch(self) -> None:
        assert self._sampler is not None
        if self._progress.epoch != self._sampler.epoch:
            raise RuntimeError(
                f"checkpointer progress epoch {self._progress.epoch} does not "
                f"match sampler epoch {self._sampler.epoch}; call "
                "sampler.set_epoch(ckpt.epoch) before iterating"
            )

    def _maybe_save(self) -> None:
        if self._interval <= 0 or (time.monotonic() - self._last_save) >= self._interval:
            self.save()

    # ----------------------------------------------------------------- saving
    def save(self) -> bool:
        """Write a checkpoint at the current committed boundary.

        Returns ``True`` on success and ``False`` after a write/serialization
        failure.  A failure emits :class:`CheckpointWriteWarning`, keeps the
        previous generation intact, and leaves training running.  Do not call
        this in the middle of an accumulation window; auto-saves are triggered
        from :meth:`commit` and :meth:`finish_epoch` instead.
        """
        try:
            payload = self._build_payload()
            generation = self._next_generation()
            _write_generation(self.checkpoint_dir, generation, payload)
            self._last_save = time.monotonic()
            return True
        except Exception as exc:  # deliberately checkpoint-scoped, not training
            message = (
                f"Checkpoint save failed ({type(exc).__name__}: {exc}); "
                "training continues with the previous checkpoint intact. "
                "Another save will be attempted at a later committed boundary. "
                "The five-minute recovery target is not guaranteed while writes fail."
            )
            logger.warning(message, exc_info=True)
            warnings.warn(message, CheckpointWriteWarning, stacklevel=3)
            return False

    def _build_payload(self) -> dict[str, Any]:
        progress = self._progress
        if self._sampler is not None:
            if progress.epoch != self._sampler.epoch:
                raise CheckpointError(
                    "cannot save: checkpointer progress epoch does not match "
                    "sampler epoch"
                )
            total = self._sampler.total_batches
            if progress.batch > total:
                raise CheckpointError(
                    f"cannot save: committed batch {progress.batch} exceeds "
                    f"total_batches {total}"
                )
            sampler_state = self._sampler.state_dict()
            sampler_state["epoch"] = progress.epoch
            sampler_state["batch_offset"] = progress.batch
            sampler_state["total_batches"] = total
            total_for_progress: int | None = total
        else:
            sampler_state = None
            total_for_progress = progress.total_batches

        return {
            "format_version": _FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "torch_version": torch.__version__,
            "progress": {
                "epoch": progress.epoch,
                "batch": progress.batch,
                "total_batches": total_for_progress,
                "batches_seen": progress.batches_seen,
            },
            "sampler": sampler_state,
            "models": {
                name: module.state_dict() for name, module in self._models.items()
            },
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self._optimizers.items()
            },
            "schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in self._schedulers.items()
            },
            "scalers": {
                name: scaler.state_dict() for name, scaler in self._scalers.items()
            },
            "components": {
                name: component.state_dict()
                for name, component in self._components.items()
            },
            "rng": _capture_rng(),
            "metadata": dict(self._metadata),
            "compatibility": dict(self._compatibility),
        }

    def _next_generation(self) -> int:
        generations = self._list_generations()
        return generations[0][0] + 1 if generations else 1

    def _list_generations(self) -> list[tuple[int, Path]]:
        if not self.checkpoint_dir.is_dir():
            return []
        found: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.iterdir():
            match = _CHECKPOINT_RE.match(path.name)
            if match and path.is_file():
                found.append((int(match.group(1)), path))
        return sorted(found, key=lambda item: item[0], reverse=True)

    # -------------------------------------------------------------- restoring
    def restore(self, *, allow_incompatible: bool = False) -> ResumeInfo | None:
        """Restore the newest usable generation.

        Returns ``None`` only when no checkpoint files exist (a legitimate first
        run).  A corrupt newest generation falls back to the previous generation
        with a visible warning.  If no generation is readable, raises
        :class:`RecoveryError`; if a readable generation is incompatible, raises
        :class:`IncompatibleCheckpointError`.  Neither situation starts fresh
        automatically: use :meth:`reset` or ``allow_incompatible=True``
        explicitly.
        """
        generations = self._list_generations()
        if not generations:
            return None

        attempts: list[tuple[int, str]] = []
        chosen: tuple[int, dict[str, Any]] | None = None
        for generation, path in generations:
            try:
                payload = _load_payload(path)
            except IncompatibleCheckpointError:
                raise
            except CheckpointCorruptError as exc:
                attempts.append((generation, str(exc)))
                _warn(
                    f"Checkpoint generation {generation} is unusable ({exc}); "
                    "trying an older generation.",
                    CheckpointRecoveryWarning,
                )
                continue
            try:
                self._validate_compatibility(payload, allow_incompatible)
            except IncompatibleCheckpointError as exc:
                attempts.append((generation, str(exc)))
                raise
            except Exception as exc:
                # A hash-valid but structurally malformed payload is treated
                # like corruption so an older generation can still be used.
                attempts.append((generation, f"malformed payload: {exc}"))
                _warn(
                    f"Checkpoint generation {generation} has a malformed payload "
                    f"({exc}); trying an older generation.",
                    CheckpointRecoveryWarning,
                )
                continue
            chosen = (generation, payload)
            break

        if chosen is None:
            detail = "; ".join(f"generation {g}: {why}" for g, why in attempts)
            raise RecoveryError(
                "No recoverable checkpoint generation was found. "
                "Refusing to start fresh automatically. "
                "Call Checkpointer.reset() only if losing the run state is intended. "
                f"Attempts: {detail}",
                attempts,
            )

        generation, payload = chosen
        recovered = any(seq > generation for seq, _ in generations)
        try:
            self._apply_payload(payload, allow_incompatible)
        except Exception as exc:
            raise CheckpointError(
                "Checkpoint was readable and compatible, but applying it failed. "
                "Live components may be partially restored; restart the process "
                "and recover from a known-good generation or reset explicitly."
            ) from exc

        self._last_save = time.monotonic()
        return ResumeInfo(
            progress=self._progress,
            generation=generation,
            metadata=dict(payload["metadata"]),
            compatibility=dict(payload["compatibility"]),
            recovered_from_previous=recovered,
        )

    def _validate_compatibility(
        self, payload: Mapping[str, Any], allow_incompatible: bool
    ) -> None:
        saved_compat = payload["compatibility"]
        if saved_compat != self._compatibility:
            if not allow_incompatible:
                raise IncompatibleCheckpointError(
                    "checkpoint compatibility metadata does not match this run: "
                    f"checkpoint={saved_compat!r}, current={self._compatibility!r}. "
                    "Pass allow_incompatible=True only for an intentional change."
                )
            _warn(
                "Compatibility metadata differs; restoring because "
                "allow_incompatible=True was requested.",
                CheckpointRecoveryWarning,
            )

        self._check_component_names(payload, "models", self._models)
        self._check_component_names(payload, "optimizers", self._optimizers)
        self._check_component_names(payload, "schedulers", self._schedulers)
        self._check_component_names(payload, "scalers", self._scalers)
        self._check_component_names(payload, "components", self._components)

        for name, module in self._models.items():
            _validate_state_structure(
                module.state_dict(), payload["models"][name], f"model {name!r}"
            )
        for name, optimizer in self._optimizers.items():
            _validate_optimizer_state(
                optimizer.state_dict(), payload["optimizers"][name], name
            )
        for name, scheduler in self._schedulers.items():
            _validate_mapping_keys(
                scheduler.state_dict(),
                payload["schedulers"][name],
                f"scheduler {name!r}",
            )
        for name, scaler in self._scalers.items():
            _validate_mapping_keys(
                scaler.state_dict(), payload["scalers"][name], f"scaler {name!r}"
            )
        for name, component in self._components.items():
            current_state = component.state_dict()
            saved_state = payload["components"][name]
            if isinstance(current_state, Mapping) and isinstance(saved_state, Mapping):
                _validate_mapping_keys(
                    current_state, saved_state, f"component {name!r}"
                )

        saved_sampler = payload["sampler"]
        if self._sampler is None:
            if saved_sampler is not None:
                raise IncompatibleCheckpointError(
                    "checkpoint contains within-epoch sampler state, but this "
                    "Checkpointer has no sampler"
                )
            return
        if saved_sampler is None:
            raise IncompatibleCheckpointError(
                "checkpoint has no sampler state, but this Checkpointer has a "
                "ResumableBatchSampler"
            )
        try:
            matches = self._sampler.validate_state_dict(
                saved_sampler, allow_config_mismatch=allow_incompatible
            )
        except ValueError as exc:
            raise IncompatibleCheckpointError(
                f"resumable sampler state is incompatible: {exc}"
            ) from exc
        if not matches:
            _warn(
                "Resumable sampler configuration differs; restoring the epoch "
                "with batch offset reset to zero because allow_incompatible=True "
                "was requested.",
                CheckpointRecoveryWarning,
            )

        progress = payload["progress"]
        if int(saved_sampler["epoch"]) != int(progress["epoch"]):
            raise IncompatibleCheckpointError(
                "checkpoint progress epoch and sampler epoch disagree"
            )
        if int(saved_sampler["batch_offset"]) != int(progress["batch"]):
            raise IncompatibleCheckpointError(
                "checkpoint progress batch and sampler committed offset disagree"
            )
        if progress["total_batches"] != saved_sampler["total_batches"]:
            raise IncompatibleCheckpointError(
                "checkpoint progress total_batches and sampler total_batches disagree"
            )

    @staticmethod
    def _check_component_names(
        payload: Mapping[str, Any], section: str, current: Mapping[str, Any]
    ) -> None:
        saved_names = set(payload[section])
        current_names = set(current)
        if saved_names != current_names:
            missing = sorted(current_names - saved_names)
            extra = sorted(saved_names - current_names)
            raise IncompatibleCheckpointError(
                f"checkpoint {section} names do not match this run "
                f"(missing in checkpoint={missing}, extra in checkpoint={extra})"
            )

    def _apply_payload(
        self, payload: Mapping[str, Any], allow_incompatible: bool
    ) -> None:
        for name, module in self._models.items():
            module.load_state_dict(payload["models"][name])
        for name, optimizer in self._optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        for name, scheduler in self._schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        for name, scaler in self._scalers.items():
            scaler.load_state_dict(payload["scalers"][name])
        for name, component in self._components.items():
            component.load_state_dict(payload["components"][name])
        if self._sampler is not None:
            self._sampler.load_state_dict(
                payload["sampler"], allow_config_mismatch=allow_incompatible
            )

        _restore_rng(payload["rng"])

        saved_progress = payload["progress"]
        batch = int(saved_progress["batch"])
        total = saved_progress["total_batches"]
        if self._sampler is not None:
            # If an intentional sampler config change reset the offset, use
            # that safe value rather than the old batch index.
            batch = self._sampler.batch_offset
            total = self._sampler.total_batches
        self._progress = Progress(
            epoch=int(saved_progress["epoch"]),
            batch=batch,
            total_batches=total,
            batches_seen=int(saved_progress["batches_seen"]),
        )

    # -------------------------------------------------------------- lifecycle
    def mark_complete(self) -> None:
        """Durably mark the run complete so a launcher will not relaunch it."""
        payload = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "epoch": self._progress.epoch,
            "batch": self._progress.batch,
            "batches_seen": self._progress.batches_seen,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.complete_marker, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    def reset(self) -> None:
        """Explicitly delete checkpoints and the completion marker.

        This is destructive and intentionally never called automatically.  It
        resets progress and sampler position but does not reinitialize live model
        or optimizer objects.
        """
        if self.checkpoint_dir.is_dir():
            for path in self.checkpoint_dir.iterdir():
                is_generation = path.name.startswith("checkpoint-") and path.suffix in {
                    ".pt",
                    ".sha256",
                }
                is_temp = (
                    path.name.startswith(_TEMP_MARKER) and path.name.endswith(".tmp")
                )
                if path.is_file() and (is_generation or is_temp):
                    path.unlink(missing_ok=True)
            _fsync_dir(self.checkpoint_dir)
        self.complete_marker.unlink(missing_ok=True)
        if self.run_dir.is_dir():
            _fsync_dir(self.run_dir)
        if self._sampler is not None:
            self._sampler.set_position(0, 0)
            total: int | None = self._sampler.total_batches
        else:
            total = None
        self._progress = Progress(0, 0, total, 0)
        self._last_save = time.monotonic()


# ---------------------------------------------------------------------------
# Validation helpers


def _normalize(
    value: Any | Mapping[str, Any] | None,
    label: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    if value is None:
        if required:
            raise ValueError(f"at least one {label} is required")
        return {}
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {"main": value}
    if required and not result:
        raise ValueError(f"at least one {label} is required")
    for name, obj in result.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} names must be non-empty strings")
        for method in ("state_dict", "load_state_dict"):
            if not callable(getattr(obj, method, None)):
                raise TypeError(
                    f"{label} {name!r} must provide callable {method}()"
                )
    return result


def _as_mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _validate_mapping_keys(current: Any, saved: Any, label: str) -> None:
    if not isinstance(current, Mapping) or not isinstance(saved, Mapping):
        raise IncompatibleCheckpointError(f"{label} state is not a mapping")
    current_keys = set(current)
    saved_keys = set(saved)
    if current_keys != saved_keys:
        missing = sorted(map(str, current_keys - saved_keys))
        extra = sorted(map(str, saved_keys - current_keys))
        raise IncompatibleCheckpointError(
            f"{label} state keys differ (missing={missing}, extra={extra})"
        )


def _validate_state_structure(current: Any, saved: Any, label: str) -> None:
    _validate_mapping_keys(current, saved, label)
    for key in current:
        current_value = current[key]
        saved_value = saved[key]
        current_tensor = torch.is_tensor(current_value)
        saved_tensor = torch.is_tensor(saved_value)
        if current_tensor != saved_tensor:
            raise IncompatibleCheckpointError(
                f"{label} state entry {key!r} changed between tensor and non-tensor"
            )
        if current_tensor and tuple(current_value.shape) != tuple(saved_value.shape):
            raise IncompatibleCheckpointError(
                f"{label} state entry {key!r} shape changed: "
                f"checkpoint={tuple(saved_value.shape)}, "
                f"current={tuple(current_value.shape)}"
            )


def _validate_optimizer_state(current: Any, saved: Any, name: str) -> None:
    _validate_mapping_keys(current, saved, f"optimizer {name!r}")
    current_groups = current.get("param_groups")
    saved_groups = saved.get("param_groups")
    if not isinstance(current_groups, list) or not isinstance(saved_groups, list):
        raise IncompatibleCheckpointError(
            f"optimizer {name!r} state has malformed param_groups"
        )
    if len(current_groups) != len(saved_groups):
        raise IncompatibleCheckpointError(
            f"optimizer {name!r} param group count changed: "
            f"checkpoint={len(saved_groups)}, current={len(current_groups)}"
        )
    for index, (current_group, saved_group) in enumerate(
        zip(current_groups, saved_groups)
    ):
        current_params = current_group.get("params") if isinstance(current_group, Mapping) else None
        saved_params = saved_group.get("params") if isinstance(saved_group, Mapping) else None
        if current_params is None or saved_params is None:
            raise IncompatibleCheckpointError(
                f"optimizer {name!r} param group {index} has no params"
            )
        if len(current_params) != len(saved_params):
            raise IncompatibleCheckpointError(
                f"optimizer {name!r} param group {index} size changed: "
                f"checkpoint={len(saved_params)}, current={len(current_params)}"
            )


def _validate_payload_shape(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise CheckpointCorruptError("checkpoint payload is not a mapping")
    missing = _REQUIRED_TOP_LEVEL - set(payload)
    if missing:
        raise CheckpointCorruptError(
            f"checkpoint payload is missing keys: {sorted(missing)}"
        )
    version = payload["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise CheckpointCorruptError("checkpoint format_version is not an int")
    if version != _FORMAT_VERSION:
        raise IncompatibleCheckpointError(
            f"checkpoint format version {version} is not supported by this "
            f"revive-me build (supported: {_FORMAT_VERSION})"
        )
    for section in (
        "models",
        "optimizers",
        "schedulers",
        "scalers",
        "components",
        "rng",
        "metadata",
        "compatibility",
    ):
        if not isinstance(payload[section], Mapping):
            raise CheckpointCorruptError(
                f"checkpoint section {section!r} is not a mapping"
            )
    progress = payload["progress"]
    if not isinstance(progress, Mapping):
        raise CheckpointCorruptError("checkpoint progress is not a mapping")
    for key in ("epoch", "batch", "batches_seen"):
        value = progress.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointCorruptError(
                f"checkpoint progress {key!r} is not a non-negative int"
            )
    total = progress.get("total_batches")
    if total is not None and (
        isinstance(total, bool) or not isinstance(total, int) or total < 0
    ):
        raise CheckpointCorruptError(
            "checkpoint progress total_batches is not a non-negative int or None"
        )
    if total is not None and progress["batch"] > total:
        raise CheckpointCorruptError(
            "checkpoint progress batch exceeds total_batches"
        )
    sampler = payload["sampler"]
    if sampler is not None:
        if not isinstance(sampler, Mapping):
            raise CheckpointCorruptError("checkpoint sampler state is not a mapping")
        required_sampler_keys = {
            "version",
            "epoch",
            "batch_offset",
            "seed",
            "dataset_size",
            "batch_size",
            "shuffle",
            "drop_last",
            "dataset_fingerprint",
            "total_batches",
        }
        missing_sampler = required_sampler_keys - set(sampler)
        if missing_sampler:
            raise CheckpointCorruptError(
                "checkpoint sampler state is missing keys: "
                f"{sorted(missing_sampler)}"
            )
    rng = payload["rng"]
    if "python" not in rng or "torch_cpu" not in rng:
        raise CheckpointCorruptError("checkpoint RNG state is incomplete")
    if not torch.is_tensor(rng["torch_cpu"]):
        raise CheckpointCorruptError("checkpoint torch CPU RNG state is not a tensor")
    cuda_rng = rng.get("torch_cuda")
    if cuda_rng is not None and (
        not isinstance(cuda_rng, (list, tuple))
        or not all(torch.is_tensor(state) for state in cuda_rng)
    ):
        raise CheckpointCorruptError("checkpoint CUDA RNG state is malformed")


# ---------------------------------------------------------------------------
# RNG


def _capture_rng() -> dict[str, Any]:
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": _np.random.get_state() if _np is not None else None,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def _restore_rng(rng: Mapping[str, Any]) -> None:
    random.setstate(rng["python"])
    numpy_state = rng.get("numpy")
    if numpy_state is not None:
        if _np is None:
            _warn(
                "Checkpoint contains NumPy RNG state but NumPy is unavailable; "
                "NumPy RNG was not restored.",
                CheckpointRecoveryWarning,
            )
        else:
            _np.random.set_state(numpy_state)
    torch.set_rng_state(rng["torch_cpu"])

    cuda_states = rng.get("torch_cuda")
    if cuda_states:
        if not torch.cuda.is_available():
            _warn(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "CUDA RNG was not restored.",
                CheckpointRecoveryWarning,
            )
        else:
            device_count = torch.cuda.device_count()
            states = list(cuda_states)
            if len(states) != device_count:
                _warn(
                    f"Checkpoint has CUDA RNG state for {len(states)} device(s), "
                    f"but {device_count} device(s) are visible; restoring the "
                    "available subset.",
                    CheckpointRecoveryWarning,
                )
                for index, state in enumerate(states[:device_count]):
                    torch.cuda.set_rng_state(state, index)
            else:
                torch.cuda.set_rng_state_all(states)


# ---------------------------------------------------------------------------
# Durable file helpers


def _load_payload(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise CheckpointCorruptError(f"missing checksum sidecar {sidecar.name}")
    try:
        expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    except (OSError, UnicodeError, IndexError) as exc:
        raise CheckpointCorruptError(
            f"cannot read checksum sidecar {sidecar.name}: {exc}"
        ) from exc
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise CheckpointCorruptError(
            f"malformed checksum sidecar {sidecar.name}"
        )
    actual = _sha256_file(path)
    if actual != expected:
        raise CheckpointCorruptError(
            f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointCorruptError(f"cannot deserialize {path.name}: {exc}") from exc
    try:
        _validate_payload_shape(payload)
    except (CheckpointCorruptError, IncompatibleCheckpointError):
        raise
    except Exception as exc:
        raise CheckpointCorruptError(
            f"malformed checkpoint payload in {path.name}: {exc}"
        ) from exc
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _HashingWriter:
    def __init__(self, handle: Any):
        self._handle = handle
        self._digest = hashlib.sha256()

    def write(self, data: Any) -> int:
        if isinstance(data, str):
            data = data.encode()
        self._digest.update(data)
        return self._handle.write(data)

    def flush(self) -> None:
        self._handle.flush()

    def tell(self) -> int:
        return self._handle.tell()

    def seek(self, *args: Any) -> int:
        return self._handle.seek(*args)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _write_generation(directory: Path, generation: int, payload: Mapping[str, Any]) -> Path:
    path = directory / f"checkpoint-{generation:08d}.pt"
    sidecar = path.with_suffix(".sha256")
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temp_path = directory / f".checkpoint-{generation:08d}.pt.{token}.tmp"
    temp_sidecar = directory / f".checkpoint-{generation:08d}.sha256.{token}.tmp"
    try:
        with temp_path.open("xb") as handle:
            writer = _HashingWriter(handle)
            torch.save(payload, writer)
            writer.flush()
            os.fsync(handle.fileno())
            digest = writer.hexdigest()
        with temp_sidecar.open("x", encoding="ascii") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        # The checkpoint appears only after it is complete and fsynced.  If we
        # stop before the sidecar rename, the new generation is ignored and the
        # previous generation remains recoverable.
        os.replace(temp_path, path)
        _fsync_dir(directory)
        os.replace(temp_sidecar, sidecar)
        _fsync_dir(directory)
    except BaseException:
        for candidate in (temp_path, temp_sidecar):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        # If the checkpoint file was already renamed but its sidecar was not,
        # the new generation is unusable.  Remove it so a later save does not
        # mistake it for a recoverable generation; the previous one is intact.
        if path.exists() and not sidecar.exists():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    _prune_generations(directory)
    return path


def _prune_generations(directory: Path) -> None:
    """Keep the two newest generations that have a checksum sidecar.

    A generation without a sidecar is incomplete (for example a process died
    between the checkpoint rename and the sidecar rename) and is not a recovery
    candidate.  Keeping two sidecar-backed generations means a failed or
    truncated newest write cannot destroy the last good state.
    """
    generations = []
    for path in directory.iterdir():
        match = _CHECKPOINT_RE.match(path.name)
        if match and path.is_file():
            generations.append((int(match.group(1)), path))
    generations.sort(key=lambda item: item[0], reverse=True)
    recoverable = [
        (sequence, path)
        for sequence, path in generations
        if path.with_suffix(".sha256").is_file()
    ]
    keep = {sequence for sequence, _ in recoverable[:2]}
    for sequence, path in generations:
        if sequence in keep:
            continue
        for candidate in (path, path.with_suffix(".sha256")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not prune old checkpoint %s: %s. Newer generations "
                    "remain intact.",
                    candidate,
                    exc,
                )
    _fsync_dir(directory)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temp_path = path.with_name(f".{path.name}.{token}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _fsync_dir(directory: Path) -> None:
    """Best-effort directory fsync; ignore filesystems that cannot support it."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise
    finally:
        os.close(descriptor)


def _warn(message: str, category: type[UserWarning]) -> None:
    logger.warning(message)
    warnings.warn(message, category, stacklevel=3)
