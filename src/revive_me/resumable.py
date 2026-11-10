"""Decorator-first entry point over the explicit :class:`Checkpointer`.

``@resumable`` gives each call of the decorated function a fresh :class:`Run`
handle.  The handle only *stages* what you register and configure; the real
:class:`Checkpointer` is built, locked and restored when ``run.epochs()`` starts,
because restore is only valid once every component and the sampler are known and
before any DataLoader iterator exists.  Your training loop, optimizer stepping,
accumulation and gradient clearing stay yours: you still tell the run when a
complete window has been committed.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from torch.utils.data import DataLoader

from .checkpoint import Checkpointer, Progress, ResumeInfo, _normalize
from .errors import (
    CheckpointError,
    CheckpointWriteWarning,
    RunIncompleteWarning,
    RunStateError,
)
from .sampler import ResumableBatchSampler, make_dataloader

__all__ = ["Run", "RunLoader", "resumable"]

logger = logging.getLogger("revive_me")

_HANDLE = "run"
F = TypeVar("F", bound=Callable[..., Any])


class _AlreadyComplete(BaseException):
    """Unwinds the body when COMPLETE is found after the lock is taken.

    A BaseException so a user's ``except Exception`` cannot swallow it.
    """


class RunLoader:
    """DataLoader wrapper that counts delivered batches in the main process.

    Created by :meth:`Run.dataloader`.  Iterate it once per epoch inside
    ``for epoch in run.epochs(n)``.  ``len(loader)`` is the number of batches
    this epoch will deliver (fewer than ``total_batches`` right after a
    mid-epoch resume) and does not change while the epoch runs.
    """

    def __init__(
        self, run: "Run", dataloader: DataLoader, sampler: ResumableBatchSampler
    ) -> None:
        self._run = run
        self._sampler = sampler
        self.dataloader = dataloader  # underlying loader; iterating it bypasses tracking
        self._len = sampler.total_batches
        self._armed = False
        self._iterated = False
        self._start = 0  # committed batches in the epoch when it began
        self._consumed = 0  # batches delivered to the loop body this epoch

    @property
    def total_batches(self) -> int:
        """Batches in a complete epoch (independent of resume position)."""
        return self._sampler.total_batches

    def __len__(self) -> int:
        return self._len

    def _begin_epoch(self, epoch: int, start: int) -> None:
        self._sampler.set_epoch(epoch)  # no-op unless the epoch changed
        self._start, self._consumed, self._iterated, self._armed = start, 0, False, True
        self._len = len(self.dataloader)

    def __iter__(self) -> Iterator[Any]:
        self._run._require_active("iterate the loader")
        if not self._armed:
            raise RunStateError(
                "iterate this loader inside `for epoch in run.epochs(...)`"
            )
        if self._iterated:
            raise RunStateError(
                "this epoch's loader was already iterated; exactly one pass per "
                "epoch is supported, because a second pass would replay from the "
                "committed epoch start"
            )
        self._iterated = True
        return self._batches()

    def _batches(self) -> Iterator[Any]:
        for batch in self.dataloader:
            self._consumed += 1
            yield batch


class Run:
    """Per-invocation handle injected by :func:`resumable` as ``run``.

    Order: ``register`` and ``dataloader`` (any order, once each), then
    ``epochs``.  ``epochs()`` takes the run-directory lock, restores the newest
    usable checkpoint into everything registered, and returns the epoch
    iterator; call ``commit`` inside it after each complete window.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        interval_seconds: float = 300.0,
        retry_seconds: float = 60.0,
        fresh: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._interval = interval_seconds
        self._retry = retry_seconds
        self._fresh = fresh
        self._pid = os.getpid()
        self._state = "configuring"  # -> "active" -> "closed"
        self._registration: dict[str, Any] | None = None
        self._loader: RunLoader | None = None
        self._ckpt: Checkpointer | None = None
        self._resumed: ResumeInfo | None = None
        self._loop_done = False

    # ------------------------------------------------------------- inspection
    @property
    def resumed(self) -> ResumeInfo | None:
        """What ``epochs()`` restored, or ``None`` for a fresh start."""
        return self._resumed

    @property
    def progress(self) -> Progress | None:
        """Committed progress; ``None`` until ``epochs()`` has started."""
        return self._ckpt.progress if self._ckpt is not None else None

    # ----------------------------------------------------------- configuration
    def register(
        self,
        *,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        components: Mapping[str, Any] | None = None,
        compatibility: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Stage the objects to save.  Nothing is restored by this call."""
        self._require_configuring("register")
        if self._registration is not None:
            raise RunStateError("register() was already called; pass everything at once")
        for label, value in (
            ("model", model),
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
            ("component", components),
        ):
            _normalize(value, label, required=label == "model")  # fail at this line
        self._registration = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "components": components,
            "compatibility": compatibility,
            "metadata": metadata,
        }

    def dataloader(
        self,
        dataset: Any,
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        dataset_fingerprint: str | None = None,
        **loader_kwargs: Any,
    ) -> RunLoader:
        """Build the run's resumable loader (see :func:`make_dataloader`)."""
        self._require_configuring("dataloader")
        if self._loader is not None:
            raise RunStateError("only one managed dataloader per run is supported")
        try:
            size = len(dataset)
        except TypeError as exc:
            raise TypeError(
                "run.dataloader needs a map-style dataset with a reliable "
                "__len__; IterableDataset is not supported"
            ) from exc
        sampler = ResumableBatchSampler(
            size,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            dataset_fingerprint=dataset_fingerprint,
        )
        loader = make_dataloader(dataset, sampler, **loader_kwargs)
        self._loader = RunLoader(self, loader, sampler)
        return self._loader

    # ---------------------------------------------------------------- training
    def epochs(self, n: int) -> Iterator[int]:
        """Take the lock, restore, and iterate epochs ``ckpt.epoch .. n-1``.

        An epoch only advances when it ended at a complete boundary: every
        batch consumed and committed.  Otherwise :class:`RunStateError` is
        raised and nothing is saved.  After a crash between an epoch's last
        commit and its end, that epoch is yielded again with an empty loader so
        your epoch-end code (scheduler, eval) still runs once.
        """
        self._require_configuring("epochs")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive int")
        if self._registration is None:
            raise RunStateError("call run.register(...) before run.epochs(...)")
        if self._loader is None:
            raise RunStateError("call run.dataloader(...) before run.epochs(...)")
        # Restore is applied only now: all components and the sampler are known,
        # and no DataLoader iterator exists yet.
        self._ckpt = ckpt = Checkpointer(
            self.run_dir,
            **self._registration,
            sampler=self._loader._sampler,
            interval_seconds=self._interval,
            retry_seconds=self._retry,
        )
        self._state = "active"
        if self._fresh:
            ckpt.reset()
        elif ckpt.is_complete():
            raise _AlreadyComplete
        else:
            self._resumed = ckpt.restore()
        return self._epoch_iter(n)

    def _epoch_iter(self, n: int) -> Iterator[int]:
        ckpt, loader = self._ckpt, self._loader
        assert ckpt is not None and loader is not None
        while ckpt.epoch < n:
            self._require_active("advance the epoch loop")
            epoch = ckpt.epoch
            loader._begin_epoch(epoch, ckpt.batch)
            yield epoch
            loader._armed = False
            if ckpt.batch != loader.total_batches:
                raise RunStateError(
                    f"epoch {epoch} ended with {ckpt.batch} of "
                    f"{loader.total_batches} batches committed "
                    f"({loader._consumed} delivered since batch {loader._start}). "
                    "Call run.commit(batches) after every complete accumulation "
                    "window, including a shorter final one, and consume the whole "
                    "loader each epoch; an epoch cut short cannot be counted as "
                    "finished."
                )
            ckpt.finish_epoch()
        self._loop_done = True

    def commit(self, batches: int = 1) -> None:
        """Commit ``batches`` consumed in a completed accumulation window.

        Call after optimizer/scaler/scheduler handling and gradient clearing.
        """
        self._require_active("commit")
        ckpt, loader = self._ckpt, self._loader
        assert ckpt is not None and loader is not None
        if not loader._armed:
            raise RunStateError("commit() is only valid inside the run.epochs() loop")
        delivered = loader._start + loader._consumed - ckpt.batch
        if isinstance(batches, int) and not isinstance(batches, bool) and batches > delivered:
            raise RunStateError(
                f"commit({batches}) but the loader has delivered only {delivered} "
                "uncommitted batch(es); commit after a window's batches were "
                "consumed, with that window's actual count"
            )
        ckpt.commit(batches)

    # --------------------------------------------------------------- lifecycle
    def _require_configuring(self, what: str) -> None:
        self._require_active(what, expect="configuring")
        if self._state == "active":
            raise RunStateError(
                f"{what}() must be called before run.epochs(): restore happens "
                "when the epoch loop starts, after every component and the "
                "sampler are known"
            )

    def _require_active(self, what: str, expect: str = "active") -> None:
        if os.getpid() != self._pid:
            raise CheckpointError(
                f"cannot {what}: this Run was inherited across fork() and does "
                "not own its run directory"
            )
        if self._state == "closed":
            raise RunStateError(f"cannot {what}: the run is closed")
        if expect == "active" and self._state != "active":
            raise RunStateError(f"cannot {what} before run.epochs() has started")

    def _finish(self) -> None:
        """Final save + COMPLETE, only after the epoch loop fully finished."""
        ckpt = self._ckpt
        if ckpt is None:
            raise RunStateError(
                "the decorated function returned without calling run.epochs(); "
                "nothing was trained or checkpointed and the run was not "
                "marked complete"
            )
        if not self._loop_done:
            warnings.warn(
                "the decorated function returned before the run.epochs() loop "
                "finished (early return or break). No final checkpoint was "
                "written and the run was NOT marked complete; the next call "
                "resumes from the last committed checkpoint.",
                RunIncompleteWarning,
                stacklevel=3,
            )
        elif ckpt.save():
            ckpt.mark_complete()
        else:
            warnings.warn(
                "training finished but the final checkpoint could not be "
                "written, so the run was NOT marked complete; the next call "
                "resumes from the last good checkpoint.",
                CheckpointWriteWarning,
                stacklevel=3,
            )

    def _close(self) -> None:
        """Release the lock without saving.  Always runs; never raises."""
        self._state = "closed"
        if self._loader is not None:
            self._loader._armed = False
        if self._ckpt is not None:
            self._ckpt.close()


def _resolve(value: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return value(*args, **kwargs) if callable(value) else value


def resumable(
    run_dir: str | os.PathLike[str] | Callable[..., str | os.PathLike[str]],
    *,
    interval_seconds: float | Callable[..., float] = 300.0,
    retry_seconds: float | Callable[..., float] = 60.0,
    fresh: bool | Callable[..., bool] = False,
) -> Callable[[F], F]:
    """Make a training function resumable; it receives ``run`` as a keyword.

    ``run_dir``, ``interval_seconds``, ``retry_seconds`` and ``fresh`` may be
    values or callables; callables receive the decorated function's own call
    arguments (``run_dir=lambda args: args.run_dir`` for ``train(args)``).

    The function must declare a keyword-only ``run`` parameter.  Each call gets
    a fresh :class:`Run`.  The wrapper:

    * skips the call, logging a warning and returning ``None``, if the run is
      already complete (``fresh=True`` deletes the old checkpoints and marker
      first; it never happens automatically);
    * returns the function's own result otherwise;
    * on normal return, writes a final checkpoint and ``COMPLETE`` only if the
      ``run.epochs()`` loop ran to its end; an early return or ``break`` warns
      (:class:`RunIncompleteWarning`) and leaves the run resumable;
    * on any exception (including ``KeyboardInterrupt``/``SystemExit``) saves
      nothing and marks nothing complete;
    * always releases the run-directory lock.

    Seed RNGs inside the function before ``run.epochs()``; restore overwrites
    RNG state at that point.
    """

    def decorate(fn: F) -> F:
        if (
            inspect.iscoroutinefunction(fn)
            or inspect.isgeneratorfunction(fn)
            or inspect.isasyncgenfunction(fn)
        ):
            raise TypeError("@resumable supports plain functions, not generators or coroutines")
        sig = inspect.signature(fn)
        param = sig.parameters.get(_HANDLE)
        if param is None or param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"{fn.__qualname__} must declare a keyword-only parameter "
                f"named {_HANDLE!r}, e.g. `def {fn.__name__}(..., *, {_HANDLE})`"
            )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _HANDLE in kwargs:
                raise TypeError(
                    f"{fn.__qualname__}() got {_HANDLE!r}; that keyword is "
                    "reserved for the handle @resumable injects"
                )
            sig.bind(*args, **kwargs, **{_HANDLE: None})  # bad call: fail before any I/O
            directory = Path(_resolve(run_dir, args, kwargs))
            want_fresh = bool(_resolve(fresh, args, kwargs))
            if not want_fresh and (directory / "COMPLETE").is_file():
                logger.warning(
                    "%s is already complete; not running %s (returning None). "
                    "Use fresh=True to delete it and start over.",
                    directory,
                    fn.__qualname__,
                )
                return None
            handle = Run(
                directory,
                interval_seconds=_resolve(interval_seconds, args, kwargs),
                retry_seconds=_resolve(retry_seconds, args, kwargs),
                fresh=want_fresh,
            )
            try:
                result = fn(*args, **kwargs, **{_HANDLE: handle})
                handle._finish()
                return result
            except _AlreadyComplete:
                logger.warning(
                    "%s became complete before training started; not running "
                    "%s (returning None).",
                    directory,
                    fn.__qualname__,
                )
                return None
            finally:
                handle._close()

        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=[p for p in sig.parameters.values() if p.name != _HANDLE]
        )
        return wrapper  # type: ignore[return-value]

    return decorate
