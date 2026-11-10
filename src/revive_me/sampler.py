"""A small resumable batch sampler for stable map-style datasets.

The sampler deliberately keeps two positions separate:

* the position used to construct the current epoch's order (``epoch``), and
* the *committed* number of batches consumed in that epoch (``batch_offset``).

``DataLoader`` workers prefetch batches, so the sampler's yielded position can
be ahead of what the training loop has actually consumed.  The checkpointer
owns the committed cursor and writes it to disk; it never derives the cursor
from how far the sampler has yielded.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, Sampler

__all__ = ["ResumableBatchSampler", "make_dataloader"]

_SAMPLER_STATE_VERSION = 1
_CONFIG_KEYS = (
    "seed",
    "dataset_size",
    "batch_size",
    "shuffle",
    "drop_last",
    "dataset_fingerprint",
    "total_batches",
)


class ResumableBatchSampler(Sampler[list[int]]):
    """Yield fixed-size batches in sequential or seeded-shuffled order.

    Parameters
    ----------
    dataset_size:
        Number of items in the map-style dataset.  It must not change during a
        run; :func:`make_dataloader` also checks ``len(dataset)``.
    batch_size:
        Fixed batch size.
    shuffle:
        Use a deterministic permutation seeded from ``seed + epoch``.
    seed:
        Base order seed.  The epoch is added so every epoch has a stable,
        distinct order.
    drop_last:
        Drop the final short batch.  The default matches the recovery contract;
        ``False`` is supported but the final batch is shorter.
    dataset_fingerprint:
        Optional user-supplied fingerprint of dataset membership (for example a
        hash of sorted sample ids).  revive-me cannot detect a same-size
        membership change without this.

    Notes
    -----
    ``batch_offset`` is the committed position captured at the start of the
    current iterator (or restored from a checkpoint).  The sampler never
    advances it from yielded or prefetched batches.  Call
    :meth:`Checkpointer.commit` after each completed accumulation window; the
    checkpointer's live committed cursor is what gets saved.  On restore, call
    ``sampler.set_epoch(epoch)``; setting the same epoch preserves the restored
    offset, setting a new epoch starts at batch zero.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        dataset_fingerprint: str | None = None,
    ) -> None:
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
            raise TypeError("dataset_size must be an int")
        if dataset_size < 0:
            raise ValueError("dataset_size must be >= 0")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an int")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")
        if dataset_fingerprint is not None and not isinstance(dataset_fingerprint, str):
            raise TypeError("dataset_fingerprint must be a str or None")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.dataset_fingerprint = dataset_fingerprint
        self.epoch = 0
        self.batch_offset = 0

    @property
    def total_batches(self) -> int:
        """Number of batches in a complete epoch."""
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch, preserving the offset when it is already current."""
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an int")
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if epoch != self.epoch:
            self.epoch = epoch
            self.batch_offset = 0

    def set_position(self, epoch: int, batch_offset: int) -> None:
        """Explicitly set epoch and committed batch offset."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative int")
        if (
            isinstance(batch_offset, bool)
            or not isinstance(batch_offset, int)
            or batch_offset < 0
        ):
            raise ValueError("batch_offset must be a non-negative int")
        if batch_offset > self.total_batches:
            raise ValueError(
                f"batch_offset {batch_offset} exceeds total_batches {self.total_batches}"
            )
        self.epoch = epoch
        self.batch_offset = batch_offset

    def _order(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(self.dataset_size))
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        # Snapshot the position so a concurrent set_epoch cannot change an
        # iterator halfway through.  DataLoader prefetch does not advance this
        # object's committed offset.
        epoch = self.epoch
        offset = self.batch_offset
        order = self._order(epoch)
        total = self.total_batches
        for batch_index in range(offset, total):
            start = batch_index * self.batch_size
            end = min(start + self.batch_size, self.dataset_size)
            if self.drop_last and end - start < self.batch_size:
                break
            yield order[start:end]

    def __len__(self) -> int:
        return max(0, self.total_batches - self.batch_offset)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _SAMPLER_STATE_VERSION,
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "dataset_fingerprint": self.dataset_fingerprint,
            "total_batches": self.total_batches,
        }

    def validate_state_dict(
        self, state: Mapping[str, Any], *, allow_config_mismatch: bool = False
    ) -> bool:
        """Validate a saved sampler state without mutating this sampler.

        Returns ``True`` when the immutable sampler configuration matches,
        ``False`` when it differs and ``allow_config_mismatch`` permits a reset.
        Raises ``ValueError`` for malformed state or a disallowed mismatch.
        """
        if not isinstance(state, Mapping):
            raise ValueError("sampler state must be a mapping")
        version = state.get("version")
        if version != _SAMPLER_STATE_VERSION:
            raise ValueError(f"unsupported sampler state version: {version!r}")
        for key in ("epoch", "batch_offset"):
            value = state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"sampler state {key!r} must be a non-negative int")
        for key in _CONFIG_KEYS:
            if key not in state:
                raise ValueError(f"sampler state is missing {key!r}")
        if isinstance(state["seed"], bool) or not isinstance(state["seed"], int):
            raise ValueError("sampler state 'seed' must be an int")
        if (
            isinstance(state["dataset_size"], bool)
            or not isinstance(state["dataset_size"], int)
            or state["dataset_size"] < 0
        ):
            raise ValueError("sampler state 'dataset_size' must be a non-negative int")
        if (
            isinstance(state["batch_size"], bool)
            or not isinstance(state["batch_size"], int)
            or state["batch_size"] <= 0
        ):
            raise ValueError("sampler state 'batch_size' must be a positive int")
        if not isinstance(state["shuffle"], bool):
            raise ValueError("sampler state 'shuffle' must be a bool")
        if not isinstance(state["drop_last"], bool):
            raise ValueError("sampler state 'drop_last' must be a bool")
        if state["dataset_fingerprint"] is not None and not isinstance(
            state["dataset_fingerprint"], str
        ):
            raise ValueError(
                "sampler state 'dataset_fingerprint' must be a str or None"
            )
        if (
            isinstance(state["total_batches"], bool)
            or not isinstance(state["total_batches"], int)
            or state["total_batches"] < 0
        ):
            raise ValueError(
                "sampler state 'total_batches' must be a non-negative int"
            )

        current = {
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "dataset_fingerprint": self.dataset_fingerprint,
            "total_batches": self.total_batches,
        }
        saved = {key: state.get(key) for key in _CONFIG_KEYS}
        matches = saved == current
        if not matches and not allow_config_mismatch:
            changed = [
                f"{key}: checkpoint={saved[key]!r}, current={current[key]!r}"
                for key in _CONFIG_KEYS
                if saved[key] != current[key]
            ]
            raise ValueError(
                "resumable sampler configuration changed; "
                + "; ".join(changed)
                + ". Use allow_incompatible=True only if resetting the epoch "
                "position is acceptable."
            )
        if matches:
            if state["batch_offset"] > self.total_batches:
                raise ValueError(
                    f"sampler batch_offset {state['batch_offset']} exceeds "
                    f"total_batches {self.total_batches}"
                )
        return matches

    def load_state_dict(
        self, state: Mapping[str, Any], *, allow_config_mismatch: bool = False
    ) -> None:
        """Restore epoch and committed batch offset.

        If the configuration changed and ``allow_config_mismatch`` is true, the
        epoch is restored but the batch offset is reset to zero.  Replaying the
        whole current epoch is the only safe choice after an order change.
        """
        matches = self.validate_state_dict(
            state, allow_config_mismatch=allow_config_mismatch
        )
        self.epoch = int(state["epoch"])
        self.batch_offset = int(state["batch_offset"]) if matches else 0


def make_dataloader(
    dataset: Any,
    sampler: ResumableBatchSampler,
    **loader_kwargs: Any,
) -> DataLoader:
    """Build an ordered DataLoader using ``sampler`` as its batch sampler.

    This helper is intentionally strict: it rejects ``IterableDataset``,
    arbitrary custom samplers, and DataLoader options that would override the
    sampler's order.  It also verifies that ``len(dataset)`` still matches the
    sampler's ``dataset_size``.

    Out-of-order worker delivery (``in_order=False``, available in newer torch
    versions) is rejected: a committed prefix cursor cannot describe a stream
    whose batches arrive in an arbitrary order.  ``in_order=True`` is accepted
    for compatibility and then omitted from the ``DataLoader`` call, because
    ordered delivery is the default in every supported torch version.
    """
    if "in_order" in loader_kwargs:
        in_order = loader_kwargs.pop("in_order")
        if not isinstance(in_order, bool):
            raise TypeError("in_order must be a bool")
        if not in_order:
            raise ValueError(
                "make_dataloader requires ordered batch delivery; "
                "in_order=False cannot be represented by a committed prefix cursor"
            )
    if isinstance(dataset, IterableDataset):
        raise TypeError(
            "IterableDataset is not supported for within-epoch resume; use a "
            "stable map-style dataset with ResumableBatchSampler"
        )
    if not isinstance(sampler, ResumableBatchSampler):
        raise TypeError(
            "make_dataloader requires a ResumableBatchSampler; arbitrary custom "
            "samplers are not supported for within-epoch resume"
        )
    forbidden = {"batch_size", "shuffle", "sampler", "batch_sampler", "drop_last"}
    present = forbidden.intersection(loader_kwargs)
    if present:
        names = ", ".join(sorted(present))
        raise TypeError(
            f"make_dataloader owns batch order/size; do not pass: {names}"
        )
    try:
        dataset_size = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "dataset must be a map-style dataset with a reliable __len__"
        ) from exc
    if dataset_size != sampler.dataset_size:
        raise ValueError(
            f"dataset length changed: sampler expects {sampler.dataset_size}, "
            f"dataset has {dataset_size}. Refusing to resume with changed membership."
        )
    return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
