"""Exceptions and warning categories raised by revive-me."""

from __future__ import annotations


class ReviveMeError(Exception):
    """Base class for all revive-me errors."""


class CheckpointError(ReviveMeError):
    """Base class for checkpoint read/write problems."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint file or its checksum could not be read reliably."""


class IncompatibleCheckpointError(CheckpointError):
    """A readable checkpoint does not match the current run."""


class CheckpointLockedError(CheckpointError):
    """Another live Checkpointer already owns this run directory."""


class RecoveryError(CheckpointError):
    """No recoverable checkpoint generation was available.

    The run must be reset explicitly or the checkpoints repaired.  revive-me
    never silently starts a fresh run in this situation.
    """

    def __init__(self, message: str, attempts: list[tuple[int, str]] | None = None):
        super().__init__(message)
        self.attempts = list(attempts or [])


class RunStateError(ReviveMeError):
    """The ``@resumable`` run handle was used out of order or incorrectly.

    Raised for misuse that would otherwise record a wrong cursor, such as
    committing batches the loader has not delivered, or finishing an epoch with
    uncommitted or unconsumed batches.
    """


class CheckpointWriteWarning(UserWarning):
    """A checkpoint write failed; training continues with older state."""


class CheckpointRecoveryWarning(UserWarning):
    """A checkpoint generation was skipped while recovering."""


class RunIncompleteWarning(UserWarning):
    """A ``@resumable`` function returned before finishing ``run.epochs()``.

    The run is left resumable: no final checkpoint was written and ``COMPLETE``
    was not created.
    """
