"""revive-me: durable PyTorch checkpoint/recovery for single-process runs.

Use ``@resumable`` (with the injected ``run`` handle) for the short path, or
:class:`Checkpointer` directly for full control of the loop.
"""

from __future__ import annotations

from ._version import __version__
from .checkpoint import Checkpointer, Progress, ResumeInfo
from .errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointLockedError,
    CheckpointRecoveryWarning,
    CheckpointWriteWarning,
    IncompatibleCheckpointError,
    RecoveryError,
    ReviveMeError,
    RunIncompleteWarning,
    RunStateError,
)
from .resumable import Run, RunLoader, resumable
from .sampler import ResumableBatchSampler, make_dataloader

__all__ = [
    "Checkpointer",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointLockedError",
    "CheckpointRecoveryWarning",
    "CheckpointWriteWarning",
    "IncompatibleCheckpointError",
    "Progress",
    "RecoveryError",
    "ResumeInfo",
    "ResumableBatchSampler",
    "ReviveMeError",
    "Run",
    "RunIncompleteWarning",
    "RunLoader",
    "RunStateError",
    "__version__",
    "make_dataloader",
    "resumable",
]
