from __future__ import annotations

import copy
import dataclasses
import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from revive_me import (
    Checkpointer,
    CheckpointError,
    CheckpointRecoveryWarning,
    CheckpointWriteWarning,
    IncompatibleCheckpointError,
    RecoveryError,
    ResumableBatchSampler,
)
from revive_me import checkpoint as checkpoint_module
from conftest import (
    Counter,
    FakeScaler,
    ToyModel,
    assert_state_equal,
    make_run,
    training_step,
)


class _FakeClock:
    """Controlled monotonic time: tests never sleep."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, start: float = 1000.0) -> None:
        self.now = start
        monkeypatch.setattr(checkpoint_module, "_now", lambda: self.now)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _batch_of(checkpoint_dir: Path, generation: int) -> int:
    path = checkpoint_dir / f"checkpoint-{generation:08d}.pt"
    return checkpoint_module._load_payload(path)["progress"]["batch"]


def _snapshot(
    ckpt: Checkpointer,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    component: Counter,
    sampler: ResumableBatchSampler,
) -> dict[str, Any]:
    return {
        "progress": ckpt.progress,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "component": copy.deepcopy(component.state_dict()),
        # The checkpoint stores the committed consumer cursor, not the
        # sampler's iterator-start offset.
        "sampler": {"epoch": ckpt.epoch, "batch_offset": ckpt.batch},
    }


def test_roundtrip_all_registered_state(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, scaler, component, sampler = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(2)
    ckpt.finish_epoch()
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    assert ckpt.save() is True
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    expected = _snapshot(ckpt, model, optimizer, scheduler, scaler, component, sampler)

    ckpt.close()  # one writer per run directory; the "restart" releases it first
    ckpt2, model2, optimizer2, scheduler2, scaler2, component2, sampler2 = make_run(
        tmp_path
    )
    info = ckpt2.restore()
    assert info is not None
    assert info.progress == expected["progress"]
    assert info.progress.epoch == 1
    assert info.progress.batch == 1
    assert info.metadata == {"config": {"lr": 0.1}}
    assert info.compatibility == {"arch": "toy-v1", "dataset": "toy-v1"}
    assert info.generation == 4
    assert info.recovered_from_previous is False

    assert_state_equal(model2.state_dict(), expected["model"])
    assert_state_equal(optimizer2.state_dict(), expected["optimizer"])
    assert_state_equal(scheduler2.state_dict(), expected["scheduler"])
    assert_state_equal(scaler2.state_dict(), expected["scaler"])
    assert_state_equal(component2.state_dict(), expected["component"])
    assert sampler2.epoch == expected["sampler"]["epoch"]
    assert sampler2.batch_offset == expected["sampler"]["batch_offset"]
    assert len(sampler2) == sampler2.total_batches - 1

    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert actual_rng == expected_rng


def test_multiple_models_and_optimizers(tmp_path: Path) -> None:
    first = ToyModel()
    second = ToyModel(3, 5)
    first_opt = torch.optim.SGD(first.parameters(), lr=0.1)
    second_opt = torch.optim.Adam(second.parameters(), lr=0.01)
    sampler = ResumableBatchSampler(9, 3, shuffle=False, drop_last=True)
    ckpt = Checkpointer(
        tmp_path,
        model={"first": first, "second": second},
        optimizer={"first": first_opt, "second": second_opt},
        sampler=sampler,
        interval_seconds=0,
    )
    first_opt.zero_grad(set_to_none=True)
    second_opt.zero_grad(set_to_none=True)
    first(torch.randn(4, 2)).pow(2).mean().backward()
    second(torch.randn(4, 3)).pow(2).mean().backward()
    first_opt.step()
    second_opt.step()
    ckpt.commit(1)
    ckpt.close()

    first2 = ToyModel()
    second2 = ToyModel(3, 5)
    first_opt2 = torch.optim.SGD(first2.parameters(), lr=0.1)
    second_opt2 = torch.optim.Adam(second2.parameters(), lr=0.01)
    sampler2 = ResumableBatchSampler(9, 3, shuffle=False, drop_last=True)
    ckpt2 = Checkpointer(
        tmp_path,
        model={"first": first2, "second": second2},
        optimizer={"first": first_opt2, "second": second_opt2},
        sampler=sampler2,
        interval_seconds=0,
    )
    assert ckpt2.restore() is not None
    assert_state_equal(first2.state_dict(), first.state_dict())
    assert_state_equal(second2.state_dict(), second.state_dict())
    assert_state_equal(first_opt2.state_dict(), first_opt.state_dict())
    assert_state_equal(second_opt2.state_dict(), second_opt.state_dict())


def test_commit_and_finish_epoch_validation(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, sampler = make_run(
        tmp_path, dataset_size=12, batch_size=3
    )
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(3)
    with pytest.raises(ValueError, match="exceed"):
        ckpt.commit(2)
    ckpt.commit(1)
    assert ckpt.progress.batch == 4
    ckpt.finish_epoch()
    assert ckpt.progress.epoch == 1
    assert ckpt.progress.batch == 0
    assert sampler.epoch == 1
    assert sampler.batch_offset == 0

    sampler.set_epoch(5)
    with pytest.raises(RuntimeError, match="does not match sampler epoch"):
        ckpt.commit(1)


def test_compatibility_rejected_and_explicit_override(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    expected_model = copy.deepcopy(model.state_dict())
    ckpt.close()

    ckpt2, model2, *_ = make_run(
        tmp_path, compatibility={"arch": "different", "dataset": "toy-v1"}
    )
    before = copy.deepcopy(model2.state_dict())
    with pytest.raises(IncompatibleCheckpointError, match="compatibility"):
        ckpt2.restore()
    assert_state_equal(model2.state_dict(), before)

    with pytest.warns(CheckpointRecoveryWarning, match="allow_incompatible"):
        info = ckpt2.restore(allow_incompatible=True)
    assert info is not None
    assert_state_equal(model2.state_dict(), expected_model)


def test_model_shape_rejected_before_apply(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    ckpt.close()

    incompatible_model = ToyModel(in_features=3)
    before = copy.deepcopy(incompatible_model.state_dict())
    ckpt2, *_ = make_run(tmp_path, model=incompatible_model)
    with pytest.raises(IncompatibleCheckpointError, match="shape changed"):
        ckpt2.restore()
    assert_state_equal(incompatible_model.state_dict(), before)


def test_sampler_membership_change_rejected_and_override_resets_offset(
    tmp_path: Path,
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    ckpt.finish_epoch()
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(2)
    ckpt.close()

    sampler2 = ResumableBatchSampler(
        15, 3, shuffle=True, seed=3, drop_last=True, dataset_fingerprint="toy-v1"
    )
    ckpt2, _, _, _, _, _, _ = make_run(tmp_path, sampler=sampler2)
    with pytest.raises(IncompatibleCheckpointError, match="sampler"):
        ckpt2.restore()

    with pytest.warns(CheckpointRecoveryWarning, match="batch offset reset"):
        info = ckpt2.restore(allow_incompatible=True)
    assert info is not None
    assert info.epoch == 1
    assert info.batch == 0
    assert sampler2.batch_offset == 0
    assert sampler2.total_batches == 5


def test_corrupt_latest_falls_back_with_warning(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    latest = ckpt.checkpoint_dir / "checkpoint-00000002.pt"
    latest.write_bytes(b"not a torch checkpoint")
    ckpt.close()

    ckpt2, model2, *_ = make_run(tmp_path)
    with pytest.warns(CheckpointRecoveryWarning, match="unusable"):
        info = ckpt2.restore()
    assert info is not None
    assert info.generation == 1
    assert info.batch == 1
    assert info.recovered_from_previous is True


def test_missing_sidecar_falls_back_with_warning(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    (ckpt.checkpoint_dir / "checkpoint-00000002.sha256").unlink()
    ckpt.close()

    ckpt2, *_ = make_run(tmp_path)
    with pytest.warns(CheckpointRecoveryWarning, match="missing checksum"):
        info = ckpt2.restore()
    assert info is not None
    assert info.generation == 1
    assert info.batch == 1


def test_both_generations_invalid_require_explicit_reset(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    for name in ("checkpoint-00000001.pt", "checkpoint-00000002.pt"):
        (ckpt.checkpoint_dir / name).write_bytes(b"broken")
    ckpt.mark_complete()
    ckpt.close()

    ckpt2, model2, *_ = make_run(tmp_path)
    before = copy.deepcopy(model2.state_dict())
    with pytest.raises(RecoveryError, match="No recoverable"):
        ckpt2.restore()
    assert_state_equal(model2.state_dict(), before)
    assert ckpt2.is_complete() is True

    ckpt2.reset()
    assert ckpt2.is_complete() is False
    assert not list(ckpt2.checkpoint_dir.glob("checkpoint-*"))
    assert ckpt2.restore() is None


def test_write_failure_warns_continues_and_keeps_prior_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(monkeypatch)
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    prior_files = sorted(p.name for p in ckpt.checkpoint_dir.glob("checkpoint-*.pt"))
    assert len(prior_files) == 1

    real_write = checkpoint_module._write_generation

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_module, "_write_generation", fail)
    with pytest.warns(CheckpointWriteWarning, match="disk full"):
        ckpt.commit(1)
    assert ckpt.progress.batch == 2
    assert sorted(p.name for p in ckpt.checkpoint_dir.glob("checkpoint-*.pt")) == prior_files
    # The prior generation is still the newest usable one (read without a second
    # owner: a run directory has exactly one writer).
    assert _batch_of(ckpt.checkpoint_dir, 1) == 1

    monkeypatch.setattr(checkpoint_module, "_write_generation", real_write)
    clock.advance(61)  # past the failed-write retry cadence
    ckpt.commit(1)
    assert _batch_of(ckpt.checkpoint_dir, 2) == 3


def test_atomic_replace_failure_leaves_prior_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    prior = sorted(p.name for p in ckpt.checkpoint_dir.glob("checkpoint-*.pt"))

    real_replace = os.replace

    def failing_replace(src: str, dst: str) -> None:
        if str(dst).endswith(".pt"):
            raise OSError("simulated rename failure")
        real_replace(src, dst)

    monkeypatch.setattr(checkpoint_module.os, "replace", failing_replace)
    with pytest.warns(CheckpointWriteWarning, match="rename failure"):
        ckpt.commit(1)
    assert sorted(p.name for p in ckpt.checkpoint_dir.glob("checkpoint-*.pt")) == prior
    assert not list(ckpt.checkpoint_dir.glob(".*.tmp"))
    ckpt.close()

    ckpt2, *_ = make_run(tmp_path)
    info = ckpt2.restore()
    assert info is not None
    assert info.batch == 1


def test_sidecar_rename_failure_removes_incomplete_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    prior = sorted(p.name for p in ckpt.checkpoint_dir.glob("checkpoint-*.pt"))

    real_replace = os.replace

    def failing_replace(src: str, dst: str) -> None:
        if str(dst).endswith(".sha256"):
            raise OSError("simulated sidecar rename failure")
        real_replace(src, dst)

    monkeypatch.setattr(checkpoint_module.os, "replace", failing_replace)
    with pytest.warns(CheckpointWriteWarning, match="sidecar rename failure"):
        ckpt.commit(1)
    assert sorted(p.name for p in ckpt.checkpoint_dir.glob("checkpoint-*.pt")) == prior
    assert not list(ckpt.checkpoint_dir.glob(".*.tmp"))
    ckpt.close()

    ckpt2, *_ = make_run(tmp_path)
    info = ckpt2.restore()
    assert info is not None
    assert info.batch == 1


def test_malformed_sampler_payload_falls_back(
    tmp_path: Path,
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    latest = ckpt.checkpoint_dir / "checkpoint-00000002.pt"
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    del payload["sampler"]["total_batches"]
    checkpoint_module._write_generation(ckpt.checkpoint_dir, 3, payload)
    ckpt.close()

    ckpt2, *_ = make_run(tmp_path)
    with pytest.warns(CheckpointRecoveryWarning, match="missing keys"):
        info = ckpt2.restore()
    assert info is not None
    assert info.generation == 2
    assert info.batch == 2


def test_rotation_keeps_two_recoverable_generations(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    for _ in range(4):
        training_step(model, optimizer, scheduler, component)
        ckpt.commit(1)
    assert len(list(ckpt.checkpoint_dir.glob("checkpoint-*.pt"))) == 2
    assert len(list(ckpt.checkpoint_dir.glob("checkpoint-*.sha256"))) == 2
    assert (ckpt.checkpoint_dir / "checkpoint-00000003.pt").is_file()
    assert (ckpt.checkpoint_dir / "checkpoint-00000004.pt").is_file()
    ckpt.close()

    ckpt2, *_ = make_run(tmp_path)
    info = ckpt2.restore()
    assert info is not None
    assert info.generation == 4
    assert info.batch == 4


def test_completion_marker_is_durable_and_resettable(tmp_path: Path) -> None:
    ckpt, *_ = make_run(tmp_path)
    assert ckpt.is_complete() is False
    ckpt.mark_complete()
    assert ckpt.is_complete() is True
    content = ckpt.complete_marker.read_text()
    assert '"epoch": 0' in content
    assert '"batch": 0' in content
    ckpt.close()

    ckpt2, *_ = make_run(tmp_path)
    assert ckpt2.is_complete() is True
    ckpt2.reset()
    assert ckpt2.is_complete() is False
    assert not ckpt2.complete_marker.exists()


def test_amp_skipped_update_still_commits_and_restores(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    sampler = ResumableBatchSampler(6, 2, shuffle=False, drop_last=True)
    ckpt = Checkpointer(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        sampler=sampler,
        interval_seconds=0,
    )

    def normal_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
            loss = model(torch.ones(4, 1)).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    normal_step()
    ckpt.commit(1)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        bad_loss = model(torch.ones(4, 1)).pow(2).mean()
    bad_loss = bad_loss * float("inf")
    scaler.scale(bad_loss).backward()
    before = scaler.get_scale()
    scaler.step(optimizer)  # skipped because of inf
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    after = scaler.get_scale()
    assert after < before
    ckpt.commit(1)
    assert ckpt.progress.batch == 2
    expected_scale = after
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    ckpt.close()

    model2 = torch.nn.Linear(1, 1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    scaler2 = torch.amp.GradScaler("cpu", enabled=True)
    sampler2 = ResumableBatchSampler(6, 2, shuffle=False, drop_last=True)
    ckpt2 = Checkpointer(
        tmp_path,
        model=model2,
        optimizer=optimizer2,
        scaler=scaler2,
        sampler=sampler2,
        interval_seconds=0,
    )
    info = ckpt2.restore()
    assert info is not None
    assert info.batch == 2
    assert scaler2.get_scale() == expected_scale
    assert_state_equal(optimizer2.state_dict(), expected_optimizer)

    # A restored skipped-scaler run can continue.
    optimizer2.zero_grad(set_to_none=True)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        loss = model2(torch.ones(4, 1)).pow(2).mean()
    scaler2.scale(loss).backward()
    scaler2.step(optimizer2)
    scaler2.update()
    optimizer2.zero_grad(set_to_none=True)
    ckpt2.commit(1)
    assert ckpt2.progress.batch == 3


def test_restore_with_no_checkpoint_is_first_run(tmp_path: Path) -> None:
    ckpt, *_ = make_run(tmp_path)
    assert ckpt.restore() is None
    assert ckpt.progress.epoch == 0
    assert ckpt.progress.batch == 0


# ---------------------------------------------------------------- retention
def _commit_steps(ckpt: Checkpointer, model, optimizer, scheduler, component, n: int = 1) -> None:
    for _ in range(n):
        training_step(model, optimizer, scheduler, component)
        ckpt.commit(1)


@pytest.mark.parametrize("lose_newest", ["delete", "corrupt"])
def test_fallback_then_save_then_lose_newest_still_recovers_older(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lose_newest: str
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component)
    gen1_model = copy.deepcopy(model.state_dict())
    _commit_steps(ckpt, model, optimizer, scheduler, component)
    directory = ckpt.checkpoint_dir
    ckpt.close()
    (directory / "checkpoint-00000002.pt").write_bytes(b"corrupt; sidecar left behind")

    ckpt2, model2, optimizer2, scheduler2, _, component2, _ = make_run(tmp_path)
    with pytest.warns(CheckpointRecoveryWarning, match="unusable"):
        info = ckpt2.restore()
    assert info is not None and info.generation == 1
    assert_state_equal(model2.state_dict(), gen1_model)

    # Continue and save.  Saving must not deserialize checkpoints, and the
    # corrupt newer generation must not cost us the good generation 1.
    def no_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("save() must not torch.load a checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_module.torch, "load", no_load)
        _commit_steps(ckpt2, model2, optimizer2, scheduler2, component2)
    assert (directory / "checkpoint-00000001.pt").is_file()
    assert (directory / "checkpoint-00000001.sha256").is_file()
    newest = directory / "checkpoint-00000003.pt"
    assert newest.is_file()
    ckpt2.close()

    if lose_newest == "delete":
        newest.unlink()
    else:
        newest.write_bytes(b"lost")
    ckpt3, model3, optimizer3, scheduler3, _, component3, _ = make_run(tmp_path)
    with pytest.warns(CheckpointRecoveryWarning):
        info3 = ckpt3.restore()
    assert info3 is not None and info3.generation == 1
    assert info3.recovered_from_previous is True
    assert_state_equal(model3.state_dict(), gen1_model)

    # Keep going: once two newer usable generations exist, bad ones are dropped.
    _commit_steps(ckpt3, model3, optimizer3, scheduler3, component3, 2)
    survivors = sorted(directory.glob("checkpoint-*.pt"))
    assert len(survivors) == 2
    for path in survivors:
        checkpoint_module._load_payload(path)  # both are loadable


def test_corruption_after_write_in_same_process_is_detected_at_prune(
    tmp_path: Path,
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component, 2)
    directory = ckpt.checkpoint_dir
    (directory / "checkpoint-00000002.pt").write_bytes(b"bit rot")
    _commit_steps(ckpt, model, optimizer, scheduler, component)
    assert (directory / "checkpoint-00000001.pt").is_file()  # not pruned
    _commit_steps(ckpt, model, optimizer, scheduler, component)
    assert sorted(p.name for p in directory.glob("checkpoint-*.pt")) == [
        "checkpoint-00000003.pt",
        "checkpoint-00000004.pt",
    ]


def test_prune_failure_keeps_every_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component, 2)
    real_unlink = Path.unlink

    def failing_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name.startswith("checkpoint-"):
            raise PermissionError("simulated")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CheckpointWriteWarning)  # save still succeeds
        _commit_steps(ckpt, model, optimizer, scheduler, component)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    directory = ckpt.checkpoint_dir
    assert len(list(directory.glob("checkpoint-*.pt"))) == 3
    ckpt.close()
    ckpt2, *_ = make_run(tmp_path)
    info = ckpt2.restore()
    assert info is not None and info.generation == 3 and info.batch == 3


def test_failed_save_after_fallback_keeps_good_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component, 2)
    directory = ckpt.checkpoint_dir
    ckpt.close()
    (directory / "checkpoint-00000002.pt").write_bytes(b"corrupt")
    ckpt2, model2, optimizer2, scheduler2, _, component2, _ = make_run(tmp_path)
    with pytest.warns(CheckpointRecoveryWarning):
        ckpt2.restore()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_module, "_write_generation", fail)
    with pytest.warns(CheckpointWriteWarning):
        _commit_steps(ckpt2, model2, optimizer2, scheduler2, component2)
    assert (directory / "checkpoint-00000001.pt").is_file()


# -------------------------------------------------------------- stale files
_HEX = "0123456789abcdef0123456789abcdef"


def test_restore_removes_only_exact_stale_files(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component, 2)
    directory = ckpt.checkpoint_dir
    ckpt.close()

    stale = [
        directory / f".checkpoint-00000003.pt.4242.{_HEX}.tmp",
        directory / f".checkpoint-00000003.sha256.4242.{_HEX}.tmp",
        directory / "checkpoint-00000009.sha256",  # sidecar without a .pt
        tmp_path / f".COMPLETE.4242.{_HEX}.tmp",
    ]
    unrelated = [
        directory / "notes.txt",
        directory / "other.tmp",
        directory / ".checkpoint-keep.tmp",
        directory / ".checkpoint-00000003.pt.4242.not-hex.tmp",
        directory / "checkpoint-00000002.pt.bak",
        directory / "checkpoint-00000002.sha256.bak",
        directory / "checkpoint-latest.pt",
        tmp_path / "COMPLETE.tmp",
        tmp_path / ".lock",
    ]
    for path in stale + unrelated:
        path.write_text("x")

    ckpt2, *_ = make_run(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # stale files are not recovery events
        info = ckpt2.restore()
    assert info is not None and info.generation == 2
    assert not any(path.exists() for path in stale)
    assert all(path.exists() for path in unrelated)
    assert (directory / "checkpoint-00000001.sha256").is_file()  # real sidecars kept


def test_stale_only_directory_is_a_first_run(tmp_path: Path) -> None:
    ckpt, *_ = make_run(tmp_path)
    orphan = ckpt.checkpoint_dir / "checkpoint-00000001.sha256"
    orphan.write_text("0" * 64 + "\n")
    assert ckpt.restore() is None
    assert not orphan.exists()


def test_failed_stale_cleanup_never_blocks_or_touches_good_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component, 2)
    directory = ckpt.checkpoint_dir
    ckpt.close()
    temp = directory / f".checkpoint-00000003.pt.1.{_HEX}.tmp"
    orphan = directory / "checkpoint-00000008.sha256"
    temp.write_text("x")
    orphan.write_text("x")

    real_unlink = Path.unlink

    def failing_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self in (temp, orphan):
            raise PermissionError("simulated")
        real_unlink(self, *args, **kwargs)

    ckpt2, model2, optimizer2, scheduler2, _, component2, _ = make_run(tmp_path)
    monkeypatch.setattr(Path, "unlink", failing_unlink)
    info = ckpt2.restore()
    assert info is not None and info.generation == 2
    _commit_steps(ckpt2, model2, optimizer2, scheduler2, component2)
    assert temp.exists() and orphan.exists()  # cleanup failed, nothing else did
    assert (directory / "checkpoint-00000003.pt").is_file()
    assert (directory / "checkpoint-00000002.pt").is_file()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    _commit_steps(ckpt2, model2, optimizer2, scheduler2, component2)  # retried
    assert not temp.exists() and not orphan.exists()
    assert len(list(directory.glob("checkpoint-*.pt"))) == 2


def test_pt_without_sidecar_is_never_counted_as_a_fallback(tmp_path: Path) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(tmp_path)
    _commit_steps(ckpt, model, optimizer, scheduler, component, 2)
    directory = ckpt.checkpoint_dir
    (directory / "checkpoint-00000002.sha256").unlink()
    _commit_steps(ckpt, model, optimizer, scheduler, component)  # generation 3
    assert (directory / "checkpoint-00000001.pt").is_file()  # still the fallback
    assert (directory / "checkpoint-00000002.pt").is_file()  # not deleted blindly


# ------------------------------------------------------------ retry cadence
def test_failed_writes_are_retried_at_bounded_cadence_and_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(monkeypatch)
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(
        tmp_path, dataset_size=3000, interval=300.0
    )
    attempts: list[float] = []
    real_write = checkpoint_module._write_generation

    def failing_write(*args: Any, **kwargs: Any) -> None:
        attempts.append(clock.now)
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_module, "_write_generation", failing_write)

    def commits(n: int) -> list[warnings.WarningMessage]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(n):
                ckpt.commit(1)
        return [w for w in caught if issubclass(w.category, CheckpointWriteWarning)]

    clock.advance(299)
    assert commits(20) == [] and attempts == []  # interval not yet due
    clock.advance(1)  # t=300: due
    assert len(commits(50)) == 1  # one attempt, not one per commit
    assert attempts == [1300.0]
    clock.advance(59)
    assert commits(50) == [] and len(attempts) == 1
    clock.advance(1)  # 60s after the failure
    assert len(commits(50)) == 1 and len(attempts) == 2
    assert not list(ckpt.checkpoint_dir.glob("checkpoint-*"))
    assert ckpt.progress.batch == 170  # training progressed throughout

    # The write recovers; the next attempt is allowed after the cadence and the
    # interval clock now counts from that last *successful* save.
    monkeypatch.setattr(checkpoint_module, "_write_generation", real_write)
    clock.advance(60)
    assert commits(1) == []
    assert _batch_of(ckpt.checkpoint_dir, 1) == 171
    saved_at = clock.now
    clock.advance(299)
    ckpt.commit(1)
    assert _batch_of(ckpt.checkpoint_dir, 1) == 171  # interval measured from success
    clock.advance(1)
    ckpt.commit(1)
    assert _batch_of(ckpt.checkpoint_dir, 2) == 173
    assert clock.now - saved_at == 300


def test_failure_cadence_also_bounds_interval_zero_and_epoch_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(monkeypatch)
    ckpt, *_ = make_run(tmp_path, dataset_size=300, batch_size=3, interval=0.0)
    count = 0

    def failing_write(*args: Any, **kwargs: Any) -> None:
        nonlocal count
        count += 1
        raise OSError("read-only filesystem")

    monkeypatch.setattr(checkpoint_module, "_write_generation", failing_write)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CheckpointWriteWarning)
        for _ in range(10):
            ckpt.commit(1)
        ckpt.finish_epoch()
        assert count == 1
        clock.advance(60)
        ckpt.commit(1)
        assert count == 2
        assert ckpt.save() is False  # explicit saves are never throttled
        assert count == 3


# --------------------------------------------------------- epoch boundaries
def _plain_loop(
    ckpt: Checkpointer,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    epochs: int,
    *,
    crash_at: tuple[int, int] | None = None,
    epoch_end: dict[int, dict[str, Any]] | None = None,
) -> None:
    for epoch in range(ckpt.epoch, epochs):
        for index, (x, y) in enumerate(loader):
            if crash_at == (epoch, index):
                raise KeyboardInterrupt  # process dies mid-epoch
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(x), y).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ckpt.commit(1)
        ckpt.finish_epoch()
        if epoch_end is not None:
            epoch_end[epoch + 1] = copy.deepcopy(model.state_dict())


def _plain_run(run_dir: Path) -> tuple[Checkpointer, torch.nn.Module, torch.optim.Optimizer, Any]:
    torch.manual_seed(0)
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    ckpt = Checkpointer(run_dir, model=model, optimizer=optimizer, interval_seconds=0)
    generator = torch.Generator().manual_seed(1)
    data = torch.utils.data.TensorDataset(
        torch.randn(12, 2, generator=generator), torch.randn(12, 1, generator=generator)
    )
    return ckpt, model, optimizer, torch.utils.data.DataLoader(data, batch_size=3, shuffle=False)


def test_without_sampler_recovery_is_epoch_boundary_and_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    ref_ckpt, ref_model, ref_opt, ref_loader = _plain_run(tmp_path / "reference")
    _plain_loop(ref_ckpt, ref_model, ref_opt, ref_loader, 3)
    ref_ckpt.close()

    run_dir = tmp_path / "run"
    ckpt, model, optimizer, loader = _plain_run(run_dir)
    epoch_end: dict[int, dict[str, Any]] = {}
    with pytest.raises(KeyboardInterrupt):
        _plain_loop(ckpt, model, optimizer, loader, 3, crash_at=(1, 2), epoch_end=epoch_end)
    assert ckpt.progress.epoch == 1 and ckpt.progress.batch == 2  # trained mid-epoch
    # Two mid-epoch commits (interval 0) wrote nothing: only epoch 0's boundary.
    directory = ckpt.checkpoint_dir
    assert [p.name for p in directory.glob("checkpoint-*.pt")] == ["checkpoint-00000001.pt"]
    assert _batch_of(directory, 1) == 0
    ckpt.close()

    ckpt2, model2, optimizer2, loader2 = _plain_run(run_dir)
    info = ckpt2.restore()
    assert info is not None and (info.epoch, info.batch) == (1, 0)
    # Weights are the end-of-epoch-0 weights, not the mid-epoch-1 weights, so
    # replaying epoch 1 from its first batch is consistent.
    assert_state_equal(model2.state_dict(), epoch_end[1])
    first_batch = next(iter(loader2))[0]
    assert torch.equal(first_batch, next(iter(ref_loader))[0])
    _plain_loop(ckpt2, model2, optimizer2, loader2, 3)
    assert_state_equal(model2.state_dict(), ref_model.state_dict())
    assert_state_equal(optimizer2.state_dict(), ref_opt.state_dict())
    # Progress equals the uninterrupted run: abandoned mid-epoch work is not counted.
    assert ckpt2.progress == ref_ckpt.progress
    ckpt2.close()


def test_without_sampler_mid_epoch_explicit_save_is_refused(tmp_path: Path) -> None:
    ckpt, model, optimizer, _ = _plain_run(tmp_path)
    ckpt.commit(1)
    with pytest.raises(CheckpointError, match="mid-epoch"):
        ckpt.save()
    assert not list(ckpt.checkpoint_dir.glob("checkpoint-*"))
    ckpt.finish_epoch()  # boundary: auto-saved
    assert len(list(ckpt.checkpoint_dir.glob("checkpoint-*.pt"))) == 1
    assert ckpt.save() is True


def test_without_sampler_rejects_legacy_mid_epoch_checkpoint(tmp_path: Path) -> None:
    ckpt, *_ = _plain_run(tmp_path)
    ckpt._progress = dataclasses.replace(ckpt.progress, batch=2)  # as v0.1.0 could save
    checkpoint_module._write_generation(ckpt.checkpoint_dir, 1, ckpt._build_payload())
    ckpt.close()
    ckpt2, *_ = _plain_run(tmp_path)
    with pytest.raises(IncompatibleCheckpointError, match="mid-epoch"):
        ckpt2.restore()
