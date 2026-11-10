from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from revive_me import (
    Checkpointer,
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

    ckpt2, *_ = make_run(tmp_path)
    info = ckpt2.restore()
    assert info is not None
    assert info.batch == 1

    monkeypatch.setattr(checkpoint_module, "_write_generation", real_write)
    ckpt.commit(1)
    # The prior failure did not update the interval clock; the next commit saved.
    ckpt3, *_ = make_run(tmp_path)
    info3 = ckpt3.restore()
    assert info3 is not None
    assert info3.batch == 3


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
