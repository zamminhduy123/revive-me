"""In-process lifecycle tests for the ``@resumable`` decorator and ``Run`` handle."""

from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

import revive_me.checkpoint as checkpoint_module
from conftest import Counter, FakeScaler, ToyModel
from revive_me import (
    Checkpointer,
    CheckpointError,
    CheckpointLockedError,
    CheckpointRecoveryWarning,
    CheckpointWriteWarning,
    IncompatibleCheckpointError,
    RecoveryError,
    ResumableBatchSampler,
    RunIncompleteWarning,
    RunStateError,
    resumable,
)

SEED = 5


class IdDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return torch.tensor([index / 10.0, 1.0]), index


def cfg(run_dir: Path, **overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = dict(
        dir=run_dir,
        epochs=2,
        size=12,
        batch=3,
        accum=1,
        compat={"arch": "toy-v1"},
        fresh=False,
        retry=60.0,
        # misbehaviour switches
        fail_after_commits=None,
        exc=RuntimeError,
        skip_last_commit=False,
        no_commit=False,
        inner_break_after=None,
        outer_break_after_epochs=None,
        return_after_commits=None,
        never_epochs=False,
        complete_before_epochs=False,
        nested=None,
        # observations
        entered=0,
        seen=[],  # list of (epoch, [ids]) per delivered batch
        windows=[],
        lens=[],
        epoch_end=[],
        resumed="unset",
        weights=None,
        last_idiom=[],
        nested_result=None,
    )
    values.update(overrides)
    ns = SimpleNamespace(**values)
    for key in ("seen", "windows", "lens", "epoch_end", "last_idiom"):
        setattr(ns, key, list(getattr(ns, key)))
    return ns


@resumable(
    run_dir=lambda c: c.dir,
    interval_seconds=0.0,
    retry_seconds=lambda c: c.retry,
    fresh=lambda c: c.fresh,
)
def train(c: SimpleNamespace, *, run):
    c.entered += 1
    torch.manual_seed(0)
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    counter = Counter()
    run.register(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=FakeScaler(),
        components={"counter": counter},
        compatibility=c.compat,
        metadata={"who": "test"},
    )
    loader = run.dataloader(
        IdDataset(c.size),
        batch_size=c.batch,
        shuffle=True,
        seed=SEED,
        drop_last=True,
        dataset_fingerprint="id-v1",
    )
    if c.never_epochs:
        return "never"
    c.lens.append(("pre", len(loader), loader.total_batches))
    pre = model.net[0].weight.detach().clone()
    if c.complete_before_epochs:
        c.dir.mkdir(parents=True, exist_ok=True)
        (c.dir / "COMPLETE").write_text("raced\n")
    epochs = run.epochs(c.epochs)
    c.weights = (pre, model.net[0].weight.detach().clone())
    c.resumed = run.resumed
    commits = 0
    for epoch in epochs:
        c.lens.append((epoch, len(loader), loader.total_batches))
        if c.nested is not None and epoch == 0:
            try:
                c.nested_result = train(c.nested)
            except Exception as exc:  # recorded and asserted by the test
                c.nested_result = exc
        pending = 0
        for i, (x, ids) in enumerate(loader):
            c.seen.append((epoch, ids.tolist()))
            if i == len(loader) - 1:
                c.last_idiom.append(epoch)
            if pending == 0:
                optimizer.zero_grad(set_to_none=True)
            model(x).pow(2).mean().backward()
            pending += 1
            if pending == c.accum:
                optimizer.step()
                scheduler.step()
                counter.count += 1
                if not c.no_commit:
                    run.commit(pending)
                    c.windows.append(pending)
                    commits += 1
                pending = 0
                if c.fail_after_commits == commits:
                    raise c.exc("boom")
                if c.return_after_commits == commits:
                    return "early"
            if c.inner_break_after is not None and i + 1 >= c.inner_break_after:
                break
        if pending and not c.skip_last_commit:
            optimizer.step()
            scheduler.step()
            counter.count += 1
            run.commit(pending)
            c.windows.append(pending)
            commits += 1
        c.epoch_end.append(epoch)
        if c.outer_break_after_epochs == len(c.epoch_end):
            break
    return {"epochs": c.epochs, "ids": c.seen}


def expected_batches(size: int = 12, batch: int = 3, epochs: int = 2):
    out = []
    for epoch in range(epochs):
        sampler = ResumableBatchSampler(size, batch, shuffle=True, seed=SEED)
        sampler.set_epoch(epoch)
        out.extend((epoch, ids) for ids in sampler)
    return out


def generations(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "checkpoints").glob("checkpoint-*.pt"))


def saved_progress(run_dir: Path) -> dict[str, int]:
    return torch.load(generations(run_dir)[-1], weights_only=False)["progress"]


def assert_released(run_dir: Path) -> None:
    Checkpointer(run_dir, model=ToyModel()).close()  # raises if a lock is held


def snapshot_files(run_dir: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(run_dir)): p.read_bytes()
        for p in run_dir.rglob("*")
        if p.is_file() and p.name != ".checkpointer.lock"
    }


# ----------------------------------------------------------------- happy path
def test_initial_run_returns_result_completes_and_releases_lock(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run")
    result = train(c)
    assert result == {"epochs": 2, "ids": c.seen}
    assert c.seen == expected_batches()
    assert c.resumed is None
    assert (c.dir / "COMPLETE").is_file()
    assert saved_progress(c.dir) == {
        "epoch": 2,
        "batch": 0,
        "total_batches": 4,
        "batches_seen": 8,
    }
    assert_released(c.dir)


def test_accumulation_with_short_final_window_commits_actual_counts(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", size=15, accum=2)  # 5 batches/epoch -> windows 2,2,1
    train(c)
    assert c.windows == [2, 2, 1, 2, 2, 1]
    assert saved_progress(c.dir)["batches_seen"] == 10
    assert (c.dir / "COMPLETE").is_file()


def test_sequential_invocations_with_distinct_dirs_are_isolated(tmp_path: Path) -> None:
    first, second = cfg(tmp_path / "a"), cfg(tmp_path / "b", epochs=1)
    train(first)
    train(second)
    assert first.resumed is None and second.resumed is None
    assert saved_progress(first.dir)["epoch"] == 2
    assert saved_progress(second.dir)["epoch"] == 1
    assert second.seen == expected_batches(epochs=1)
    assert_released(first.dir)
    assert_released(second.dir)


def test_missing_marker_with_final_checkpoint_completes_without_replay(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    train(cfg(run_dir))
    (run_dir / "COMPLETE").unlink()  # crash between final save and marker
    again = cfg(run_dir)
    train(again)
    assert again.seen == []  # nothing replayed
    assert (run_dir / "COMPLETE").is_file()


# ------------------------------------------------------------ already complete
def test_already_complete_skips_body_and_never_touches_files(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run")
    train(c)
    before = snapshot_files(c.dir)
    again = cfg(c.dir)
    assert train(again) is None
    assert again.entered == 0
    assert snapshot_files(c.dir) == before


def test_complete_marker_with_missing_checkpoints_is_not_overwritten(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run")
    train(c)
    for path in generations(c.dir):
        path.unlink()
        path.with_suffix(".sha256").unlink()
    again = cfg(c.dir)
    assert train(again) is None
    assert again.entered == 0
    assert generations(c.dir) == []
    assert (c.dir / "COMPLETE").is_file()


def test_complete_appearing_after_lock_is_taken_skips_training(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", complete_before_epochs=True)
    assert train(c) is None
    assert c.entered == 1 and c.seen == []
    assert generations(c.dir) == []
    assert_released(c.dir)


def test_fresh_deletes_complete_run_and_starts_over(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run")
    train(c)
    again = cfg(c.dir, fresh=True)
    train(again)
    assert again.resumed is None
    assert again.seen == expected_batches()
    assert (c.dir / "COMPLETE").is_file()
    assert saved_progress(c.dir)["batches_seen"] == 8


# ------------------------------------------------------- decorator contract
def test_signature_name_and_docstring_are_preserved_without_handle(tmp_path: Path) -> None:
    @resumable(run_dir=tmp_path / "r")
    def fn(a, b=2, *, run):
        """doc"""

    assert list(inspect.signature(fn).parameters) == ["a", "b"]
    assert fn.__name__ == "fn" and fn.__doc__ == "doc"
    assert fn.__wrapped__ is not None


def test_reserved_handle_and_bad_calls_fail_before_any_io(tmp_path: Path) -> None:
    ran = []
    run_dir = tmp_path / "never"

    @resumable(run_dir=lambda a: ran.append("dir") or run_dir)
    def fn(a, *, run):
        ran.append("body")

    with pytest.raises(TypeError, match="reserved"):
        fn(1, run=object())
    with pytest.raises(TypeError):
        fn()  # missing argument
    with pytest.raises(TypeError):
        fn(1, 2)  # surplus positional
    assert ran == [] and not run_dir.exists()


def test_decoration_requires_keyword_only_run_and_a_plain_function(tmp_path: Path) -> None:
    for build in (
        lambda: resumable(run_dir=tmp_path)(lambda a: None),
        lambda: resumable(run_dir=tmp_path)(lambda a, run: None),
    ):
        with pytest.raises(TypeError, match="keyword-only"):
            build()

    def gen(*, run):
        yield 1

    with pytest.raises(TypeError, match="generators"):
        resumable(run_dir=tmp_path)(gen)


def test_run_dir_callable_works_with_keyword_call_and_path_value(tmp_path: Path) -> None:
    def tiny(run):
        run.register(model=ToyModel())
        loader = run.dataloader(IdDataset(4), batch_size=2)
        for _ in run.epochs(1):
            for _batch in loader:
                pass
            run.commit(2)
        return "done"

    @resumable(run_dir=lambda args: args.dir, interval_seconds=0)
    def by_callable(args, *, run):
        return tiny(run)

    @resumable(run_dir=tmp_path / "static", interval_seconds=0)
    def by_value(*, run):
        return tiny(run)

    # the callable receives the call arguments, positionally or by keyword
    assert by_callable(SimpleNamespace(dir=tmp_path / "x")) == "done"
    assert by_callable(args=SimpleNamespace(dir=tmp_path / "y")) == "done"
    assert by_value() == "done"
    for name in ("x", "y", "static"):
        assert (tmp_path / name / "COMPLETE").is_file()


# ----------------------------------------------------- abnormal exits
@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_exception_saves_nothing_extra_releases_lock_and_resumes(
    tmp_path: Path, exc: type[BaseException]
) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=3, exc=exc)
    with pytest.raises(exc):
        train(c)
    assert not (c.dir / "COMPLETE").exists()
    # interval 0 saves at each commit; the exception added no unsafe final save
    assert saved_progress(c.dir) == {
        "epoch": 0,
        "batch": 3,
        "total_batches": 4,
        "batches_seen": 3,
    }
    assert len(c.seen) == 3
    assert_released(c.dir)

    resumed = cfg(c.dir)
    train(resumed)
    assert resumed.resumed.batch == 3
    assert c.seen + resumed.seen == expected_batches()
    assert (c.dir / "COMPLETE").is_file()


def test_early_return_warns_leaves_run_incomplete_and_resumes(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", return_after_commits=2)
    with pytest.warns(RunIncompleteWarning):
        assert train(c) == "early"
    assert not (c.dir / "COMPLETE").exists()
    assert saved_progress(c.dir)["batch"] == 2
    assert len(generations(c.dir)) == 2  # one per commit, no final save
    assert_released(c.dir)

    resumed = cfg(c.dir)
    train(resumed)
    assert c.seen + resumed.seen == expected_batches()


def test_outer_break_after_epoch_body_is_incomplete_and_epoch_end_reruns(
    tmp_path: Path,
) -> None:
    c = cfg(tmp_path / "run", outer_break_after_epochs=1)
    with pytest.warns(RunIncompleteWarning):
        train(c)
    assert not (c.dir / "COMPLETE").exists()
    assert saved_progress(c.dir)["epoch"] == 0  # finish_epoch was never reached
    assert_released(c.dir)

    resumed = cfg(c.dir)
    train(resumed)
    # epoch 0 is yielded again with nothing left to consume, so epoch-end work
    # runs again rather than being skipped
    assert resumed.epoch_end == [0, 1]
    assert [e for e, _ in resumed.seen] == [1] * 4
    assert c.seen + resumed.seen == expected_batches()


def test_function_that_never_starts_the_epoch_loop_is_an_error(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", never_epochs=True)
    with pytest.raises(RunStateError, match="without calling run.epochs"):
        train(c)
    assert not (c.dir / "COMPLETE").exists()
    assert_released(c.dir)


# ------------------------------------------------ epoch boundary enforcement
@pytest.mark.parametrize(
    "overrides",
    [
        dict(no_commit=True),  # never committed
        dict(size=15, accum=2, skip_last_commit=True),  # partial final window
        dict(inner_break_after=2),  # break inner loader, outer loop continues
    ],
    ids=["missing-commit", "partial-window", "inner-break"],
)
def test_epoch_cannot_finish_with_uncommitted_or_unconsumed_batches(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    c = cfg(tmp_path / "run", **overrides)
    with pytest.raises(RunStateError, match="batches committed"):
        train(c)
    assert not (c.dir / "COMPLETE").exists()
    assert_epoch_never_advanced(c.dir)
    assert_released(c.dir)


def assert_epoch_never_advanced(run_dir: Path) -> None:
    for path in generations(run_dir):
        assert torch.load(path, weights_only=False)["progress"]["epoch"] == 0


def test_breaking_at_the_last_batch_after_committing_it_is_accepted(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", epochs=1, inner_break_after=4)
    train(c)
    assert (c.dir / "COMPLETE").is_file()


@pytest.mark.parametrize(
    "mode, match",
    [
        ("commit-before-epochs", "before run.epochs"),
        ("iterate-before-epochs", "before run.epochs"),
        ("commit-ahead", "delivered only 1"),
        ("second-iteration", "already iterated"),
        ("register-late", "before run.epochs"),
        ("epochs-twice", "before run.epochs"),
    ],
)
def test_misuse_is_rejected_with_a_clear_error(
    tmp_path: Path, mode: str, match: str
) -> None:
    @resumable(run_dir=tmp_path / "r", interval_seconds=0)
    def fn(*, run):
        model = ToyModel()
        if mode == "commit-before-epochs":
            run.commit(1)
        run.register(model=model)
        loader = run.dataloader(IdDataset(8), batch_size=2)
        if mode == "iterate-before-epochs":
            iter(loader)
        epochs = run.epochs(1)
        for _ in epochs:
            if mode == "commit-ahead":
                next(iter(loader))
                run.commit(2)
            elif mode == "second-iteration":
                list(loader)
                iter(loader)
            elif mode == "register-late":
                run.register(model=model)
            elif mode == "epochs-twice":
                run.epochs(1)

    with pytest.raises(RunStateError, match=match):
        fn()
    assert not (tmp_path / "r" / "COMPLETE").exists()
    assert_released(tmp_path / "r")


def test_commit_outside_the_epoch_body_is_rejected(tmp_path: Path) -> None:
    @resumable(run_dir=tmp_path / "r", interval_seconds=0)
    def fn(*, run):
        run.register(model=ToyModel())
        loader = run.dataloader(IdDataset(4), batch_size=2)
        for _ in run.epochs(1):
            list(loader)
            run.commit(2)
        run.commit(1)  # loop finished: no epoch is open

    with pytest.raises(RunStateError, match="inside the run.epochs"):
        fn()
    assert not (tmp_path / "r" / "COMPLETE").exists()  # exception: no final save


def test_register_and_dataloader_validation(tmp_path: Path) -> None:
    @resumable(run_dir=tmp_path / "r")
    def fn(mode, *, run):
        if mode == "twice":
            run.register(model=ToyModel())
            run.register(model=ToyModel())
        elif mode == "bad-model":
            run.register(model=object())
        elif mode == "no-register":
            run.dataloader(IdDataset(4), batch_size=2)
            run.epochs(1)
        elif mode == "no-loader":
            run.register(model=ToyModel())
            run.epochs(1)
        elif mode == "two-loaders":
            run.dataloader(IdDataset(4), batch_size=2)
            run.dataloader(IdDataset(4), batch_size=2)
        elif mode == "iterable":

            class Stream(IterableDataset):
                def __iter__(self):
                    yield 1

            run.dataloader(Stream(), batch_size=2)

    for mode, error in (
        ("twice", RunStateError),
        ("bad-model", TypeError),
        ("no-register", RunStateError),
        ("no-loader", RunStateError),
        ("two-loaders", RunStateError),
        ("iterable", TypeError),
    ):
        with pytest.raises(error):
            fn(mode)
    assert_released(tmp_path / "r")


# -------------------------------------------------- restore order and lengths
def test_restore_applies_at_epochs_after_sampler_known_before_iterator(
    tmp_path: Path,
) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=2)
    with pytest.raises(RuntimeError):
        train(c)
    saved = torch.load(generations(c.dir)[-1], weights_only=False)

    resumed = cfg(c.dir)
    train(resumed)
    pre, post = resumed.weights
    assert not torch.equal(pre, post)  # register/dataloader restored nothing
    assert torch.equal(post, saved["models"]["main"]["net.0.weight"])
    # the first delivered batch is exactly the committed offset, so the sampler
    # state was in place before the DataLoader iterator was created
    assert resumed.seen[0] == expected_batches()[2]
    assert c.seen + resumed.seen == expected_batches()


def test_len_is_snapshotted_per_epoch_and_idiom_last_batch_works(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=1)
    with pytest.raises(RuntimeError):
        train(c)
    resumed = cfg(c.dir)
    train(resumed)
    # before restore: full epoch; resumed epoch 0: remaining 3 of 4; epoch 1: 4
    assert resumed.lens == [("pre", 4, 4), (0, 3, 4), (1, 4, 4)]
    assert resumed.last_idiom == [0, 1]
    assert c.last_idiom == []  # killed before the last batch


def test_restored_end_of_epoch_yields_empty_epoch_with_zero_length(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=4)  # all of epoch 0 committed
    with pytest.raises(RuntimeError):
        train(c)
    assert saved_progress(c.dir)["batch"] == 4
    resumed = cfg(c.dir)
    train(resumed)
    assert resumed.lens[1] == (0, 0, 4)
    assert resumed.epoch_end == [0, 1]
    assert [e for e, _ in resumed.seen] == [1] * 4


# ------------------------------------------------------------ failed writes
def test_transient_write_failures_warn_and_training_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = checkpoint_module._write_generation
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "_write_generation", flaky)
    c = cfg(tmp_path / "run", retry=0.0)
    with pytest.warns(CheckpointWriteWarning):
        result = train(c)
    assert result is not None and c.seen == expected_batches()
    assert (c.dir / "COMPLETE").is_file()
    assert saved_progress(c.dir)["epoch"] == 2


@pytest.mark.filterwarnings("ignore::revive_me.CheckpointWriteWarning")
def test_failing_final_write_does_not_mark_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fail(*args: Any, **kwargs: Any):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(checkpoint_module, "_write_generation", always_fail)
    c = cfg(tmp_path / "run")
    with pytest.warns(CheckpointWriteWarning, match="NOT marked complete"):
        assert train(c) is not None  # training itself ran to the end
    assert c.seen == expected_batches()
    assert not (c.dir / "COMPLETE").exists()
    assert generations(c.dir) == []
    assert_released(c.dir)


# ------------------------------------------------ corrupt/incompatible restore
def corrupt(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-40] + b"x" * 40)


def test_corrupt_newest_generation_falls_back_through_wrapper(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=3)
    with pytest.raises(RuntimeError):
        train(c)
    corrupt(generations(c.dir)[-1])

    resumed = cfg(c.dir)
    with pytest.warns(CheckpointRecoveryWarning):
        train(resumed)
    assert resumed.resumed.recovered_from_previous
    assert resumed.resumed.batch == 2
    assert (c.dir / "COMPLETE").is_file()


def test_all_invalid_checkpoints_refuse_fresh_start_and_change_nothing(
    tmp_path: Path,
) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=3)
    with pytest.raises(RuntimeError):
        train(c)
    for path in generations(c.dir):
        corrupt(path)
    before = snapshot_files(c.dir)

    again = cfg(c.dir)
    with pytest.warns(CheckpointRecoveryWarning), pytest.raises(RecoveryError):
        train(again)
    assert again.seen == []
    assert snapshot_files(c.dir) == before
    assert not (c.dir / "COMPLETE").exists()
    assert_released(c.dir)


def test_incompatible_checkpoint_is_refused(tmp_path: Path) -> None:
    c = cfg(tmp_path / "run", fail_after_commits=2)
    with pytest.raises(RuntimeError):
        train(c)
    other = cfg(c.dir, compat={"arch": "different"})
    with pytest.raises(IncompatibleCheckpointError):
        train(other)
    assert other.seen == []
    assert_released(c.dir)


# ------------------------------------------------------- locks and ownership
def test_nested_same_dir_is_rejected_different_dir_is_independent(tmp_path: Path) -> None:
    same = cfg(tmp_path / "run")
    same.nested = cfg(same.dir)
    train(same)
    assert isinstance(same.nested_result, CheckpointLockedError)
    assert same.nested.seen == []
    assert (same.dir / "COMPLETE").is_file()  # outer unaffected

    other = cfg(tmp_path / "outer")
    other.nested = cfg(tmp_path / "inner")
    train(other)
    assert other.nested_result is not None
    assert other.nested.seen == expected_batches()
    assert (tmp_path / "inner" / "COMPLETE").is_file()
    assert (tmp_path / "outer" / "COMPLETE").is_file()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_inherited_run_in_forked_child_cannot_commit(tmp_path: Path) -> None:
    @resumable(run_dir=tmp_path / "run", interval_seconds=0)
    def fn(*, run):
        run.register(model=ToyModel())
        loader = run.dataloader(IdDataset(4), batch_size=2)
        for _ in run.epochs(1):
            for _batch in loader:
                pid = os.fork()
                if pid == 0:
                    try:
                        run.commit(1)
                        code = 1
                    except CheckpointError:
                        code = 0
                    os._exit(code)
                _, status = os.waitpid(pid, 0)
                assert os.waitstatus_to_exitcode(status) == 0
                run.commit(1)
        return "ok"

    assert fn() == "ok"
    assert (tmp_path / "run" / "COMPLETE").is_file()
