"""Fresh-process interruption/continuation through the ``@resumable`` decorator."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from conftest import assert_state_equal
from test_recovery_subprocess import PROJECT_ROOT, _base_args, _env, _expected_ids, _load

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
DECORATED = FIXTURES / "decorated_fixture.py"
LOW_LEVEL = FIXTURES / "train_fixture.py"
STATE_KEYS = ("model", "optimizer", "scheduler", "scaler", "component")


def _run(fixture: Path, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
        done = subprocess.run(
            [sys.executable, str(fixture), *args],
            cwd=PROJECT_ROOT,
            env=_env(),
            stdout=out,
            stderr=err,
            text=True,
            timeout=timeout,
        )
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(done.args, done.returncode, out.read(), err.read())


def _common(workers: int, seed: int) -> dict[str, object]:
    return {
        "epochs": 3,
        "dataset-size": 30,
        "batch-size": 5,
        "accum": 4,
        "workers": workers,
        "seed": seed,
        "save-interval": 0,
        "no-shuffle": False,
    }


def _assert_continuation_matches_uninterrupted(
    tmp_path: Path, common: dict[str, object], kill: dict[str, object], seed: int
) -> tuple[dict, dict]:
    expected_ids = _expected_ids(30, 5, 3, seed, shuffle=True)

    ref_path = tmp_path / "uninterrupted.pt"
    done = _run(DECORATED, _base_args(tmp_path / "uninterrupted", **common, reference=ref_path))
    assert done.returncode == 0 and "result=trained" in done.stdout, done.stderr
    reference = _load(ref_path)

    run_dir = tmp_path / "interrupted"
    prefix_path = tmp_path / "interrupted.pt"
    killed = _run(DECORATED, _base_args(run_dir, **common, **kill, reference=prefix_path))
    assert killed.returncode == 17, killed.stderr
    assert not (run_dir / "COMPLETE").exists()
    newest = sorted((run_dir / "checkpoints").glob("*.pt"))[-1]
    durable = torch.load(newest, weights_only=False)["progress"]

    suffix_path = tmp_path / "resumed.pt"
    resumed = _run(DECORATED, _base_args(run_dir, **common, reference=suffix_path))
    assert resumed.returncode == 0 and "result=trained" in resumed.stdout, resumed.stderr
    assert (run_dir / "COMPLETE").is_file()
    prefix, suffix = _load(prefix_path), _load(suffix_path)
    prefix["durable"] = (durable["epoch"], durable["batch"], durable["batches_seen"])

    assert reference["progress"] == {"epoch": 3, "batch": 0, "batches_seen": 18}
    assert suffix["progress"] == reference["progress"]
    assert prefix["ids_seen"] + suffix["ids_seen"] == expected_ids
    assert suffix["ids_seen"] == expected_ids[len(prefix["ids_seen"]) :]
    assert reference["component"]["window_sizes"] == [4, 2, 4, 2, 4, 2]
    assert reference["scheduler"]["last_epoch"] == 6
    assert any("momentum_buffer" in s for s in reference["optimizer"]["state"].values())
    for key in STATE_KEYS:  # deterministic CPU fixture: no dropout / random transforms
        assert_state_equal(suffix[key], reference[key])
    return prefix, suffix


@pytest.mark.parametrize("workers", [0, 2])
def test_decorated_kill_after_commit_then_continue_matches_uninterrupted(
    tmp_path: Path, workers: int
) -> None:
    prefix, _ = _assert_continuation_matches_uninterrupted(
        tmp_path, _common(workers, 13), {"kill-after-commits": 5}, 13
    )
    # epoch 2, four batches in; the short final window had not been committed
    assert prefix["progress"] == {"epoch": 2, "batch": 4, "batches_seen": 16}
    assert prefix["component"]["window_sizes"] == [4, 2, 4, 2, 4]
    assert prefix["durable"] == (2, 4, 16)


def test_decorated_mid_window_kill_with_prefetch_workers_continues_correctly(
    tmp_path: Path,
) -> None:
    prefix, _ = _assert_continuation_matches_uninterrupted(
        tmp_path,
        _common(2, 21),
        {"kill-after-commits": 2, "kill-mid-after": 2},
        21,
    )
    # Killed two microbatches into epoch 1's first window.  The snapshot is the
    # last commit; the durable cursor is the epoch boundary (finish_epoch ran),
    # not the prefetched or half-accumulated position.
    assert prefix["progress"] == {"epoch": 0, "batch": 6, "batches_seen": 6}
    assert prefix["durable"] == (1, 0, 6)


def test_decorated_uninterrupted_run_equals_low_level_run(tmp_path: Path) -> None:
    common = _common(2, 17)
    high, low = tmp_path / "high.pt", tmp_path / "low.pt"
    assert _run(DECORATED, _base_args(tmp_path / "h", **common, reference=high)).returncode == 0
    assert _run(LOW_LEVEL, _base_args(tmp_path / "l", **common, reference=low)).returncode == 0
    high_state, low_state = _load(high), _load(low)
    assert high_state["progress"] == low_state["progress"]
    assert high_state["ids_seen"] == low_state["ids_seen"]
    for key in STATE_KEYS:
        assert_state_equal(high_state[key], low_state[key])


def test_decorated_restore_positions_sampler_before_the_first_iterator(tmp_path: Path) -> None:
    common = _common(2, 7)
    run_dir = tmp_path / "run"
    killed = _run(
        DECORATED,
        _base_args(run_dir, **common, **{"kill-after-commits": 1, "kill-mid-after": 2}),
    )
    assert killed.returncode == 17, killed.stderr
    before = sorted(p.name for p in (run_dir / "checkpoints").iterdir())

    # epoch 0 has 6 batches (windows of 4 then 2); the kill landed mid-window
    # after the first full window of epoch 0 committed 4 batches.
    observe = tmp_path / "observe.pt"
    inspected = _run(DECORATED, _base_args(run_dir, **common, inspect=True, observe=observe))
    assert inspected.returncode == 0 and "result=inspected" in inspected.stdout, inspected.stderr
    observed = _load(observe)
    order = torch.randperm(30, generator=torch.Generator().manual_seed(7)).tolist()
    assert observed["progress"] == {"epoch": 0, "batch": 4, "batches_seen": 4}
    assert observed["first_ids"] == order[20:25]
    assert observed["len"] == 2  # remaining batches; total stays 6
    assert not (run_dir / "COMPLETE").exists()  # returning early never completes
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == before


def test_decorated_complete_run_skips_and_fresh_restarts_explicitly(tmp_path: Path) -> None:
    common = _common(0, 3)
    run_dir = tmp_path / "run"
    first = _run(DECORATED, _base_args(run_dir, **common))
    assert first.returncode == 0 and "result=trained" in first.stdout, first.stderr
    files = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file() and p.name != ".checkpointer.lock"}

    again = _run(DECORATED, _base_args(run_dir, **common))
    assert again.returncode == 0 and "result=None" in again.stdout, again.stderr
    assert "already complete" in again.stderr
    assert files == {
        p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file() and p.name != ".checkpointer.lock"
    }

    reference = tmp_path / "fresh.pt"
    fresh = _run(DECORATED, _base_args(run_dir, **common, fresh=True, reference=reference))
    assert fresh.returncode == 0 and "result=trained" in fresh.stdout, fresh.stderr
    assert _load(reference)["ids_seen"] == _expected_ids(30, 5, 3, 3, shuffle=True)
    assert (run_dir / "COMPLETE").is_file()
