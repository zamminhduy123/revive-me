from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from conftest import assert_state_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "train_fixture.py"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    # Redirect to files rather than pipes.  A killed fixture can leave
    # DataLoader workers briefly holding inherited pipe descriptors; waiting
    # for pipe EOF would add their teardown latency to every kill test.
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(
        mode="w+"
    ) as stderr:
        completed = subprocess.run(
            [sys.executable, str(FIXTURE), *args],
            cwd=PROJECT_ROOT,
            env=_env(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
        )
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(
            completed.args,
            completed.returncode,
            stdout.read(),
            stderr.read(),
        )


def _base_args(run_dir: Path, **overrides: object) -> list[str]:
    values: dict[str, object] = {
        "epochs": 2,
        "dataset-size": 40,
        "batch-size": 4,
        "accum": 2,
        "workers": 0,
        "seed": 11,
        "save-interval": 0,
        "no-shuffle": True,
    }
    values.update(overrides)
    args = ["--run-dir", str(run_dir)]
    for key, value in values.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def _load(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _expected_ids(
    dataset_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
    *,
    shuffle: bool,
) -> list[int]:
    """Expected committed sample order for a drop_last run."""
    ids: list[int] = []
    for epoch in range(epochs):
        if shuffle:
            order = torch.randperm(
                dataset_size, generator=torch.Generator().manual_seed(seed + epoch)
            ).tolist()
        else:
            order = list(range(dataset_size))
        for start in range(0, dataset_size - batch_size + 1, batch_size):
            ids.extend(order[start : start + batch_size])
    return ids


def test_kill_after_commit_then_fresh_process_resume_matches_state_and_ids(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    reference = tmp_path / "reference.pt"
    first = _run(
        _base_args(
            run_dir,
            **{
                "kill-after-commits": 3,
                "reference": reference,
            },
        )
    )
    assert first.returncode == 17, first.stderr
    assert reference.is_file()

    observe = tmp_path / "observe.pt"
    second = _run(
        _base_args(
            run_dir,
            **{
                "inspect": True,
                "observe": observe,
            },
        )
    )
    assert second.returncode == 0, second.stderr
    observed = _load(observe)
    expected = _load(reference)

    assert observed["progress"] == expected["progress"]
    assert observed["progress"]["batch"] == 6
    assert observed["progress"]["epoch"] == 0
    assert observed["first_ids"] == [24, 25, 26, 27]
    assert observed["sampler_actual"] == {"epoch": 0, "batch_offset": 6}
    assert observed["generation"] == 3

    assert_state_equal(observed["model"], expected["model"])
    assert_state_equal(observed["optimizer"], expected["optimizer"])
    assert_state_equal(observed["scheduler"], expected["scheduler"])
    assert_state_equal(observed["scaler"], expected["scaler"])
    assert_state_equal(observed["component"], expected["component"])


def test_prefetch_workers_and_partial_accumulation_kill_resume_at_committed_boundary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    reference = tmp_path / "reference.pt"
    first = _run(
        _base_args(
            run_dir,
            **{
                "dataset-size": 48,
                "accum": 4,
                "workers": 2,
                "seed": 7,
                "no-shuffle": False,
                "kill-after-commits": 2,
                "kill-mid-after": 2,
                "reference": reference,
            },
        )
    )
    assert first.returncode == 17, first.stderr

    observe = tmp_path / "observe.pt"
    second = _run(
        _base_args(
            run_dir,
            **{
                "dataset-size": 48,
                "accum": 4,
                "workers": 2,
                "seed": 7,
                "no-shuffle": False,
                "inspect": True,
                "observe": observe,
            },
        )
    )
    assert second.returncode == 0, second.stderr
    observed = _load(observe)
    expected = _load(reference)

    # Two completed four-batch windows were committed before the kill.  The two
    # batches of the partial third window (and any DataLoader prefetch) must not
    # appear in the checkpoint.
    assert observed["progress"] == expected["progress"]
    assert observed["progress"]["batch"] == 8
    assert observed["progress"]["epoch"] == 0
    permutation = torch.randperm(
        48, generator=torch.Generator().manual_seed(7)
    ).tolist()
    assert observed["first_ids"] == permutation[32:36]
    assert observed["sampler_actual"] == {"epoch": 0, "batch_offset": 8}

    assert_state_equal(observed["model"], expected["model"])
    assert_state_equal(observed["optimizer"], expected["optimizer"])
    assert_state_equal(observed["scheduler"], expected["scheduler"])
    assert_state_equal(observed["scaler"], expected["scaler"])
    assert_state_equal(observed["component"], expected["component"])


def test_full_run_writes_marker_and_resumes_completed_generation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    result = _run(
        _base_args(
            run_dir,
            **{
                "dataset-size": 20,
                "accum": 3,
                "epochs": 2,
            },
        )
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "COMPLETE").is_file()

    observe = tmp_path / "observe.pt"
    inspected = _run(
        _base_args(
            run_dir,
            **{
                "dataset-size": 20,
                "accum": 3,
                "epochs": 2,
                "inspect": True,
                "observe": observe,
            },
        )
    )
    assert inspected.returncode == 0, inspected.stderr
    observed = _load(observe)
    # Two epochs of five batches each; the last accumulation window is short
    # (two batches) and is still committed.
    assert observed["progress"]["epoch"] == 2
    assert observed["progress"]["batch"] == 0
    assert observed["progress"]["batches_seen"] == 10
    # The checkpoint is at the start of the next epoch; sequential order means
    # the next DataLoader batch is the beginning of epoch 2.
    assert observed["first_ids"] == [0, 1, 2, 3]


def test_kill_after_committed_windows_then_continue_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    """Kill after a commit, resume with workers, and finish across an epoch."""
    common: dict[str, object] = {
        "epochs": 3,
        "dataset-size": 30,
        "batch-size": 5,
        "accum": 4,
        "workers": 2,
        "seed": 13,
        "save-interval": 0,
        "no-shuffle": False,
    }
    expected_ids = _expected_ids(30, 5, 3, 13, shuffle=True)

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_ref = tmp_path / "uninterrupted.pt"
    uninterrupted = _run(
        _base_args(uninterrupted_dir, **common, reference=uninterrupted_ref)
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr
    reference = _load(uninterrupted_ref)

    run_dir = tmp_path / "interrupted"
    interrupted_ref = tmp_path / "interrupted.pt"
    first = _run(
        _base_args(
            run_dir,
            **common,
            **{"kill-after-commits": 5},
            reference=interrupted_ref,
        )
    )
    assert first.returncode == 17, first.stderr

    resumed_ref = tmp_path / "resumed.pt"
    resumed = _run(_base_args(run_dir, **common, reference=resumed_ref))
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed generation" in resumed.stdout
    assert (run_dir / "COMPLETE").is_file()

    prefix = _load(interrupted_ref)
    suffix = _load(resumed_ref)

    # Five completed windows: epoch 2, four batches into the epoch.  The
    # checkpoint must not include the remaining short window yet.
    assert prefix["progress"] == {"epoch": 2, "batch": 4, "batches_seen": 16}
    assert prefix["component"]["steps"] == 5
    assert prefix["component"]["window_sizes"] == [4, 2, 4, 2, 4]
    assert prefix["scheduler"]["last_epoch"] == 5
    assert suffix["progress"] == reference["progress"]
    assert suffix["progress"] == {"epoch": 3, "batch": 0, "batches_seen": 18}
    assert prefix["ids_seen"] == expected_ids[: 16 * 5]
    assert suffix["ids_seen"] == expected_ids[16 * 5 :]
    assert prefix["ids_seen"] + suffix["ids_seen"] == expected_ids

    # The fixture's windows are 4, 2, 4, 2, 4, 2: the final window of every
    # epoch is short and still stepped and committed.
    assert reference["component"]["steps"] == 6
    assert reference["component"]["window_sizes"] == [4, 2, 4, 2, 4, 2]
    assert reference["scheduler"]["last_epoch"] == 6
    assert reference["optimizer"]["state"]
    assert any(
        "momentum_buffer" in state
        for state in reference["optimizer"]["state"].values()
    )

    # DataLoader worker base-seed draws are not part of the comparison.  The
    # dataset has no random transforms and the model has no dropout, so the
    # different iterator-creation RNG state cannot change these registered
    # values.
    for key in ("model", "optimizer", "scheduler", "scaler", "component"):
        assert_state_equal(suffix[key], reference[key])


def test_mid_window_kill_with_prefetch_workers_then_continue_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    """Kill inside a window with prefetched workers, then finish the run."""
    common: dict[str, object] = {
        "epochs": 3,
        "dataset-size": 30,
        "batch-size": 5,
        "accum": 4,
        "workers": 2,
        "seed": 21,
        "save-interval": 0,
        "no-shuffle": False,
    }
    expected_ids = _expected_ids(30, 5, 3, 21, shuffle=True)

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_ref = tmp_path / "uninterrupted.pt"
    uninterrupted = _run(
        _base_args(uninterrupted_dir, **common, reference=uninterrupted_ref)
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr
    reference = _load(uninterrupted_ref)

    run_dir = tmp_path / "interrupted"
    interrupted_ref = tmp_path / "interrupted.pt"
    first = _run(
        _base_args(
            run_dir,
            **common,
            **{"kill-after-commits": 2, "kill-mid-after": 2},
            reference=interrupted_ref,
        )
    )
    assert first.returncode == 17, first.stderr

    resumed_ref = tmp_path / "resumed.pt"
    resumed = _run(_base_args(run_dir, **common, reference=resumed_ref))
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed generation" in resumed.stdout
    assert (run_dir / "COMPLETE").is_file()

    prefix = _load(interrupted_ref)
    suffix = _load(resumed_ref)

    # Two complete windows (4 + 2) ended epoch 0.  The kill happened two
    # microbatches into epoch 1's first window, so the durable cursor is the
    # epoch boundary, not the prefetched or partially-backwarded position.
    assert prefix["progress"] == {"epoch": 1, "batch": 0, "batches_seen": 6}
    assert prefix["component"]["steps"] == 2
    assert prefix["component"]["window_sizes"] == [4, 2]
    assert prefix["scheduler"]["last_epoch"] == 2
    assert suffix["progress"] == reference["progress"]
    assert suffix["progress"] == {"epoch": 3, "batch": 0, "batches_seen": 18}
    assert prefix["ids_seen"] == expected_ids[: 6 * 5]
    assert suffix["ids_seen"] == expected_ids[6 * 5 :]
    assert prefix["ids_seen"] + suffix["ids_seen"] == expected_ids

    assert reference["component"]["steps"] == 6
    assert reference["component"]["window_sizes"] == [4, 2, 4, 2, 4, 2]
    assert reference["scheduler"]["last_epoch"] == 6
    assert reference["optimizer"]["state"]
    assert any(
        "momentum_buffer" in state
        for state in reference["optimizer"]["state"].values()
    )

    for key in ("model", "optimizer", "scheduler", "scaler", "component"):
        assert_state_equal(suffix[key], reference[key])
