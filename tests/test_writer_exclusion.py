"""Writer exclusion and interrupted-write recovery, using real processes."""

from __future__ import annotations

import copy
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import assert_state_equal, make_run, training_step
from revive_me import (
    Checkpointer,
    CheckpointError,
    CheckpointLockedError,
    CheckpointRecoveryWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "writer_fixture.py"
LAUNCHER = PROJECT_ROOT / "scripts" / "relaunch.py"
DATASET = 300  # must match writer_fixture.py


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _fixture_cmd(run_dir: Path, mode: str) -> list[str]:
    return [sys.executable, str(FIXTURE), "--run-dir", str(run_dir), "--mode", mode]


def _spawn(cmd: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _readline(proc: subprocess.Popen[str], timeout: float = 90.0) -> str:
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    assert ready, "subprocess produced no output in time"
    line = proc.stdout.readline()  # type: ignore[union-attr]
    assert line, f"subprocess exited early: {proc.stderr.read()}"  # type: ignore[union-attr]
    return line.strip()


def _finish(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=30)
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()


def _run_fixture(run_dir: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _fixture_cmd(run_dir, mode),
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )


def _open(run_dir: Path) -> Checkpointer:
    return make_run(run_dir, interval=0.0, dataset_size=DATASET)[0]


# ------------------------------------------------------------------ same process
def test_second_owner_in_same_process_is_rejected_until_close(tmp_path: Path) -> None:
    first = _open(tmp_path)
    stale = first.checkpoint_dir / ".checkpoint-00000007.pt.1.0123456789abcdef0123456789abcdef.tmp"
    stale.write_bytes(b"in-flight write of the live owner")

    with pytest.raises(CheckpointLockedError, match="already owned"):
        _open(tmp_path)
    # The failed attempt neither cleaned the live owner's file nor released the
    # lock (flock is per open file description, so its failed fd cannot drop it).
    assert stale.is_file()
    with pytest.raises(CheckpointLockedError):
        _open(tmp_path)

    first.close()
    first.close()  # idempotent
    second = _open(tmp_path)
    second.close()


def test_context_manager_releases_and_closed_instance_is_unusable(tmp_path: Path) -> None:
    with _open(tmp_path) as ckpt:
        with pytest.raises(CheckpointLockedError):
            _open(tmp_path)
    for call in (ckpt.save, ckpt.restore, ckpt.mark_complete, ckpt.reset):
        with pytest.raises(CheckpointError, match="closed"):
            call()
    _open(tmp_path).close()


def test_different_run_dirs_are_independent(tmp_path: Path) -> None:
    with _open(tmp_path / "a"), _open(tmp_path / "b"):
        pass


def test_lock_file_is_not_the_launcher_lock_and_survives_reset(tmp_path: Path) -> None:
    with _open(tmp_path) as ckpt:
        ckpt.save()
        ckpt.reset()
        assert (tmp_path / ".checkpointer.lock").is_file()
        assert not (tmp_path / ".lock").exists()


# ----------------------------------------------------------------- real processes
def test_competing_process_rejected_then_allowed_after_release(tmp_path: Path) -> None:
    holder = _spawn(_fixture_cmd(tmp_path, "hold"))
    try:
        assert _readline(holder) == "READY"
        contender = _run_fixture(tmp_path, "try")
        assert contender.returncode == 42, contender.stderr
        assert contender.stdout.startswith("LOCKED")
        with pytest.raises(CheckpointLockedError):
            _open(tmp_path)
        holder.stdin.close()  # type: ignore[union-attr]
        assert holder.wait(timeout=60) == 0
    finally:
        _finish(holder)
    _open(tmp_path).close()


def test_sigkilled_owner_releases_lock(tmp_path: Path) -> None:
    holder = _spawn(_fixture_cmd(tmp_path, "hold"))
    try:
        assert _readline(holder) == "READY"
        with pytest.raises(CheckpointLockedError):
            _open(tmp_path)
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=30)
    finally:
        _finish(holder)
    _open(tmp_path).close()


def test_os_exit_owner_releases_lock(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path, "die")
    assert result.returncode == 17, result.stderr
    _open(tmp_path).close()


def test_lingering_forked_child_does_not_keep_dead_owners_lock(tmp_path: Path) -> None:
    """A DataLoader-style forked child must not pin the lock after its parent dies."""
    owner = _spawn(_fixture_cmd(tmp_path, "fork-die"))
    child = 0
    try:
        child = int(_readline(owner).split()[1])
        assert owner.wait(timeout=30) == 17
        os.kill(child, 0)  # the child is still alive...
        _open(tmp_path).close()  # ...yet the lock is free
    finally:
        _finish(owner)
        if child:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


# ----------------------------------------------------------------- launcher + library
def test_launcher_and_library_locks_do_not_deadlock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--run-dir", str(run_dir), "--retry-delay", "0",
         "--", *_fixture_cmd(run_dir, "complete")],
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "COMPLETE").is_file()
    assert (run_dir / ".lock").is_file()
    assert (run_dir / ".checkpointer.lock").is_file()
    assert "refusing" not in result.stderr


def test_manual_owner_cannot_collide_with_supervised_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    launcher = _spawn(
        [sys.executable, str(LAUNCHER), "--run-dir", str(run_dir),
         "--", *_fixture_cmd(run_dir, "hold")]
    )
    try:
        assert _readline(launcher) == "READY"  # child holds the library lock
        with pytest.raises(CheckpointLockedError):
            _open(run_dir)  # a manual script on the same run directory
        second = subprocess.run(
            [sys.executable, str(LAUNCHER), "--run-dir", str(run_dir), "--", "true"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert second.returncode == 75, second.stderr  # launcher lock still works
        launcher.stdin.close()  # type: ignore[union-attr]
        assert launcher.wait(timeout=60) == 0
    finally:
        _finish(launcher)
    _open(run_dir).close()


# ------------------------------------------------------------ interrupted writes
@pytest.mark.parametrize(
    ("mode", "exit_code", "leftovers"),
    [
        # checkpoint renamed, sidecar not: an unusable generation 3 + a temp sidecar
        ("crash-sidecar", 23, ["checkpoint-00000003.pt", ".checkpoint-00000003.sha256."]),
        # killed while streaming the temp file: only a temp checkpoint remains
        ("crash-torch-save", 24, [".checkpoint-00000003.pt."]),
    ],
)
def test_abrupt_write_interruption_cleans_and_keeps_good_fallback(
    tmp_path: Path, mode: str, exit_code: int, leftovers: list[str]
) -> None:
    ckpt, model, optimizer, scheduler, _, component, _ = make_run(
        tmp_path, interval=0.0, dataset_size=DATASET
    )
    for _ in range(2):
        training_step(model, optimizer, scheduler, component)
        ckpt.commit(1)
    gen2_model = copy.deepcopy(model.state_dict())
    ckpt.close()

    crashed = _run_fixture(tmp_path, mode)
    assert crashed.returncode == exit_code, crashed.stderr
    names = [p.name for p in (tmp_path / "checkpoints").iterdir()]
    for prefix in leftovers:
        assert any(name.startswith(prefix) for name in names), (prefix, names)

    ckpt2, model2, optimizer2, scheduler2, _, component2, _ = make_run(
        tmp_path, interval=0.0, dataset_size=DATASET
    )
    if mode == "crash-sidecar":
        with pytest.warns(CheckpointRecoveryWarning, match="missing checksum"):
            info = ckpt2.restore()
    else:
        info = ckpt2.restore()
    assert info is not None and info.generation == 2
    assert_state_equal(model2.state_dict(), gen2_model)
    assert not list((tmp_path / "checkpoints").glob(".*.tmp"))  # stale temps removed

    # A later save must not sacrifice the good generation 2 for the unusable 3.
    training_step(model2, optimizer2, scheduler2, component2)
    ckpt2.commit(1)
    directory = tmp_path / "checkpoints"
    assert (directory / "checkpoint-00000002.pt").is_file()
    newest = max(directory.glob("checkpoint-*.pt"))
    newest.unlink()  # lose the newest generation
    ckpt2.close()

    ckpt3, model3, *_ = make_run(tmp_path, interval=0.0, dataset_size=DATASET)
    if mode == "crash-sidecar":  # the unusable generation 3 is still skipped
        with pytest.warns(CheckpointRecoveryWarning):
            info3 = ckpt3.restore()
    else:
        info3 = ckpt3.restore()
    assert info3 is not None and info3.generation == 2
    assert_state_equal(model3.state_dict(), gen2_model)
    ckpt3.close()


# ------------------------------------------------------------ fork ownership
def test_inherited_checkpointer_never_owns_or_releases_locks_after_fork(tmp_path: Path) -> None:
    """A forked child's copy cannot write, and its close/__del__ cannot release
    a descriptor number that a child-created Checkpointer has since reused."""
    run_dir = tmp_path / "parent-a"
    owner = _spawn(_fixture_cmd(run_dir, "fork-inherit"))
    child = 0
    try:
        report = json.loads(_readline(owner))
        child = report["child"]
        assert "error" not in report, report
        # The child's fresh owners really reused the inherited descriptor numbers.
        assert report["fresh_fds"] == report["inherited_fds"]
        assert report["ops"] == {
            name: "refused"
            for name in ("save", "restore", "commit", "finish_epoch", "mark_complete",
                         "reset", "__enter__")
        }
        # Child-created owners stay exclusive after inherited close() / __del__.
        assert report["after_close"] is True
        assert report["after_del"] is True
        # The parent stays the exclusive owner of both of its directories.
        assert report["child_sees_parent_locked"] == [True, True]
        assert report["parent_dirs_locked"] == [True, True]
        # Independent processes agree, with the child still alive.
        os.kill(child, 0)
        for name in ("parent-a", "parent-b", "fresh-a", "fresh-b"):
            with pytest.raises(CheckpointLockedError):
                _open(tmp_path / name)
        # The inherited save()/commit()/mark_complete() wrote nothing.
        assert not list((run_dir / "checkpoints").glob("checkpoint-*"))
        assert not (run_dir / "COMPLETE").exists()
        owner.stdin.close()  # type: ignore[union-attr]
        assert owner.wait(timeout=60) == 0
    finally:
        _finish(owner)
        if child:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for name in ("parent-a", "parent-b", "fresh-a", "fresh-b"):
        _open(tmp_path / name).close()
