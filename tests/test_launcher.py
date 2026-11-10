from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "relaunch.py"


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_complete_marker_suppresses_launch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "COMPLETE").write_text("done\n")
    sentinel = tmp_path / "ran"
    code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')"
    result = _run(
        [
            "--run-dir",
            str(run_dir),
            "--",
            sys.executable,
            "-c",
            code,
            str(sentinel),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "not launching" in result.stderr
    assert not sentinel.exists()


def test_retry_budget_is_initial_plus_three_retries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter = tmp_path / "count"
    code = (
        "import pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "n=int(p.read_text())+1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "sys.exit(1)\n"
    )
    result = _run(
        [
            "--run-dir",
            str(run_dir),
            "--retry-delay",
            "0",
            "--",
            sys.executable,
            "-c",
            code,
            str(counter),
        ]
    )
    assert result.returncode == 1, result.stderr
    assert int(counter.read_text()) == 4
    assert "giving up after 4 attempt(s)" in result.stderr


def test_success_after_retries_creates_marker_and_stops(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter = tmp_path / "count"
    code = (
        "import pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1]); marker=pathlib.Path(sys.argv[2])\n"
        "n=int(p.read_text())+1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "if n < 3:\n"
        "    sys.exit(1)\n"
        "marker.write_text('complete\\n')\n"
        "sys.exit(0)\n"
    )
    result = _run(
        [
            "--run-dir",
            str(run_dir),
            "--retry-delay",
            "0",
            "--",
            sys.executable,
            "-c",
            code,
            str(counter),
            str(run_dir / "COMPLETE"),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert int(counter.read_text()) == 3
    assert (run_dir / "COMPLETE").is_file()


def test_clean_exit_without_marker_is_not_retried(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter = tmp_path / "count"
    code = (
        "import pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "n=int(p.read_text())+1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "sys.exit(0)\n"
    )
    result = _run(
        [
            "--run-dir",
            str(run_dir),
            "--retry-delay",
            "0",
            "--",
            sys.executable,
            "-c",
            code,
            str(counter),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert int(counter.read_text()) == 1
    assert "not retrying" in result.stderr


def test_lock_prevents_duplicate_concurrent_launcher(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sentinel = tmp_path / "ran"
    lock_fd = os.open(run_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(
            [
                "--run-dir",
                str(run_dir),
                "--",
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')",
                str(sentinel),
            ]
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert result.returncode == 75, result.stderr
    assert "refusing duplicate run" in result.stderr
    assert not sentinel.exists()


def test_marker_appearing_after_failed_attempt_stops_retry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter = tmp_path / "count"
    code = (
        "import pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1]); marker=pathlib.Path(sys.argv[2])\n"
        "n=int(p.read_text())+1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "marker.write_text('complete\\n')\n"
        "sys.exit(1)\n"
    )
    result = _run(
        [
            "--run-dir",
            str(run_dir),
            "--retry-delay",
            "0",
            "--",
            sys.executable,
            "-c",
            code,
            str(counter),
            str(run_dir / "COMPLETE"),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert int(counter.read_text()) == 1
