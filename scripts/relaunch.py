#!/usr/bin/env python3
"""Run a training command with a bounded retry policy and a completion marker.

This launcher is deliberately separate from the revive_me library.  It:

* refuses to start if ``<run-dir>/COMPLETE`` exists,
* holds an exclusive ``flock`` on ``<run-dir>/.lock`` so two launchers cannot
  run the same run directory concurrently,
* runs the command, then retries up to three times, sixty seconds apart by
  default, only when the command fails and the completion marker is absent.

Retry budget is per launcher invocation.  A systemd start at boot or a manual
``systemctl start`` begins a fresh budget.  There is no persistent retry file;
the completion marker is the durable stop condition.  A command that exits 0
without writing COMPLETE is treated as a clean intentional stop and is not
retried.

Usage::

    python scripts/relaunch.py --run-dir /srv/run \\
        -- python examples/train.py --run-dir /srv/run

Do not run this script and another supervisor for the same run directory at the
same time; the lock prevents overlap, but only one supervisor should own it.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

EX_TEMPFAIL = 75


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a training command with bounded retries and a completion marker."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory; COMPLETE and .lock live here.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Automatic retries after the initial attempt (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=60.0,
        help="Seconds between retries (default: 60).",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Optional working directory for the training command.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, after '--'.",
    )
    args = parser.parse_args(argv)
    if args.max_retries < 0 or args.max_retries > 3:
        parser.error("--max-retries must be between 0 and 3 (bounded retry policy)")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be >= 0")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing training command after '--'")
    args.command = command
    return args


def _log(message: str) -> None:
    print(f"relaunch: {message}", file=sys.stderr, flush=True)


def _sleep_interruptible(seconds: float, state: dict[str, int | None]) -> None:
    deadline = time.monotonic() + seconds
    while state["signal"] is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    marker = run_dir / "COMPLETE"
    lock_path = run_dir / ".lock"

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log(f"cannot create run directory {run_dir}: {exc}")
        return 1
    if marker.is_file():
        _log(f"{marker} exists; run is complete, not launching")
        return 0

    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        _log(f"cannot open lock {lock_path}: {exc}")
        return 1

    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log(f"another launcher holds {lock_path}; refusing duplicate run")
            return EX_TEMPFAIL
        try:
            os.ftruncate(lock_fd, 0)
            os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(lock_fd)
        except OSError as exc:
            _log(f"cannot record launcher pid in {lock_path}: {exc}")
            return 1

        if args.cwd is not None:
            try:
                os.chdir(args.cwd)
            except OSError as exc:
                _log(f"cannot change directory to {args.cwd}: {exc}")
                return 1

        state: dict[str, int | None] = {"signal": None}
        child_holder: dict[str, subprocess.Popen[bytes] | None] = {"child": None}

        def _handle_signal(signum: int, _frame: object) -> None:
            state["signal"] = signum
            child = child_holder["child"]
            if child is not None and child.poll() is None:
                child.terminate()

        previous_handlers: dict[int, object] = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_handlers[sig] = signal.signal(sig, _handle_signal)
            except (ValueError, OSError):
                pass

        try:
            attempts = 0
            while True:
                if marker.is_file():
                    _log(f"{marker} exists; run is complete, stopping")
                    return 0
                if state["signal"] is not None:
                    return 128 + int(state["signal"])

                attempts += 1
                _log(f"attempt {attempts}: {' '.join(args.command)}")
                try:
                    child = subprocess.Popen(args.command)
                except FileNotFoundError as exc:
                    _log(f"command not found: {exc}")
                    returncode = 127
                except OSError as exc:
                    _log(f"cannot start command: {exc}")
                    returncode = 126
                else:
                    child_holder["child"] = child
                    try:
                        returncode = child.wait()
                    finally:
                        child_holder["child"] = None
                    if returncode < 0:
                        returncode = 128 - returncode

                if state["signal"] is not None:
                    return 128 + int(state["signal"])
                if marker.is_file():
                    _log(f"{marker} exists; run is complete")
                    return 0
                if returncode == 0:
                    _log(
                        "training command exited 0 without COMPLETE; "
                        "not retrying (clean intentional stop)"
                    )
                    return 0
                if attempts > args.max_retries:
                    _log(
                        f"giving up after {attempts} attempt(s), last exit "
                        f"code {returncode}; the run is not complete"
                    )
                    return returncode
                _log(
                    f"attempt {attempts} failed with exit code {returncode}; "
                    f"retrying in {args.retry_delay:g}s "
                    f"({args.max_retries - attempts} retry(ies) left)"
                )
                _sleep_interruptible(args.retry_delay, state)
        finally:
            for sig, handler in previous_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
