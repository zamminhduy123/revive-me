#!/usr/bin/env python3
"""Subprocess fixture for writer-exclusion and interrupted-write tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ for conftest

from conftest import make_run, training_step  # noqa: E402

from revive_me import CheckpointError, CheckpointLockedError  # noqa: E402
from revive_me import checkpoint as checkpoint_module  # noqa: E402

MODES = (
    "hold",
    "die",
    "fork-die",
    "fork-inherit",
    "try",
    "complete",
    "crash-sidecar",
    "crash-torch-save",
)


def _locked(run_dir: Path) -> bool:
    try:
        make_run(run_dir, interval=0.0, dataset_size=300)[0].close()
    except CheckpointLockedError:
        return True
    return False


def _land_on(fd: int) -> list[int]:
    """Make ``fd`` the lowest free descriptor so the next open() reuses it."""
    fillers = []
    while True:
        new = os.open(os.devnull, os.O_RDONLY)
        if new == fd:
            os.close(new)
            return fillers
        assert new < fd, f"descriptor {fd} is not free"
        fillers.append(new)


def _fork_inherit(run_dir: Path, owners: list) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """Child gets inherited owners plus fresh ones that reuse their descriptors."""
    results, control = os.pipe(), os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(results[0])
        os.close(control[1])  # else the child would never see EOF on control
        report: dict[str, object] = {}
        keep = []
        try:
            a, b = owners
            report["inherited_fds"] = [a._lock_fd, b._lock_fd]
            ops = {}
            for name in ("save", "restore", "commit", "finish_epoch", "mark_complete",
                         "reset", "__enter__"):
                try:
                    getattr(a, name)()
                    ops[name] = "ALLOWED"
                except CheckpointError as exc:
                    ops[name] = "refused"
                    assert "inherited" in str(exc), exc
            report["ops"] = ops
            fresh = []
            for inherited, suffix in ((a, "fresh-a"), (b, "fresh-b")):
                keep += _land_on(inherited._lock_fd)
                fresh.append(make_run(run_dir.parent / suffix, interval=0.0, dataset_size=300)[0])
            report["fresh_fds"] = [f._lock_fd for f in fresh]
            a.close()  # inherited close, descriptor number now owned by fresh-a
            a.close()
            report["after_close"] = _locked(run_dir.parent / "fresh-a")
            b.__del__()  # inherited destructor, descriptor now owned by fresh-b
            report["after_del"] = _locked(run_dir.parent / "fresh-b")
            report["child_sees_parent_locked"] = [
                _locked(run_dir), _locked(run_dir.parent / "parent-b"),
            ]
            os.write(results[1], json.dumps(report).encode() + b"\n")
            os.read(control[0], 1)  # stay alive until the parent releases us
        except BaseException as exc:
            os.write(results[1], json.dumps({"error": repr(exc)}).encode() + b"\n")
        finally:
            os._exit(0)
    os.close(results[1])
    os.close(control[0])
    return pid, results, control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    args = parser.parse_args()

    try:
        ckpt, model, optimizer, scheduler, _, component, _ = make_run(
            args.run_dir, interval=0.0, dataset_size=300
        )
    except CheckpointLockedError as exc:
        print(f"LOCKED {exc}", flush=True)
        return 42
    if args.mode == "try":
        print("ACQUIRED", flush=True)
        ckpt.close()
        return 0

    if args.mode == "fork-inherit":
        other = make_run(args.run_dir.parent / "parent-b", interval=0.0, dataset_size=300)[0]
        child, results, control = _fork_inherit(args.run_dir, [ckpt, other])
        report = json.loads(os.fdopen(results[0]).readline())
        report["child"] = child
        report["parent_dirs_locked"] = [_locked(args.run_dir), _locked(args.run_dir.parent / "parent-b")]
        print(json.dumps(report), flush=True)
        sys.stdin.read()  # until the test closes stdin
        os.close(control[1])
        os.waitpid(child, 0)
        ckpt.close()
        other.close()
        return 0

    if args.mode == "hold":
        print("READY", flush=True)
        sys.stdin.read()  # until the test closes stdin
        ckpt.close()
        return 0
    if args.mode == "die":
        print("READY", flush=True)
        os._exit(17)
    if args.mode == "fork-die":
        pid = os.fork()
        if pid == 0:  # a lingering child, like a DataLoader worker
            devnull = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(devnull, fd)
            time.sleep(60)
            os._exit(0)
        print(f"READY {pid}", flush=True)
        os._exit(17)

    if args.mode == "complete":
        training_step(model, optimizer, scheduler, component)
        ckpt.commit(1)
        ckpt.finish_epoch()
        ckpt.mark_complete()
        ckpt.close()
        return 0

    # Interrupted-write modes: restore (when a prior run exists), then die in
    # the middle of the next save.
    ckpt.restore()
    if args.mode == "crash-sidecar":
        real_replace = os.replace

        def replace(src: object, dst: object) -> None:
            if str(dst).endswith(".sha256"):
                os._exit(23)  # checkpoint renamed, sidecar not yet
            real_replace(src, dst)

        checkpoint_module.os.replace = replace  # type: ignore[assignment]
    else:

        def partial_save(obj: object, writer: object) -> None:
            writer.write(b"partial checkpoint bytes")  # type: ignore[attr-defined]
            writer.flush()  # type: ignore[attr-defined]
            os._exit(24)  # temp file exists, nothing renamed

        checkpoint_module.torch.save = partial_save  # type: ignore[assignment]
    training_step(model, optimizer, scheduler, component)
    ckpt.commit(1)
    return 99  # unreachable: the save above kills the process


if __name__ == "__main__":
    raise SystemExit(main())
