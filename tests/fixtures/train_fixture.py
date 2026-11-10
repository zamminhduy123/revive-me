#!/usr/bin/env python3
"""Subprocess fixture for kill/resume tests.  Not a library example."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from revive_me import Checkpointer, ResumableBatchSampler, make_dataloader


class IdDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return torch.tensor([float(index)], dtype=torch.float32), index


class Counter:
    """Registered extra state with a nontrivial, growing value."""

    def __init__(self) -> None:
        self.steps = 0
        self.window_sizes: list[int] = []

    def state_dict(self) -> dict[str, object]:
        return {"steps": self.steps, "window_sizes": list(self.window_sizes)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.steps = int(state["steps"])  # type: ignore[arg-type]
        self.window_sizes = [
            int(value) for value in state["window_sizes"]  # type: ignore[union-attr]
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dataset-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-interval", type=float, default=0.0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--kill-after-commits", type=int, default=None)
    parser.add_argument("--kill-mid-after", type=int, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--observe", type=Path, default=None)
    args = parser.parse_args()
    if args.accum < 1:
        parser.error("--accum must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.kill_mid_after is not None:
        if args.kill_after_commits is None:
            parser.error("--kill-mid-after requires --kill-after-commits")
        if not 1 <= args.kill_mid_after < args.accum:
            parser.error("--kill-mid-after must be in [1, accum)")
    if args.kill_after_commits is not None and args.kill_after_commits < 0:
        parser.error("--kill-after-commits must be >= 0")
    if args.inspect and args.observe is None:
        parser.error("--inspect requires --observe")
    return args


def build(args: argparse.Namespace):
    sampler = ResumableBatchSampler(
        args.dataset_size,
        args.batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        drop_last=True,
        dataset_fingerprint="fixture-v1",
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    component = Counter()
    ckpt = Checkpointer(
        args.run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        components={"counter": component},
        metadata={"fixture": "kill-resume", "seed": args.seed},
        compatibility={"arch": "linear-1", "dataset": "fixture-v1"},
        interval_seconds=args.save_interval,
    )
    return ckpt, model, optimizer, scheduler, scaler, component, sampler


def snapshot(
    ckpt: Checkpointer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    component: Counter,
    *,
    ids_seen: list[int],
) -> dict[str, object]:
    return {
        "progress": {
            "epoch": ckpt.epoch,
            "batch": ckpt.batch,
            "batches_seen": ckpt.progress.batches_seen,
        },
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "component": copy.deepcopy(component.state_dict()),
        "ids_seen": list(ids_seen),
    }


def make_loader(
    dataset: IdDataset, sampler: ResumableBatchSampler, args: argparse.Namespace
):
    loader_kwargs: dict[str, object] = {"num_workers": args.workers}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        # POSIX fork keeps worker startup fast while still exercising real
        # prefetched worker processes; macOS spawn costs ~10s per loader.
        if sys.platform != "win32":
            loader_kwargs["multiprocessing_context"] = "fork"
    return make_dataloader(dataset, sampler, **loader_kwargs)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = IdDataset(args.dataset_size)
    ckpt, model, optimizer, scheduler, scaler, component, sampler = build(args)

    if args.inspect:
        info = ckpt.restore()
        if info is None:
            raise SystemExit("inspect requested but no checkpoint was found")
        sampler.set_epoch(ckpt.epoch)
        loader = make_loader(dataset, sampler, args)
        first_ids: list[int] = []
        if len(loader) > 0:
            _, ids = next(iter(loader))
            first_ids = [int(value) for value in ids.tolist()]
        observed = snapshot(
            ckpt,
            model,
            optimizer,
            scheduler,
            scaler,
            component,
            ids_seen=[],
        )
        observed["first_ids"] = first_ids
        observed["generation"] = info.generation
        observed["sampler_actual"] = {
            "epoch": sampler.epoch,
            "batch_offset": sampler.batch_offset,
        }
        assert args.observe is not None
        torch.save(observed, args.observe)
        print(
            f"inspect: epoch={ckpt.epoch} batch={ckpt.batch} "
            f"first_ids={first_ids}"
        )
        return 0

    info = ckpt.restore()
    if info is None:
        print("fresh run")
    else:
        print(
            f"resumed generation={info.generation} epoch={info.epoch} "
            f"batch={info.batch} recovered={info.recovered_from_previous}"
        )

    loader = make_loader(dataset, sampler, args)
    committed_ids: list[int] = []
    window_ids: list[int] = []
    commits = 0
    batches_since_commit = 0

    def write_reference() -> None:
        if args.reference is not None:
            torch.save(
                snapshot(
                    ckpt,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    component,
                    ids_seen=committed_ids,
                ),
                args.reference,
            )

    def optimizer_step(batch_count: int) -> None:
        nonlocal commits, batches_since_commit, window_ids
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        component.steps += 1
        component.window_sizes.append(batch_count)
        ckpt.commit(batch_count)
        committed_ids.extend(window_ids)
        window_ids = []
        commits += 1
        batches_since_commit = 0
        write_reference()
        if (
            args.kill_after_commits is not None
            and args.kill_mid_after is None
            and commits >= args.kill_after_commits
        ):
            os._exit(17)

    for epoch in range(ckpt.epoch, args.epochs):
        sampler.set_epoch(epoch)
        pending = 0
        # Gradients accumulate across the whole window.  This is deliberately
        # not zeroed per microbatch.
        optimizer.zero_grad(set_to_none=True)
        for x, y in loader:
            window_ids.extend(int(value) for value in y.tolist())
            with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
                loss = nn.functional.mse_loss(model(x), y.float().unsqueeze(1))
            # Normalize by the configured window size.  A shorter final window
            # contributes a smaller update, but is still stepped and committed
            # with its actual batch count.
            scaler.scale(loss / args.accum).backward()
            pending += 1
            batches_since_commit += 1
            if (
                args.kill_mid_after is not None
                and commits >= int(args.kill_after_commits or 0)
                and batches_since_commit >= args.kill_mid_after
            ):
                os._exit(17)
            if pending == args.accum:
                optimizer_step(pending)
                pending = 0
        if pending:
            optimizer_step(pending)
        ckpt.finish_epoch()
        write_reference()

    if ckpt.save():
        ckpt.mark_complete()
        print(f"completed epoch={ckpt.epoch} batches_seen={ckpt.progress.batches_seen}")
    write_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
