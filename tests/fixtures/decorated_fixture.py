#!/usr/bin/env python3
"""Subprocess fixture: the train_fixture loop adapted to ``@resumable``.

Same model, data, accumulation, kill switches and reference snapshots as
``train_fixture.py`` so the decorated path can be compared against it.  Not a
library example.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

import torch
from torch import nn

from revive_me import RunIncompleteWarning, resumable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fixture import Counter, IdDataset  # noqa: E402


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
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--kill-after-commits", type=int, default=None)
    parser.add_argument("--kill-mid-after", type=int, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--observe", type=Path, default=None)
    return parser.parse_args()


@resumable(
    run_dir=lambda a: a.run_dir,
    interval_seconds=lambda a: a.save_interval,
    fresh=lambda a: a.fresh,
)
def train(args: argparse.Namespace, *, run) -> str:
    torch.manual_seed(args.seed)
    dataset = IdDataset(args.dataset_size)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    component = Counter()
    run.register(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        components={"counter": component},
        metadata={"fixture": "kill-resume", "seed": args.seed},
        compatibility={"arch": "linear-1", "dataset": "fixture-v1"},
    )
    loader_kwargs: dict[str, object] = {"num_workers": args.workers}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "fork"
    loader = run.dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        drop_last=True,
        dataset_fingerprint="fixture-v1",
        **loader_kwargs,
    )

    committed_ids: list[int] = []
    window_ids: list[int] = []
    state = {"commits": 0, "since_commit": 0}

    def snapshot(ids_seen: list[int]) -> dict[str, object]:
        progress = run.progress
        return {
            "progress": {
                "epoch": progress.epoch,
                "batch": progress.batch,
                "batches_seen": progress.batches_seen,
            },
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()),
            "component": copy.deepcopy(component.state_dict()),
            "ids_seen": list(ids_seen),
        }

    def write_reference() -> None:
        if args.reference is not None:
            torch.save(snapshot(committed_ids), args.reference)

    def optimizer_step(batch_count: int) -> None:
        nonlocal window_ids
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        component.steps += 1
        component.window_sizes.append(batch_count)
        run.commit(batch_count)
        committed_ids.extend(window_ids)
        window_ids = []
        state["commits"] += 1
        state["since_commit"] = 0
        write_reference()
        if (
            args.kill_after_commits is not None
            and args.kill_mid_after is None
            and state["commits"] >= args.kill_after_commits
        ):
            os._exit(17)

    for epoch in run.epochs(args.epochs):
        if args.inspect:
            # Restore has been applied.  Pull one batch to show the sampler was
            # positioned before the iterator existed, then stop (incomplete).
            _, ids = next(iter(loader))
            observed = snapshot([])
            observed["first_ids"] = [int(v) for v in ids.tolist()]
            observed["generation"] = run.resumed.generation
            observed["len"] = len(loader)
            torch.save(observed, args.observe)
            return "inspected"
        pending = 0
        optimizer.zero_grad(set_to_none=True)
        for x, y in loader:
            window_ids.extend(int(v) for v in y.tolist())
            with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
                loss = nn.functional.mse_loss(model(x), y.float().unsqueeze(1))
            scaler.scale(loss / args.accum).backward()
            pending += 1
            state["since_commit"] += 1
            if (
                args.kill_mid_after is not None
                and state["commits"] >= int(args.kill_after_commits or 0)
                and state["since_commit"] >= args.kill_mid_after
            ):
                os._exit(17)
            if pending == args.accum:
                optimizer_step(pending)
                pending = 0
        if pending:
            optimizer_step(pending)

    write_reference()  # after the loop: progress is (epochs, 0) like the low-level fixture
    return "trained"


def main() -> int:
    args = parse_args()
    warnings.simplefilter("ignore", RunIncompleteWarning)  # --inspect stops early on purpose
    result = train(args)
    print(f"result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
