#!/usr/bin/env python3
"""Single-device training with gradient accumulation and AMP, via ``@resumable``.

Run from the repository root after installing the package::

    python examples/train.py --run-dir runs/example --epochs 3 --save-interval 300

Interrupt it and run the same command again: it resumes at the last committed
accumulation window.  The loop below is still a normal PyTorch loop; the
decorator and ``run`` handle only register objects, build the resumable loader,
iterate epochs, and record each completed window with ``run.commit``.  The
explicit equivalent is ``examples/train_lowlevel.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from revive_me import resumable


class RunningLoss:
    """Tiny registered extra component showing custom state_dict support."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def state_dict(self) -> dict[str, float | int]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.total = float(state["total"])
        self.count = int(state["count"])

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


def build_dataset(size: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(size, 1, generator=generator)
    y = 2.0 * x + 1.0
    return TensorDataset(x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-interval",
        type=float,
        default=300.0,
        help="Seconds between synchronous saves at committed boundaries.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Explicitly delete existing checkpoints and COMPLETE marker.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.dataset_size < 1:
        parser.error("--dataset-size must be >= 1")
    if args.batch_size < 1 or args.accum < 1:
        parser.error("--batch-size and --accum must be >= 1")
    return args


@resumable(
    run_dir=lambda args: args.run_dir,
    interval_seconds=lambda args: args.save_interval,
    fresh=lambda args: args.fresh,
)
def train(args: argparse.Namespace, *, run) -> float:
    torch.manual_seed(args.seed)  # before run.epochs(): restore overwrites RNG state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    dataset = build_dataset(args.dataset_size)
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    running_loss = RunningLoss()

    run.register(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        components={"running_loss": running_loss},
        metadata={"lr": args.lr, "batch_size": args.batch_size, "accum": args.accum},
        compatibility={"arch": "linear-1", "dataset": "synthetic-linear-v1"},
    )
    loader_kwargs: dict[str, object] = {"num_workers": args.workers}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
    loader = run.dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
        dataset_fingerprint="synthetic-linear-v1",
        **loader_kwargs,
    )

    def optimizer_step(batch_count: int) -> None:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # Do not advance a scheduler on an AMP-skipped optimizer update.
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        run.commit(batch_count)

    epochs = run.epochs(args.epochs)  # takes the lock and restores here
    if run.resumed is None:
        print(f"Starting fresh in {args.run_dir}")
    else:
        info = run.resumed
        print(
            f"Resumed generation {info.generation}: epoch {info.epoch}, "
            f"committed batch {info.batch}, batches seen {info.progress.batches_seen}"
        )

    for epoch in epochs:
        pending = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = nn.functional.mse_loss(model(x), y)
            # Gradients accumulate until optimizer_step below.  This example
            # sums the window's losses; divide by the window size (or
            # args.accum) if you want mean-of-window gradient scaling.
            scaler.scale(loss).backward()
            running_loss.total += float(loss.detach())
            running_loss.count += 1
            pending += 1
            if pending == args.accum:
                optimizer_step(pending)
                pending = 0
        # A shorter final accumulation window is still a coherent boundary.
        if pending:
            optimizer_step(pending)
        print(
            f"epoch {epoch + 1}/{args.epochs} done; "
            f"mean loss {running_loss.mean:.6f}; "
            f"committed batches seen {run.progress.batches_seen}"
        )
    return running_loss.mean


def main() -> int:
    args = parse_args()
    result = train(args)
    if result is None:
        print(
            f"{args.run_dir / 'COMPLETE'} exists; run is already complete. "
            "Pass --fresh to start over explicitly."
        )
    else:
        print(f"Run complete; final mean loss {result:.6f}; marker {args.run_dir / 'COMPLETE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
