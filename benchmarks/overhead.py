#!/usr/bin/env python3
"""Measure synchronous checkpoint write cost on this machine.

The percentage is the measured save time divided by a nominal 300-second
interval plus save time.  It is a local measurement, not a product guarantee:
model size, optimizer state, filesystem, and hardware dominate the result.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from revive_me import Checkpointer, ResumableBatchSampler


class Counter:
    def __init__(self) -> None:
        self.count = 0

    def state_dict(self) -> dict[str, int]:
        return {"count": self.count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.count = int(state["count"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--interval", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    layers: list[nn.Module] = []
    in_features = 64
    for _ in range(args.depth):
        layers.extend([nn.Linear(in_features, args.width), nn.ReLU()])
        in_features = args.width
    layers.append(nn.Linear(in_features, 1))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = ResumableBatchSampler(10_000, 32, shuffle=True, seed=0)
    component = Counter()

    with tempfile.TemporaryDirectory(prefix="revive-me-bench-") as directory:
        ckpt = Checkpointer(
            Path(directory),
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            components={"counter": component},
            interval_seconds=args.interval,
        )
        # Warm up filesystem and serialization paths.
        assert ckpt.save()
        durations: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            assert ckpt.save()
            durations.append(time.perf_counter() - start)
        checkpoint_files = sorted(ckpt.checkpoint_dir.glob("checkpoint-*.pt"))
        checkpoint_bytes = checkpoint_files[-1].stat().st_size
        mean = statistics.mean(durations)
        median = statistics.median(durations)
        worst = max(durations)
        overhead = mean / (args.interval + mean) * 100.0
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"checkpoint file: {checkpoint_bytes / (1024 * 1024):.2f} MiB")
        print(f"saves measured: {len(durations)}")
        print(f"mean save: {mean * 1000:.2f} ms")
        print(f"median save: {median * 1000:.2f} ms")
        print(f"worst save: {worst * 1000:.2f} ms")
        print(
            f"nominal overhead at {args.interval:g}s interval: "
            f"{overhead:.4f}% (measured save time / (interval + save time))"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
