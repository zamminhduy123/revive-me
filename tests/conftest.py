from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from revive_me import Checkpointer, ResumableBatchSampler


class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 4) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Counter:
    def __init__(self) -> None:
        self.count = 0
        self.history: list[int] = []

    def state_dict(self) -> dict[str, Any]:
        return {"count": self.count, "history": list(self.history)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.count = int(state["count"])
        self.history = list(state["history"])


class FakeScaler:
    def __init__(self, scale: float = 8.0) -> None:
        self.scale = scale
        self.skipped = 0

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.scale, "skipped": self.skipped}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale = float(state["scale"])
        self.skipped = int(state["skipped"])


def make_run(
    run_dir: Path,
    *,
    dataset_size: int = 12,
    batch_size: int = 3,
    seed: int = 3,
    interval: float = 0.0,
    compatibility: dict[str, Any] | None = None,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
    sampler: ResumableBatchSampler | None = None,
    component: Counter | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[
    Checkpointer,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    Any,
    Counter,
    ResumableBatchSampler,
]:
    model = model or ToyModel()
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = scheduler or torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.5
    )
    scaler = scaler or FakeScaler()
    component = component or Counter()
    sampler = sampler or ResumableBatchSampler(
        dataset_size,
        batch_size,
        shuffle=True,
        seed=seed,
        drop_last=True,
        dataset_fingerprint="toy-v1",
    )
    ckpt = Checkpointer(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        components={"counter": component},
        metadata=metadata or {"config": {"lr": 0.1}},
        compatibility=compatibility or {"arch": "toy-v1", "dataset": "toy-v1"},
        interval_seconds=interval,
    )
    return ckpt, model, optimizer, scheduler, scaler, component, sampler


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    component: Counter | None = None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(4, 2)).pow(2).mean()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if component is not None:
        component.count += 1
        component.history.append(component.count)


def assert_state_equal(actual: Any, expected: Any) -> None:
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        assert torch.is_tensor(actual) and torch.is_tensor(expected)
        assert torch.equal(actual, expected)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in actual:
            assert_state_equal(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
        return
    assert actual == expected
