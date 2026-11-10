from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, Sampler, TensorDataset

from revive_me import ResumableBatchSampler, make_dataloader


class _Iterable(IterableDataset):
    def __iter__(self):
        yield torch.tensor([0.0])


class _CustomSampler(Sampler[list[int]]):
    def __iter__(self):
        yield [0]


def test_sequential_order() -> None:
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    assert list(sampler) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert sampler.total_batches == 3
    assert len(sampler) == 3


def test_seeded_shuffle_is_deterministic() -> None:
    sampler = ResumableBatchSampler(10, 3, shuffle=True, seed=5, drop_last=True)
    order = torch.randperm(
        10, generator=torch.Generator().manual_seed(5)
    ).tolist()
    assert list(sampler) == [order[0:3], order[3:6], order[6:9]]
    other = ResumableBatchSampler(10, 3, shuffle=True, seed=5, drop_last=True)
    assert list(sampler) == list(other)


def test_offset_slices_without_touching_dataset() -> None:
    class ExplodingDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, index: int):
            raise AssertionError("sampler must not read dataset items")

    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    sampler.set_position(0, 1)
    assert list(sampler) == [[3, 4, 5], [6, 7, 8]]
    assert len(sampler) == 2
    # The exploding dataset is only here to make the intent explicit; the
    # sampler itself never receives it.
    assert len(ExplodingDataset()) == 10


def test_set_epoch_preserves_same_offset_and_resets_new_epoch() -> None:
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    sampler.set_position(2, 1)
    sampler.set_epoch(2)
    assert sampler.batch_offset == 1
    sampler.set_epoch(3)
    assert sampler.batch_offset == 0
    assert sampler.epoch == 3


def test_drop_last_false_keeps_short_final_batch() -> None:
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=False)
    assert list(sampler) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert sampler.total_batches == 4


def test_state_roundtrip_and_config_mismatch() -> None:
    sampler = ResumableBatchSampler(
        12, 3, shuffle=True, seed=9, drop_last=True, dataset_fingerprint="a"
    )
    sampler.set_position(4, 2)
    state = sampler.state_dict()
    restored = ResumableBatchSampler(
        12, 3, shuffle=True, seed=9, drop_last=True, dataset_fingerprint="a"
    )
    assert restored.validate_state_dict(state) is True
    restored.load_state_dict(state)
    assert restored.epoch == 4
    assert restored.batch_offset == 2

    changed = ResumableBatchSampler(
        15, 3, shuffle=True, seed=9, drop_last=True, dataset_fingerprint="b"
    )
    with pytest.raises(ValueError, match="configuration changed"):
        changed.load_state_dict(state)
    assert changed.validate_state_dict(state, allow_config_mismatch=True) is False
    changed.load_state_dict(state, allow_config_mismatch=True)
    assert changed.epoch == 4
    assert changed.batch_offset == 0


def test_offset_beyond_total_rejected() -> None:
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.set_position(0, 4)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.load_state_dict(
            {
                **sampler.state_dict(),
                "batch_offset": 4,
            }
        )


def test_sampler_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        ResumableBatchSampler(-1, 3)
    with pytest.raises(ValueError):
        ResumableBatchSampler(10, 0)
    with pytest.raises(TypeError):
        ResumableBatchSampler(10, 3, dataset_fingerprint=123)  # type: ignore[arg-type]


def test_make_dataloader_rejects_unsupported_inputs() -> None:
    dataset = TensorDataset(torch.arange(10).float().unsqueeze(1))
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    with pytest.raises(TypeError, match="IterableDataset"):
        make_dataloader(_Iterable(), sampler)
    with pytest.raises(TypeError, match="ResumableBatchSampler"):
        make_dataloader(dataset, _CustomSampler())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="do not pass"):
        make_dataloader(dataset, sampler, shuffle=True)
    with pytest.raises(TypeError, match="do not pass"):
        make_dataloader(dataset, sampler, batch_size=4)
    with pytest.raises(ValueError, match="length changed"):
        make_dataloader(
            TensorDataset(torch.arange(11).float().unsqueeze(1)), sampler
        )


def test_make_dataloader_delivers_ordered_batches() -> None:
    dataset = TensorDataset(torch.arange(10).float().unsqueeze(1))
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    loader = make_dataloader(dataset, sampler, num_workers=0)
    seen = [int(batch[0][0, 0]) for batch in loader]
    assert seen == [0, 3, 6]
    sampler.set_position(0, 1)
    loader = make_dataloader(dataset, sampler, num_workers=0)
    assert [int(batch[0][0, 0]) for batch in loader] == [3, 6]


def test_make_dataloader_rejects_out_of_order_worker_delivery() -> None:
    dataset = TensorDataset(torch.arange(10).float().unsqueeze(1))
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    with pytest.raises(ValueError, match="ordered batch delivery"):
        make_dataloader(dataset, sampler, num_workers=2, in_order=False)
    with pytest.raises(TypeError, match="in_order"):
        make_dataloader(dataset, sampler, num_workers=2, in_order="yes")  # type: ignore[arg-type]

    # in_order=True is redundant but accepted; it is not forwarded to older
    # torch versions whose DataLoader predates the argument.
    loader = make_dataloader(dataset, sampler, num_workers=2, in_order=True)
    assert [int(batch[0][0, 0]) for batch in loader] == [0, 3, 6]


def test_make_dataloader_does_not_forward_redundant_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = TensorDataset(torch.arange(10).float().unsqueeze(1))
    sampler = ResumableBatchSampler(10, 3, shuffle=False, drop_last=True)
    captured: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, dataset, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("revive_me.sampler.DataLoader", FakeLoader)
    make_dataloader(dataset, sampler, in_order=True)
    assert "in_order" not in captured
    make_dataloader(dataset, sampler)
    assert "in_order" not in captured
