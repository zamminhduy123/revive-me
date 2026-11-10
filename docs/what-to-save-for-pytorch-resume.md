# What to save besides model weights for PyTorch resume

Scope: single-process PyTorch training loops. revive-me is a private, unpublished
repository; the API shown is the one in `src/revive_me/resumable.py`
(`@resumable` with the injected `run` handle) and is exercised by
[`examples/train.py`](../examples/train.py).

## Why weights alone are not enough

A `model.state_dict()` restores the parameters, and a loop restarted from it
trains, but not the same run. PyTorch's own
[saving and loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
says a checkpoint meant for resuming training must contain more than the model's
`state_dict`: the optimizer's `state_dict` as well (it holds buffers that are
updated as the model trains), and optionally the epoch and other items.

| State | What goes wrong if it is missing | In revive-me |
| --- | --- | --- |
| Optimizer (momentum, Adam moments, ...) | training restarts with empty optimizer state | `optimizer=` |
| LR scheduler | the learning-rate schedule restarts | `scheduler=` |
| AMP `GradScaler` | loss-scale history is lost | `scaler=` |
| Python, NumPy, torch CPU and CUDA RNG | different dropout/augmentation/shuffle draws | saved automatically |
| Epoch and position inside the epoch | the epoch restarts and repeats data | committed cursor, below |
| EMA, running metrics, other custom state | that state silently resets | `components=` |
| What the checkpoint was made for | a changed architecture or dataset loads wrongly | `compatibility=`, `metadata=` |

NumPy's RNG is saved only if NumPy is installed. CUDA RNG restore is best effort
if the number of visible devices differs.

## The API

```python
import torch
from revive_me import resumable

@resumable(run_dir=lambda args: args.run_dir, interval_seconds=300)
def train(args, *, run):                         # keyword-only `run`, injected
    torch.manual_seed(args.seed)                 # seed BEFORE run.epochs(): restore overwrites RNG
    model, optimizer, scheduler = ...
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    run.register(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        components={"ema": ema},                 # anything with state_dict()/load_state_dict()
        compatibility={"arch": "resnet18-v1"},   # mismatch -> IncompatibleCheckpointError
        metadata={"config": config},             # picklable, stored with the checkpoint
    )
    loader = run.dataloader(dataset, batch_size=32, shuffle=True, seed=1234,
                            drop_last=True, num_workers=4)

    for epoch in run.epochs(args.epochs):        # lock + restore happen here
        pending = 0
        for x, y in loader:
            ...                                  # forward/backward as usual
            pending += 1
            if pending == accumulation_steps:
                ...                              # optimizer/scaler/scheduler step, zero_grad
                run.commit(pending)              # one completed accumulation window
                pending = 0
        if pending:                              # shorter final window
            ...
            run.commit(pending)
```

- `run.register(model=..., optimizer=..., scheduler=..., scaler=..., components=..., compatibility=..., metadata=...)`
  stages what to save; only `model` is required, and each of the first four takes
  one object or a `{name: object}` mapping. Nothing is restored yet.
- `run.dataloader(dataset, batch_size=..., shuffle=True, seed=0, drop_last=True, dataset_fingerprint=None, **loader_kwargs)`
  replaces `DataLoader(...)`. Other `DataLoader` options (`num_workers`,
  `prefetch_factor`, `pin_memory`, ...) pass through.
- `run.epochs(n)` takes the run-directory lock, restores the newest usable
  checkpoint into everything registered, and iterates the remaining epochs.
- `run.commit(n)` records `n` consumed batches after the window's optimizer,
  scaler, scheduler and `zero_grad` handling. This is also where automatic saves
  happen, once `interval_seconds` has elapsed.

A skipped AMP update still consumed its batches. Detect it the way
`examples/train.py` does, step the scheduler only when the update was not skipped,
and commit either way:

```python
old_scale = scaler.get_scale()
scaler.step(optimizer); scaler.update()
optimizer.zero_grad(set_to_none=True)
if scaler.get_scale() >= old_scale:   # the scale did not shrink: no skipped update
    scheduler.step()
run.commit(pending)
```

## Data position: committed cursor, not prefetch

DataLoader workers read ahead. The checkpoint stores the **committed** cursor
(the epoch and how many batches have been committed with `run.commit`), not how
many batches were prefetched or delivered, so batches that were prefetched or
delivered but not committed are processed again after a restart. The epoch order is derived from
`seed + epoch`, and a resume starts at the committed batch index in that order
without replaying skipped dataset items.

## Accumulation boundaries

Checkpoints are only coherent between optimizer updates. Call `run.commit(n)`
after a full window, with the actual count (a shorter final window commits its
smaller count). Never save in the middle of a window; the library only saves from
`commit` and at epoch end, and the decorator raises `RunStateError` if an epoch
ends with uncommitted or unconsumed batches. A crash mid-window resumes from the
previous committed boundary.

## Limits

- Map-style datasets with a reliable `len`, one managed loader, ordered
  delivery. `IterableDataset`, custom samplers, `in_order=False` and distributed
  training are not supported by the decorator.
- Not bit-exact replay in general: DataLoader worker RNG is not saved, and
  nondeterministic kernels or changed data can change the next numbers. The
  guarantee is a consistent state at the committed boundary. On the repository's
  deterministic CPU test fixture, an interrupted-then-resumed run matched an
  uninterrupted one; that is not a claim about other models, GPUs or kernels.
- Compatibility checks are best effort (key/shape checks plus the mapping you
  supply); they cannot prove that arbitrary code or same-size data is unchanged.
- Checkpoints are loaded with `torch.load(..., weights_only=False)`; only load
  checkpoints you produced or trust.

See the [README](../README.md) for the complete contract, and
[Resume PyTorch training after a server reboot](resume-pytorch-training-after-reboot.md)
for restart procedure.
