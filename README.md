# revive-me

Explicit, durable checkpoint/recovery for plain single-process PyTorch training.

revive-me is a small library addon, not a training engine. You keep your loop,
your optimizer, and your data pipeline. You explicitly register stateful objects
and explicitly commit completed accumulation windows. In exchange, you get
synchronous checkpoints at coherent boundaries, committed within-epoch resume,
two-generation rotation, and a separate bounded-retry launcher for reboot
recovery.

It does **not** promise bit-exact replay. DataLoader worker RNG, nondeterministic
kernels, and changed data can change the exact next numbers. It does promise a
consistent state at the committed boundary.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

Requires Python 3.10+ and `torch>=2.3` (the example uses the unified
`torch.amp.GradScaler`/`torch.amp.autocast` API). NumPy is optional; if
installed, its RNG state is saved and restored.

## Quick start

```python
from torch.utils.data import DataLoader
from revive_me import Checkpointer, ResumableBatchSampler, make_dataloader

sampler = ResumableBatchSampler(
    dataset_size=len(dataset),
    batch_size=32,
    shuffle=True,
    seed=1234,
    drop_last=True,
    dataset_fingerprint="dataset-v1",  # optional; recommended
)

model = ...
optimizer = ...
scheduler = ...
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

ckpt = Checkpointer(
    "runs/exp1",                       # COMPLETE lives here; checkpoints/ below it
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    sampler=sampler,
    components={"ema": ema},           # any object with state_dict/load_state_dict
    metadata={"config": config},       # arbitrary picklable metadata
    compatibility={"arch": "resnet18-v1", "data": "dataset-v1"},
    interval_seconds=300.0,
)

# Restore before creating the DataLoader iterator.
info = ckpt.restore()
if info is not None:
    print(info.epoch, info.batch, info.recovered_from_previous)

loader = make_dataloader(dataset, sampler, num_workers=4, prefetch_factor=2)

for epoch in range(ckpt.epoch, epochs):
    sampler.set_epoch(epoch)  # same epoch preserves the restored committed offset
    pending = 0
    for x, y in loader:
        ...
        scaler.scale(loss).backward()
        pending += 1
        if pending == accumulation_steps:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:  # no AMP skip
                scheduler.step()
            ckpt.commit(pending)                 # coherent boundary
            pending = 0
    if pending:                                  # shorter final window
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        ckpt.commit(pending)
    ckpt.finish_epoch()

ckpt.save()               # explicit final save at a committed boundary
ckpt.mark_complete()      # durable stop marker for the launcher
```

A complete runnable version is in [`examples/train.py`](examples/train.py).

## The safe-boundary contract

1. **Restore first, then create the DataLoader iterator.** The sampler is mutated
   by `restore()`; a DataLoader or iterator created before that can capture the
   wrong position.
2. **Call `commit(n)` only at the end of a completed accumulation window**, after
   optimizer/scaler/scheduler handling and after clearing gradients. `n` is the
   number of batches actually consumed in that window.
3. **A shorter final window commits its actual count**, for example `commit(2)`
   when `accumulation_steps=4` but only two batches remain. The snippets above
   sum the microbatch losses; divide by the window size (or the configured
   accumulation count) if you want mean-of-window gradient scaling. A shorter
   window then contributes a proportionally smaller update but is still a
   coherent, committed boundary.
4. **An AMP-skipped optimizer update still consumed batches.** Do the
   `scaler.step`/`scaler.update`/zero-grad handling, optionally skip the scheduler
   step, then commit the consumed batches.
5. **A partial accumulation window is never auto-saved.** Auto-saves happen only
   inside `commit()` and `finish_epoch()`. Do not put a save in an unconditional
   `finally` block.
6. **Do not call explicit `save()` mid-window.** It writes the last committed
   cursor, but live RNG/state may already include partial work. Auto-save from
   `commit()` is the safe path.
7. **Call `finish_epoch()` at epoch end.** It advances the epoch and resets the
   sampler offset. If a crash happened after committing the final batch but
   before `finish_epoch()`, restore yields an empty epoch; call `finish_epoch()`
   and continue.

`commit()` uses a committed consumer cursor. It does not use how many batches
the sampler or DataLoader workers have prefetched. Replaying discarded Dataset
items just to skip ahead is not done; the sampler builds the epoch order and
starts at the committed batch index.

## What is saved

- models, optimizers, schedulers, AMP GradScalers (a single object or a mapping
  of names to objects)
- Python RNG, NumPy RNG if available, torch CPU RNG, and CUDA RNG if available
- epoch, committed batch in epoch, total batches, cumulative batches seen
- sampler epoch, order seed/configuration, committed batch offset, dataset size
  and optional fingerprint
- registered extra components with `state_dict()` / `load_state_dict()`
- arbitrary picklable `metadata` and a user `compatibility` mapping

The checkpoint is written to `run_dir/checkpoints/` as a complete file plus a
SHA-256 sidecar. `run_dir/COMPLETE` is the launcher stop marker.

## Recovery behavior

| Situation | Behavior |
| --- | --- |
| No checkpoint files | `restore()` returns `None`; first run is allowed |
| Newest generation corrupt / truncated / sidecar missing | Visible `CheckpointRecoveryWarning`, falls back to the previous generation |
| All generations corrupt | Raises `RecoveryError`; **does not start fresh** |
| Readable but incompatible | Raises `IncompatibleCheckpointError`; **does not start fresh** |
| Intentional compatibility change | `restore(allow_incompatible=True)` |
| Intentional sampler membership change | `restore(allow_incompatible=True)` resets the current epoch offset to zero |
| User decides to discard a broken run | `ckpt.reset()` (destructive; also removes `COMPLETE`) |

Compatibility checks are best effort: model key/shape checks, optimizer param
group sizes, scheduler/scaler/component key checks, sampler configuration, and
your explicit `compatibility` mapping. revive-me cannot verify that arbitrary
code, dataset contents, or same-size dataset membership are equivalent. Use
`dataset_fingerprint` and `compatibility` for things you can check yourself.

All candidates are loaded and validated before any live object is modified, so a
corrupt newest generation does not partially overwrite live state before
fallback. If an unexpected exception occurs while applying an otherwise valid
checkpoint, live objects may be partially modified; restart the process and
recover explicitly. This limit is inherent in in-place `load_state_dict()`.

## Checkpoint durability and rotation

A save writes:

1. a temporary checkpoint file in the same directory,
2. `flush()` + `fsync()` on the file,
3. a temporary checksum sidecar,
4. `os.replace()` to the generation filename and a directory `fsync`,
5. `os.replace()` for the sidecar and another directory `fsync`,
6. pruning older generations.

Generation filenames are `checkpoint-00000001.pt` with matching
`checkpoint-00000001.sha256`. The newest two sidecar-backed generations are
retained. A failed write does not delete or rename either retained generation.
A process killed between the checkpoint rename and the sidecar rename leaves an
incomplete generation that restore skips with a warning.

- Saves are synchronous, so training pauses for the write.
- With successful saves, typical loss is at most the save interval plus the time
  to the next safe boundary. With write failures, that target is not guaranteed;
  the library warns and retries at a later committed boundary.
- The checkpoint directory must be on persistent storage that survives reboot
  (not `/tmp`, not an ephemeral container layer). Ensure the mount is available
  before the service starts.
- Only two generations are retained. If both are corrupt, recovery refuses to
  start fresh and requires an explicit reset.

### Measured overhead (not a guarantee)

On the development machine (macOS arm64, local APFS/SSD, CPU, Python 3.12.7,
torch 2.15.0), `benchmarks/overhead.py --iterations 5 --width 1024 --depth 8`
produced:

```text
model parameters: 7,414,785
checkpoint file: 28.30 MiB
saves measured: 5
mean save: 17.07 ms
median save: 17.11 ms
worst save: 17.27 ms
nominal overhead at 300s interval: 0.0057% (measured save time / (interval + save time))
```

This is a local measurement of one model/filesystem, not a product guarantee.
Large models, optimizer state, network filesystems, or slow disks can make
synchronous saves exceed the 5% target.

## Sampler restrictions

`ResumableBatchSampler` supports stable map-style datasets with a fixed batch
size and sequential or seeded-shuffled order. `drop_last=True` is the normal
choice; `drop_last=False` is supported with a shorter final batch.

- The order is derived from `seed + epoch`; the committed offset is a batch
  index, not a sample count.
- `make_dataloader()` rejects `IterableDataset`, arbitrary custom samplers, and
  options that would override batch size/order (`batch_size`, `shuffle`,
  `sampler`, `batch_sampler`, `drop_last`). It also rejects
  `in_order=False`: a committed prefix cursor cannot describe a stream whose
  batches arrive in arbitrary order. `in_order=True` is accepted and omitted
  from the `DataLoader` call so older torch versions without that argument
  keep working (ordered delivery is their default).
- `make_dataloader()` checks `len(dataset)` against the sampler's
  `dataset_size`. A same-size membership change is not detectable unless you
  supply and update `dataset_fingerprint`.
- Default DataLoader delivery is ordered. If you reorder or drop batches
  yourself, the committed cursor is no longer meaningful; revive-me cannot
  detect that.
- DataLoader workers prefetch. Their random augmentation state is not saved.
  Resume is consistent at the committed batch boundary but not bit-exact.

## Launcher, systemd, and reboot

The launcher is separate from the library and uses only the standard library:

```bash
python scripts/relaunch.py \
  --run-dir /srv/revive-me/runs/exp1 \
  -- /srv/revive-me/.venv/bin/python /srv/revive-me/examples/train.py \
     --run-dir /srv/revive-me/runs/exp1
```

Behavior:

- Exits immediately if `<run-dir>/COMPLETE` exists.
- Holds an exclusive `flock` on `<run-dir>/.lock` for its whole lifetime,
  including retry sleeps. A second launcher exits with code 75.
- Runs the command, then retries up to **three** times, **60 seconds** apart,
  only on failure and only if `COMPLETE` is still absent. That is one initial
  attempt plus three retries.
- The retry budget is per launcher invocation and resets on the next service
  start or boot. There is no persistent retry counter.
- A command that exits 0 without `COMPLETE` is treated as a clean intentional
  stop and is not retried.
- On SIGTERM/SIGINT it forwards termination to the child and stops retrying.

`deploy/revive-me.service` is an example unit. It uses `Restart=no` on purpose:
the launcher owns bounded retries, and `systemctl enable` (plus
`WantedBy=multi-user.target`) is what starts the run again after a server reboot.
`Restart=` alone does not survive a reboot. See [`deploy/README.md`](deploy/README.md)
for installation steps and the `@reboot` cron alternative. Nothing in this
repository installs a service or edits a crontab.

The run directory passed to the launcher and to `Checkpointer` must be the same
persistent path. The systemd unit sets `WorkingDirectory`, environment file, and
absolute paths explicitly; adapt every `/absolute/path` and `CHANGE_ME`.

## Example

```bash
python examples/train.py \
  --run-dir runs/example \
  --epochs 3 \
  --dataset-size 64 \
  --batch-size 8 \
  --accum 2 \
  --workers 0 \
  --save-interval 300
```

The example shows gradient accumulation, a shorter final accumulation window,
AMP GradScaler handling (including skipped updates), a registered custom
component, resume before DataLoader creation, a final save, and `mark_complete()`.
It sums the losses of the microbatches in a window (the effective gradient is
scaled by the window size); divide the loss by the window size if you need
mean-of-window scaling. Use `--fresh` only when you intentionally want to delete
the run's checkpoints and completion marker.

## Tests and verification

```bash
python -m pytest -q
python examples/train.py --run-dir /tmp/revive-me-smoke --epochs 1 \
  --dataset-size 12 --batch-size 4 --accum 2 --workers 0 --save-interval 0
python benchmarks/overhead.py --iterations 5 --width 1024 --depth 8
```

The test suite covers:

- a subprocess forced to `os._exit(17)` after a committed window, followed by a
  fresh process that restores and checks registered state plus expected next
  sample IDs
- an interrupted run continued in a fresh process and compared end-to-end with
  an uninterrupted reference: momentum optimizer state, scheduler, AMP scaler,
  registered extra state, committed progress, remaining sample IDs, epoch
  transitions, and short final accumulation windows
- kills both after a committed window and in the middle of a window with
  prefetched DataLoader workers; the checkpoint remains at the previous
  committed offset and the resumed run matches the uninterrupted reference
- a full run with a shorter final accumulation window
- AMP-skipped optimizer updates still committing consumed batches
- rejection of out-of-order `DataLoader` delivery (`in_order=False`)
- latest-generation corruption fallback and both-invalid recovery refusal
- missing checksum sidecars, malformed payloads, and atomic-replace/sidecar
  rename failures preserving the prior generation
- compatibility and dataset-membership rejection, including explicit override
- rotation retaining two generations
- write failures warning while training progress continues and the prior
  checkpoint stays usable
- launcher completion suppression, retry budget, clean-exit behavior, and lock
  exclusion

**Executed on this machine:** 40 tests passed in about 14 seconds with Python
3.12.7, torch 2.15.0, NumPy 2.5.4, macOS arm64 CPU. The example and editable
install smoke checks passed. Not executed: a real server reboot, systemd on
Linux, GPU/CUDA recovery, or a production-length five-minute interval run. A
process kill is not evidence of reboot survival; checkpoint persistence depends
on the storage/mount setup described above. The end-to-end comparison is scoped
to this deterministic CPU fixture (no dropout or random transforms); it is not
a claim of general bit-exact replay.

## Limitations and security

- Single process, CPU or one GPU. No distributed training, Lightning/HF
  adapters, cloud/object storage, or internal training engine.
- Not bit-exact: worker RNG, nondeterministic kernels, and data changes can alter
  replay. The guarantee is consistent state at a committed boundary.
- CUDA RNG restore is best effort if the visible device count differs; without
  CUDA it is skipped with a warning. NumPy RNG is skipped with a warning if
  NumPy is unavailable.
- Checkpoints are trusted input. Loading uses `torch.load(..., weights_only=False)`
  because components and metadata may be arbitrary Python objects; a malicious
  checkpoint can execute code. Only load checkpoints you produced or trust.
- Validation cannot prove arbitrary code or dataset equivalence. Model
  architecture changes are caught by key/shape checks where possible; other
  changes need `compatibility`/`dataset_fingerprint` or an explicit reset.
- `reset()` deletes checkpoints and `COMPLETE`; it does not reinitialize live
  model/optimizer objects.
- If the final save fails, `COMPLETE` can still be written if the user calls
  `mark_complete()`; check warnings/logs if the final checkpoint matters.
