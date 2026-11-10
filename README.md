# revive-me

Durable checkpoint/recovery for plain single-process PyTorch training.

revive-me is a small library addon, not a training engine. You keep your loop,
your optimizer, and your data pipeline. Two entry points sit on one backend:

- **`@resumable` decorator + `run` handle** (recommended): a few clear edits to an
  existing loop. You still register your objects and commit each completed
  accumulation window yourself; it cannot discover them for you.
- **`Checkpointer`** (low level): every step explicit, full control.

In exchange you get synchronous checkpoints at coherent boundaries, committed
within-epoch resume, two-generation rotation, one-writer locking, and a separate
bounded-retry launcher for reboot recovery.

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

## Quick start: `@resumable`

```python
import torch
from revive_me import resumable

@resumable(run_dir=lambda args: args.run_dir, interval_seconds=300)
def train(args, *, run):                       # 1. keyword-only `run` parameter
    torch.manual_seed(args.seed)               #    seed BEFORE run.epochs(): restore overwrites RNG
    model = ...
    optimizer = ...
    scheduler = ...
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    run.register(model=model, optimizer=optimizer,         # 2. say what to save
                 scheduler=scheduler, scaler=scaler,
                 components={"ema": ema},                  #    any state_dict/load_state_dict object
                 compatibility={"arch": "resnet18-v1"},
                 metadata={"config": config})
    loader = run.dataloader(dataset, batch_size=32,        # 3. replaces DataLoader(...)
                            shuffle=True, seed=1234, drop_last=True,
                            num_workers=4, prefetch_factor=2)

    for epoch in run.epochs(args.epochs):                  # 4. replaces range(args.epochs)
        pending = 0
        for x, y in loader:
            ...                                            #    your loop, unchanged
            scaler.scale(loss).backward()
            pending += 1
            if pending == accumulation_steps:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                run.commit(pending)                        # 5. after each COMPLETE window
                pending = 0
        if pending:                                        #    shorter final window
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            run.commit(pending)
        # epoch-end work (eval, epoch scheduler) goes here, as before

train(args)   # first call trains; after a crash the same call resumes
```

A complete runnable version is [`examples/train.py`](examples/train.py).

### What you actually change

| Before (plain loop) | After |
| --- | --- |
| `def train(args):` | `@resumable(run_dir=...)` and `def train(args, *, run):` |
| objects created as usual | one `run.register(model=..., optimizer=..., ...)` call |
| `DataLoader(dataset, batch_size=..., shuffle=True, ...)` | `run.dataloader(dataset, batch_size=..., shuffle=True, seed=..., ...)` |
| `for epoch in range(epochs):` | `for epoch in run.epochs(epochs):` |
| (nothing) | `run.commit(n)` after every optimizer window, including a short final one |

Not changed: your forward/backward code, accumulation logic, AMP handling, optimizer
and scheduler stepping, gradient clearing, and epoch-end work. Without
accumulation, `run.commit()` (default `1`) goes right after `optimizer.step()`
and `zero_grad()`. The decorator does not find your model, optimizer or
boundaries on its own, and it does not patch the optimizer; those edits are the
price of knowing what to save and where the state is coherent.

### Lifecycle

- **Order.** `run.register(...)` and `run.dataloader(...)` only *stage* things (once
  each, either order). `run.epochs(n)` then takes the run-directory lock, checks
  for `COMPLETE`, and restores the newest usable checkpoint into everything
  registered, all before any DataLoader iterator exists. Call them in that order;
  anything else raises `RunStateError`. `register()` does not restore.
- **Fresh `run` per call.** No module-level state. Passing `run=` yourself raises
  `TypeError`. `run_dir`, `interval_seconds`, `retry_seconds` and `fresh` accept a
  value or a callable that receives the decorated function's own call arguments.
- **Commit rules.** `run.commit(n)` takes the number of batches in the finished
  window. It raises if the loader has not yet delivered that many uncommitted
  batches.
- **Epoch boundaries are verified.** An epoch advances only if every batch was
  consumed and committed. A missing `commit`, a partial final window, or a loop
  that `break`s out of the loader early while the outer loop continues raises
  `RunStateError`; nothing is saved and nothing is marked complete. Breaking right
  after consuming *and committing* the last batch is accepted. Iterate the loader
  exactly once per epoch.
- **`len(loader)`** is a per-epoch snapshot: batches this epoch will deliver
  (fewer than a full epoch right after a mid-epoch resume) and it does not shrink
  as you commit, so `i == len(loader) - 1` idioms work. `loader.total_batches` is
  the full epoch.
- **Resuming at an epoch end.** If the crash happened after an epoch's last commit
  but before it was closed, `run.epochs()` yields that epoch again with an empty
  loader (`len(loader) == 0`), so your epoch-end code runs. Keep epoch-end side
  effects idempotent or registered as components.
- **Normal return is not completion.** After your function returns, the wrapper
  writes a final checkpoint and `COMPLETE` **only if** the `run.epochs()` loop ran
  to its end. An early `return` or `break` emits `RunIncompleteWarning`, writes
  nothing, and leaves the run resumable. If the final write fails, `COMPLETE` is
  not written (warning; the next call resumes from the last good checkpoint).
- **Exceptions** (including `KeyboardInterrupt`, `SystemExit`) propagate with no
  save and no `COMPLETE`. The lock is always released.
- **Already complete.** If `COMPLETE` exists, the call logs a warning and returns
  `None` without running the body (a function that legitimately returns `None`
  cannot be told apart by return value; check `run_dir/COMPLETE`). A missing or
  corrupt checkpoint never overwrites a completed run. `fresh=True` (for example
  `fresh=lambda args: args.fresh`) explicitly deletes checkpoints and the marker
  first.
- **One run directory, one writer.** The lock is taken when `run.epochs()` starts,
  not at call time, so a second instance on the same directory fails with
  `CheckpointLockedError` there, after its own setup work. A nested call on a
  different directory is independent. A forked child (for example a DataLoader
  worker) holding an inherited `run` cannot commit.
- **Recovery errors surface at `run.epochs()`**: all generations corrupt raises
  `RecoveryError`, a readable but incompatible checkpoint raises
  `IncompatibleCheckpointError`. Neither starts fresh.
- **Not supported by the decorator in v1:** a loader other than the one
  `run.dataloader()` builds, more than one managed loader, `IterableDataset`,
  custom samplers, `in_order=False`, and the sampler-free epoch-boundary mode
  (use `Checkpointer` for that). `persistent_workers=True` is untested.

## Low-level API: `Checkpointer`

The decorator is a thin layer over this; all of it stays available and unchanged.

```python
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
ckpt.close()              # release the writer lock (or: `with Checkpointer(...) as ckpt:`)
```

A complete runnable version is in [`examples/train_lowlevel.py`](examples/train_lowlevel.py).
The explicit API leaves epoch-boundary, completion and error handling to you; the
decorator adds the checks described above.

## The safe-boundary contract

These rules apply to both entry points; with `@resumable` the first, seventh and
completion steps are done for you (and checked), while `commit` placement stays yours.

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
   `commit()` is the safe path. Without a sampler, `save()` with committed
   batches in the current epoch raises `CheckpointError` (see below).
7. **Call `finish_epoch()` at epoch end.** It advances the epoch and resets the
   sampler offset. If a crash happened after committing the final batch but
   before `finish_epoch()`, restore yields an empty epoch; call `finish_epoch()`
   and continue.

### Without a sampler: epoch-boundary recovery only

Pass `sampler=` (a `ResumableBatchSampler`) for within-epoch recovery. Without
one, an ordinary `DataLoader` always starts an epoch at its first batch, so the
only coherent recovery point is an epoch boundary:

- `commit()` only advances the in-epoch counter; it never auto-saves.
- Auto-saves happen in `finish_epoch()` (and on the interval there).
- Explicit `save()` with batches committed in the current epoch raises
  `CheckpointError`; call it after `finish_epoch()`.
- After a crash, `restore()` returns the last finished epoch (`batch == 0`) with
  weights from that boundary, and your loop re-runs the epoch from its start.
  Lost work is everything since the last finished epoch; the five-minute target
  does **not** apply. Checkpoints written mid-epoch without a sampler by older
  builds are refused with `IncompatibleCheckpointError`.

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
| Intentional compatibility-mapping change | `restore(allow_incompatible=True)` |
| Intentional sampler membership change | `restore(allow_incompatible=True)` resets the current epoch offset to zero |
| Different models/optimizers/components or changed state shapes/keys | Always `IncompatibleCheckpointError`; `allow_incompatible` does not relax this |
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

## One writer per run directory

A `Checkpointer` takes an exclusive, non-blocking `flock` on
`<run-dir>/.checkpointer.lock` **before** it creates, cleans or writes anything,
and raises `CheckpointLockedError` if another owner (another process, a manually
started script, or another instance in the same process) holds it. This is
separate from the launcher's `<run-dir>/.lock`, so a launcher-supervised run and
the library never contend.

- Release it with `ckpt.close()` or `with Checkpointer(...) as ckpt:`. `close()`
  is idempotent and saves nothing; a closed instance refuses `save`, `restore`,
  `reset` and `mark_complete`.
- The kernel releases the lock when the owning process dies (kill, `os._exit`,
  crash). Forked children (for example DataLoader workers) close their copy of the
  descriptor, so a lingering worker cannot pin a dead owner's lock.
- **API compatibility:** code that created a second `Checkpointer` for the same
  run directory while the first was alive (for example to simulate a restart in
  one process) must now `close()` the first. A new `retry_seconds` constructor
  argument and `close()`/context-manager support are additive.
- **Platform limits:** POSIX only (Linux, macOS), local filesystems. On platforms
  without `fcntl` the constructor raises `CheckpointError`. `flock` over NFS and
  some FUSE mounts may be advisory-only or emulated; keep the run directory on a
  local disk. The lock is advisory: it excludes other revive-me owners, not
  arbitrary programs, and builds that predate the lock do not take it.
  `.checkpointer.lock` is left in place after `close()` (deleting lock files is
  racy); it is harmless.

## Checkpoint durability and rotation

A save writes:

1. a temporary checkpoint file in the same directory,
2. `flush()` + `fsync()` on the file,
3. a temporary checksum sidecar,
4. `os.replace()` to the generation filename and a directory `fsync`,
5. `os.replace()` for the sidecar and another directory `fsync`,
6. pruning older generations.

Generation filenames are `checkpoint-00000001.pt` with matching
`checkpoint-00000001.sha256`. Retention keeps the newest two **usable**
generations: older files are deleted only once two newer generations pass their
checksum. A generation just written (or chosen by `restore()`) is known good;
others are verified by SHA-256 only, cached by size/mtime, and no checkpoint is
deserialized on save. A corrupt or sidecar-less generation never counts, so after
a fallback the last good generation survives the next save (the corrupt one is
dropped once two newer good ones exist). Pruning failures are logged and leave
files in place. A failed write does not delete or rename either retained
generation. A process killed between the checkpoint rename and the sidecar
rename leaves an incomplete generation that restore skips with a warning.

Abandoned files are cleaned by `restore()` and each save, under the writer lock
(so no live writer can own them): exactly `.checkpoint-<gen>.(pt|sha256).<pid>.
<32 hex>.tmp`, `checkpoint-<gen>.sha256` with no matching `.pt`, and
`.COMPLETE.<pid>.<32 hex>.tmp` in the run directory. Anything else is left
alone, and a failed delete never blocks training or touches a good generation.
A `.pt` whose sidecar is missing is not deleted on sight (it cannot be verified);
it is treated as unusable and removed by normal retention.

- Saves are synchronous, so training pauses for the write.
- With successful saves, typical loss is at most the save interval plus the time
  to the next safe boundary. With write failures, that target is not guaranteed;
  the library warns, keeps training, and retries at a later committed boundary
  no sooner than `retry_seconds` (default 60) after the last failed attempt, so a
  full disk costs one attempt per minute rather than one per commit. Explicit
  `save()` is never throttled. The interval clock counts from the last
  *successful* save or restore.
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
  The library installs no signal handler and does not checkpoint on SIGTERM;
  work since the last committed save is lost.

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

The example is the `@resumable` form of a plain loop. It shows gradient
accumulation, a shorter final accumulation window, AMP GradScaler handling
(including skipped updates), a registered custom component, and the automatic
final save and `COMPLETE` marker. It sums the losses of the microbatches in a
window (the effective gradient is scaled by the window size); divide the loss by
the window size if you need mean-of-window scaling. Use `--fresh` only when you
intentionally want to delete the run's checkpoints and completion marker. Run the
same command again after an interruption (or after the example finishes, to see
the skip). [`examples/train_lowlevel.py`](examples/train_lowlevel.py) is the same
program written against `Checkpointer` directly (`--run-dir`, `--fresh` and the
other flags are identical).

## Tests and verification

```bash
python -m pytest -q
python examples/train.py --run-dir /tmp/revive-me-smoke --epochs 1 \
  --dataset-size 12 --batch-size 4 --accum 2 --workers 0 --save-interval 0
python examples/train_lowlevel.py --run-dir /tmp/revive-me-smoke-low --epochs 1 \
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
- rotation retaining two generations, including corrupt-latest fallback, then a
  save, then losing the newest generation (recovery from the older one)
- abandoned-file cleanup (exact patterns only, failed cleanup, live-owner files
  preserved) and real subprocesses killed mid-write
- failed-write retry cadence with a controlled clock (no sleeps)
- epoch-boundary recovery without a sampler compared to an uninterrupted run
- writer exclusion across real processes: competing owners rejected, SIGKILL /
  `os._exit` release the lock, a lingering forked child does not pin it, and the
  launcher lock and library lock do not deadlock
- write failures warning while training progress continues and the prior
  checkpoint stays usable
- launcher completion suppression, retry budget, clean-exit behavior, and lock
  exclusion

The `@resumable` layer adds (`tests/test_resumable.py`, in-process, and
`tests/test_resumable_subprocess.py`, fresh processes):

- initial run, returned value, sequential invocations with distinct run
  directories, keyword/conflict handling, signature preservation, and lock release
  after every path
- already-complete skip (also with missing checkpoints and when `COMPLETE`
  appears after the lock is taken) and explicit `fresh=True` restart
- exceptions (`RuntimeError`, `KeyboardInterrupt`, `SystemExit`), early `return`,
  outer `break`, and a function that never starts the epoch loop: no extra save,
  no `COMPLETE`, resumable afterwards
- missing commit, partial final window, inner-loader `break` with the outer loop
  continuing, commit ahead of delivered batches, out-of-order API use
- restore applied at `run.epochs()` (not `register()`), with the sampler positioned
  before the first iterator; per-epoch `len(loader)` snapshot; resume at an epoch end
- failed checkpoint writes still training, a failing final write not marking
  complete, corrupt-newest fallback, all-invalid refusal that changes no files,
  incompatible-checkpoint refusal
- same-directory nested runs rejected, different-directory nested runs
  independent, an inherited `run` in a forked child unable to commit
- real subprocesses: `os._exit(17)` after a commit and mid-window with two
  prefetch workers, then continuation in a fresh process compared with an
  uninterrupted decorated run (model, momentum optimizer, scheduler, AMP scaler,
  extra component, progress, expected sample IDs, short final windows, epoch
  transitions); an uninterrupted decorated run compared with the low-level run

**Executed on this machine:** 113 tests passed in about 43 seconds with Python
3.12.7, torch 2.15.0, NumPy 2.5.4, macOS arm64 CPU (the 67 low-level tests plus
46 for the decorator layer). `pip check` was clean. Smoke checks passed:
`examples/train.py` after `SIGKILL` and resume with `--workers 0` and `--workers 2`;
a wheel installed with `pip install --no-deps --target` into a temporary directory
and used from another working directory; `scripts/relaunch.py` running the
decorated example (including the no-relaunch-after-`COMPLETE` check) with workers 0
and 2. Not executed: a real server reboot, systemd on
Linux, GPU/CUDA recovery, or a production-length five-minute interval run. A
process kill is not evidence of reboot survival; checkpoint persistence depends
on the storage/mount setup described above. The end-to-end comparison is scoped
to this deterministic CPU fixture (no dropout or random transforms); it is not
a claim of general bit-exact replay.

## Limitations and security

- Single process, CPU or one GPU. No distributed training, Lightning/HF
  adapters, cloud/object storage, or internal training engine.
- `@resumable` needs `run.register`, `run.dataloader`, `run.epochs` and a
  `run.commit` per completed window; it does not discover models, optimizers or
  accumulation boundaries, and it is not a zero-edit wrapper. One managed
  loader per run; map-style datasets only. The sampler-free epoch-boundary mode
  is low level only.
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
