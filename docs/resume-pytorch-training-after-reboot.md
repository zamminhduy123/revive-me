# Resume PyTorch training after a server reboot

Scope: a single-process PyTorch training script (CPU or one GPU) on a POSIX
machine with a local filesystem. revive-me is not on PyPI and the repository is
currently private, so everything below runs from a checkout you are authorized
to use.

## Two separate jobs

A reboot ends every process. Continuing afterwards needs two independent things:

1. **Restore state.** Training saved a checkpoint to persistent storage earlier,
   and the next run loads it: model, optimizer, scheduler, AMP scaler, RNG state,
   your registered extras, and the data position. In revive-me this is the
   `@resumable` decorator (or `Checkpointer`). It happens when the training
   command starts.
2. **Start the command again.** Something has to run the same training command
   after boot. revive-me does not do this inside the library. The separate
   `scripts/relaunch.py` launcher and the templates in [`deploy/`](../deploy/README.md)
   (systemd unit, `@reboot` cron line) are for that.

Without job 1 the script starts from step 0. Without job 2 nothing runs and the
checkpoint sits unused.

## What you lose, and what can fail

- **Lost work.** Everything since the last successful save. Saves happen at
  committed accumulation boundaries (`run.commit(...)`) once `interval_seconds`
  (default 300) has elapsed, so the typical loss is at most the interval plus the
  time to the next boundary. If a write fails, the library warns, keeps training
  and retries no sooner than `retry_seconds` (default 60); the interval then no
  longer bounds the loss.
- **No save on shutdown.** The library installs no signal handler and does not
  checkpoint on SIGTERM, so an orderly `shutdown` loses the same work as a power
  cut.
- **Non-persistent storage looks like a fresh start.** If the run directory is
  on `/tmp`, a container layer or a mount that is not ready, there is nothing to
  restore and training starts from scratch without an error. Put the run
  directory on storage that survives reboot and is mounted before the service
  starts.
- **Refusals instead of silent fresh starts:**

| Situation | Result |
| --- | --- |
| Newest checkpoint corrupt or incomplete | `CheckpointRecoveryWarning`, falls back to the previous generation |
| Every generation corrupt | `RecoveryError`, does not start fresh |
| Checkpoint does not match the registered model/optimizer/etc. | `IncompatibleCheckpointError`, does not start fresh |
| Another process owns the run directory | `CheckpointLockedError` |
| Run already finished (`COMPLETE` exists) | the call returns `None`; the launcher does not start the command |
| Missing commits or an incomplete inner loader while the epoch loop continues | `RunStateError`; no additional save or completion marker |
| Early return or outer-loop `break` | `RunIncompleteWarning`; no final save or completion marker |

## 1. Set up a checkout

```bash
git clone https://github.com/zamminhduy123/revive-me.git   # needs access to the private repository
cd revive-me
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

Python 3.10+ and `torch>=2.3` are required. `pip install -e .` is enough to run
the examples; `[test]` adds pytest and NumPy. To use a specific CUDA/CPU build of
torch, install that first so the `torch>=2.3` requirement is already satisfied.

## 2. The code you run

[`examples/train.py`](../examples/train.py) is a complete decorated loop on a tiny
synthetic problem. The edits it makes to a plain loop are the decorator, one
`run.register(...)`, `run.dataloader(...)` instead of `DataLoader(...)`,
`run.epochs(n)` instead of `range(n)`, and `run.commit(n)` after each optimizer
window. See the [README](../README.md#quick-start-resumable) for the pattern and
[what to save](what-to-save-for-pytorch-resume.md) for the details.

## 3. Interrupt and resume on one machine

This shows checkpoint restore only. It uses a temporary directory, so nothing
persistent is touched. `SIGKILL` stops the process abruptly, like a crash; it is
**not** a validated power-loss or reboot test: page cache, filesystem and disk
behavior, and the boot launcher are not exercised. Nothing here reboots the
machine.

Run from the repository root with the venv active:

```bash
TMP=$(mktemp -d)
RUN="$TMP/run"
ARGS="--run-dir $RUN --epochs 1500 --dataset-size 64 --batch-size 8 --accum 2 --workers 0 --save-interval 0"

# Start training, wait for the first checkpoint, let it run a bit, then kill it.
python examples/train.py $ARGS > "$TMP/first.log" 2>&1 &
PID=$!
until [ -n "$(ls "$RUN/checkpoints" 2>/dev/null)" ]; do sleep 0.1; done
sleep 2
kill -KILL "$PID"
wait "$PID" || echo "first process exited with status $?"   # 137 = SIGKILL
test ! -e "$RUN/COMPLETE" && echo "no COMPLETE: the run is unfinished"

# Same command again: it resumes.
python examples/train.py $ARGS | sed -n '1p;$p'
test -e "$RUN/COMPLETE" && echo "COMPLETE present"

# And once more: a finished run is not started again.
python examples/train.py $ARGS
```

`--save-interval 0` saves at every commit so the short demo has something to
restore; the default is 300 seconds. Expect output like the following (numbers
vary with timing):

```text
Resumed generation 2832: epoch 566, committed batch 4, batches seen 4532
Run complete; final mean loss 0.000732; marker <tmp>/run/COMPLETE
<tmp>/run is already complete; not running train (returning None). Use fresh=True to delete it and start over.
<tmp>/run/COMPLETE exists; run is already complete. Pass --fresh to start over explicitly.
```

If the kill lands during a save you may also see `Checkpoint generation N is
unusable ... trying an older generation.`: the expected fallback to the previous
generation. On this deterministic CPU fixture the final mean loss of the resumed
run matched an uninterrupted run (`0.000732` on the machine where this page was
checked); this is a property of the fixture, not a general bit-exact guarantee.
`tests/test_resumable_subprocess.py` covers the comparison. Remove the temporary
directory with `rm -rf "$TMP"` when done.

## 4. Add the launcher

`scripts/relaunch.py` is what a boot service runs. It exits if `COMPLETE` exists,
otherwise runs your command and retries a failing command up to three more times,
60 seconds apart (retry budget resets on every launcher start). To rehearse the
"machine died, then boot started it again" sequence without a reboot, kill the
launcher and the training process together, then start the launcher again:

```bash
TMP=$(mktemp -d)
RUN="$TMP/run"
TRAIN="python examples/train.py --run-dir $RUN --epochs 1500 --dataset-size 64 --batch-size 8 --accum 2 --workers 0 --save-interval 0"

python scripts/relaunch.py --run-dir "$RUN" -- $TRAIN > "$TMP/launcher.log" 2>&1 &
LAUNCHER=$!
until [ -n "$(ls "$RUN/checkpoints" 2>/dev/null)" ]; do sleep 0.1; done
sleep 2
pkill -KILL -P "$LAUNCHER"     # the training child
kill -KILL "$LAUNCHER"         # the launcher
wait "$LAUNCHER" 2>/dev/null || true

python scripts/relaunch.py --run-dir "$RUN" -- $TRAIN 2>&1 | sed -n '1,2p;$p'   # what boot would do
python scripts/relaunch.py --run-dir "$RUN" -- $TRAIN; echo "launcher exit $?"
```

The first launcher call prints the attempt line, the `Resumed generation ...` line
and, at the end, `run is complete`. The last command prints `run is complete, not
launching` and `launcher exit 0`. Again, a SIGKILL rehearsal, not a reboot.

## 5. Real boot setup

[`deploy/README.md`](../deploy/README.md) describes the systemd unit
(`revive-me.service`) and the `@reboot` cron alternative. Both are templates:
nothing in this repository installs a service or edits a crontab. Before relying
on them:

- Replace every `/absolute/path`, `User`, `Group` and the run directory; use the
  same persistent run directory for the launcher and the training command.
- `systemctl enable` is what starts the service after a reboot. The unit uses
  `Restart=no` on purpose because the launcher owns the bounded retries.
- If the run directory is on a separate mount, declare the dependency
  (`RequiresMountsFor=`) as the template notes.
- Rehearse once on your own hardware at a time you choose: start training, let
  it pass a checkpoint, reboot, then check `journalctl -u revive-me` for the
  `Resumed generation ...` line.

What has been validated: the test suite (113 tests at commit `bdcd0dd`) and the
kill-and-resume procedures above, on one macOS arm64 CPU machine. Not validated:
a real server reboot, systemd on Linux, GPU/CUDA recovery, NFS or other network
filesystems. The lock is POSIX `flock` on local filesystems only.
