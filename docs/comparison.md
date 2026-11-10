# Comparison: native PyTorch, revive-me, and Ravex

A factual side-by-side of three ways to resume a PyTorch training loop. It is not
a ranking and contains no speed, reliability, popularity or overhead comparison.

**Evidence levels.** Only the revive-me column describes code tested in this
repository (and only as stated). The native and Ravex columns restate their
authors' documentation as read on 2026-11-10; nothing from them was installed,
run or audited here. "Not stated" means the page I read does not say, which is not
the same as "not supported". Ravex releases frequently, so re-check its page.

Sources:

- Native: PyTorch tutorial,
  [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
  (section "Saving & Loading a General Checkpoint for Inference and/or Resuming Training").
- Ravex: [PyPI project page, version 0.11.2](https://pypi.org/project/ravex/)
  (the version served on 2026-11-10; PyPI lists it as uploaded 2026-11-08).
- revive-me: this repository at commit `bdcd0dd`, [README](../README.md) and tests.

| | Native PyTorch recipe | revive-me | Ravex (documented, 0.11.2) |
| --- | --- | --- | --- |
| Availability | part of PyTorch | private repository, not on PyPI; use from an authorized checkout | public on PyPI (`pip install ravex`); Apache-2.0 |
| Python / torch | whatever you run | Python 3.10+, `torch>=2.3` | Python 3.11+; does not depend on torch |
| Integration edits | you write the save and load code: a dict of `epoch`, model and optimizer `state_dict`s, loss, written with `torch.save` and loaded with `torch.load` | decorator `@resumable`, one `run.register(...)`, `run.dataloader(...)` in place of `DataLoader`, `run.epochs(n)` in place of `range(n)`, `run.commit(n)` after each optimizer window; objects must be registered explicitly | "one decorator on the function that trains" (`@ravex.train_loop`); objects are noticed as they are built, `ravex.track(...)` names them when detection is a guess; it advises setting `max_steps` |
| State covered | what you put in the dict; the page shows model, optimizer, epoch, loss | model, optimizer, scheduler, AMP scaler, Python/NumPy/torch/CUDA RNG, epoch and committed batch, extra `state_dict` objects, metadata | optimizer, LR scheduler, AMP loss scale, torch/CUDA/Python/NumPy RNG, dataset position (per its page) |
| Scheduler, scaler, RNG, data position | not part of the recipe on that page | yes (see above) | yes (see above) |
| Data position | not covered by the recipe | committed-batch cursor through one managed map-style loader with an ordered, seeded sampler; no `IterableDataset`, custom sampler, or multiple managed loaders | its page says position is replayed through a dataset index sampler and that an `IterableDataset`, which has none, cannot be replayed |
| When it saves | whenever your code calls `torch.save` | synchronously at the next committed accumulation boundary after `interval_seconds` has elapsed (default 300), plus epoch-boundary and final saves when due | every `checkpoint_every` steps (default 100) with `torch_save`; per-step deltas with the default Moonclip engine; also a final checkpoint on SIGTERM unless disabled |
| Write safety | your responsibility; not discussed on that page | temp file, fsync, SHA-256 sidecar, two generations, fallback to the previous one, refusal to start fresh when all are corrupt | not stated on the page beyond "a failed checkpoint is lost, logged, and tried again"; byte writing is delegated to its Moonclip engine or `torch.save` |
| Restart | you rerun the script and your load code | rerun the same command; plus a separate launcher (`scripts/relaunch.py`) and systemd/cron templates for boot | rerun the same command; boot or service setup not described on the page |
| Storage | any path you choose | local filesystem, POSIX `flock` | local, S3 or R2 (per its page, through its Moonclip engine) |
| Scope | single or multiple processes, your design | single process, CPU or one GPU; no DDP/FSDP | documents DDP and FSDP resume; Lightning/Hugging Face Trainer interplay noted with limits |
| Determinism statement | none on that page | consistent state at the committed boundary; not bit-exact in general (worker RNG, nondeterministic kernels) | its page says plain loops, DDP and FSDP replay exactly, with listed exceptions (for example Trainer/Lightning with shuffle and dropout) |
| Environment validation | not applicable (documentation) | 113 tests passed at `bdcd0dd`: Python 3.12.7, torch 2.15.0, macOS arm64, CPU, local APFS. Not run: real reboot, GPU, Linux/systemd, NFS | its page says unit tests run on any OS and distributed/integration tests (SIGKILL, `torchrun`) need Linux; results not independently checked here |
| Notable limits | all of it is manual | not zero-edit; POSIX/local FS only; unpublished; single process | per its page: `IterableDataset` position, sharded model has no final checkpoint at exit, pre-1.0 (0.x) versioning |

## Unknown or not independently tested

- Ravex behavior of any kind, including its claimed exact replay, SIGTERM
  handling, S3/R2 storage and DDP/FSDP resume.
- How either tool behaves across a real machine reboot. Neither page describes
  one; revive-me's own reboot path is untested here.
- Overhead, throughput, checkpoint size or reliability comparisons. None were
  measured, and revive-me's one local save-time benchmark in the README is not
  comparable to anything here.
- Whether revive-me supports a workload Ravex documents (DDP, FSDP,
  `IterableDataset`, object storage): it does not currently support them.

## Reading the differences

The recipe asks the most of you and promises nothing about atomicity, rotation or
data position. Ravex's documented interface describes the smallest code change (one
decorator, plus a recommended `max_steps`) and a wider scope (multi-GPU, object
storage, SIGTERM handling); its claims are unverified here. revive-me asks for
explicit registration and commit placement in exchange for known save points and
tested refusal behavior, within a narrow, currently unpublished scope.
