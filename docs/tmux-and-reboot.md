# Does tmux keep training alive after a reboot?

**No.** tmux protects a running program from a lost terminal or SSH connection. It
does not protect it from the machine restarting, and it does not save training
state.

According to tmux's own [Getting Started guide](https://github.com/tmux/tmux/wiki/Getting-Started),
tmux keeps its state in a single main process, the tmux server. Detaching, or a
dropped connection, leaves that server and the programs inside it running; its
stated use is to "protect running programs on a remote server from connection
drops". A reboot ends all processes, including that server and your training
process, and the guide does not describe surviving one. (The reboot part is
standard operating-system behavior, not a tmux claim.)

| Event | tmux session | Training process | What brings it back |
| --- | --- | --- | --- |
| Close the terminal or SSH drops | keeps running | keeps running | `tmux attach` |
| Detach (`C-b d`) | keeps running | keeps running | `tmux attach` |
| Training crashes or is killed (OOM, `kill -9`) | still there | gone | a checkpoint plus rerunning the command |
| `tmux kill-server` / killing the session | gone | gone | a checkpoint plus rerunning the command |
| Server reboot | gone | gone | a checkpoint on persistent storage plus something that starts the command at boot |

So surviving a reboot takes two things that tmux provides neither of:

1. A checkpoint of the full training state (weights **and** optimizer, scheduler,
   AMP scaler, RNG state, data position) on storage that survives reboot. See
   [what to save](what-to-save-for-pytorch-resume.md).
2. A launcher that runs the training command again after boot, for example a
   systemd unit or `@reboot` cron line running `scripts/relaunch.py`. See
   [Resume PyTorch training after a server reboot](resume-pytorch-training-after-reboot.md)
   and [`deploy/`](../deploy/README.md).

You can still run the launched command inside tmux to watch it; just do not count
on tmux for recovery.

revive-me covers the first part and ships templates for the second, for
single-process CPU or one-GPU PyTorch loops. It is currently a private,
unpublished repository (install from a checkout you are authorized to use; see the
[README](../README.md#install)). A real reboot, systemd on Linux and GPU recovery
have not been tested; see the [limitations](../README.md#limitations-and-security).
