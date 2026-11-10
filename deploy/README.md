# revive-me deploy notes

These files are templates.  Nothing here installs a service, modifies a crontab,
or contacts an external system.

## systemd (recommended for reboot survival)

1. Copy `revive-me.service` to `/etc/systemd/system/`.
2. Replace `User`, `Group`, every `/absolute/path`, and the run directory.
3. Ensure the run directory is on storage that survives reboot (not `/tmp`, not
   an ephemeral container layer).  If it is a separate mount, add
   `RequiresMountsFor=` and the appropriate `After=` target.
4. `sudo systemctl daemon-reload`
5. `sudo systemctl enable --now revive-me.service`
6. Check `systemctl is-enabled revive-me.service` and `journalctl -u revive-me`.

`enable` creates the boot symlink; `Restart=no` is intentional because
`relaunch.py` owns the bounded retry policy.  The unit is started again at the
next boot.

## @reboot cron (alternative)

Use the line in `crontab.example` only if you are not using the systemd unit.
It starts once at boot.  The launcher still performs up to three retries sixty
seconds apart.  Do not edit a user's crontab automatically.

## Completion and retry semantics

- `<run-dir>/COMPLETE` is written by `Checkpointer.mark_complete()`.  If it
  exists, `relaunch.py` exits without starting training.
- `<run-dir>/.lock` is held with `flock` for the whole launcher lifetime,
  including retry sleeps.  A second launcher exits with code 75.
- `<run-dir>/.checkpointer.lock` is the library's writer lock, held by the
  training process (not the launcher) while a `Checkpointer` is open.  It stops
  a manually started script from writing into a supervised run's checkpoints;
  the kernel releases it when the training process dies.  The launcher never
  touches it.
- On stop, systemd's SIGTERM is forwarded by the launcher and terminates the
  training command.  revive-me does not checkpoint on SIGTERM.
- One launcher invocation makes the initial attempt plus up to three retries,
  sixty seconds apart.  The retry budget resets on each launcher/service start
  (including a boot).  A command that exits 0 without COMPLETE is treated as a
  clean intentional stop and is not retried.
- A write failure inside the library warns and continues training.  It does not
  update the completion marker.
