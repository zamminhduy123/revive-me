# revive-me documentation

revive-me: PyTorch checkpointing and resume for single-process training loops.
It saves everything a loop needs to continue (not just weights) at consistent
boundaries, restores it when the same command runs again, and ships a separate
launcher plus deploy templates for starting that command after a server reboot.

Availability: the repository is currently private and the package is not on PyPI.
These docs assume a checkout you are authorized to use; see the
[README](../README.md#install) for setup.

## Tutorials

| Question | Page |
| --- | --- |
| How do I continue a PyTorch run after the server reboots? | [Resume PyTorch training after a server reboot](resume-pytorch-training-after-reboot.md) |
| Does tmux keep training alive across a reboot? | [Does tmux keep training alive after a reboot?](tmux-and-reboot.md) |
| What must a resumable checkpoint contain besides the weights? | [What to save besides model weights for PyTorch resume](what-to-save-for-pytorch-resume.md) |

## Reference and background

- [Comparison](comparison.md): native PyTorch checkpoint recipe, revive-me, and
  Ravex's documented interface.
- [README](../README.md): full API, safe-boundary contract, recovery behavior,
  limitations.
- [deploy/README.md](../deploy/README.md): systemd and `@reboot` cron templates.
- [AI discoverability strategy and release checklist](ai-discoverability.md):
  internal planning notes, not user documentation.
