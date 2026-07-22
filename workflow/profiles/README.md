# Execution profiles

Execution profiles keep scheduler-specific settings out of workflow rules.

- `scf/`: Berkeley Statistics Slurm policy for artifact generation and
  validation. See [`scf/README.md`](scf/README.md) before submitting jobs.

Run publication without the SCF profile. The `publish` target is reserved for
the author-approved public Hugging Face release and is guarded against Slurm
execution. See [`../../docs/hugging-face-release.md`](../../docs/hugging-face-release.md).
