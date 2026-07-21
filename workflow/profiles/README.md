# Execution profiles

Execution profiles keep scheduler-specific settings out of workflow rules.

- `scf/`: Berkeley Statistics Slurm policy for artifact generation and
  validation. See [`scf/README.md`](scf/README.md) before submitting jobs.

Run publication without the SCF profile. The `publish` target is reserved for
one intentional local process after artifact validation.
