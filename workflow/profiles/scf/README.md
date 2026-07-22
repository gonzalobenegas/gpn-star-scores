# Berkeley SCF profile

This workflow profile uses the modern Snakemake Slurm executor and the locked
project environment. It is intentionally limited to CPU partitions:

- `epurdom` runs reference preparation, Polars scans and rewrites, shard
  validation, benchmarks, all BigWig generation/validation/reporting rules,
  and the environment smoke test.
- `high` runs inventory aggregation and non-BigWig reporting jobs.

The profile never selects a GPU server or a lab partition outside this policy,
and workflow rules never request GPU resources. `epurdom` is preemptible, so
material outputs must be written to a temporary sibling, validated, and
atomically renamed before a job succeeds.

SCF removes completed jobs from `squeue` too quickly for reliable BigWig
production polling. BigWig production commands must pass
`--slurm-status-command=sacct`; see the issue #7 execution command in
`docs/bigwig-benchmark.md`.

## Setup

Work from a checkout on a shared filesystem and install the exact locked
environment there:

```bash
uv sync --locked --group dev --python 3.13
```

Choose a durable shared project-scratch directory. Do not set `scratch_root`
to node-local `/tmp` or `/var/tmp`:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --workflow-profile workflow/profiles/scf \
  artifacts \
  --config scratch_root=/path/to/shared/project/scratch
```

The profile limits the workflow to 64 concurrent jobs, retries failures twice,
reruns incomplete outputs, retains centralized Slurm logs under `logs/slurm/`,
and writes reports under `logs/slurm-efficiency/` at the end of each run. Shard
validation, optional Parquet rewrites, and chromosome BigWigs use explicit
Slurm arrays.
Portable resources stay in rules rather than this scheduler profile. New heavy
rules start at four threads, 4 GB, four hours, and 1 GB temporary disk using
`resource_policy.heavy_initial` in the workflow config. After a representative
pilot, set runtime to twice the observed p95 bounded to 30 minutes–6 hours and
memory to 1.5 times peak RSS with a 4 GB minimum. Record the source job IDs and
measurements with any resource update.

## Cluster smoke test

Before production work, submit the two-chromosome smoke target:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --workflow-profile workflow/profiles/scf \
  scf_smoke \
  --config scratch_root=/path/to/shared/project/scratch
```

The two jobs run on `epurdom`, import Snakemake, the Slurm
executor, Polars, PyArrow, and pyBigWig, execute a small Polars lazy query with
the standard `polars-runtime-32` build, and atomically write one validated JSON
report per chromosome.

The latest committed pilot evidence is
[`reports/scf-smoke-pilot.json`](../../../reports/scf-smoke-pilot.json),
including job IDs, wall time, peak RSS as reported by `sacct`, and scheduler
efficiency. The smoke resources remain at the policy minimum because the
reported peaks are below 4 GB and twice the observed p95 is below 30 minutes.

The smoke jobs intentionally submit separately. Slurm executor 2.7.1 has an
[open array-output bug](https://github.com/snakemake/snakemake-executor-plugin-slurm/issues/447):
non-first wildcard tasks can run the correct command but validate the first
task's output. The production rules remain explicitly selected for arrays as
required by the release plan, but production array execution is blocked until
an upstream fix is released, pinned, and passes this workflow's SCF pilot.
An authorized production run that cannot wait for that fix may submit the same
restart units as individual jobs with `--slurm-array-jobs=`. Record that
override and the resulting job IDs with the production evidence; it does not
qualify as the required array pilot.

## Exceptional partition override

For a short, targeted debugging run only, override one rule on the command
line. Target only that rule so the command-line setting cannot remove the
profile's partition assignments for unrelated jobs:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --workflow-profile workflow/profiles/scf \
  validate_source_shard \
  --set-resources validate_source_shard:slurm_partition=high
```

There is no automatic fallback to a GPU or lab partition. Never use this
override to select `yss`, `gpu`, or `jsteinhardt`.

## Publication boundary

Slurm only produces and validates artifacts. After all release gates pass,
run `publish` without this profile from one intentional process:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --cores 1 \
  publish
```

The target does not perform an upload until a publication issue adds an
author-approved upload rule.
