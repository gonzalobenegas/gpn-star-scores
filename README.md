# GPN-Star genome-wide scores

Workflows and release metadata for validating and publishing the GPN-Star
genome-wide scores. The public data release will live at
`songlab/gpn-star-scores` on Hugging Face.

## Development setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/) for a locked
Python environment:

```bash
uv sync --locked
uv run --locked pre-commit install
```

Run the complete local quality gate before committing:

```bash
uv run --locked pre-commit run --all-files
```

That command checks file hygiene, Ruff linting and formatting, Snakefmt, and
the complete fast pytest suite. Slow, network, cluster, and production-data
tests use the `slow`, `network`, `cluster`, and `production_data` markers and
are not part of the pre-commit gate.

The workflow scaffold should also dry-run locally:

```bash
uv run --locked snakemake --snakefile workflow/Snakefile --cores 1 --dry-run
```

Release artifacts and publication are separate targets. Build artifacts with
`artifacts`; invoke `publish` only from one intentional local process after all
release gates pass. Berkeley SCF users should follow the committed
[`workflow/profiles/scf`](workflow/profiles/scf/README.md) profile, including
its two-chromosome environment smoke test, before submitting production work.

## Repository layout

- `src/gpn_star_scores/`: importable Python transformation, validation,
  benchmark, and reporting logic.
- `tests/`: hermetic tests and small documented fixtures.
- `workflow/Snakefile`: workflow entry point.
- `workflow/rules/`: DAG definitions and thin calls into the Python package.
- `workflow/envs/`: pinned Conda environments for external command-line tools.
- `workflow/config/`: portable workflow configuration.
- `workflow/profiles/`: execution profiles, including the planned SCF profile.

Python, Snakemake, Polars, PyArrow, pyBigWig, and development tools are locked
in `uv.lock`. Rule-local Conda environments are reserved for non-Python tools
such as UCSC utilities; they do not duplicate the Python project environment.

## Staged score inventory

Issue #8's production workflow is opt-in because its inputs are immutable,
shared staged data rather than repository fixtures. Copy
`workflow/config/inventory.example.yaml` outside Git, replace its generic
paths, capacity evidence, and pilot-derived resource values, then run:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/inventory.yaml \
  --cores 1 \
  --dry-run
```

The workflow reads each expected Parquet shard and supplied Ensembl FASTA but
never writes beside them. Each FASTA must retain its original pinned filename
and match an author-approved SHA-256. The workflow prepares memory-mappable
reference contigs under the configured scratch output, validates each shard as
a chromosome-level restart unit, and atomically promotes:

- one JSON validation record per shard under `shards/`;
- reference provenance and SHA-256 records under `references/`;
- `release/manifest.json`, containing checksums, schemas, row counts, physical
  Parquet layout, coordinate bounds, and validation results;
- `release/summary.md`, the human-readable status and blocker report.

The Box root currently reports 333,761,247,733 bytes. A read-only API inventory
on 2026-07-20 found that the 290 current Parquet files themselves total
333,761,235,219 bytes; the 12,514-byte difference is consistent with the README
and its retained versions. Keep `expected_shard_bytes` unset until an author
approves which release number is canonical, then record that decision in the
production config and pull request evidence.
