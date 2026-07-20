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
tests are marked separately and are not part of the pre-commit gate.

The workflow scaffold should also dry-run locally:

```bash
uv run --locked snakemake --snakefile workflow/Snakefile --cores 1 --dry-run
```

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
