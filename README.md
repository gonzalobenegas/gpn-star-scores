# GPN-Star genome-wide scores

Workflows and release metadata for validating and publishing the GPN-Star
genome-wide scores. The public data release is
[`songlab/gpn-star-scores`](https://huggingface.co/datasets/songlab/gpn-star-scores)
on Hugging Face.

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

The issue #5 Parquet layout benchmark is opt-in, uses the committed SCF profile,
and hash-validates its inputs against issue #8's inventory manifest before any
rewrite or benchmark. Production remains gated on that manifest plus
issue-specific pilot resources.
Its candidates, measurement method, selection rule, staging boundary,
and production configuration are documented in
[`docs/parquet-layout-benchmark.md`](docs/parquet-layout-benchmark.md).

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
never writes beside them. It rejects an output root that resolves inside the
immutable staged source tree. Each FASTA must retain its original pinned
filename and match an author-approved SHA-256. The workflow prepares
memory-mappable reference contigs under the configured scratch output,
validates each shard as a chromosome-level restart unit, and atomically
promotes:

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

## BigWig tracks

Issue #7's opt-in generation and validation workflow is described in
[`docs/bigwig-benchmark.md`](docs/bigwig-benchmark.md). It consumes the merged
inventory, Parquet-layout, and SCF-profile interfaces from issues #8, #5, and
#9, benchmarks WIG conversion against direct streaming, and produces five
validated final tracks for each score set. Final BigWigs use the configured
three-decimal visualization precision; the Parquet files remain the canonical
full-precision scores.

Issue #15's additive raw calibrated-LLR workflow is documented in
[`docs/raw-calibrated-llr.md`](docs/raw-calibrated-llr.md). It reads only the
immutable LLR Parquet shards and produces `llr_A`, `llr_C`, `llr_G`, and
`llr_T` for all eight score sets. Alternate alleles retain signed
`llr_calibrated`; the reference allele is an explicit zero.
`abs_llr_calibrated` is not used or derived. Generation, validation, and
publication are isolated from the 40 immutable v1 BigWigs, so the post-v1
workflow neither rebuilds nor revalidates them.
The public browser catalog contains 72 BigWigs: the 40 v1 tracks retain
immutable artifact revision `5c799b2ec6aa089f0caa8294ae72adb4510f81ae`,
and the 32 raw LLR tracks use additive artifact revision
`47e7f051113abab49f04f43f9107cae2cbbfd34d`. The published dataset card and
hub metadata are at revision `6fbdd6e8754080c08b9db34a78282e6ac04398b7`.
The measured issue #15 resource and validation evidence is recorded in
[`reports/raw-llr-pilot`](reports/raw-llr-pilot/README.md).

## Hugging Face release

Issue #4's public release workflow is documented in
[`docs/hugging-face-release.md`](docs/hugging-face-release.md). It consumes the
complete source-layout and BigWig validation evidence, generates a checksummed
dataset card with 16 explicit Parquet configurations, and publishes only from
one intentional non-Slurm process. The final validation runs without
credentials and records the immutable Hugging Face commit SHA after checking
remote identities, direct Polars pushdown, counted Parquet range reads, dataset
card rendering, and HTTP byte ranges
for all 40 BigWigs. It records Dataset Viewer readiness separately so hosted
preview indexing does not block access to the public release.

## UCSC track hub

Issue #6's opt-in workflow is documented in
[`docs/ucsc-track-hub.md`](docs/ucsc-track-hub.md). It builds one
multi-assembly hub with eight model groups. Each group contains a conventional
one-dimensional entropy signal, one stacked A/C/G/T sequence-logo view, and
the issue #15 raw signed-LLR composite after that extension is enabled.
Entropy defaults to `dense`. LLR defaults to `full`, with positive values blue
and negative values red. The derived sequence logo retains nucleotide colors.
Existing v1 URLs stay pinned to their immutable issue #4 revision, while the
32 additive URLs pin their own artifact revision. Extension validation checks
only the new BigWigs; `hubCheck` still validates the complete hub structure.
Afterward, a separate approval-gated target can update the public dataset card
and `ucsc/` metadata in one commit.
The committed
[`reports/ucsc-track-hub-preflight`](reports/ucsc-track-hub-preflight/README.md)
records the passing production preflight, measured resources, anonymous public
validation, and base/zoom browser rendering for all eight model groups. The
public hub is available from the dataset's stable
[`resolve/main/ucsc/hub.txt`](https://huggingface.co/datasets/songlab/gpn-star-scores/resolve/main/ucsc/hub.txt)
entry URL, while all 40 BigWigs remain pinned to the immutable issue #4
artifact revision.

## End-to-end release QA

Issue #2's opt-in [`qa` workflow](docs/end-to-end-qa.md) repeats anonymous
public artifact, Polars, BigWig, Dataset Viewer, and UCSC hub checks and
reconciles them with the complete 290-shard inventory, 40-track audit, locked
environment, SCF profile, and Slurm efficiency evidence. It writes an immutable
release record while keeping the `v1.0.0` tag behind a separate exact-commit
author-approval gate. The workflow never creates or pushes a tag on its own.
The committed
[`reports/v1.0.0-qa-preflight`](reports/v1.0.0-qa-preflight/README.md)
records the passing anonymous artifact, Viewer, Polars, and UCSC checks plus the
published bounded-join metadata correction. Routine generated dataset-card
updates use the separate approval-gated `publish_dataset_card` target, which
uploads only `README.md` and performs no BigWig checks; the full `publish_hub`
validation remains reserved for hub or track changes.
