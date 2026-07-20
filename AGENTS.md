# Repository guidance for agents

## Purpose and scope

This repository builds, validates, and publishes the GPN-Star genome-wide
score release. Keep workflow code, small test fixtures, manifests, reports,
and release documentation in Git. Keep full-size source data and generated
Parquet, WIG, bedGraph, and BigWig artifacts outside Git.

The Python package lives in `src/gpn_star_scores/`, tests live in `tests/`,
and the Snakemake workflow lives in `workflow/`. Use `uv` and the committed
`uv.lock` for the Python project environment. Use rule-local Conda environments
only for external command-line tools that are not Python project dependencies.

Work from a GitHub issue and keep each branch and pull request focused on that
issue. Read the release epic, the issue, and any merged dependencies before
implementing. State assumptions when repository evidence is incomplete; do
not silently turn them into scientific or release decisions.

When an agent creates an issue, apply the `agent-generated` label and begin its
body with:

> 🤖 Agent-generated draft (Codex). Validate scientific assumptions and release metadata before implementation.

## Data invariants

- Treat the staged Box files as immutable source data. Work from a
  configurable, read-only staged path and never rename, rewrite, or delete the
  source files.
- Preserve supplied chromosome names and one-based `Int64` positions in
  Parquet outputs.
- Entropy rows have `chrom String`, `pos Int64`, `ref String`, and
  `entropy_calibrated Float32`.
- LLR rows have the same keys plus `alt String`, `llr_calibrated Float32`, and
  `abs_llr_calibrated Float32`.
- Treat `abs_llr_calibrated` as an independently supplied score. Do not derive
  it from `llr_calibrated`.
- Preserve source scores as `Float32`. Use `Float64` only for numerically
  sensitive intermediate calculations, such as the logo softmax and entropy,
  and explicitly cast final track values to `Float32`.
- Do not assume source sorting, row-group layout, encodings, statistics, or
  page indexes. Measure them from the staged files before deciding to rewrite.
- Parquet uses the supplied assembly chromosome names and one-based positions.
  BigWig uses UCSC chromosome names and zero-based half-open intervals. Keep
  coordinate conversion explicit and covered by boundary tests.
- Scientific interpretations, calibrated-logo transformations, assembly
  mappings, and changes to public score semantics require author review.

## Workflow and execution

- Keep Snakefiles focused on the DAG, named inputs and outputs, configuration,
  resources, logs, Conda environments, and thin calls. Put non-trivial Python
  transformations, validation, benchmarking, and reporting in importable
  functions under `src/gpn_star_scores/`, with pytest coverage. Do not add
  standalone Python scripts under `workflow/`.
- Run Snakemake through the locked project environment with
  `uv run --locked snakemake`. Python `run:` blocks use that project
  environment; rule-local Conda environments belong on `shell`, `script`, or
  wrapper rules that need external tools.
- Keep Snakemake rules scheduler-neutral. Put SCF partition names and
  scheduler-specific settings in the committed SCF profile.
- Local execution must remain supported. Production cluster execution uses
  CPU resources only: heavy parallel work belongs on `epurdom`, while short
  finalization work may use `high`. Never request `yss`, `gpu`, or
  `jsteinhardt`, and never declare a GPU resource.
- Declare portable `threads`, `mem_mb`, `runtime`, and temporary-disk
  resources on applicable rules. Base production time and memory requests on
  recorded pilot measurements rather than guesses.
- Use chromosome-level or shard-level restart units. Avoid whole-genome jobs
  when a safely restartable smaller unit is available.
- Write every material output to a temporary sibling path, validate it, and
  atomically rename it into place. Interrupted or incomplete outputs must be
  safe to rerun.
- Use shared project scratch for durable intermediates. Do not assume
  node-local `/tmp` is available on `epurdom`.
- Separate artifact generation from publication. Parallel jobs may generate
  and validate artifacts; one intentional process performs remote uploads.

## Validation and evidence

- Before committing, run `uv run --locked pre-commit run --all-files`. This is
  the normal fast gate and includes Ruff linting and formatting, Snakefmt, and
  the complete fast pytest suite.
- Mark slow, network, cluster, and production-data tests explicitly. Keep them
  out of the fast pre-commit gate and document how and where they were run.
- Start with small synthetic fixtures and local dry-runs. Exercise relevant
  schema, chromosome-gap, coordinate-boundary, and failure-restart behavior
  before production-scale work.
- Run the checks relevant to the changed files. Do not claim a command passed
  unless it was run successfully; document missing tools or unavailable
  infrastructure as explicit limitations.
- For cluster changes, complete the smallest representative SCF pilot and
  record job identifiers, wall time, peak RSS, and scheduler efficiency before
  setting production resources.
- For benchmarks, commit the reproducible workflow, machine-readable summary,
  and decision rationale. Keep bulky inputs and outputs in managed artifact
  storage, not Git.
- Pull requests must map every applicable issue acceptance criterion to
  concrete code, tests, reports, or an explicitly documented blocker.

## Safety and publication

- Never commit credentials, access tokens, signed URLs, personal paths, or
  sensitive cluster configuration. Use configuration and environment
  mechanisms documented by the workflow.
- Do not commit full-scale generated data or transient logs. Small,
  intentionally documented test fixtures are allowed.
- Box access is read-only for this project. Do not modify Box objects.
- Creating or updating a private staging dataset must be authorized by the
  relevant issue. Making the Hugging Face repository public, changing public
  artifacts, creating a release tag, or deleting remote data requires an
  explicit author approval at that step.

## Pull requests and review

- Use one issue, one focused branch, and one pull request. Open draft pull
  requests when useful for visibility, and mark them ready only when their
  acceptance evidence is present.
- Keep generated artifacts and unrelated cleanup out of implementation diffs.
- Before author review, run a fresh, read-only review against the base branch.
  The reviewer should check acceptance criteria, data semantics, failure and
  restart safety, reproducibility, tests, resource safety, and unintended
  external effects. Return confirmed findings to the implementation session
  and review the resulting diff again.
- Agent review complements rather than replaces author review. An author must
  approve scientific semantics, release metadata, public visibility, and the
  `v1.0.0` tag.
