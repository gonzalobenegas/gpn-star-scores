# Parquet layout benchmark

This is the reproducible implementation plan for
[issue #5](https://github.com/gonzalobenegas/gpn-star-scores/issues/5). A
production selection remains gated on a complete source-validation manifest
from issue #8 and issue-specific `epurdom` pilot evidence. The issue #8
inventory workflow and Berkeley SCF profile from issue #9 are merged. The
profile provides the scheduler policy used here, and its
documented Slurm array-output blocker must be cleared before production array
execution. The default workflow therefore keeps the benchmark disabled.

## Inputs and author decisions

The production run uses the complete entropy and LLR shards for `gg6 chr32`
and both complete score shards for one representative `hg38-v100` chromosome.
The author approved chromosome 22 as that representative human chromosome on
2026-07-21.

The staged source root is configurable and read-only. Candidate output,
benchmark reports, and logs belong on durable shared project scratch. The
workflow never writes to the staged root and rejects an output path that
resolves to its source path.

The three candidates are fixed in code:

1. `source`: use the source file unchanged.
2. `zstd-262144`: Zstandard level 3, 262,144-row groups, statistics and page
   indexes for every column, content-defined chunking, and dictionary encoding
   only for `chrom`, `ref`, and `alt` when present.
3. `zstd-1048576`: the same physical settings with 1,048,576-row groups.

Rewrites stream complete shards, retain input row order, and preserve the
Arrow schema. Every final value is compared to the source, with `Float32`
values compared by bit pattern so that the independently supplied
`abs_llr_calibrated` score is never recomputed. A candidate is written to a
temporary sibling, its physical layout and exact values are validated, and it
is atomically promoted only after both checks pass.
Rewrite reports separate write and validation time and record the promoted
candidate's SHA-256 so the staging copy can be matched to the local
artifact before remote measurements.

## Measurement method

Each case/candidate/access combination runs these queries:

- first, middle, and last one-based intervals at widths of 1 kb and 1 Mb;
- a projected scan of calibrated score columns;
- a sparse join on the available `(chrom, pos, ref[, alt])` keys; and
- a full scan that consumes every column.

Each query gets exactly one warm-up followed by five measured repetitions.
Reports contain all five durations, their median and inclusive linearly
interpolated p95, result row counts, file size, environment versions, and
process peak RSS. They also capture generation time, Slurm job/array IDs and
partition when present, and the exact sparse keys. Local and staged-Hugging-Face
runs are separate. Timed HF queries use PyArrow against a seekable counted file
so row-group statistics drive HTTP range reads; passing a Python file object
directly to Polars would materialize the whole object and is therefore not used
for transferred-byte measurements. HF reports record the response-body bytes
fetched by the range reader and include the configured range-cache block size.

The sparse keys are deterministic: up to the configured count are taken from
the union of the first, middle, and last 1 Mb intervals. Setup queries run
before the required warm-up and are excluded from timing.

Direct `pl.scan_parquet("hf://...")` checks independently require an optimized
plan with a selection and projection and collect one row. Those checks are
inputs to the final selection and cannot be replaced by a manual assertion.

## Functional evidence

`inventory_manifest` points directly to issue #8's `release/manifest.json`.
Before any rewrite or benchmark, `validate_source_shard` requires the manifest
to describe all 290 expected shards with no missing, unexpected, unreported,
or invalid source shards. For each configured case it then matches the
canonical relative path and score type, requires zero ordering violations,
and hashes the current staged file to prove its size and SHA-256 still match
the manifest. The resulting source-evidence reports, including the manifest
SHA-256, are inputs to every downstream job and to final selection.

The inventory manifest's overall `release_ready` value is recorded but is not
a benchmark gate: issue #8 can block release for organization-capacity
evidence that does not change whether a source shard is valid for a staged
layout benchmark. Its complete shard-validation section is mandatory.

`staging_checks` contains only the human-observed staging Dataset Viewer
result plus the identities of the staged objects. Each passing result requires
a nonblank Viewer URL and check time and an `artifacts` entry for every case.
Each entry's pinned `hf://` URI and SHA-256 must match the HF benchmark report
and either issue #8's source hash or the generated rewrite report.
Local benchmark reports derive row-group position-statistics usability for
every candidate. Exact rewrite equality propagates validated source ordering
to rewritten candidates. Direct `hf://` behavior is derived exclusively from
workflow-generated Polars check reports. Any similarly named assertions in
the staging file are ignored.

```json
{
  "source": {
    "dataset_viewer": true,
    "evidence": {
      "dataset_viewer_url": "https://huggingface.co/datasets/OWNER/PRIVATE-STAGING",
      "dataset_viewer_checked_at": "YYYY-MM-DDTHH:MM:SSZ",
      "artifacts": {
        "CASE_ID": {
          "uri": "hf://datasets/OWNER/PRIVATE-STAGING@REVISION/path.parquet",
          "sha256": "64_LOWERCASE_HEX_CHARACTERS"
        }
      }
    }
  },
  "zstd-262144": {
    "dataset_viewer": true,
    "evidence": {
      "dataset_viewer_url": "https://huggingface.co/datasets/OWNER/PRIVATE-STAGING",
      "dataset_viewer_checked_at": "YYYY-MM-DDTHH:MM:SSZ",
      "artifacts": {
        "CASE_ID": {
          "uri": "hf://datasets/OWNER/PRIVATE-STAGING@REVISION/path.parquet",
          "sha256": "64_LOWERCASE_HEX_CHARACTERS"
        }
      }
    }
  },
  "zstd-1048576": {
    "dataset_viewer": true,
    "evidence": {
      "dataset_viewer_url": "https://huggingface.co/datasets/OWNER/PRIVATE-STAGING",
      "dataset_viewer_checked_at": "YYYY-MM-DDTHH:MM:SSZ",
      "artifacts": {
        "CASE_ID": {
          "uri": "hf://datasets/OWNER/PRIVATE-STAGING@REVISION/path.parquet",
          "sha256": "64_LOWERCASE_HEX_CHARACTERS"
        }
      }
    }
  }
}
```

Replace `CASE_ID` with every configured case; the abbreviated example shows
the repeated shape rather than a production record. The staging upload is a
separate, intentional process. The benchmark workflow only reads remote
objects. Configure every `hf://` URI with an immutable staging revision. Keep
the repository private by default; public staging requires explicit author
approval. Supply `HF_TOKEN` when required and record the staged hashes, Dataset
Viewer URL, and check time. Selection verifies each recorded object identity.
Do not place
credentials or signed URLs in configuration or reports.

## Selection rule

The issue's thresholds are applied mechanically. To remove an ambiguity in
the issue text, the implementation declares the following aggregation rules,
which the author approved on 2026-07-21:

- size is the sum of all complete benchmark shard sizes;
- remote range time is the median of all 30 measured interval repetitions per
  shard (six interval queries times five), pooled across benchmark shards;
- local and HF full-scan medians are computed separately and both must remain
  within 10% of source; and
- exact threshold equality qualifies (`25%`, `5%`, and `10%` are inclusive).

If source passes all functional checks, it stays unchanged unless a rewrite
is at least 25% faster for the aggregate remote range median, is no more than
5% larger, and is no more than 10% slower for either full-scan access mode. If
rewriting is functionally required, the fastest functional remote-range
candidate is chosen among exact rewrites within 5% of the smallest rewrite and
within both full-scan limits. If no candidate qualifies, the result is
explicitly blocked for author review.

The workflow writes `selection.json` and
`dataset-card-parquet-benchmark.md`. The latter is ready to insert into the
dataset card after issue #8's complete production manifest clears the source
validation gate.

## Recorded pilot

The provisional 2026-07-21 `epurdom` run, public staging identities, aggregate
measurements, resource evidence, and explicit production blockers are recorded
in [`reports/parquet-layout-benchmark-pilot`](../reports/parquet-layout-benchmark-pilot/README.md).
The pilot points to retaining source files, but is not a production selection
because the complete issue #8 inventory manifest remains outstanding.

## Running

Start from `workflow/config/parquet-benchmark.example.yaml`, replace its path
and URI placeholders, and point `inventory_manifest` at the
completed issue #8 production manifest. The committed resource values come
from the author-approved chr22 `epurdom` pilot. After the SCF profile's current
array blocker is cleared, run the local dry-run and then the SCF artifact
workflow through the locked environment:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/parquet-benchmark.yaml \
  --cores 1 \
  --dry-run

uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --workflow-profile workflow/profiles/scf \
  --configfile /path/to/parquet-benchmark.yaml \
  all
```

Record the Slurm job identifiers, wall time, peak RSS, and scheduler efficiency
beside the generated reports before changing production resource requests.
