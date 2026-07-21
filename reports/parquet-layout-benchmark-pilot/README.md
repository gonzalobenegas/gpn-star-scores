# Parquet layout benchmark pilot

Status: **pilot complete; production selection blocked**

The 2026-07-21 `epurdom` pilot completed all four provisional cases, all three
candidates, and both local and Hugging Face access modes. The evidence points
to keeping the source files unchanged, but that outcome is not final until a
complete issue #8 inventory manifest validates the sources. On 2026-07-21 the
author approved chromosome 22 as the representative `hg38-v100` chromosome and
approved the documented cross-case aggregation interpretation.

## Provisional result

| Candidate | Total bytes | HF interval median | HF interval bytes | Local full scan | HF full scan | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `source` | 1,033,206,375 | 0.366 s | 10,453,073 | 0.166 s | 2.806 s | Provisional choice |
| `zstd-262144` | 971,587,077 | 0.384 s | 5,715,825 | 0.232 s | 2.487 s | Ineligible |
| `zstd-1048576` | 970,968,353 | 0.382 s | 8,054,631 | 0.208 s | 2.199 s | Ineligible |

The rewrites save about 6% of aggregate size and reduce transferred bytes, but
neither improves the aggregate median remote interval time: the 262k and 1M
layouts are respectively 4.9% and 4.3% slower than source. They also make the
aggregate local full scan 39.5% and 25.0% slower, exceeding the declared 10%
limit. Consequently neither rewrite reaches the issue's performance threshold.

Every rewrite passed bitwise source-to-output equality, including independent
`abs_llr_calibrated` values. All 12 local candidate reports found usable
position statistics. All 12 direct lazy `hf://` Polars predicate/projection
checks passed.

## Staging and Viewer

The author approved public benchmark staging. The 12 Parquet objects
(2,975,761,805 bytes) are immutable at data revision
`09b3f9a5d7ed9ae7acdd583f8e76d0c474a4642f` in
[`songlab/gpn-star-scores-layout-benchmark-staging`](https://huggingface.co/datasets/songlab/gpn-star-scores-layout-benchmark-staging).
Viewer-only dataset-card metadata is revision
`78efdb0f9a7b6e25f2568a879304836af75dd8b1`. Each of the 12 explicit Viewer
configurations returned the expected entropy or LLR schema and 100 preview
rows. [`staging-checks.json`](staging-checks.json) records the Viewer check,
pinned artifact URIs, and SHA-256 identities without credentials or signed
URLs.

## Execution evidence

Local generation and measurements ran as jobs `3345354`, `3345355`, `3345357`,
and `3345358`. Corrected serialized HF measurements ran as jobs `3345430`
through `3345441`; all completed successfully. The longest local job took four
minutes and peaked at 21,478,580 KiB RSS. The longest remote job took 8 minutes
55 seconds; remote peak RSS was 14,934,604 KiB. The production example therefore
uses 32,768 MB and 30 minutes for benchmark jobs, and 4,096 MB and 30 minutes
for rewrites, following the repository's 1.5x memory and minimum-runtime policy.
The initial local 4 GB request was demonstrably too small despite completing;
the remote 32 GB request peaked at 44.5% utilization. CPU efficiency ranged
from 19.5% to 62.3% locally and 10.2% to 21.6% remotely, consistent with the
remote jobs being HTTP-I/O-bound. These values come from Slurm `ElapsedRaw`,
`TotalCPU`, allocated CPU count, requested memory, and `MaxRSS` accounting.

An initial remote attempt was discarded after real measurements showed that
passing a Python file object directly to Polars materialized the complete
object for every query. The corrected implementation reuses one Hugging Face
filesystem per job and uses PyArrow row-group reads for counted remote
measurements; a 1 kb live probe then transferred 7,990,149 bytes from a
140,690,567-byte object. Direct Polars `hf://` behavior remains a separate
required validation. Regression tests cover filesystem reuse, counter reset,
and selective row-group reads.

[`summary.json`](summary.json) contains the machine-readable aggregate metrics,
job IDs, resource evidence, functional checks, staging revisions, provisional
rationale, and explicit blockers.
