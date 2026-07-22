# BigWig generation benchmark

Status: implementation, local synthetic validation, the SCF benchmark, and
40-track production validation are complete. Issues #5, #8, and #9 are closed
and their committed interfaces are consumed directly. The sanitized benchmark,
resource, size, and validation evidence is committed under
`reports/bigwig-benchmark-pilot/`; full-size artifacts remain outside Git.

## Candidates and score semantics

The benchmark applies the same transformation and validation to both methods:

1. Emit variable-step WIG files and convert them with `wigToBigWig`, following
   the upstream `logits.smk` implementation, then combine disjoint chromosome
   BigWigs with `bigWigCat`.
2. Stream sorted Parquet batches directly into per-chromosome `pyBigWig`
   files. Contiguous positions use fixed-step entries and isolated positions
   use variable-step entries. A bounded-window `pyBigWig` repack creates the
   final tracks because production-scale `bigWigCat` is not reliable for
   direct-writer inputs.

The benchmark retains exact Float32 values so method selection compares like
with like. Final browser tracks are explicitly rounded to the configured three
decimal places to preserve the staged source precision while reducing storage;
Parquet remains the canonical
full-precision score product. Values remain Float32 in BigWig.

For each LLR position, the reference nucleotide receives logit zero and the
three independently supplied `llr_calibrated` values become alternate logits.
The stable softmax and base-2 entropy calculation use Float64 because those
intermediate operations are numerically sensitive. Final A/C/G/T information
heights are cast to Float32. `abs_llr_calibrated` is neither used nor derived.
These tracks are calibrated-LLR-derived logos, not raw model probabilities.

## Input contract and output layout

Production generation is opt-in. It requires:

- issue #8's complete 290-shard inventory manifest with zero validation or
  ordering failures;
- issue #5's finalized `selection.json`, selecting the immutable `source`
  candidate and naming that exact inventory-manifest SHA-256;
- issue #7's benchmark `selection.json`; and
- positive, pilot-derived resources in the production config.

The implementation rejects a missing, partial, mismatched, or rewritten-source
contract. Each chromosome is a restart unit that creates five temporary,
validated, atomically promoted BigWigs with the same ordered full-assembly
header required by `bigWigCat`, plus one validation report. Final output
contains five tracks per score set under `final/<score-set>/`: `entropy.bw`,
`A.bw`, `C.bw`, `G.bw`, and `T.bw`. `gg6` is recorded as UCSC assembly
`galGal6`, and `tair10` as UCSC assembly hub `araTha1`; source chromosome names
map explicitly to `chr`-prefixed browser names.

## Benchmark design and decision

The example config benchmarks complete entropy and LLR shards for `gg6 chr32`
and `gpn-star-hg38-v100-200m chr22`, matching issue #5's representative pilot.
For every case and method it runs one excluded warm-up and five measured
repetitions. GNU time records wall time and peak RSS. Scratch sampling reports
the peak transient bytes in addition to final artifacts, and final BigWig
bytes are recorded separately.

Within a case, wall time is the median of measured repetitions and resources
use the maximum observed value. Across cases, wall medians are summed, peak
RSS and scratch are the maximum, and final sizes are summed. This treats the
declared case set as one representative workload while retaining conservative
resource peaks.

Direct writing wins only when it is correct and either:

- is at least 20% faster; or
- reduces peak scratch by at least 80% and is no more than 20% slower.

Threshold comparisons are inclusive. Otherwise the WIG baseline wins.

## Validation

Both benchmark candidates and production outputs use deterministic samples
that always include the first and last source positions. Validation requires:

- strict source ordering and in-range one-based Int64 positions;
- exact source-to-UCSC coordinate conversion to zero-based, one-base-wide
  intervals;
- absent values at a detected source gap;
- exact Float32 agreement in benchmark and chromosome restart units;
- agreement with configured three-decimal rounding for every stored
  deterministic/random sample in final browser tracks, including the first and
  last source positions;
- absence at the stored first source gap for every chromosome that has a gap;
- expected chromosome sizes and covered-base counts;
- readable `pyBigWig` files and successful `bigWigInfo` output; and
- at least one default BigWig zoom level before and after concatenation.

The post-assembly audit is a separate restart unit, so existing final BigWigs
can be rechecked without regenerating chromosome artifacts. The final aggregate
fails unless all 40 audited score-set/track reports are valid, use the benchmark
winner, and refer to the same inventory used by the benchmark selection.

## Running on SCF

Copy `workflow/config/bigwig.example.yaml` outside Git and replace its generic
paths and resource placeholders. First dry-run the benchmark selection target,
then submit it with the SCF profile. The pinned Slurm executor still has an
open multi-wildcard array-output bug, so current production runs must disable
arrays explicitly with `--slurm-array-jobs=` until the upstream fix is pinned
and piloted:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/bigwig.yaml \
  --workflow-profile workflow/profiles/scf \
  --slurm-array-jobs= \
  --slurm-status-command=sacct \
  /shared/project/scratch/gpn-star-scores/bigwig-benchmark/selection.json
```

Use the same two Slurm flags for the production `artifacts` target. The SCF
profile pins every BigWig rule to `epurdom`; `high` is not compatible with the
locked numeric runtime used by these rules and failed the production pilot
with an illegal-instruction error. `sacct` is required because completed jobs
leave SCF's `squeue` before Snakemake reliably observes their terminal state.

After recording pilot job IDs, wall time, peak RSS, transient scratch, and
scheduler efficiency, set production resources from those measurements and
target `artifacts`. The workflow only generates and validates local artifacts;
it does not upload, publish, delete remote data, or create a release tag.

The 2026-07-21 production run selected direct streaming, retained conservative
24,576 MB chromosome-build and 49,152 MB finalizer memory requests, and
validated all 40 three-decimal tracks. The final files total 190,384,237,902
bytes (177.309 GiB). See `reports/bigwig-benchmark-pilot/README.md` for the
human-readable record and `summary.json` for machine-readable evidence.
