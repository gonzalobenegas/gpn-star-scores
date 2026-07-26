# Raw calibrated-LLR BigWigs

Issue #15 adds 32 browser artifacts to the public GPN-Star score release:
`llr_A`, `llr_C`, `llr_G`, and `llr_T` for each of the eight score sets. This
is an additive post-v1 workflow. It does not regenerate, rewrite, or revalidate
the 40 entropy and derived-logo BigWigs already released in v1.

## Score and coordinate contract

For every covered genomic position, the source LLR shard must contain exactly
the three non-reference alleles with one shared reference allele. Each output
track uses:

- the independently supplied `llr_calibrated` Float32 value when its allele is
  an alternate;
- an explicit Float32 zero when its allele is the reference; and
- no value at source coordinate gaps.

`abs_llr_calibrated` is not read, derived, or substituted. Source Parquet keeps
its supplied assembly chromosome names and one-based `Int64` positions.
BigWig output uses UCSC chromosome names and zero-based, half-open one-base
intervals. Chromosome restart units preserve exact Float32 values. The final
stream-copy step rounds visualization values to three decimals, matching the
v1 browser-artifact contract; Parquet remains canonical.

## Build and focused validation

Copy `workflow/config/raw-llr.example.yaml` outside Git and replace its shared
paths and resource placeholders. The workflow requires the immutable issue #8
inventory, the issue #5 source-layout selection, and issue #7's accepted
direct-streaming selection. Run a smallest representative SCF pilot before
setting production resource requests.

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/raw-llr.yaml \
  --profile workflow/profiles/scf \
  artifacts
```

The DAG creates chromosome-level restart units, four final BigWigs per score
set, one audited final report per new track, and:

```text
raw-llr/
├── final/<score-set>/{llr_A,llr_C,llr_G,llr_T}.bw
├── audit-reports/<score-set>/{llr_A,llr_C,llr_G,llr_T}.json
├── validation.json
└── validation.md
```

The focused validation proves:

- exact chromosome headers, base coverage, and zoom levels;
- deterministic source-to-BigWig Float32 samples at chromosome boundaries and
  across row-group/batch boundaries;
- explicit one-based to zero-based coordinate conversion;
- source gaps remain absent;
- signed values and reference-zero baselines are retained;
- the final three-decimal values match chromosome validation evidence; and
- the manifest covers exactly 32 new tracks and records their sizes and
  SHA-256 identities.

The aggregate report explicitly records that no v1 BigWig was revalidated.
The raw workflow has no dependency on issue #7 final tracks or audit reports.
The committed
[`raw-LLR pilot report`](../reports/raw-llr-pilot/README.md) records the
representative job IDs, measured resource use, retained requests, signed-value
counts, and four-track v100 validation evidence.

## Additive Hugging Face publication

`publish_raw_llr` is a manual local target and refuses to run under Slurm. It
requires an issue #15 approval record bound to:

- the exact current public base revision;
- the exact aggregate byte count from `validation.json`;
- the aggregate candidate digest returned by
  `raw_llr_candidate_sha256`;
- the Hugging Face public storage policy; and
- the author, date, operation, and issue evidence URL.

The publisher creates one optimistic Hugging Face commit with 32
`CommitOperationAdd` operations. It does not delete files, upload Parquet,
replace v1 BigWigs, or update metadata in that commit. Anonymous
post-publication validation checks only the 32 new LFS identities and HTTP
byte-range responses and records the immutable artifact revision.
Before creating the commit, the publisher enumerates the approved base
revision and rejects any candidate path that already exists, preventing an
`Add` operation from silently becoming an overwrite.

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/raw-llr.yaml \
  --cores 1 \
  publish_raw_llr
```

If the commit succeeds but anonymous validation is interrupted, the durable
publication report records the new revision and the command fails without
creating the success marker. Resume the same revision with
`gpn_star_scores.raw_llr_publication --validate-existing-revision <SHA>` and
the same paths and approval arguments. Recovery rejects a mismatched
publisher report and never creates a second commit.

## UCSC presentation

After artifact publication, configure the hub with `raw_llr_validation`,
`raw_llr_artifact_revision`, and the exact `source_revision` used to render the
expanded dataset card. Existing entropy/logo artifacts remain pinned to the
immutable v1 artifact revision; only `llr_{A,C,G,T}` uses the new revision.
After issue #28, the entropy BigWigs remain in Hugging Face but are no longer
referenced by the hub. The 32 logo URLs retain the v1 revision.

The browser presentation adapts the
[UCSC CADD v1.7 track](https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=caddSuper1_7)
organization:

- one `compositeTrack` per model with separate A, C, G, and T child rows;
- `dense` on the composite, inherited by every child;
- `negateValues on`, so UCSC displays `-llr_calibrated` without changing the
  BigWig values;
- automatic scaling disabled with a default 0–10 viewing range;
- a default/minimum height of 16 pixels;
- an always-visible zero baseline;
- `mean+whiskers` windowing; and
- dense grayscale rendering, with muted blue (`60,60,140`) for positive
  displayed `-LLR` and muted red (`140,60,60`) for negative displayed `-LLR`
  when expanded.

Negative source LLR therefore appears as positive `-LLR` in blue, while
positive source LLR appears as negative `-LLR` in red. Higher displayed values
correspond to more-negative source LLR and therefore greater constraint or a
larger predicted functional effect.

The existing derived sequence logo remains unchanged, defaults to `full` at 16
pixels, and continues using nucleotide colors. Initial artifact validation ran
`hubCheck -noTracks -checkSettings` for the complete metadata structure and
scoped BigWig range, header, base, and zoom checks to the 32 new tracks.
Presentation-only follow-ups use metadata-only validation and do not request
any BigWig. Public hub updates use the existing optimistic, approval-gated
`publish_hub` target and preserve all 72 artifact identities.

CADD's native trackDb uses `mouseOverFunction noAverage`, but UCSC's public
hub settings validator rejects that native-only setting. The hub therefore
uses the supported default mouse-over behavior.
