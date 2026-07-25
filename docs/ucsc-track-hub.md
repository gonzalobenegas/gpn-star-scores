# UCSC track hub

Issue #6 adds one multi-assembly UCSC track hub to the public
[`songlab/gpn-star-scores`](https://huggingface.co/datasets/songlab/gpn-star-scores)
dataset. One hub URL keeps discovery and versioning aligned with the single
release repository. UCSC routes each browser session to the matching
assembly-specific `trackDb.txt`; tracks from different assemblies are never
overlaid.

## Browser organization

The hub contains six UCSC databases and eight model groups:

| Release assembly | UCSC database | Model groups |
| --- | --- | ---: |
| `hg38` | `hg38` | 3 |
| `ce11` | `ce11` | 1 |
| `dm6` | `dm6` | 1 |
| `gg6` | `galGal6` | 1 |
| `tair10` | `GCF_000001735.4` | 1 |
| `mm39` | `mm39` | 1 |

Each model group has three user-facing tracks after the issue #15 extension:

- **Entropy** is a conventional one-dimensional quantitative BigWig graph,
  displayed as bars with automatic scaling and mean windowing when zoomed out.
  Its default visibility is `full`.
- **Sequence logo** is one stacked `multiWig` with `logo on`. Its A, C, G, and
  T BigWigs are implementation subtracks rather than four separate top-level
  plots.
- **Raw calibrated LLR** is a CADD-inspired composite with separate A, C, G,
  and T rows. It defaults to `full`, shares group scaling, displays the zero
  line, uses `mean+whiskers` windowing, and colors positive values muted blue
  (`60,60,140`) and negative values muted red (`140,60,60`) to match UCSC's
  hg38 `phyloP100way` track.

The logo follows the established GPN hub colors: A green (`0,128,0`), C blue
(`0,0,255`), G orange (`255,166,0`), and T red (`255,0,0`). This retains the
working display recipe from the older human and Arabidopsis hubs while using
the standard `hub.txt` → `genomes.txt` → per-assembly `trackDb.txt` layout.
The layout and settings follow the
[UCSC track-hub guide](https://genome.ucsc.edu/goldenPath/help/hgTrackHubHelp.html)
and
[trackDb definition](https://genome.ucsc.edu/goldenPath/help/trackDb/trackDbHub.html).

The entropy track contains the supplied `entropy_calibrated` field. The logo
sets the reference logit to zero, assigns the three supplied
`llr_calibrated` values to alternate bases, computes a stable Float64 softmax
and base-2 entropy `H`, and displays `p(base) * (2 - H)`. Final browser values
are Float32 rounded to three decimals. Parquet remains the canonical
full-precision score product, and `abs_llr_calibrated` is not used or derived.
The descriptions do not invent biological directionality or calibration
semantics that remain pending author review.

Issue #15 adds those raw calibrated-LLR tracks as 32 explicitly versioned
artifacts without replacing either v1 view. Their reference allele is an
explicit zero; alternate alleles retain signed `llr_calibrated`.
`abs_llr_calibrated` remains unused.

Entropy, the sequence logo, the LLR composite, and its four allele rows default
to `full` so the complete selected model and signed colors are visible
immediately.

## Generated layout

The workflow atomically builds:

```text
metadata/
├── README.md
├── manifest/
│   ├── release.json             # v2 combined 72-track catalog
│   ├── ucsc-hub.json
│   └── raw-llr-validation.json  # when issue #15 is enabled
└── ucsc/
    ├── hub.txt
    ├── genomes.txt
    ├── description.html
    └── <ucsc-assembly>/
        ├── trackDb.txt
        └── <track>.html
```

Every `bigDataUrl` pins an immutable BigWig artifact revision rather than
`main`. The 40 v1 tracks retain the issue #4 revision, while the 32 issue #15
tracks pin their additive artifact revision. The public entry URL follows
`main` so validated metadata updates can be delivered without silently
changing either artifact set:

```text
https://huggingface.co/datasets/songlab/gpn-star-scores/resolve/main/ucsc/hub.txt
```

The generated dataset card links that URL prominently and provides eight
model-specific browser launch links. Each launch link uses the same hub while
hiding unrelated tracks and opening the selected model's entropy, logo, and
raw-LLR views when the extension is enabled. `manifest/ucsc-hub.json` records
every generated control-file checksum, all 72 pinned BigWig URLs and
identities, the six database names, and all eight model groups and launch
URLs. The v2 `manifest/release.json` combines the trusted 40-track v1 catalog
with the 32 focused raw-LLR identities and records their two immutable
artifact revisions without revalidating the v1 files.
For each score set, the hub manifest also records a
`raw_llr_validation_url` that explicitly hides entropy and logo and opens only
the raw-LLR composite. Use those eight URLs for issue #15 rendering evidence;
the user-facing launch links intentionally open all views.

## Build and validation

Copy `workflow/config/hub.example.yaml` outside Git and replace its paths and
resource placeholders. `release_manifest` must be issue #4's validated
`manifest/release.json`; `artifact_revision` is the immutable commit containing
the 40 v1 BigWigs. Configure `raw_llr_validation`,
`raw_llr_artifact_revision`, and the exact `source_revision` together to add
issue #15 and render the expanded dataset card. Use shared project scratch for
`udc_cache_root`, not node-local `/tmp`.

Run the local preflight without the SCF profile:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/hub.yaml \
  --software-deployment-method conda \
  --cores 1 \
  artifacts
```

The validation rule uses pinned UCSC v482 tools and the explicit official HTTPS
trackDb settings specification URL, avoiding the utility's failing implicit
HTTP discovery path. It requires:

- `hubCheck -noTracks -checkSettings` to pass for the complete hub metadata;
- valid anonymous HTTP byte ranges for the configured validation scope;
- the exact expected UCSC chromosome names in every checked BigWig header and
  consistent chromosome lengths within each score set;
- a covered representative base for each of the eight score sets; and
- non-empty direct `bigWigSummary` values at that base and across a
  surrounding zoom window.

For the issue #15 extension, range, header, base, and zoom queries cover only
the 32 new raw-LLR tracks; the 40 immutable v1 BigWigs are not revalidated.
`hubCheck -noTracks` still covers the complete hub structure without opening
the 40 previous BigWigs. After publication, the
validator repeats that focused check against the immutable hub commit and
verifies every published hub/control file byte-for-byte without credentials.
A final manual browser pass still records that each assembly/model group
renders at base and zoomed-out scales; automation supplements rather than
replaces that visual check.

CADD's native `mouseOverFunction noAverage` setting is intentionally omitted:
`hubCheck -noTracks -checkSettings` rejects it as unsupported in public hubs.
The issue #15 LLR composite, its four allele rows, and its launch links use
`full` by default so the configured signed colors are visible.

The issue #6 read-only production run passed all of these automated checks for
6 assemblies, 8 score sets, and 40 tracks. Its counts, representative loci,
tool versions, and measured resources are committed under
[`reports/ucsc-track-hub-preflight`](../reports/ucsc-track-hub-preflight/README.md).
The original preflight did not claim publication or manual browser rendering;
the publication follow-up is recorded separately below.

The first approved metadata commit published revision
`6671186db8e07c2e87d8f2eb8496c7be5d5b1c7e`; its anonymous automated
validation passed. Live rendering then exposed one release-metadata blocker:
UCSC returns HTTP 500 for `db=araTha1` even without this hub, whereas the older
Arabidopsis hub and a direct control use the working TAIR10 GenArk identifier
`hub_2660163_GCF_000001735.4`. Seven other model groups rendered correctly at
base and zoomed-out scales.

The approved second commit published
`8ea5b82c19a61691629f9084b805758a6a0ba1c9` and passed the complete anonymous
automated validation. Fresh-session rendering then established that the
`hub_2660163_` prefix from the older hub is session-generated: it falls back to
hg38 when this hub is connected alone. The stable `GCF_000001735.4` database
renders the intended TAIR10.1 assembly directly.

The approved final correction published revision
`2cb55ca6ceb4bddbe4314d2edd0fe370b200fde8`. Anonymous validation passed for
all 35 metadata files and all 40 pinned BigWigs, with no credentials sent.
Manual UCSC rendering then passed at base and zoomed-out scales for all eight
model groups, including TAIR10.1 through `GCF_000001735.4`.

## Approval-gated publication

Building and validating the hub is read-only with respect to Hugging Face.
`publish_hub` is a separate local target and never runs through Slurm. It
requires an approval record from the issue owning the metadata update
(issue #15 for the raw-LLR extension), containing the approver, date, evidence
URL, exact candidate digest, and exact expected remote base revision. It
refuses a stale base, a private repository, missing validation evidence, or a
Slurm environment.

After explicit author approval is recorded, run:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/hub.yaml \
  --software-deployment-method conda \
  --cores 1 \
  publish_hub
```

The publisher adds the dataset card, hub manifest, and `ucsc/` tree in one
Hugging Face commit using optimistic concurrency against the approved base.
It never uploads or rewrites Parquet or BigWig data and never deletes remote
files. The publication rule runs the project Python interpreter inside the
pinned UCSC-tool environment so anonymous post-validation has `hubCheck`,
`bigWigInfo`, and `bigWigSummary` available.

If the commit succeeds but post-validation is interrupted, the publication
report preserves the new immutable revision and the command fails loudly. The
workflow declares only `publication.complete` as its success output, so
Snakemake can remove the failed-job marker without deleting the durable
recovery report. Resume without creating another commit with the module's
`validate-existing` subcommand, using the same approval record, base revision,
metadata tree, and preserved report and final revision. Recovery rejects a
missing or mismatched publisher-created report instead of asserting an
unverified base-to-final single-commit relationship. It always repeats the
anonymous immutable-revision validation before writing the success marker, so
changed local metadata cannot be certified by an older validation report.
