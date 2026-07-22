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

Each model group has two user-facing tracks:

- **Entropy** is a conventional one-dimensional quantitative BigWig graph,
  displayed as bars with automatic scaling and mean windowing when zoomed out.
- **Sequence logo** is one stacked `multiWig` with `logo on`. Its A, C, G, and
  T BigWigs are implementation subtracks rather than four separate top-level
  plots.

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

Issue #15's raw calibrated-LLR tracks are not part of the current 40-BigWig
release. They can be added later as explicitly versioned quantitative tracks
without replacing either v1 view.

## Generated layout

The workflow atomically builds:

```text
metadata/
├── README.md
├── manifest/ucsc-hub.json
└── ucsc/
    ├── hub.txt
    ├── genomes.txt
    ├── description.html
    └── <ucsc-assembly>/
        ├── trackDb.txt
        └── <track>.html
```

Every `bigDataUrl` pins the immutable BigWig artifact revision rather than
`main`. The public entry URL follows `main` so validated metadata updates can
be delivered without silently changing the underlying v1 values:

```text
https://huggingface.co/datasets/songlab/gpn-star-scores/resolve/main/ucsc/hub.txt
```

The generated dataset card links that URL prominently and provides eight
model-specific browser launch links. Each launch link uses the same hub while
hiding unrelated tracks and opening both the selected model's entropy and logo
views. `manifest/ucsc-hub.json` records every generated control-file checksum,
all 40 BigWig URLs and identities, the six database names, and all eight model
groups and launch URLs.

## Build and validation

Copy `workflow/config/hub.example.yaml` outside Git and replace its paths and
resource placeholders. `release_manifest` must be issue #4's validated
`manifest/release.json`; `artifact_revision` is the immutable commit containing
the 40 BigWigs. Use shared project scratch for `udc_cache_root`, not node-local
`/tmp`.

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

- `hubCheck -checkSettings` to pass for the complete hub;
- valid anonymous HTTP byte ranges for all 40 pinned BigWigs;
- the exact expected UCSC chromosome names in every BigWig header and
  consistent chromosome lengths across the five tracks in each score set;
- a covered representative base for each of the eight score sets; and
- non-empty direct `bigWigSummary` values for entropy and A/C/G/T at that base
  and across a surrounding zoom window.

This proves that each track stanza resolves to the same remote BigWig queried
directly at base and summary scales. After publication, the validator repeats
the complete check against the immutable hub commit and verifies every
published hub/control file byte-for-byte without credentials. A final manual
browser pass still records that each assembly/model group renders at base and
zoomed-out scales; automation supplements rather than replaces that visual
check.

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
requires an issue #6 approval record containing the approver, date, evidence
URL, and exact expected remote base revision. It refuses a stale base, a
private repository, missing validation evidence, or a Slurm environment.

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
report preserves the new immutable revision and the command fails loudly.
Resume without creating another commit with the module's `validate-existing`
subcommand, using the same approval record, base revision, metadata tree, and
the preserved final revision.
