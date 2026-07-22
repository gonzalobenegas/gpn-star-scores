# End-to-end release QA

Issue #2 is the final blocking issue in the v1 release epic. Its opt-in `qa`
target repeats public checks against immutable revisions, reconciles those
results with the complete production evidence chain, and writes machine- and
human-readable release records. It does not create or push a Git tag.

## What the evidence chain proves

The release record validates all of these relationships before it can report a
passing QA result:

- the release catalog and inventory contain exactly 290 source Parquet shards;
- every inventory record has the required entropy or LLR schema, `Int64`
  positions, `Float32` scores, zero null/non-finite/content-error counts, a
  source SHA-256, and a verified reference FASTA;
- every published Parquet identity matches its inventory path, size, SHA-256,
  row count, assembly, chromosome, and score type;
- the final source-layout selection references that exact inventory;
- all 40 released BigWigs reference the same inventory and retain the complete
  sampled-value, gap, chromosome-boundary, covered-base, and zoom-level audit;
- a fresh anonymous Hub API pass reconciles all 330 data-object identities,
  reruns representative range queries, checks all 40 BigWig HTTP ranges, and
  verifies both dataset-card source and rendering;
- all four published Polars examples execute against the immutable public
  release revision;
- fresh anonymous hub validation checks all 35 metadata files, all 40 pinned
  BigWigs, chromosome headers, representative base values, and zoom summaries;
  and
- the committed manual UCSC evidence covers base and zoom rendering for every
  one of the eight model groups.

Because the anonymous public Parquet SHA-256s equal the hashes of the fully
validated source files, exact published value and schema identity follows
without downloading and rescanning 333 GB a second time. The source scan and
the public identity comparison remain independently recorded steps.

## Inputs and execution

Copy `workflow/config/qa.example.yaml` outside Git. Keep the issue #4 release
metadata and issue #6 hub metadata trees read-only, and set `workflow_commit`
to the exact clean candidate commit. Fill the resource requests only after the
smallest complete issue #2 pilot has measured the four public Polars examples;
the projected and multi-chromosome examples intentionally run exactly as
published and may materialize large frames.

Then dry-run and execute locally with the pinned UCSC environment:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/qa.yaml \
  --software-deployment-method conda \
  --cores 1 \
  --dry-run qa

uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/qa.yaml \
  --software-deployment-method conda \
  --cores 1 \
  qa
```

The output root contains `public-release.json`, `public-hub.json`,
`release-record.json`, and `release-record.md`. Material outputs and network
logs remain outside Git. The release record stores source and release-manifest
checksums, the immutable Hugging Face artifact and hub revisions, the GitHub
workflow commit, `uv.lock` and SCF-profile checksums, SCF run identifiers,
Slurm efficiency evidence, limitations, and waivers.

When an immutable full public-hub report already exists, set
`qa.public_hub_report` to that committed report. The release record consumes
and revalidates its identities instead of repeating the same 40-track network
sweep. Omit the setting only when an intentional fresh hub audit is required.

## Routine dataset-card edits

The heavy `publish_hub` target is for changes to hub structure, track metadata,
or pinned artifact identities. It is not a documentation publishing workflow
and is never triggered by pre-commit or CI.

For a generated README-only correction, build the metadata normally and use
the explicit `publish_dataset_card` target. Its publisher rejects a changed
base revision, verifies that the existing public hub manifest is byte-identical
to the candidate, commits only `README.md`, and anonymously checks the final
README bytes and rendered dataset page. It performs zero BigWig requests. Both
publication targets remain manual and approval-gated. Each approval records
the exact operation, base revision, and candidate SHA-256, so README approval
cannot authorize full-hub publication. Full-hub approval uses an aggregate
digest of every submitted path and file identity, including `README.md`. If
rendering is temporarily unavailable after the commit,
`validate-existing-card` resumes validation for that exact revision and writes
the success marker without creating a second commit.

## Dataset-card join safety gate

Issue #2 planning found that the original one-row variant-join example has no
predicate on its right-hand scan. Polars' optimized public plan therefore
reads all 24 human LLR shards, with an estimated 16,592,702,040 rows. Running
that example literally would be an unsafe release check rather than useful
evidence.

The corrected generated card selects the chromosome shard and bounds the
position interval before joining. The release record requires those exact
markers in the public card and verifies the card's SHA-256 against the fresh
anonymous hub report. It therefore refuses to certify the existing unbounded
card. Publishing the metadata-only correction, recording its new immutable hub
revision, and repeating anonymous hub validation require explicit author
approval; the QA workflow never performs that update.

## Viewer and capacity waivers

The published source inventory has one narrow numeric-capacity waiver recorded
by issue #4. It does not waive any data, checksum, schema, BigWig, or public
access check.

Issue #4 also records the author's decision that hosted Dataset Viewer
readiness is a non-blocking convenience while issue #17 tracks server-side
processing. The issue #2 public preflight subsequently found all 16
configurations ready, with complete `/splits`, `/first-rows`, and `/is-valid`
checks, so the example configuration contains no Viewer waiver. The record
rejects a stale waiver, and it rejects pending configurations without the
exact approval and follow-up evidence.

Known scientific-interpretation wording and raw-LLR browser tracks remain
separately tracked by issues #19 and #15. They do not change v1 score values,
schemas, coordinates, or the documented logo transformation.

## Author approval and tag

Keep `tag_approval: null` through implementation, QA, and independent review.
The resulting record is valid but says `ready_to_tag: false`.

After a GPN-Star author approves the exact workflow commit, artifact revision,
hub revision, and `v1.0.0` tag on issue #2, copy that approval into the external
QA configuration and rerun `qa`. Review the new record, verify that the
worktree is clean and still at the approved commit, and create the local tag:

```bash
uv run --locked python -m gpn_star_scores.qa tag \
  --release-record /path/to/qa/release-record.json \
  --repository-root /path/to/gpn-star-scores
```

The tag command refuses a missing or mismatched approval, a changed `HEAD`, a
dirty worktree, Slurm execution, or an existing tag pointing elsewhere. It is
idempotent when `v1.0.0` already points to the approved commit. Pushing the tag
is a separate intentional remote action and still requires explicit author
approval at that step.
