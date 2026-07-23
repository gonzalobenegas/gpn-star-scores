# Hugging Face release

Issue #4 publishes the complete release to the public
[`songlab/gpn-star-scores`](https://huggingface.co/datasets/songlab/gpn-star-scores)
dataset repository. Public-first publication is author-approved in issue #4.
The public issue #5 layout-benchmark staging repository established the same
unauthenticated access model for representative Parquet artifacts; it remains
benchmark evidence and is not reused as the final repository.

## Local preflight

Copy `workflow/config/release.example.yaml` outside Git and replace its generic
paths and resource placeholders. The configured inputs are:

- issue #8's complete 290-shard inventory manifest;
- issue #5's selection of the unchanged source Parquet layout;
- issue #7's 40 final BigWigs and aggregate validation report.

Run the local artifact target without the SCF profile:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/release.yaml \
  --cores 1 \
  artifacts
```

The preflight checks every expected local path and byte size, requires the
three input reports to reference the same inventory SHA-256, and hashes every
final BigWig. It atomically writes `metadata/README.md` and the `metadata/manifest/`
files below `release.output_root`. The generated dataset card defines exactly
16 explicit Parquet configurations, so BigWig and future UCSC hub files cannot
be inferred as tabular data.

The release manifest contains no credentials, signed URLs, or local filesystem
paths. Parquet checksums come from the complete source inventory; BigWig
checksums are calculated during preflight.

The retained production inventory has one release-level blocker: Hugging Face
does not expose a numeric public-storage quota for the organization. Issue #4
records author approval for the exact 524,145,473,121-byte public artifact
plan. `capacity_approval` is a narrow waiver accepted only when numeric
capacity is the manifest's sole blocker; it cannot waive source validation,
checksums, BigWig validation, or any public post-upload check. The approval also
records 52,414,547,313 bytes of planning headroom and the current Hugging Face
public-storage policy URL.

## Intentional publication

Authenticate through the standard Hugging Face token store or `HF_TOKEN`.
Never put a token in the YAML configuration, a log, or a report. Then invoke
the explicit target from one non-Slurm process:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile /path/to/release.yaml \
  --cores 4 \
  publish
```

The publisher creates or reuses the exact public dataset repository, uploads
each score-set/type directory and each score-set BigWig directory through the
current `huggingface_hub` `upload_folder` path, and uploads the dataset card
last. Hugging Face's Xet-backed folder upload chunks, deduplicates, retries,
and resumes already committed content. The workflow never deletes remote
files and refuses to run when `SLURM_JOB_ID` is set.

## Public validation

The final validation pins the returned 40-character Hugging Face commit SHA
and disables implicit credentials. It requires:

- all 290 Parquet and 40 BigWig remote sizes and LFS SHA-256 values to match
  `manifest/release.json`;
- the public dataset page and dataset-card source to load;
- every BigWig to return HTTP `206` with a valid `Content-Range` for a byte
  range request;
- representative direct Polars `hf://` scans to return interval results while
  retaining predicate/projection pushdown; and
- separately counted PyArrow/HfFileSystem interval queries to transfer fewer
  bytes than the corresponding chromosome object.

The report keeps these two checks distinct: the counted range-reader bytes are
not presented as Polars transfer measurements. This is consistent with
[Polars' cloud-scan guidance](https://docs.pola.rs/user-guide/io/cloud-storage/),
which documents that lazy cloud scans apply predicate and projection pushdown
before downloading data.

Dataset Viewer readiness is a reported, non-blocking convenience check. The
validator probes all 16 configurations without credentials and records either
their exact schemas and non-empty previews or their pending/error responses.
Use `viewer_required: true` only for a later hosted-preview gate. Issue #17
tracks server-side Viewer processing independently of the public data release.

`release.output_root/publication.json` records the immutable final commit SHA
and the complete validation evidence. It is written only after every required
public check passes. Full artifacts and runtime reports remain outside Git.

If upload completed but validation was interrupted, revalidate the same commit
without uploading again:

```bash
uv run --locked python -m gpn_star_scores.release validate-existing \
  --metadata-root /path/to/release/metadata \
  --report /path/to/release/publication.json \
  --revision 40_CHARACTER_COMMIT_SHA
```

## Post-v1 raw-LLR extension

Issue #15 publishes 32 additional `llr_A`, `llr_C`, `llr_G`, and `llr_T`
BigWigs through the separate
[`raw calibrated-LLR workflow`](raw-calibrated-llr.md). That publisher uses
one optimistic commit containing only the new files. It neither uploads
Parquet nor changes the 40 v1 BigWigs. Its approval is bound to the exact
current public revision, incremental byte total, and candidate digest.
Anonymous validation checks only the new LFS identities and byte-range
responses. The README and UCSC metadata are updated afterward through the
hub's own approval-gated commit, with existing v1 URLs left pinned to their
original artifact revision.

The downstream [issue #6 workflow](ucsc-track-hub.md) owns `ucsc/`, `hubCheck`,
browser rendering, and representative browser-value comparisons. Its files can
be added to the same public repository without changing the 16 explicit table
configurations.
