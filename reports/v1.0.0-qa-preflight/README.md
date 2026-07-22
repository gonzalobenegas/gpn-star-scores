# GPN-Star v1.0.0 QA preflight

Status: **public QA valid; release blocked on exact candidate identifiers and
final tag approval**

The 2026-07-22 issue #2 preflight reconciled the complete source and production
evidence, then repeated public validation without Box, GitHub, or Hugging Face
credentials.

## Passing checks

- All 290 published Parquet files match the fully validated source manifest by
  path, byte size, SHA-256, row count, assembly, chromosome, score type, and
  coordinate bounds. The inventory covers 51,402,120,888 rows and
  333,761,235,219 bytes, with 145 exact entropy schemas and 145 exact LLR
  schemas.
- All 40 BigWigs link to the same inventory and its 742,400 sampled-value and
  680 gap checks. Their public LFS identities and all 40 HTTP byte ranges pass.
- Both representative public Parquet range reads transfer less than their
  chromosome objects and retain direct Polars predicate/projection pushdown.
- Dataset Viewer is now ready: `/splits` lists all 16 configs with `train`, all
  16 `/first-rows` schemas and previews pass, `/is-valid` reports preview and
  Viewer support, and no configs are pending or failed.
- The corrected four-example candidate passes against immutable artifact
  revision `5c799b2ec6aa089f0caa8294ae72adb4510f81ae`, returning 1,001 interval
  rows, 100,271,839 projected ce11 rows, 140,963,613 filtered dm6 rows, and one
  joined variant row.
- Fresh final-hub validation at
  `340a04b4ccc95d68ad3be4fc3b08d725f29842e4` passes for all 35 metadata files,
  `hubCheck`, 40 HTTP ranges, 40 chromosome headers, and eight representative
  base/zoom queries. The existing manual rendering evidence covers all eight
  model groups.

## Dataset-card blocker resolved

The previous public variant-join example scanned `*.parquet` on the right side
of a one-row left join. Its optimized plan had no selection and estimated a
scan of 16,592,702,040 rows across all 24 human LLR shards. Executing that
version literally would have been an unsafe and wasteful release check.

The corrected card selects `llr_chr22.parquet` and filters the exact position
interval before joining. After explicit author approval, the metadata-only
correction was published from exact base revision `2cb55ca6...` to immutable
revision `340a04b4...`. The post-publication report verifies the corrected
README identity and repeats the one-time final hub sweep. No Parquet or BigWig
object changed.

## Resources and evidence

The consolidated public release QA completed in 61.87 seconds with
11,409,068 KiB peak RSS. The example configuration requests 17,408 MB and the
30-minute policy minimum. Hub validation reuses the measured issue #6
cold-cache request of 1,024 MB and 15 minutes.

[`summary.json`](summary.json) records all immutable identifiers, counts,
resource evidence, evidence-file hashes, the capacity waiver, and the remaining
tag blocker. Full runtime reports remain in managed scratch and are
identified by SHA-256 in the summary.

Routine future dataset-card edits use the separate `publish_dataset_card`
target. It commits only `README.md`, verifies the unchanged hub manifest and
the public card's bytes/rendering, and performs zero BigWig checks. The heavy
`publish_hub` target remains available only for structural hub or track
changes. A live check of the corrected revision completed in 0.92 seconds.

The final `v1.0.0` tag remains blocked until the final workflow commit is
recorded and a GPN-Star author approves that exact commit together with the two
immutable Hugging Face revisions on issue #2.
