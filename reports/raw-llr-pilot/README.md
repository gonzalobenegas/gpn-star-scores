# Raw calibrated-LLR production pilot

Status: **production, publication, and public hub validation complete**

The issue #15 pilot exercised the complete generation and finalization path
for `gpn-star-hg38-v100-200m`. It read only the immutable LLR Parquet shards
and produced `llr_A`, `llr_C`, `llr_G`, and `llr_T`. The 40 v1 BigWigs were
neither inputs nor validation targets.

## Scientific and artifact checks

All four final tracks passed full chromosome-header, covered-base, zoom-level,
boundary, deterministic sampled-value, gap-absence, and SHA-256 checks before
atomic promotion. Each track covers 2,934,747,438 positions across the 24
expected hg38 chromosomes and has 10 zoom levels. The pilot performed 98,304
source-to-final value checks and 96 explicit gap checks.

The source A/C/G/T matrix contained 4,814,818,526 negative values,
3,983,535,239 positive values, and 2,940,635,987 zeros. Those counts include
the explicit reference-zero baseline at every covered position.
`abs_llr_calibrated` was not read, derived, or substituted. Final visualization
values are Float32 rounded to three decimals; Parquet remains canonical.

| Track | Bytes | SHA-256 |
| --- | ---: | --- |
| `llr_A` | 11,008,752,801 | `cae5f13abcc923860ba599d702193f315d460f288bebc319d0419145d19827dd` |
| `llr_C` | 11,590,980,877 | `9425c3d6b4f76ad21e7fb53290f1353f5950778f6d30735f97f516f967e7478e` |
| `llr_G` | 11,602,594,318 | `2eefc34f652655179fd704a40acf16d50fcdf621a67e41fae32721177e00301d` |
| `llr_T` | 10,989,717,672 | `bdc49134826e00c7c35ed91534f58047f877dd39304970a4887fafbc2ccfba2c` |

The four files total 45,192,045,668 bytes. Snakemake removed all regenerable
chromosome BigWigs after their consumers succeeded; no sibling temporary
outputs remained.

## SCF evidence and retained resources

The smallest representative job, v100 chromosome 22 (`3350217`), completed in
116 seconds at 1,878.8 MiB peak RSS and 90.73% CPU efficiency. The full
score-set run used workflow ID
`gpn-star_4dea360a-2dc7-4c67-b52c-0fe6629888d7`:

- chromosome jobs `3350219`–`3350242` completed in 208–922 seconds, peaked at
  1,209.1–10,906.6 MiB RSS, and used 22.88–76.94% CPU efficiency;
- finalizer jobs `3350246`–`3350249` completed in 2,196–2,243 seconds, peaked
  at 24,418.4–25,442.7 MiB RSS, and used 69.51–70.66% CPU efficiency.

The pilot initially selected 16,384 MB and 30 minutes for chromosome jobs,
49,152 MB and 60 minutes for finalizers, and the 4,096 MB policy minimum for
audits and aggregation. Full production evidence below supersedes the runtime
and audit-memory assumptions: retained requests are 60 minutes for chromosome
jobs, 120 minutes for finalizers, 16,384 MB for audits, and 4,096 MB for
aggregation. Temporary-disk requests remain 16,384, 24,576, 1,024, and 1,024
MB respectively.

Jobs `3350214`–`3350216` failed before reading source data because the first
ignored config file was placed in node-local `/tmp`, which compute nodes could
not see. Moving that config into the shared worktree resolved the operational
error; those attempts created no artifacts and supplied no scientific
evidence.

## Full 32-track production

The production aggregate passed on 2026-07-23 for exactly 32 new BigWigs:

- 198,020,546,809 total bytes;
- 593,920 source-to-final sampled-value checks;
- 544 explicit source-gap checks;
- 19,339,670,405 negative, 19,196,780,608 positive, and 12,865,669,875
  zero source-matrix values;
- direct generation, three-decimal Float32 visualization values, an explicit
  reference-zero baseline, and no use of `abs_llr_calibrated`; and
- machine-readable scope `new_raw_llr_tracks_only` with
  `existing_v1_files_checked: 0`.

The quota-safe recovery workflow
`gpn-star_8a5c52e4-a239-404b-a516-de97e343555b` completed the 14 missing
finalizers as jobs `3350761`, `3350762`, `3350848`, `3350849`, `3350950`,
`3350960`, `3350970`, `3351041`, `3351063`, `3351109`, `3351111`,
`3351159`, `3351162`, and `3351308`. Their wall times were 735–2,139
seconds, peak RSS was 8,973–25,007 MiB, and one-core CPU efficiency was
76.2–90.2%. The 14 focused audits completed in 68–434 seconds and peaked at
4,091–11,289 MiB; aggregate job `3351468` completed in 9 seconds. All 32
final files were present after completion, no private sibling output remained,
and scratch returned to its 989 GiB baseline.

The production run corrected two resource assumptions:

- the initial 64-job run
  `gpn-star_90c42704-5e00-47d6-885f-a1a8ae072f94` exposed 30-minute
  chromosome-build timeouts under heavy shared-filesystem contention, so
  chromosome builds retain 60 minutes; and
- focused audits used up to 11,289 MiB, so future audit requests are 16,384
  MB rather than the 4,096 MB policy minimum.

The resumed 24-job run
`gpn-star_c9734eb3-d4cb-4a09-9db4-3eb2adb0b4b3` completed the remaining
chromosome builds but concurrent finalizers exhausted the 1,024 GiB user
scratch quota. Diagnostic job `3350715` recorded `Disk quota exceeded`.
Sixteen abandoned private sibling directories from canceled attempts occupied
about 35 GiB; only those exact directories were removed. Commit `39a4603`
made finalizer cleanup SIGTERM-aware. The successful recovery then used two
concurrent jobs, 120-minute finalizer limits, and atomic promotion; each
successful finalizer reclaimed its chromosome inputs before another wave.

## Public artifact and hub publication

The approved raw-only publisher added exactly 32 files in one Hugging Face
commit from `340a04b4ccc95d68ad3be4fc3b08d725f29842e4` to immutable artifact
revision `47e7f051113abab49f04f43f9107cae2cbbfd34d`. The approved candidate
digest was
`173fc9c1db97150a0b6a41585824b187aff12ca6c0bae60f82367d484e554461`.
Anonymous validation passed 32 identity and 32 HTTP range checks, sent no
credentials, and checked zero prior v1 files.

The additive 72-track manifest retains the 40 v1 BigWigs at revision
`5c799b2ec6aa089f0caa8294ae72adb4510f81ae` and pins the new tracks to the
raw artifact revision above. Its exact approved publication-candidate digest
was `89e1e88ebcf69d55824eae6c166c256eda29613a5dd44ff86fddf2e2280532c7`.
One metadata commit published 45 files from the raw artifact revision to hub
revision `6fbdd6e8754080c08b9db34a78282e6ac04398b7`.

Local and anonymous immutable-revision hub validation both passed
`hubCheck -noTracks -checkSettings`, 32 HTTP range checks, 32 chromosome
header checks, and base plus zoom queries for all eight score sets. The scope
was `new_raw_llr_tracks_only`; the 40 v1 BigWigs were not opened.

UCSC's documented `hgRenderTracks` endpoint returned 16 PNGs at HTTP 200 for
the exact hub revision: a representative base and zoom view for every score
set. The URLs explicitly hid entropy and logo, so zero v1 tracks were
rendered. TrackDb and launch-link validation establish `dense` as the default.
Because UCSC renders compact quantitative tracks in grayscale, the
signed-color pass applied a view-only `full` override to the four raw rows.
Every zoom image contained both exact blue (`0,0,255`) and exact red
(`255,0,0`) pixels, and visual inspection confirmed separate A/C/G/T rows,
zero baselines, and signed bars across all eight score sets.

CADD's native `mouseOverFunction noAverage` setting was removed after the
official public-hub settings validator rejected it as unsupported. The hub
retains the supported CADD-inspired structure, shared autoscaling, zero line,
and `mean+whiskers` windowing.

The machine-readable details are in [`summary.json`](summary.json).
