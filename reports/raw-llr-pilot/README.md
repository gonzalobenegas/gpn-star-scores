# Raw calibrated-LLR production pilot

Status: **pilot complete; production and publication pending**

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

The retained chromosome request is 16,384 MB, 30 minutes, and 16,384 MB
temporary disk, giving about a 1.5x measured memory margin. Finalizers retain
49,152 MB, 60 minutes, and 24,576 MB temporary disk, matching the measured
pilot and established v1 repack envelope. Audits and aggregation use the
4,096 MB policy minimum.

Jobs `3350214`–`3350216` failed before reading source data because the first
ignored config file was placed in node-local `/tmp`, which compute nodes could
not see. Moving that config into the shared worktree resolved the operational
error; those attempts created no artifacts and supplied no scientific
evidence.

The machine-readable details are in
[`summary.json`](summary.json). Full 32-track production and both public
publication commits are intentionally recorded later.
