# BigWig benchmark and production evidence

Status: **benchmark and production complete**

The 2026-07-21 `epurdom` benchmark compared the issue #7 WIG baseline with
direct `pyBigWig` streaming on four representative entropy and LLR cases. Both
methods passed coordinate, gap, boundary, zoom-level, and sampled-value checks.
Direct writing was selected under the issue's declared threshold.

| Method | Aggregate median wall | Peak RSS | Peak scratch | Final bytes | Correct |
| --- | ---: | ---: | ---: | ---: | :---: |
| WIG | 452.02 s | 1,263,374,336 | 3,485,139,726 | 1,340,594,859 | yes |
| Direct | 115.75 s | 1,304,629,248 | 0 | 987,133,512 | yes |

Direct writing was 74.39% faster and eliminated the measured WIG scratch
intermediate. The benchmark comprised 20 measured repetitions per method
across `gg6` chromosome 32 and `gpn-star-hg38-v100-200m` chromosome 22,
covering entropy and LLR-derived tracks. Benchmark jobs were `3346263` through
`3346270`; selection reporting completed in job `3346341`.

## Precision decision

Parquet remains the canonical full-precision product. The browser BigWigs are
Float32 tracks rounded to three decimal places. A read-only sample of one
million `entropy_calibrated` and one million `llr_calibrated` values from the
staged v100 chromosome 22 files found that every value was unchanged by
three-decimal rounding. Only 9.9628% of entropy values and 10.0135% of LLR
values were unchanged at two decimals. The author therefore approved uniform
three-decimal visualization precision for entropy and A/C/G/T tracks.

The calibrated-logo transformation still uses Float64 for the numerically
sensitive stable softmax and base-2 entropy calculation, then casts the final
heights to Float32 before BigWig writing.

## Production implementation

The UCSC `bigWigCat` v482 binary segfaulted on production-scale direct-writer
inputs even though the chromosome files passed `pyBigWig` and `bigWigInfo`
validation. Finalization therefore uses a bounded-window `pyBigWig` repack,
preserving gaps and the full ordered assembly header while applying the
declared visualization precision. The final artifact is written in a hidden
sibling directory, fully validated, and atomically renamed into place.

An exact unrounded v100 C baseline (job `3346523`) completed in 23m43s, peaked
at about 30.0 GB RSS, and produced 15,709,527,766 bytes. A discarded
two-decimal pilot (job `3346604`) completed in 26m10s, peaked at about 23.1 GB
RSS, and produced 5,387,778,947 bytes. The first three-decimal production
pilot, m447 C (job `3346847`), completed in 27m40s, peaked at 26,497,940 KiB
RSS, and produced 8,213,327,305 bytes. Production finalizers therefore retain
the conservative 49,152 MB, 60 minute, and 24,576 MB temporary-disk requests.

SCF production uses chromosome restart units on `epurdom`. Snakemake's Slurm
array path remains disabled because of the recorded executor-plugin limitation.
SCF also removes completed jobs immediately from `squeue --states=all`; using
`sacct` for status polling reliably preserves terminal states and allows the
workflow to clean temporary chromosome BigWigs.

The largest chromosome build observed about 15.6 GB RSS, so later production
builds requested 24,576 MB rather than the initial 16,896 MB. The retained
requests are 24,576 MB/30 minutes/16,384 MB temporary disk for chromosome
builds and 49,152 MB/60 minutes/24,576 MB temporary disk for finalizers. These
are intentionally conservative restart-safe requests; smaller assemblies used
far less memory.

## Production result

All 40 tracks passed aggregate validation against manifest
`de1b00e6099574dd2f74a0702b8870332f0c7dc6b2fffe9b6648398c1bef52e4`.
The aggregate report records `valid=true`, `track_count=40`, direct generation,
and three-decimal visualization precision. Every final report passed sampled
source agreement after declared rounding, chromosome-size and covered-base
checks, `pyBigWig` and `bigWigInfo` reads, and zoom-level checks.

| Score set | UCSC assembly | Checks/track | Zooms | A bytes | C bytes | G bytes | T bytes | Entropy bytes | Total bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ce11` | `ce11` | 6 | 8 | 318,476,737 | 317,715,521 | 317,087,187 | 321,790,168 | 359,573,908 | 1,634,643,521 |
| `dm6` | `dm6` | 7 | 8 | 437,562,936 | 426,041,312 | 426,553,810 | 437,254,806 | 485,419,641 | 2,212,832,505 |
| `gg6` | `galGal6` | 34 | 9 | 3,030,771,833 | 2,968,923,783 | 2,969,862,722 | 3,037,241,668 | 3,582,383,710 | 15,589,183,716 |
| `gpn-star-hg38-m447-200m` | `hg38` | 24 | 10 | 8,404,455,701 | 8,213,327,305 | 8,231,401,501 | 8,396,085,451 | 10,661,247,977 | 43,906,517,935 |
| `gpn-star-hg38-p243-200m` | `hg38` | 24 | 10 | 8,121,381,543 | 7,963,566,849 | 7,984,617,636 | 8,115,723,650 | 10,372,109,541 | 42,557,399,219 |
| `gpn-star-hg38-v100-200m` | `hg38` | 24 | 10 | 8,375,839,576 | 8,073,559,597 | 8,060,221,355 | 8,394,788,567 | 9,897,991,293 | 42,802,400,388 |
| `mm39` | `mm39` | 21 | 10 | 7,894,331,393 | 7,580,753,302 | 7,567,781,947 | 7,917,281,649 | 8,884,865,693 | 39,845,013,984 |
| `tair10` | `araTha1` | 5 | 8 | 362,243,601 | 346,859,928 | 347,541,680 | 364,488,987 | 415,112,438 | 1,836,246,634 |

The 40 final files total 190,384,237,902 bytes (190.384 GB or 177.309
GiB). The complete issue #7 scratch tree, including the 333,761,235,219-byte
immutable stage and retained inventory evidence, used 536,868,102,896 bytes
(about 500.0 GiB) after cleanup. Snakemake deleted every regenerable
chromosome BigWig after its consumers succeeded; no chromosome BigWigs,
partial outputs, or sibling temporary outputs remained. The retained exact
v100 C baseline adds one report JSON but is not a release track.

Production execution evidence:

| Score set or stage | Slurm run ID | Job IDs |
| --- | --- | --- |
| m447 | split pilot/continuation | `3346847`, `3346884`–`3346887` |
| v100 | `gpn-star_3f2036fd-065e-4833-9bbd-d8a4eb496e5e` | `3346913`–`3346945` |
| p243 | `gpn-star_e39e360d-3355-4d72-9f68-7020862d37ce` | `3346963`–`3346999` |
| mm39 | `gpn-star_ee849be4-d729-47ca-b763-fc87ea66c4b7` | `3347043`–`3347098` |
| gg6 | `gpn-star_65fa575d-b31d-469a-bb74-f591e69e351b` | `3347125`–`3347176` |
| dm6 | `gpn-star_14454b6e-ca24-4ad5-8c04-a1bbf93a65b5` | `3347184`–`3347198` |
| tair10 | `gpn-star_47422227-e255-4f98-b0d7-104e57bd2f26` | `3347204`–`3347218` |
| ce11 | `gpn-star_633e1a49-5e3d-478b-bd01-8548f0171aa2` | `3347224`–`3347236` |
| aggregate validation | `gpn-star_128fb1fd-ad71-4970-9cd6-1d1b6a04b263` | `3347239` |

No Hugging Face upload, public visibility change, or release tag was performed.
