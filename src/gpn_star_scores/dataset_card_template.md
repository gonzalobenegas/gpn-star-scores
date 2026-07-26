---
license: apache-2.0
pretty_name: GPN-Star genome-wide scores
tags:
  - biology
  - genomics
  - variant-effect-prediction
size_categories:
  - 10B<n<100B
configs:
  - config_name: gpn-star-hg38-v100-200m-entropy
    data_files:
      - split: train
        path: data/gpn-star-hg38-v100-200m/entropy/*.parquet
  - config_name: gpn-star-hg38-v100-200m-llr
    data_files:
      - split: train
        path: data/gpn-star-hg38-v100-200m/llr/*.parquet
  - config_name: gpn-star-hg38-m447-200m-entropy
    data_files:
      - split: train
        path: data/gpn-star-hg38-m447-200m/entropy/*.parquet
  - config_name: gpn-star-hg38-m447-200m-llr
    data_files:
      - split: train
        path: data/gpn-star-hg38-m447-200m/llr/*.parquet
    default: true
  - config_name: gpn-star-hg38-p243-200m-entropy
    data_files:
      - split: train
        path: data/gpn-star-hg38-p243-200m/entropy/*.parquet
  - config_name: gpn-star-hg38-p243-200m-llr
    data_files:
      - split: train
        path: data/gpn-star-hg38-p243-200m/llr/*.parquet
  - config_name: mm39-entropy
    data_files:
      - split: train
        path: data/mm39/entropy/*.parquet
  - config_name: mm39-llr
    data_files:
      - split: train
        path: data/mm39/llr/*.parquet
  - config_name: gg6-entropy
    data_files:
      - split: train
        path: data/gg6/entropy/*.parquet
  - config_name: gg6-llr
    data_files:
      - split: train
        path: data/gg6/llr/*.parquet
  - config_name: dm6-entropy
    data_files:
      - split: train
        path: data/dm6/entropy/*.parquet
  - config_name: dm6-llr
    data_files:
      - split: train
        path: data/dm6/llr/*.parquet
  - config_name: ce11-entropy
    data_files:
      - split: train
        path: data/ce11/entropy/*.parquet
  - config_name: ce11-llr
    data_files:
      - split: train
        path: data/ce11/llr/*.parquet
  - config_name: tair10-entropy
    data_files:
      - split: train
        path: data/tair10/entropy/*.parquet
  - config_name: tair10-llr
    data_files:
      - split: train
        path: data/tair10/llr/*.parquet
---

# GPN-Star genome-wide scores

@@REVIEW_NOTICE@@

[![GPN-Star tracks in the UCSC Genome Browser](https://raw.githubusercontent.com/gonzalobenegas/gpn-star-scores/main/docs/images/gpn-star-ucsc-example.png)](https://genome.ucsc.edu/s/gbenegas/gpn%2Dstar%2Dexample)

Genome-wide, mutation-rate-calibrated GPN-Star constraint and variant scores for
eight score sets covering human, mouse, chicken, *D. melanogaster*,
*C. elegans*, and *A. thaliana*. Canonical scores are available as
chromosome-sharded Parquet. Hugging Face hosts 72 BigWigs; a multi-assembly
UCSC track hub references the 64 logo/LLR views.

## Table of contents

1. [Overview and quick links](#overview-and-quick-links)
2. [What is included](#what-is-included)
3. [What the scores mean](#what-the-scores-mean)
   1. [LLR and absLLR](#llr-and-absllr)
   2. [Entropy](#entropy)
4. [Query the scores in code](#query-the-scores-in-code)
5. [Use the UCSC Genome Browser](#use-the-ucsc-genome-browser)
6. [File layout, implementation, and provenance](#file-layout-implementation-and-provenance)
7. [License and citation](#license-and-citation)

## Overview and quick links

| Resource | Link |
| --- | --- |
| Files | [Browse all dataset files](https://huggingface.co/datasets/songlab/gpn-star-scores/tree/main) |
| Default Dataset Viewer config | [Human mammalian-model LLR](https://huggingface.co/datasets/songlab/gpn-star-scores/viewer/gpn-star-hg38-m447-200m-llr) |
| UCSC hub | [Load the multi-assembly GPN-Star hub](https://genome.ucsc.edu/cgi-bin/hgTracks?hubUrl=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fsonglab%2Fgpn-star-scores%2Fresolve%2Fmain%2Fucsc%2Fhub.txt) |
| Human browser quick start | [Open UCSC's current default hg38 context plus GPN-Star-M tracks][ucsc-m447] |
| Source code | [`gonzalobenegas/gpn-star-scores`](https://github.com/gonzalobenegas/gpn-star-scores) |
| Immutable source snapshot | [`@@SOURCE_REVISION@@`](https://github.com/gonzalobenegas/gpn-star-scores/tree/@@SOURCE_REVISION@@) |
| Paper | [Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models](https://doi.org/10.1101/2025.09.21.677619) |
| Immutable public metadata base | [`@@PUBLIC_METADATA_REVISION@@`](https://huggingface.co/datasets/songlab/gpn-star-scores/tree/@@PUBLIC_METADATA_REVISION@@) |
| Release manifest | [`manifest/release.json`](https://huggingface.co/datasets/songlab/gpn-star-scores/blob/@@PUBLIC_METADATA_REVISION@@/manifest/release.json) |

Parquet is the canonical analysis product. It retains the supplied chromosome
names, one-based positions, and `Float32` score values. BigWigs are display
products with explicit coordinate conversion and three-decimal values where
documented.

## What is included

- **Eight score sets:** three GPN-Star models on hg38 plus mm39, gg6, dm6,
  ce11, and tair10.
- **16 explicit Parquet configurations:** one LLR and one entropy configuration
  for each score set.
- **290 canonical chromosome shards** containing @@TOTAL_ROWS_FORMATTED@@ rows.
- **72 browser BigWigs:** 40 v1 entropy and sequence-logo artifacts plus 32
  signed A/C/G/T LLR artifacts.
- **One multi-assembly UCSC hub** spanning six working UCSC databases and
  referencing 64 logo/LLR BigWigs. Entropy BigWigs remain downloadable but are
  not displayed in the hub.
- **Machine-readable manifests** containing file identities, sizes, SHA-256
  checksums, row counts, and immutable revision provenance.

### Score sets, assemblies, and browser databases

A **score set** is one model's released directory. A **Parquet assembly name**
identifies the reference assembly and chromosome naming used by its canonical
tables. A **UCSC database name** selects the compatible browser database; it
can differ from the Parquet assembly name.

| Score set | Organism/model | Parquet assembly | Working UCSC database | Model |
| --- | --- | --- | --- | --- |
| `gpn-star-hg38-v100-200m` | Human, 100-vertebrate alignment | `hg38` | `hg38` | [GPN-Star-V](https://huggingface.co/songlab/gpn-star-hg38-v100-200m) |
| `gpn-star-hg38-m447-200m` | Human, 447-mammal alignment | `hg38` | `hg38` | [GPN-Star-M](https://huggingface.co/songlab/gpn-star-hg38-m447-200m) |
| `gpn-star-hg38-p243-200m` | Human, 243-primate alignment | `hg38` | `hg38` | [GPN-Star-P](https://huggingface.co/songlab/gpn-star-hg38-p243-200m) |
| `mm39` | Mouse, 35-way mammal alignment | `mm39` | `mm39` | [GPN-Star mm39](https://huggingface.co/songlab/gpn-star-mm39-v35-85m) |
| `gg6` | Chicken, 77-way vertebrate alignment | `gg6` | `galGal6` | [GPN-Star galGal6](https://huggingface.co/songlab/gpn-star-galGal6-v77-85m) |
| `dm6` | *D. melanogaster*, 124-way insect alignment | `dm6` | `dm6` | [GPN-Star dm6](https://huggingface.co/songlab/gpn-star-dm6-i124-85m) |
| `ce11` | *C. elegans*, 135-way nematode alignment | `ce11` | `ce11` | [GPN-Star ce11](https://huggingface.co/songlab/gpn-star-ce11-n135-25m) |
| `tair10` | *A. thaliana*, 18-way plant alignment | `tair10` | `GCF_000001735.4` | [GPN-Star TAIR10](https://huggingface.co/songlab/gpn-star-tair10-b18-25m) |

### Coordinate conventions

- **Parquet:** supplied assembly chromosome names without an added `chr` prefix
  and one-based `pos: Int64` coordinates.
- **BigWig and UCSC:** UCSC chromosome names and zero-based, half-open
  intervals.

For example, Parquet position `chrom="22", pos=20_000_001` is represented in a
BigWig as the one-base interval `chr22:20_000_000-20_000_001`. Keep this
conversion explicit when comparing table and browser values.

## What the scores mean

The interpretations in this section come from the source Box
[`README.md`](https://app.box.com/file/2154252568578), file `2154252568578`,
version `2495703203934`. The interpretation and example blocks are reproduced
verbatim.

### LLR and absLLR

`llr_calibrated` is a variant-level score. Each LLR table has one row for every
genomic position × non-reference A/C/G/T allele, normally three alternate rows
per covered position. The score compares the alternate allele with the
reference allele.

| Column | Type | Meaning |
| --- | --- | --- |
| `chrom` | String | Supplied assembly chromosome name |
| `pos` | Int64 | One-based genomic position |
| `ref` | String | Reference nucleotide |
| `alt` | String | Non-reference alternate nucleotide |
| `llr_calibrated` | Float32 | Mutation-rate-calibrated log-likelihood ratio, alternate versus reference |
| `abs_llr_calibrated` | Float32 | Mutation-rate-calibrated absolute LLR |

**`llr_calibrated`:** More negative = more constrained or larger effect.

**`abs_llr_calibrated`:** Magnitude of the variant's effect relative to a neutral substitution.
Positive = larger effect than neutral; negative = smaller. Useful when the direction of effect
is not relevant.

**Example:**
```
chrom     pos ref alt  llr_calibrated  abs_llr_calibrated
   21 5010065   T   A          -1.774               1.774
   21 5010065   T   C          -1.550               1.550
   21 5010065   T   G          -1.670               1.670
```

Release data note: `abs_llr_calibrated` is an **independently supplied
calibrated score**. It must not be calculated as `abs(llr_calibrated)`, and
matching magnitudes in example rows do not imply a derivation.

The release exposes three distinct LLR-related representations:

1. **Canonical Parquet `llr_calibrated`:** the supplied full-precision
   variant-level `Float32` value.
2. **Signed A/C/G/T LLR BigWigs:** browser values that retain each alternate
   allele's `llr_calibrated`; the reference allele is assigned an explicit zero.
   UCSC displays these tracks as `-LLR` without modifying the files. These tracks
   are rounded to three decimal places.
3. **A/C/G/T sequence-logo BigWigs:** visualization heights derived from
   calibrated LLRs. They are not LLRs, probabilities, or canonical score
   products.

### Entropy

`entropy_calibrated` is a position-level score with one row per covered genomic
position.

| Column | Type | Meaning |
| --- | --- | --- |
| `chrom` | String | Supplied assembly chromosome name |
| `pos` | Int64 | One-based genomic position |
| `ref` | String | Reference nucleotide |
| `entropy_calibrated` | Float32 | Mutation-rate-calibrated positional entropy |

**Interpretation:** ~1.0 = neutral; <1.0 = constrained; the lower the more constrained.

**Example:**
```
chrom     pos ref  entropy_calibrated
   21 5010065   T               0.486
   21 5010066   A               0.644
   21 5010067   A               0.591
```

## Query the scores in code

This example joins UKB fine-mapped variants with GPN-Star (M) LLR scores. It
filters each chromosome shard to the exact requested positions before joining.

The example contains 11,400 variants across the 22 autosomes. On an AMD EPYC
7543 CPU node with 16 allocated CPUs and Polars 1.42.1, the remote version took
10 minutes 41 seconds and the local version took 5.94 seconds.

The local version requires 22 Parquet shards totaling about 52 GiB.

Runtime depends not only on the number of variants, but also on their genomic
locality. Variants clustered within one gene generally touch fewer Parquet row
groups and run faster than sparse variants spread across a chromosome or the
whole genome.

```python
import polars as pl

keys = ["chrom", "pos", "ref", "alt"]

variants = pl.read_parquet(
    "hf://datasets/songlab/ukb_finemapped_nc_traitgym/test.parquet",
    columns=keys,
)

# Remote scores (no full score download required):
score_root = (
    "hf://datasets/songlab/gpn-star-scores/"
    "data/gpn-star-hg38-m447-200m/llr"
)

# Local scores (use this instead after downloading the LLR shards):
# score_root = "/path/to/gpn-star-scores/data/gpn-star-hg38-m447-200m/llr"

results = []

for chrom in variants.get_column("chrom").unique(maintain_order=True):
    chrom_variants = variants.filter(pl.col("chrom") == chrom)
    positions = chrom_variants.get_column("pos").unique().to_list()

    scores = (
        pl.scan_parquet(f"{score_root}/llr_chr{chrom}.parquet")
        .filter(pl.col("pos").is_in(positions))
    )

    results.append(
        chrom_variants.lazy()
        .join(scores, on=keys, how="left")
        .collect(engine="streaming")
    )

annotated = pl.concat(results)
```

## Use the UCSC Genome Browser

The stable
[multi-assembly hub URL](https://huggingface.co/datasets/songlab/gpn-star-scores/resolve/main/ucsc/hub.txt)
loads eight GPN-Star model groups across six working UCSC databases. Each model
group includes:

- four derived A/C/G/T sequence-logo tracks; and
- four signed A/C/G/T tracks displayed as `-LLR`, with an explicit zero for the
  reference
  allele.

Entropy BigWigs remain available in the dataset but are not part of the UCSC
hub. The logo and `-LLR` default to 16 pixels high. The `-LLR` rows default to
a dense grayscale 0–10 view and use UCSC's display-time negation: higher
displayed values correspond to more-negative source LLR and therefore greater
constraint or a larger predicted functional effect. When expanded, negative
source LLR appears as positive `-LLR` in muted blue (`60,60,140`), while
positive source LLR appears as negative `-LLR` in muted red (`140,60,60`).

### Model-specific launch links

The links below start from UCSC's clean default settings with
`ignoreCookie=1`, retain UCSC's native default context tracks, and add one
selected GPN-Star model group with the sequence logo at `full` and `-LLR` at
`dense`. For assemblies with multiple GPN-Star models, the other GPN-Star
groups are explicitly hidden so that only the selected model is displayed.
The links intentionally do **not** use `hideTracks=1`, because that parameter
would remove the native context shown in the default Genome Browser view.

The links do not set `position`. Each one therefore inherits whatever locus and
native context tracks UCSC currently defines as the default for that database.
If UCSC changes a default locus or adds default tracks in the future, the link
inherits those changes while continuing to add the selected GPN-Star model
group.

| Score set | UCSC database | Browser | Position policy |
| --- | --- | --- | --- |
| `gpn-star-hg38-v100-200m` | `hg38` | [Open default context + GPN-Star-V tracks][ucsc-v100] | Current UCSC default (dynamic) |
| `gpn-star-hg38-m447-200m` **(default)** | `hg38` | [Open default context + GPN-Star-M tracks][ucsc-m447] | Current UCSC default (dynamic) |
| `gpn-star-hg38-p243-200m` | `hg38` | [Open default context + GPN-Star-P tracks][ucsc-p243] | Current UCSC default (dynamic) |
| `mm39` | `mm39` | [Open default context + GPN-Star tracks][ucsc-mm39] | Current UCSC default (dynamic) |
| `gg6` | `galGal6` | [Open default context + GPN-Star tracks][ucsc-gg6] | Current UCSC default (dynamic) |
| `dm6` | `dm6` | [Open default context + GPN-Star tracks][ucsc-dm6] | Current UCSC default (dynamic) |
| `ce11` | `ce11` | [Open default context + GPN-Star tracks][ucsc-ce11] | Current UCSC default (dynamic) |
| `tair10` | `GCF_000001735.4` | [Open default context + GPN-Star tracks][ucsc-tair10] | Current UCSC default (dynamic) |

This dynamic-default behavior is intentional. It prioritizes a familiar,
up-to-date UCSC context over reproducing one historical locus forever.

Genomic span is controlled by the `position` interval. Image width in pixels
can be controlled with UCSC's `pix` parameter, but the primary interactive
links leave pixel width responsive to the user's browser.

## File layout, implementation, and provenance

<details>
<summary>Complete public file layout</summary>

```text
data/
  <score-set>/
    entropy/
      entropy_chr<chrom>.parquet
    llr/
      llr_chr<chrom>.parquet
bigwig/
  <score-set>/
    entropy.bw
    A.bw
    C.bw
    G.bw
    T.bw
    llr_A.bw
    llr_C.bw
    llr_G.bw
    llr_T.bw
manifest/
  release.json
  raw-llr-validation.json
  ucsc-hub.json
ucsc/
  hub.txt
  genomes.txt
  <ucsc-database>/
    trackDb.txt
README.md
```

</details>

### Browser transformation details

- Entropy BigWigs contain `entropy_calibrated`.
- Signed LLR BigWigs retain alternate-allele `llr_calibrated`, assign the
  reference allele an explicit zero for display, and do not read or derive
  `abs_llr_calibrated`.
- For the derived sequence-logo tracks, the reference nucleotide receives
  logit zero and the three alternate nucleotides receive their independently
  supplied `llr_calibrated` values. A stable `Float64` softmax produces base
  weights, and each height is `p(base) * (2 - H)` for base-2 entropy `H`.
- Final browser values are stored as `Float32` and rounded to three decimals.
  They are visualization values rather than canonical scores or raw model
  probabilities.

### Immutable provenance

| Component | Immutable identity |
| --- | --- |
| Original interpretation source | Box `README.md`, file `2154252568578`, version `2495703203934`, SHA-1 `4b553074826e6a711d5308409ac8b1a0129d9f66` |
| Canonical Parquet and v1 entropy/logo artifact revision | [`@@V1_ARTIFACT_REVISION@@`](https://huggingface.co/datasets/songlab/gpn-star-scores/tree/@@V1_ARTIFACT_REVISION@@) |
| Signed LLR BigWig artifact revision | [`@@RAW_LLR_ARTIFACT_REVISION@@`](https://huggingface.co/datasets/songlab/gpn-star-scores/tree/@@RAW_LLR_ARTIFACT_REVISION@@) |
| README publication base | [`@@PUBLIC_METADATA_REVISION@@`](https://huggingface.co/datasets/songlab/gpn-star-scores/tree/@@PUBLIC_METADATA_REVISION@@) |
| Dataset-card source implementation commit | [`@@SOURCE_REVISION@@`](https://github.com/gonzalobenegas/gpn-star-scores/tree/@@SOURCE_REVISION@@) |

The release manifest records Parquet and BigWig identities, sizes, checksums,
row counts, and artifact revisions. The UCSC manifest records browser database
mappings, track URLs, and validation scope. Metadata-only card publication
does not upload, rewrite, or delete any Parquet or BigWig.

## License and citation

The dataset is released under the Apache License 2.0. Please cite:

Ye C, Benegas G, Albors C, Li JC, Prillo S, Fields PD, Clarke B, Song YS.
[Predicting functional constraints across evolutionary timescales with
phylogeny-informed genomic language models](https://doi.org/10.1101/2025.09.21.677619).
bioRxiv (2025). doi: `10.1101/2025.09.21.677619`.

```bibtex
@article{ye2025predicting,
  title={Predicting functional constraints across evolutionary timescales with
    phylogeny-informed genomic language models},
  author={Ye, Chengzhong and Benegas, Gonzalo and Albors, Carlos and Li,
    Jianan Canal and Prillo, Sebastian and Fields, Peter D and Clarke, Brian
    and Song, Yun S},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.09.21.677619}
}
```

[ucsc-v100]: @@UCSC_V100_URL@@
[ucsc-m447]: @@UCSC_M447_URL@@
[ucsc-p243]: @@UCSC_P243_URL@@
[ucsc-mm39]: @@UCSC_MM39_URL@@
[ucsc-gg6]: @@UCSC_GG6_URL@@
[ucsc-dm6]: @@UCSC_DM6_URL@@
[ucsc-ce11]: @@UCSC_CE11_URL@@
[ucsc-tair10]: @@UCSC_TAIR10_URL@@
