"""Authoritative catalog for the GPN-Star score release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class AssemblySpec:
    """Reference assembly and its expected release chromosomes."""

    name: str
    chromosomes: tuple[str, ...]
    fasta_url: str

    @property
    def fasta_filename(self) -> str:
        """Return the immutable Ensembl release filename from ``fasta_url``."""

        return Path(urlparse(self.fasta_url).path).name


@dataclass(frozen=True)
class ScoreSetSpec:
    """A directory of entropy and LLR shards tied to one assembly."""

    name: str
    assembly: str


@dataclass(frozen=True)
class ShardSpec:
    """One expected score shard."""

    score_set: str
    assembly: str
    score_type: str
    chrom: str

    @property
    def relative_path(self) -> Path:
        """Return the path expected below the configured staged source root."""

        return (
            Path(self.score_set)
            / self.score_type
            / (f"{self.score_type}_chr{self.chrom}.parquet")
        )


ASSEMBLIES: dict[str, AssemblySpec] = {
    "hg38": AssemblySpec(
        name="hg38",
        chromosomes=tuple(str(i) for i in range(1, 23)) + ("X", "Y"),
        fasta_url=(
            "http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/"
            "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
        ),
    ),
    "ce11": AssemblySpec(
        name="ce11",
        chromosomes=("I", "II", "III", "IV", "V", "X"),
        fasta_url=(
            "http://ftp.ensembl.org/pub/release-107/fasta/"
            "caenorhabditis_elegans/dna/"
            "Caenorhabditis_elegans.WBcel235.dna_sm.toplevel.fa.gz"
        ),
    ),
    "dm6": AssemblySpec(
        name="dm6",
        chromosomes=("2L", "2R", "3L", "3R", "4", "X", "Y"),
        fasta_url=(
            "https://ftp.ensembl.org/pub/release-107/fasta/"
            "drosophila_melanogaster/dna/"
            "Drosophila_melanogaster.BDGP6.32.dna_sm.toplevel.fa.gz"
        ),
    ),
    "gg6": AssemblySpec(
        name="gg6",
        chromosomes=(
            tuple(str(i) for i in range(1, 29))
            + tuple(str(i) for i in range(30, 34))
            + ("W", "Z")
        ),
        fasta_url=(
            "http://ftp.ensembl.org/pub/release-106/fasta/gallus_gallus/dna/"
            "Gallus_gallus.GRCg6a.dna_sm.toplevel.fa.gz"
        ),
    ),
    "mm39": AssemblySpec(
        name="mm39",
        chromosomes=tuple(str(i) for i in range(1, 20)) + ("X", "Y"),
        fasta_url=(
            "https://ftp.ensembl.org/pub/release-107/fasta/mus_musculus/dna/"
            "Mus_musculus.GRCm39.dna_sm.primary_assembly.fa.gz"
        ),
    ),
    "tair10": AssemblySpec(
        name="tair10",
        chromosomes=("1", "2", "3", "4", "5"),
        fasta_url=(
            "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/"
            "arabidopsis_thaliana/dna/"
            "Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa.gz"
        ),
    ),
}


SCORE_SETS: tuple[ScoreSetSpec, ...] = (
    ScoreSetSpec("gpn-star-hg38-v100-200m", "hg38"),
    ScoreSetSpec("gpn-star-hg38-m447-200m", "hg38"),
    ScoreSetSpec("gpn-star-hg38-p243-200m", "hg38"),
    ScoreSetSpec("ce11", "ce11"),
    ScoreSetSpec("dm6", "dm6"),
    ScoreSetSpec("gg6", "gg6"),
    ScoreSetSpec("tair10", "tair10"),
    ScoreSetSpec("mm39", "mm39"),
)

SCORE_TYPES = ("entropy", "llr")
EXPECTED_SHARD_COUNT = 290

# Box reports this for folder 368859688578. The difference from the current
# Parquet sum is consistent with README.md and retained versions, so this is
# folder-level provenance rather than an expected Parquet sum.
BOX_REPORTED_FOLDER_BYTES = 333_761_247_733

# A read-only Box API inventory on 2026-07-20 found this sum for the 290 current
# Parquet objects. Production validation must recompute it from the local stage.
BOX_CURRENT_PARQUET_BYTES = 333_761_235_219


def expected_shards() -> tuple[ShardSpec, ...]:
    """Return every expected assembly/model, score type, and chromosome shard."""

    shards = tuple(
        ShardSpec(score_set.name, score_set.assembly, score_type, chrom)
        for score_set in SCORE_SETS
        for score_type in SCORE_TYPES
        for chrom in ASSEMBLIES[score_set.assembly].chromosomes
    )
    if len(shards) != EXPECTED_SHARD_COUNT:
        raise AssertionError(
            f"release catalog defines {len(shards)} shards, "
            f"expected {EXPECTED_SHARD_COUNT}"
        )
    return shards


def score_set_assembly(score_set: str) -> str:
    """Resolve a score-set directory to its assembly."""

    for spec in SCORE_SETS:
        if spec.name == score_set:
            return spec.assembly
    raise KeyError(f"unknown score set: {score_set}")


def get_shard_spec(score_set: str, score_type: str, chrom: str) -> ShardSpec:
    """Resolve and validate one shard key against the release catalog."""

    assembly = score_set_assembly(score_set)
    if score_type not in SCORE_TYPES:
        raise KeyError(f"unknown score type: {score_type}")
    if chrom not in ASSEMBLIES[assembly].chromosomes:
        raise KeyError(f"unexpected chromosome {chrom!r} for {assembly}")
    return ShardSpec(score_set, assembly, score_type, chrom)
