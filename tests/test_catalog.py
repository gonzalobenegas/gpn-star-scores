from gpn_star_scores.catalog import (
    ASSEMBLIES,
    BOX_CURRENT_PARQUET_BYTES,
    BOX_REPORTED_FOLDER_BYTES,
    EXPECTED_SHARD_COUNT,
    SCORE_SETS,
    expected_shards,
)


def test_release_catalog_has_all_290_unique_shards() -> None:
    shards = expected_shards()

    assert len(shards) == EXPECTED_SHARD_COUNT == 290
    assert len({shard.relative_path for shard in shards}) == len(shards)
    assert {shard.assembly for shard in shards} == set(ASSEMBLIES)


def test_release_catalog_uses_original_pinned_ensembl_fastas() -> None:
    expected_filenames = {
        "hg38": "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz",
        "ce11": "Caenorhabditis_elegans.WBcel235.dna_sm.toplevel.fa.gz",
        "dm6": "Drosophila_melanogaster.BDGP6.32.dna_sm.toplevel.fa.gz",
        "gg6": "Gallus_gallus.GRCg6a.dna_sm.toplevel.fa.gz",
        "mm39": "Mus_musculus.GRCm39.dna_sm.primary_assembly.fa.gz",
        "tair10": "Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa.gz",
    }

    assert {
        assembly: spec.fasta_filename for assembly, spec in ASSEMBLIES.items()
    } == expected_filenames
    assert "/release-107/" in ASSEMBLIES["hg38"].fasta_url
    assert "/release-106/" in ASSEMBLIES["gg6"].fasta_url
    assert "/release-60/" in ASSEMBLIES["tair10"].fasta_url


def test_box_folder_and_current_parquet_byte_counts_are_distinct() -> None:
    assert BOX_REPORTED_FOLDER_BYTES == 333_761_247_733
    assert BOX_CURRENT_PARQUET_BYTES == 333_761_235_219
    assert BOX_REPORTED_FOLDER_BYTES - BOX_CURRENT_PARQUET_BYTES == 12_514


def test_score_sets_name_the_exact_release_models() -> None:
    assert {score_set.name: score_set.model_id for score_set in SCORE_SETS} == {
        "gpn-star-hg38-v100-200m": "songlab/gpn-star-hg38-v100-200m",
        "gpn-star-hg38-m447-200m": "songlab/gpn-star-hg38-m447-200m",
        "gpn-star-hg38-p243-200m": "songlab/gpn-star-hg38-p243-200m",
        "ce11": "songlab/gpn-star-ce11-n135-25m",
        "dm6": "songlab/gpn-star-dm6-i124-85m",
        "gg6": "songlab/gpn-star-galGal6-v77-85m",
        "tair10": "songlab/gpn-star-tair10-b18-25m",
        "mm39": "songlab/gpn-star-mm39-v35-85m",
    }
