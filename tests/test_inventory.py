from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gpn_star_scores.catalog import ASSEMBLIES, ShardSpec, expected_shards
from gpn_star_scores.inventory import (
    atomic_write_json,
    build_manifest,
    ensure_output_root_outside_source,
    inspect_shard,
    inspect_shard_to_json,
    prepare_reference,
    render_summary,
    sha256_file,
    write_release_outputs,
)


def _write_reference(path: Path, sequence: str = "ACGT") -> Path:
    path.write_bytes(sequence.encode("ascii"))
    return path


def _write_entropy(
    path: Path,
    *,
    chrom: list[str] | None = None,
    pos: list[int | None] | None = None,
    ref: list[str | None] | None = None,
    score: list[float | None] | None = None,
    score_type: pa.DataType = pa.float32(),
    write_statistics: bool = True,
) -> Path:
    table = pa.table(
        {
            "chrom": pa.array(chrom or ["1", "1", "1", "1"], pa.string()),
            "pos": pa.array(pos or [1, 2, 3, 4], pa.int64()),
            "ref": pa.array(ref or ["A", "C", "G", "T"], pa.string()),
            "entropy_calibrated": pa.array(score or [0.1, 0.2, 0.3, 0.4], score_type),
        }
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=2,
        write_statistics=write_statistics,
        write_page_index=True,
    )
    return path


def _write_llr(path: Path) -> Path:
    # Alternate order deliberately differs at every position, and the supplied
    # abs-LLR values are not derived from llr_calibrated.
    table = pa.table(
        {
            "chrom": pa.array(["1"] * 12, pa.string()),
            "pos": pa.array([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3, pa.int64()),
            "ref": pa.array(["A"] * 3 + ["C"] * 3 + ["G"] * 3 + ["T"] * 3),
            "alt": pa.array(
                ["T", "C", "G", "A", "T", "G", "T", "A", "C", "C", "G", "A"]
            ),
            "llr_calibrated": pa.array(np.linspace(-2, 2, 12), pa.float32()),
            "abs_llr_calibrated": pa.array(
                [-0.5, 0.1, 1.2, 0.0, -0.3, 0.8, 1.0, -1.0, 0.2, 0.7, -0.2, 0.4],
                pa.float32(),
            ),
        }
    )
    pq.write_table(table, path, compression="snappy", row_group_size=5)
    return path


def test_entropy_inventory_records_schema_layout_checksum_and_bounds(
    tmp_path: Path,
) -> None:
    source = _write_entropy(tmp_path / "entropy_chr1.parquet")
    reference = _write_reference(tmp_path / "1.seq")
    shard = ShardSpec("test", "test", "entropy", "1")

    record = inspect_shard(source, shard, reference, batch_size=2)

    assert record["valid"]
    assert record["size"] == source.stat().st_size
    assert record["sha256"] == sha256_file(source)
    assert record["schema"] == [
        {"name": "chrom", "type": "string", "nullable": True},
        {"name": "pos", "type": "int64", "nullable": True},
        {"name": "ref", "type": "string", "nullable": True},
        {"name": "entropy_calibrated", "type": "float", "nullable": True},
    ]
    assert record["parquet"]["num_rows"] == 4
    assert record["parquet"]["num_row_groups"] == 2
    assert record["parquet"]["columns"]["pos"]["compression_codecs"] == ["ZSTD"]
    assert record["parquet"]["columns"]["pos"]["statistics"]["row_groups_present"] == 2
    assert (
        record["parquet"]["columns"]["pos"]["page_index"]["column_index_row_groups"]
        == 2
    )
    assert record["content"]["coordinate_bounds"] == {"min": 1, "max": 4}


def test_missing_physical_statistics_are_not_reported_as_complete(
    tmp_path: Path,
) -> None:
    source = _write_entropy(tmp_path / "entropy_chr1.parquet", write_statistics=False)
    reference = _write_reference(tmp_path / "1.seq")

    record = inspect_shard(
        source,
        ShardSpec("test", "test", "entropy", "1"),
        reference,
        batch_size=2,
    )

    statistics = record["parquet"]["columns"]["pos"]["statistics"]
    assert record["valid"]
    assert statistics["row_groups_present"] == 0
    assert statistics["null_count"] == 0
    assert not statistics["null_count_complete"]


def test_infinite_statistics_do_not_abort_non_finite_reporting(
    tmp_path: Path,
) -> None:
    source = _write_entropy(
        tmp_path / "entropy_chr1.parquet",
        score=[0.1, float("inf"), 0.3, 0.4],
    )
    reference = _write_reference(tmp_path / "1.seq")

    record = inspect_shard(
        source,
        ShardSpec("test", "test", "entropy", "1"),
        reference,
        batch_size=2,
    )

    assert not record["valid"]
    assert (
        record["parquet"]["columns"]["entropy_calibrated"]["statistics"]["max"]
        == "+Infinity"
    )
    assert record["content"]["non_finite_counts"] == {"entropy_calibrated": 1}
    assert "non_finite_scores" in {error["check"] for error in record["errors"]}


def test_llr_alt_order_is_unconstrained_and_groups_can_cross_batches(
    tmp_path: Path,
) -> None:
    source = _write_llr(tmp_path / "llr_chr1.parquet")
    reference = _write_reference(tmp_path / "1.seq")
    shard = ShardSpec("test", "test", "llr", "1")

    record = inspect_shard(source, shard, reference, batch_size=2)

    assert record["valid"]
    assert record["content"]["llr_group_errors"] == {
        "wrong_size": 0,
        "inconsistent_ref": 0,
        "wrong_alt_set": 0,
    }


def test_entropy_content_failures_are_reported_without_rewriting_source(
    tmp_path: Path,
) -> None:
    source = _write_entropy(
        tmp_path / "entropy_chr1.parquet",
        chrom=["1", "2", "1", "1"],
        pos=[1, 2, 2, 5],
        ref=["A", "C", "T", "A"],
        score=[0.1, float("nan"), 0.3, 0.4],
    )
    before = source.read_bytes()
    reference = _write_reference(tmp_path / "1.seq")
    shard = ShardSpec("test", "test", "entropy", "1")

    record = inspect_shard(source, shard, reference, batch_size=2)

    checks = {error["check"] for error in record["errors"]}
    assert not record["valid"]
    assert {
        "chromosome",
        "coordinates",
        "reference_match",
        "non_finite_scores",
        "entropy_order",
    } <= checks
    assert source.read_bytes() == before

    output = tmp_path / "invalid.json"
    inspect_shard_to_json(source, shard, reference, output, batch_size=2)
    assert "non_finite_scores" in {
        error["check"] for error in json.loads(output.read_text())["errors"]
    }


def test_nulls_and_wrong_schema_are_explicit_failures(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "1.seq")
    shard = ShardSpec("test", "test", "entropy", "1")
    null_source = _write_entropy(tmp_path / "null.parquet", pos=[1, None, 3, 4])
    wrong_schema_source = _write_entropy(
        tmp_path / "float64.parquet", score_type=pa.float64()
    )

    null_record = inspect_shard(null_source, shard, reference, batch_size=2)
    schema_record = inspect_shard(wrong_schema_source, shard, reference, batch_size=2)

    assert "nulls" in {error["check"] for error in null_record["errors"]}
    assert schema_record["content"] is None
    assert [error["check"] for error in schema_record["errors"]] == ["schema"]


def test_llr_duplicate_alt_and_position_order_fail(tmp_path: Path) -> None:
    source = tmp_path / "llr_chr1.parquet"
    table = pa.table(
        {
            "chrom": pa.array(["1"] * 6, pa.string()),
            "pos": pa.array([2, 2, 2, 1, 1, 1], pa.int64()),
            "ref": pa.array(["C"] * 3 + ["A"] * 3, pa.string()),
            "alt": pa.array(["A", "G", "N", "C", "G", "T"], pa.string()),
            "llr_calibrated": pa.array([0.1] * 6, pa.float32()),
            "abs_llr_calibrated": pa.array([0.2] * 6, pa.float32()),
        }
    )
    pq.write_table(table, source, row_group_size=4)
    reference = _write_reference(tmp_path / "1.seq")

    record = inspect_shard(
        source, ShardSpec("test", "test", "llr", "1"), reference, batch_size=4
    )

    checks = {error["check"] for error in record["errors"]}
    assert "llr_order" in checks
    assert "alt_allele" in checks
    assert "llr_alt_set" in checks


def test_shard_json_is_promoted_atomically(tmp_path: Path) -> None:
    source = _write_entropy(tmp_path / "entropy_chr1.parquet")
    reference = _write_reference(tmp_path / "1.seq")
    output = tmp_path / "nested" / "record.json"

    inspect_shard_to_json(
        source,
        ShardSpec("test", "test", "entropy", "1"),
        reference,
        output,
        batch_size=2,
    )

    assert json.loads(output.read_text())["valid"]
    assert not list(output.parent.glob(".record.json.*.tmp"))


def test_missing_shard_produces_an_explicit_presence_record(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "1.seq")

    record = inspect_shard(
        tmp_path / "missing.parquet",
        ShardSpec("test", "test", "entropy", "1"),
        reference,
    )

    assert not record["valid"]
    assert record["size"] is None
    assert record["errors"] == [
        {"check": "presence", "count": 1, "message": "expected shard is missing"}
    ]


def test_prepare_reference_uses_exact_filename_and_atomic_directory(
    tmp_path: Path,
) -> None:
    assembly = ASSEMBLIES["ce11"]
    fasta = tmp_path / assembly.fasta_filename
    with gzip.open(fasta, "wt", encoding="ascii") as handle:
        for chrom in assembly.chromosomes:
            handle.write(f">{chrom} chromosome\nacgt\n")
        handle.write(">scaffold extra\nNN\n")
    output = tmp_path / "prepared" / "ce11"

    expected_sha256 = sha256_file(fasta)
    prepare_reference(fasta, output, assembly, expected_sha256)

    assert (output / "I.seq").read_bytes() == b"ACGT"
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["expected_url"] == assembly.fasta_url
    assert provenance["expected_sha256"] == expected_sha256
    assert provenance["source_sha256"] == expected_sha256
    assert provenance["identity_verified"]
    assert provenance["ignored_contigs"] == 1
    assert provenance["contigs"] == dict.fromkeys(assembly.chromosomes, 4)
    assert not list(output.parent.glob(".ce11.*"))
    with pytest.raises(FileExistsError):
        prepare_reference(fasta, output, assembly, expected_sha256)


def test_reference_sha256_mismatch_is_rejected_before_output(tmp_path: Path) -> None:
    assembly = ASSEMBLIES["ce11"]
    fasta = tmp_path / assembly.fasta_filename
    with gzip.open(fasta, "wt", encoding="ascii") as handle:
        for chrom in assembly.chromosomes:
            handle.write(f">{chrom}\nACGT\n")
    output = tmp_path / "prepared" / "ce11"

    with pytest.raises(ValueError, match="FASTA SHA-256"):
        prepare_reference(fasta, output, assembly, "0" * 64)

    assert not output.exists()


def test_failed_reference_preparation_leaves_no_partial_output(tmp_path: Path) -> None:
    assembly = ASSEMBLIES["ce11"]
    fasta = tmp_path / assembly.fasta_filename
    with gzip.open(fasta, "wt", encoding="ascii") as handle:
        handle.write(">I\nACGT\n")
    output = tmp_path / "prepared" / "ce11"

    with pytest.raises(ValueError, match="missing"):
        prepare_reference(fasta, output, assembly, sha256_file(fasta))

    assert not output.exists()
    assert not list(output.parent.glob(".ce11.*"))


def _fake_complete_inventory(tmp_path: Path) -> tuple[Path, list[Path], list[Path]]:
    source_root = tmp_path / "source"
    report_root = tmp_path / "reports"
    shard_reports = []
    for shard in expected_shards():
        source = source_root / shard.relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x")
        report = report_root / f"{len(shard_reports)}.json"
        atomic_write_json(
            report,
            {
                "path": shard.relative_path.as_posix(),
                "size": 1,
                "valid": True,
                "errors": [],
            },
        )
        shard_reports.append(report)

    reference_reports = []
    for assembly, spec in ASSEMBLIES.items():
        report = report_root / f"reference-{assembly}.json"
        atomic_write_json(
            report,
            {
                "assembly": assembly,
                "expected_url": spec.fasta_url,
                "source_filename": spec.fasta_filename,
                "expected_sha256": "0" * 64,
                "source_sha256": "0" * 64,
                "identity_verified": True,
            },
        )
        reference_reports.append(report)
    return source_root, shard_reports, reference_reports


def test_output_root_must_be_outside_immutable_source(tmp_path: Path) -> None:
    source_root = tmp_path / "staged"
    source_root.mkdir()

    with pytest.raises(ValueError, match="outside the immutable source_root"):
        ensure_output_root_outside_source(source_root, source_root)
    with pytest.raises(ValueError, match="outside the immutable source_root"):
        ensure_output_root_outside_source(source_root, source_root / "inventory")
    source_alias = tmp_path / "staged-alias"
    source_alias.symlink_to(source_root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the immutable source_root"):
        ensure_output_root_outside_source(source_root, source_alias / "inventory")

    ensure_output_root_outside_source(source_root, tmp_path / "scratch" / "inventory")


def test_manifest_accounts_for_every_shard_and_capacity_evidence(
    tmp_path: Path,
) -> None:
    source_root, shard_reports, reference_reports = _fake_complete_inventory(tmp_path)

    manifest = build_manifest(
        source_root,
        shard_reports,
        reference_reports,
        expected_shard_bytes=290,
        hugging_face_capacity={
            "confirmed": True,
            "evidence": "Private capacity approval ABC-123",
            "confirmed_by": "author",
            "confirmed_at": "2026-07-20",
            "current_storage_bytes": 600,
            "planned_release_bytes": 300,
            "reserved_headroom_bytes": 100,
            "approved_capacity_bytes": 1_000,
        },
    )

    assert manifest["source"]["reported_shards"] == 290
    assert manifest["source"]["total_shard_bytes"] == 290
    assert manifest["validation"] == {
        "valid_shards": 290,
        "invalid_shards": 0,
        "release_ready": True,
        "blockers": [],
    }


def test_manifest_rejects_incomplete_or_underbudgeted_capacity_evidence(
    tmp_path: Path,
) -> None:
    source_root, shard_reports, reference_reports = _fake_complete_inventory(tmp_path)
    valid_capacity = {
        "organization": "songlab",
        "confirmed": True,
        "evidence": "Private capacity approval ABC-123",
        "confirmed_by": "author",
        "confirmed_at": "2026-07-20",
        "current_storage_bytes": 600,
        "planned_release_bytes": 300,
        "reserved_headroom_bytes": 100,
        "approved_capacity_bytes": 1_000,
    }
    invalid_overrides = [
        {"planned_release_bytes": 0},
        {"organization": "another-organization"},
        {"confirmed": 1},
        {"evidence": "   "},
        {"confirmed_by": None},
        {"confirmed_at": None},
        {"approved_capacity_bytes": True},
    ]

    for capacity_override in invalid_overrides:
        capacity = valid_capacity | capacity_override
        manifest = build_manifest(
            source_root,
            shard_reports,
            reference_reports,
            expected_shard_bytes=290,
            hugging_face_capacity=capacity,
        )

        assert not manifest["hugging_face_capacity"]["sufficient"]
        assert not manifest["validation"]["release_ready"]
        assert any(
            "Hugging Face organization capacity" in blocker
            for blocker in manifest["validation"]["blockers"]
        )
        if capacity_override == {"confirmed": 1}:
            assert "- Confirmed: no" in render_summary(manifest)
        if capacity_override == {"approved_capacity_bytes": True}:
            assert "- Approved capacity: not recorded" in render_summary(manifest)
        if capacity_override == {"evidence": "   "}:
            assert "- Evidence: not recorded" in render_summary(manifest)


def test_release_directory_is_atomic_and_capacity_is_a_blocker(
    tmp_path: Path,
) -> None:
    source_root, shard_reports, reference_reports = _fake_complete_inventory(tmp_path)
    output = tmp_path / "release"

    write_release_outputs(
        source_root,
        shard_reports,
        reference_reports,
        output,
        expected_shard_bytes=290,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    summary = (output / "summary.md").read_text()
    assert not manifest["validation"]["release_ready"]
    assert "Hugging Face organization capacity" in summary
    assert not list(tmp_path.glob(".release.*"))
