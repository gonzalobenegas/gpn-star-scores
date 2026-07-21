from __future__ import annotations

import json
import hashlib
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gpn_star_scores.catalog import expected_shards
from gpn_star_scores.parquet_benchmark import (
    _benchmark_source,
    _execute_hf_query,
    LayoutCandidate,
    QuerySpec,
    atomic_write_json,
    benchmark_parquet_candidate,
    inspect_position_statistics,
    interval_query_specs,
    rewrite_parquet_candidate,
    select_layout,
    validate_benchmark_source,
    verify_exact_values,
    write_selection_outputs,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def _entropy_table(rows: int = 12) -> pa.Table:
    return pa.table(
        {
            "chrom": pa.array(["32"] * rows, type=pa.string()),
            "pos": pa.array(range(1, rows + 1), type=pa.int64()),
            "ref": pa.array((["A", "C", "G", "T"] * rows)[:rows]),
            "entropy_calibrated": pa.array(
                [index / 7 for index in range(rows)], type=pa.float32()
            ),
        }
    )


def _llr_table() -> pa.Table:
    return pa.table(
        {
            "chrom": pa.array(["32"] * 6, type=pa.string()),
            "pos": pa.array([1, 1, 1, 2, 2, 2], type=pa.int64()),
            "ref": pa.array(["A", "A", "A", "C", "C", "C"]),
            "alt": pa.array(["C", "G", "T", "A", "G", "T"]),
            "llr_calibrated": pa.array(
                [-0.0, -2.0, 3.0, 4.0, -5.0, 6.0], type=pa.float32()
            ),
            # Deliberately not abs(llr): this independently supplied score must
            # survive the rewrite unchanged.
            "abs_llr_calibrated": pa.array(
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0], type=pa.float32()
            ),
        }
    )


@pytest.mark.parametrize(
    ("score_type", "table", "expected_dictionary_columns"),
    [
        ("entropy", _entropy_table(), ["chrom", "ref"]),
        ("llr", _llr_table(), ["alt", "chrom", "ref"]),
    ],
)
def test_rewrite_is_exact_and_has_declared_layout(
    tmp_path: Path,
    score_type: str,
    table: pa.Table,
    expected_dictionary_columns: list[str],
) -> None:
    source = tmp_path / f"{score_type}-source.parquet"
    output = tmp_path / f"{score_type}-candidate.parquet"
    report_path = tmp_path / f"{score_type}-rewrite.json"
    pq.write_table(table, source, row_group_size=3)
    original_source = source.read_bytes()

    rewrite_parquet_candidate(
        source,
        output,
        report_path,
        case=f"test-{score_type}",
        score_type=score_type,
        candidate=LayoutCandidate("test-layout", 4),
    )

    verify_exact_values(source, output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exact_value_equality"] is True
    assert report["source_unchanged"] is True
    assert report["write_seconds"] >= 0
    assert report["validation_seconds"] >= 0
    assert len(report["output_sha256"]) == 64
    expected_row_groups = [4, 4, 4] if score_type == "entropy" else [4, 2]
    assert report["physical_layout"]["row_group_rows"] == expected_row_groups
    assert (
        report["physical_layout"]["dictionary_columns"] == expected_dictionary_columns
    )
    assert report["physical_layout"]["compression"] == "ZSTD"
    assert report["physical_layout"]["compression_level"] == 3
    assert report["physical_layout"]["content_defined_chunking"] is True
    assert report["physical_layout"]["position_statistics"]["usable"] is True
    assert source.read_bytes() == original_source
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_failed_validation_preserves_existing_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "candidate.parquet"
    report = tmp_path / "rewrite.json"
    pq.write_table(_entropy_table(), source)
    output.write_bytes(b"previous complete output")

    def reject_candidate(*_: object) -> None:
        raise ValueError("exact comparison failed")

    monkeypatch.setattr(
        "gpn_star_scores.parquet_benchmark.verify_exact_values", reject_candidate
    )
    with pytest.raises(ValueError, match="exact comparison failed"):
        rewrite_parquet_candidate(
            source,
            output,
            report,
            case="test-entropy",
            score_type="entropy",
            candidate=LayoutCandidate("test-layout", 4),
        )

    assert output.read_bytes() == b"previous complete output"
    assert not report.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_interval_queries_are_one_based_and_boundary_safe() -> None:
    queries = interval_query_specs(101, 2_000_100)
    by_name = {query.name: query for query in queries}

    assert (
        by_name["interval-first-1000"].start,
        by_name["interval-first-1000"].end,
    ) == (
        101,
        1_100,
    )
    assert (by_name["interval-last-1000"].start, by_name["interval-last-1000"].end) == (
        1_999_101,
        2_000_100,
    )
    assert all(
        query.start is not None
        and query.end is not None
        and query.start >= 101
        and query.end <= 2_000_100
        for query in queries
    )


def test_position_statistics_reject_out_of_order_row_groups(tmp_path: Path) -> None:
    table = _entropy_table(4).set_column(
        1, "pos", pa.array([5, 6, 1, 2], type=pa.int64())
    )
    path = tmp_path / "unsorted.parquet"
    pq.write_table(table, path, row_group_size=2, write_statistics=True)

    statistics = inspect_position_statistics(path)

    assert statistics["complete"] is True
    assert statistics["monotonic"] is False
    assert statistics["usable"] is False


def test_local_benchmark_records_required_queries_and_repetitions(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "entropy.parquet"
    report_path = tmp_path / "benchmark.json"
    pq.write_table(_entropy_table(30), parquet_path, row_group_size=5)

    benchmark_parquet_candidate(
        str(parquet_path),
        report_path,
        case="gg6-chr32-entropy",
        candidate="source",
        access="local",
        sparse_key_count=9,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["warmups"] == 1
    assert report["repetitions"] == 5
    assert len(report["queries"]) == 9
    assert {query["kind"] for query in report["queries"]} == {
        "interval",
        "projection",
        "sparse_join",
        "full_scan",
    }
    assert all(len(query["duration_seconds"]) == 5 for query in report["queries"])
    assert all(query["transferred_bytes"] is None for query in report["queries"])
    assert report["position_statistics"]["usable"] is True


def test_hf_benchmark_source_reuses_filesystem_and_resets_counter() -> None:
    class CountingBytesIO(BytesIO):
        def __init__(self, filesystem: object) -> None:
            super().__init__(b"abcdef")
            self.filesystem = filesystem

        def read(self, size: int = -1) -> bytes:
            content = super().read(size)
            self.filesystem.counter.bytes += len(content)
            return content

    class FakeHfFileSystem:
        def __init__(self) -> None:
            self.open_count = 0
            self.counter = None

        def open(self, path: str, mode: str) -> CountingBytesIO:
            assert path == "datasets/owner/repo@revision/data.parquet"
            assert mode == "rb"
            self.open_count += 1
            return CountingBytesIO(self)

    filesystem = FakeHfFileSystem()
    uri = "hf://datasets/owner/repo@revision/data.parquet"
    measured = []
    for size in (2, 4):
        with _benchmark_source(
            uri,
            "hf",
            hf_token="unused",
            hf_block_size=4,
            hf_filesystem=filesystem,
        ) as (source, transferred_bytes):
            assert source.read(size) == b"abcdef"[:size]
            measured.append(transferred_bytes())

    assert filesystem.open_count == 2
    assert measured == [2, 4]


def test_hf_interval_query_reads_selected_row_groups(tmp_path: Path) -> None:
    parquet_path = tmp_path / "range-test.parquet"
    pq.write_table(
        _entropy_table(10_000),
        parquet_path,
        row_group_size=1_000,
        compression="NONE",
        write_statistics=True,
    )

    class CountingBytesIO(BytesIO):
        def __init__(self, content: bytes) -> None:
            super().__init__(content)
            self.bytes_read = 0

        def read(self, size: int = -1) -> bytes:
            content = super().read(size)
            self.bytes_read += len(content)
            return content

    content = parquet_path.read_bytes()
    source = CountingBytesIO(content)
    result = _execute_hf_query(
        source,
        QuerySpec("interval-first-10", "interval", start=1, end=10),
        pl.DataFrame(),
    )

    assert result.height == 10
    assert source.bytes_read < len(content)


def _complete_inventory_manifest(
    target_relative_path: str, source_path: Path
) -> dict[str, object]:
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    records = []
    for shard in expected_shards():
        relative_path = shard.relative_path.as_posix()
        is_target = relative_path == target_relative_path
        records.append(
            {
                "path": relative_path,
                "score_type": shard.score_type,
                "size": source_path.stat().st_size if is_target else 0,
                "sha256": source_sha256 if is_target else "0" * 64,
                "content": {"order_violations": 0},
                "valid": True,
                "errors": [],
            }
        )
    return {
        "manifest_version": 1,
        "source": {
            "expected_shards": 290,
            "reported_shards": 290,
            "discovered_parquet_files": 290,
            "missing_paths": [],
            "unexpected_paths": [],
            "unreported_paths": [],
        },
        "validation": {
            "valid_shards": 290,
            "invalid_shards": 0,
            "release_ready": False,
            "blockers": ["capacity evidence pending"],
        },
        "shards": records,
    }


def test_source_evidence_is_derived_from_complete_inventory_manifest(
    tmp_path: Path,
) -> None:
    relative_path = "gg6/entropy/entropy_chr32.parquet"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    pq.write_table(_entropy_table(), source_path)
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(
        manifest_path, _complete_inventory_manifest(relative_path, source_path)
    )
    evidence_path = tmp_path / "source-evidence.json"

    validate_benchmark_source(
        manifest_path,
        source_path,
        evidence_path,
        case="gg6-chr32-entropy",
        relative_path=relative_path,
        score_type="entropy",
    )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["inventory_valid"] is True
    assert evidence["position_sorted"] is True
    assert evidence["inventory_release_ready"] is False
    assert (
        evidence["source_sha256"]
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )

    source_path.write_bytes(source_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="does not match inventory"):
        validate_benchmark_source(
            manifest_path,
            source_path,
            tmp_path / "must-not-exist.json",
            case="gg6-chr32-entropy",
            relative_path=relative_path,
            score_type="entropy",
        )


def _benchmark_report(
    path: Path,
    *,
    candidate: str,
    access: str,
    size: int,
    interval_seconds: float,
    full_scan_seconds: float,
) -> None:
    atomic_write_json(
        path,
        {
            "candidate": candidate,
            "case": "gg6-chr32-entropy",
            "access": access,
            "uri": (
                f"hf://datasets/test/staging@{'a' * 40}/{candidate}.parquet"
                if access == "hf"
                else f"/local/{candidate}.parquet"
            ),
            "file_size_bytes": size,
            "position_statistics": {"usable": True} if access == "local" else None,
            "queries": [
                {
                    "kind": "interval",
                    "duration_seconds": [interval_seconds] * 5,
                },
                {
                    "kind": "full_scan",
                    "duration_seconds": [full_scan_seconds] * 5,
                },
            ],
        },
    )


def _selection_fixture(
    tmp_path: Path,
    *,
    rewrite_interval: float,
    source_functional: bool = True,
) -> tuple[list[Path], list[Path], list[Path], Path, list[Path]]:
    benchmark_paths = []
    metrics = {
        "source": (100, 10.0, 10.0),
        "zstd-262144": (105, rewrite_interval, 11.0),
        "zstd-1048576": (100, 8.0, 10.5),
    }
    candidate_sha256 = {
        "source": "b" * 64,
        "zstd-262144": "c" * 64,
        "zstd-1048576": "d" * 64,
    }
    for candidate, (size, interval_seconds, full_scan_seconds) in metrics.items():
        for access in ("local", "hf"):
            path = tmp_path / f"{candidate}-{access}.json"
            _benchmark_report(
                path,
                candidate=candidate,
                access=access,
                size=size,
                interval_seconds=interval_seconds,
                full_scan_seconds=full_scan_seconds,
            )
            benchmark_paths.append(path)

    rewrite_paths = []
    for candidate in ("zstd-262144", "zstd-1048576"):
        path = tmp_path / f"{candidate}-rewrite.json"
        atomic_write_json(
            path,
            {
                "case": "gg6-chr32-entropy",
                "candidate": {"name": candidate},
                "exact_value_equality": True,
                "output_sha256": candidate_sha256[candidate],
                "write_seconds": 12.5,
                "peak_rss_bytes": 4096,
            },
        )
        rewrite_paths.append(path)

    staging = {
        candidate: {
            "dataset_viewer": source_functional if candidate == "source" else True,
            "evidence": {
                "dataset_viewer_url": "https://huggingface.co/datasets/test/staging",
                "dataset_viewer_checked_at": "2026-07-21T00:00:00Z",
                "artifacts": {
                    "gg6-chr32-entropy": {
                        "sha256": candidate_sha256[candidate],
                        "uri": (
                            f"hf://datasets/test/staging@{'a' * 40}/{candidate}.parquet"
                        ),
                    }
                },
            },
        }
        for candidate in metrics
    }
    staging_path = tmp_path / "staging.json"
    atomic_write_json(staging_path, staging)
    evidence_path = tmp_path / "source-evidence.json"
    atomic_write_json(
        evidence_path,
        {
            "case": "gg6-chr32-entropy",
            "relative_path": "gg6/entropy/entropy_chr32.parquet",
            "source_size_bytes": 100,
            "source_sha256": candidate_sha256["source"],
            "position_sorted": True,
            "inventory_valid": True,
            "inventory_manifest_sha256": "a" * 64,
        },
    )
    hf_validation_paths = []
    for candidate in metrics:
        path = tmp_path / f"{candidate}-hf-validation.json"
        atomic_write_json(
            path,
            {
                "case": "gg6-chr32-entropy",
                "candidate": candidate,
                "passed": True,
            },
        )
        hf_validation_paths.append(path)
    return (
        benchmark_paths,
        rewrite_paths,
        [evidence_path],
        staging_path,
        hf_validation_paths,
    )


def test_selection_keeps_functional_source_below_improvement_threshold(
    tmp_path: Path,
) -> None:
    (
        benchmark_paths,
        rewrite_paths,
        evidence_paths,
        staging_path,
        hf_paths,
    ) = _selection_fixture(tmp_path, rewrite_interval=7.51)

    selection = select_layout(
        benchmark_paths, rewrite_paths, evidence_paths, staging_path, hf_paths
    )

    assert selection["status"] == "selected"
    assert selection["selected_candidate"] == "source"
    assert "Kept source files unchanged" in selection["rationale"]


def test_selection_applies_inclusive_thresholds_and_writes_dataset_card_text(
    tmp_path: Path,
) -> None:
    (
        benchmark_paths,
        rewrite_paths,
        evidence_paths,
        staging_path,
        hf_paths,
    ) = _selection_fixture(tmp_path, rewrite_interval=7.5)
    json_path = tmp_path / "selection.json"
    markdown_path = tmp_path / "selection.md"

    write_selection_outputs(
        benchmark_paths,
        rewrite_paths,
        evidence_paths,
        staging_path,
        json_path,
        markdown_path,
        hf_validation_reports=hf_paths,
    )

    selection = json.loads(json_path.read_text(encoding="utf-8"))
    assert selection["selected_candidate"] == "zstd-262144"
    assert selection["candidates"]["zstd-262144"][
        "remote_interval_ratio_to_source"
    ] == pytest.approx(0.75)
    assert "## Parquet layout benchmark" in markdown_path.read_text(encoding="utf-8")


def test_selection_rewrites_when_source_fails_dataset_viewer(tmp_path: Path) -> None:
    (
        benchmark_paths,
        rewrite_paths,
        evidence_paths,
        staging_path,
        hf_paths,
    ) = _selection_fixture(tmp_path, rewrite_interval=7.0, source_functional=False)

    selection = select_layout(
        benchmark_paths, rewrite_paths, evidence_paths, staging_path, hf_paths
    )

    assert selection["status"] == "selected"
    assert selection["selected_candidate"] == "zstd-262144"
    assert "functionally required" in selection["rationale"]


def test_staged_hash_mismatch_disqualifies_rewrite(tmp_path: Path) -> None:
    (
        benchmark_paths,
        rewrite_paths,
        evidence_paths,
        staging_path,
        hf_paths,
    ) = _selection_fixture(tmp_path, rewrite_interval=7.5)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["zstd-262144"]["evidence"]["artifacts"]["gg6-chr32-entropy"]["sha256"] = (
        "0" * 64
    )
    atomic_write_json(staging_path, staging)

    selection = select_layout(
        benchmark_paths, rewrite_paths, evidence_paths, staging_path, hf_paths
    )

    assert selection["selected_candidate"] == "source"
    assert selection["candidates"]["zstd-262144"]["functional"] is False
    assert (
        "staged_artifacts_match"
        in selection["candidates"]["zstd-262144"]["failed_functional_checks"]
    )


def test_selection_blocks_when_source_order_is_not_confirmed(tmp_path: Path) -> None:
    (
        benchmark_paths,
        rewrite_paths,
        evidence_paths,
        staging_path,
        hf_paths,
    ) = _selection_fixture(tmp_path, rewrite_interval=7.0)
    evidence = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
    evidence["position_sorted"] = False
    atomic_write_json(evidence_paths[0], evidence)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["source"]["position_sorted"] = True
    atomic_write_json(staging_path, staging)

    selection = select_layout(
        benchmark_paths, rewrite_paths, evidence_paths, staging_path, hf_paths
    )

    assert selection["status"] == "blocked"
    assert selection["selected_candidate"] is None
    assert any("cannot repair" in blocker for blocker in selection["blockers"])


def test_enabled_workflow_builds_complete_benchmark_dag(tmp_path: Path) -> None:
    source_root = tmp_path / "stage"
    source_path = source_root / "synthetic" / "entropy.parquet"
    source_path.parent.mkdir(parents=True)
    pq.write_table(_entropy_table(), source_path)
    staging_path = tmp_path / "staging.json"
    atomic_write_json(
        staging_path,
        {
            candidate: {
                "dataset_viewer": True,
                "evidence": {
                    "dataset_viewer_url": "https://huggingface.co/datasets/test/staging",
                    "dataset_viewer_checked_at": "2026-07-21T00:00:00Z",
                },
            }
            for candidate in ("source", "zstd-262144", "zstd-1048576")
        },
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, {})
    config_path = tmp_path / "config.yaml"
    revision = "a" * 40
    config_path.write_text(
        f"""\
parquet_benchmark:
  enabled: true
  source_root: {source_root}
  output_root: {tmp_path / "output"}
  inventory_manifest: {manifest_path}
  cases:
    - id: synthetic-entropy
      score_type: entropy
      relative_path: synthetic/entropy.parquet
  remote_uris:
    synthetic-entropy:
      source: hf://datasets/owner/staging@{revision}/source.parquet
      zstd-262144: hf://datasets/owner/staging@{revision}/zstd-262144.parquet
      zstd-1048576: hf://datasets/owner/staging@{revision}/zstd-1048576.parquet
  staging_checks: {staging_path}
  resources:
    rewrite: {{mem_mb: 4096, runtime: 240, disk_mb: 1024}}
    benchmark: {{mem_mb: 4096, runtime: 240, disk_mb: 1024}}
    report: {{mem_mb: 4096, runtime: 30, disk_mb: 1024}}
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            "workflow/Snakefile",
            "--configfile",
            str(config_path),
            "--cores",
            "1",
            "--dry-run",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "validate_source_shard" in result.stdout
    assert "rewrite_parquet_shard" in result.stdout
    assert "benchmark_parquet_shard" in result.stdout
    assert "validate_parquet_shard" in result.stdout
    assert "render_report" in result.stdout
