import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gpn_star_scores.catalog import EXPECTED_SHARD_COUNT, expected_shards
from gpn_star_scores.bigwig import ChromosomeSpec
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.tracks import (
    assembly_chromosome_sizes_from_contract,
    build_score_type_tracks,
    chromosome_spec_from_contract,
    load_track_input_contract,
    render_track_benchmark,
    ucsc_assembly_name,
    ucsc_chromosome_name,
    validate_score_type_tracks,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def _write_contract(tmp_path: Path) -> tuple[Path, Path]:
    records = []
    for shard in expected_shards():
        rows = 3 if shard.score_type == "llr" else 1
        records.append(
            {
                "path": shard.relative_path.as_posix(),
                "score_set": shard.score_set,
                "assembly": shard.assembly,
                "score_type": shard.score_type,
                "chrom": shard.chrom,
                "size": 1,
                "sha256": "a" * 64,
                "valid": True,
                "errors": [],
                "parquet": {"num_rows": rows},
                "content": {"order_violations": 0, "reference_length": 100},
            }
        )
    manifest = {
        "manifest_version": 1,
        "source": {
            "expected_shards": EXPECTED_SHARD_COUNT,
            "reported_shards": EXPECTED_SHARD_COUNT,
            "discovered_parquet_files": EXPECTED_SHARD_COUNT,
            "missing_paths": [],
            "unexpected_paths": [],
            "unreported_paths": [],
        },
        "validation": {
            "valid_shards": EXPECTED_SHARD_COUNT,
            "invalid_shards": 0,
            "release_ready": False,
        },
        "shards": records,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    selection = {
        "report_version": 1,
        "status": "selected",
        "selected_candidate": "source",
        "source_inventory": {
            "valid": True,
            "manifest_sha256": sha256_file(manifest_path),
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    return manifest_path, selection_path


def test_track_contract_uses_catalog_and_inventory_lengths(tmp_path: Path) -> None:
    manifest, selection = _write_contract(tmp_path)

    contract = load_track_input_contract(manifest, selection)
    chromosome = chromosome_spec_from_contract(contract, "gg6", "32")
    header_sizes = assembly_chromosome_sizes_from_contract(contract, "gg6")

    assert chromosome.source_name == "32"
    assert chromosome.ucsc_name == "chr32"
    assert chromosome.length == 100
    assert len(header_sizes) == 34
    assert header_sizes["chr32"] == 100
    assert list(header_sizes)[:2] == ["chr1", "chr2"]
    assert ucsc_assembly_name("gg6") == "galGal6"
    assert ucsc_assembly_name("tair10") == "araTha1"
    assert ucsc_chromosome_name("X") == "chrX"


def test_direct_score_type_build_and_sample_validation(tmp_path: Path) -> None:
    source = tmp_path / "entropy.parquet"
    pq.write_table(
        pa.table(
            {
                "chrom": pa.array(["1", "1"]),
                "pos": pa.array([1, 3], type=pa.int64()),
                "ref": pa.array(["A", "G"]),
                "entropy_calibrated": pa.array([0.25, 1.75], type=pa.float32()),
            }
        ),
        source,
    )
    chromosome = ChromosomeSpec("1", "chr1", 3)
    outputs, stats = build_score_type_tracks(
        source,
        tmp_path / "tracks",
        score_type="entropy",
        method="direct",
        chromosome=chromosome,
        batch_size=1,
    )

    validation = validate_score_type_tracks(
        source,
        outputs,
        score_type="entropy",
        chromosome=chromosome,
        expected_position_count=2,
        sample_count=2,
        batch_size=1,
    )
    assert stats.position_count == 2
    assert validation["float32_exact"]
    assert validation["first_gap_position"] == 2
    assert validation["gap_checks"] == {"entropy": True}


def test_render_track_benchmark_aggregates_case_medians(tmp_path: Path) -> None:
    reports = []
    for case in ("case-a", "case-b"):
        for method, wall, scratch in (
            ("wig", 10.0, 100),
            ("direct", 8.0, 90),
        ):
            path = tmp_path / f"{case}-{method}.json"
            path.write_text(
                json.dumps(
                    {
                        "report_version": 1,
                        "case": case,
                        "method": method,
                        "inventory_manifest_sha256": "a" * 64,
                        "summary": {
                            "method": method,
                            "measured_repetitions": 5,
                            "median_wall_seconds": wall,
                            "peak_rss_bytes": 50,
                            "peak_scratch_bytes": scratch,
                            "final_bytes": 25,
                            "correct": True,
                        },
                    }
                )
            )
            reports.append(path)
    output_json = tmp_path / "selection.json"
    output_markdown = tmp_path / "selection.md"

    render_track_benchmark(reports, output_json, output_markdown)

    selection = json.loads(output_json.read_text())
    assert selection["selected_method"] == "direct"
    assert selection["aggregates"]["direct"]["median_wall_seconds"] == 16.0
    assert selection["aggregates"]["wig"]["median_wall_seconds"] == 20.0
    assert "Selected method: `direct`" in output_markdown.read_text()


def test_enabled_workflow_builds_all_track_rules(tmp_path: Path) -> None:
    source_root = tmp_path / "stage"
    for shard in expected_shards():
        path = source_root / shard.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    manifest, parquet_selection = _write_contract(tmp_path)
    track_selection = tmp_path / "track-selection.json"
    track_selection.write_text(
        json.dumps(
            {
                "report_version": 1,
                "status": "selected",
                "selected_method": "direct",
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""\
bigwig:
  enabled: true
  source_root: {source_root}
  output_root: {tmp_path / "output"}
  inventory_manifest: {manifest}
  parquet_selection: {parquet_selection}
  selection_report: {track_selection}
  batch_size: 4
  sample_count: 2
  benchmark:
    enabled: true
    output_root: {tmp_path / "benchmark"}
    repetitions: 1
    cases:
      - id: gg6-chr32-entropy
        score_set: gg6
        score_type: entropy
        chrom: "32"
  resources:
    benchmark: {{mem_mb: 4096, runtime: 240, disk_mb: 1024}}
    report: {{mem_mb: 4096, runtime: 30, disk_mb: 1024}}
    build_chromosome: {{mem_mb: 4096, runtime: 240, disk_mb: 1024}}
    concatenate: {{mem_mb: 4096, runtime: 30, disk_mb: 1024}}
    aggregate: {{mem_mb: 4096, runtime: 30, disk_mb: 1024}}
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
    assert "benchmark_bigwig_method" in result.stdout
    assert "render_bigwig_report" in result.stdout
    assert "build_chromosome_bigwig" in result.stdout
    assert "concatenate_bigwig" in result.stdout
    assert "aggregate_validation" in result.stdout
