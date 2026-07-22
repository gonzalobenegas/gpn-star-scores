import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyBigWig
import pytest

from gpn_star_scores.catalog import ASSEMBLIES, EXPECTED_SHARD_COUNT, expected_shards
from gpn_star_scores.bigwig import (
    ChromosomeSpec,
    validate_bigwig,
    write_entropy_bigwig,
)
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.tracks import (
    assembly_chromosome_sizes_from_contract,
    audit_final_track_bigwig,
    build_score_type_tracks,
    chromosome_spec_from_contract,
    load_track_input_contract,
    render_track_benchmark,
    stream_concatenate_bigwigs,
    ucsc_assembly_name,
    ucsc_chromosome_name,
    validate_score_type_tracks,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def _write_contract(tmp_path: Path) -> tuple[Path, Path]:
    records = []
    for shard in expected_shards():
        rows = 6 if shard.score_type == "llr" else 2
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


def test_stream_concatenate_bigwigs_rounds_values_and_preserves_gaps(
    tmp_path: Path,
) -> None:
    chromosome_sizes = {"chr1": 3, "chr2": 2}
    inputs = []
    for source_chrom, ucsc_chrom, length, positions, values in (
        ("1", "chr1", 3, [1, 3], [0.256, 1.754]),
        ("2", "chr2", 2, [2], [0.504]),
    ):
        source = tmp_path / f"{ucsc_chrom}.parquet"
        pq.write_table(
            pa.table(
                {
                    "chrom": pa.array([source_chrom] * len(positions)),
                    "pos": pa.array(positions, type=pa.int64()),
                    "ref": pa.array(["A"] * len(positions)),
                    "entropy_calibrated": pa.array(values, type=pa.float32()),
                }
            ),
            source,
        )
        output = tmp_path / f"{ucsc_chrom}.bw"
        write_entropy_bigwig(
            [source],
            output,
            ChromosomeSpec(source_chrom, ucsc_chrom, length),
            batch_size=1,
            header_chromosome_sizes=chromosome_sizes,
        )
        inputs.append(output)

    combined = tmp_path / "combined.bw"
    stream_concatenate_bigwigs(
        inputs,
        combined,
        chromosome_sizes,
        list(chromosome_sizes),
        batch_size=2,
        value_decimals=2,
    )

    summary = validate_bigwig(
        combined,
        chromosome_sizes,
        expected_bases_covered=3,
    )
    assert summary.zoom_levels >= 1
    with pyBigWig.open(str(combined)) as bigwig:
        assert bigwig.values("chr1", 0, 3) == pytest.approx(
            [0.26, float("nan"), 1.75], nan_ok=True
        )
        assert bigwig.values("chr2", 0, 2) == pytest.approx(
            [float("nan"), 0.5], nan_ok=True
        )

    with pytest.raises(ValueError, match="value_decimals"):
        stream_concatenate_bigwigs(
            inputs,
            tmp_path / "invalid-precision.bw",
            chromosome_sizes,
            list(chromosome_sizes),
            value_decimals=-1,
        )


def test_final_audit_checks_every_sample_and_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, selection = _write_contract(tmp_path)
    contract = load_track_input_contract(manifest, selection)
    chromosome_sizes = assembly_chromosome_sizes_from_contract(contract, "ce11")
    chromosome_reports = []
    source_values = [np.float32(0.1234), np.float32(0.5678)]
    rounded_values = [float(np.round(value, decimals=3)) for value in source_values]

    for chrom in ASSEMBLIES["ce11"].chromosomes:
        report = tmp_path / f"{chrom}.json"
        report.write_text(
            json.dumps(
                {
                    "report_version": 1,
                    "valid": True,
                    "method": "direct",
                    "score_set": "ce11",
                    "inventory_manifest_sha256": contract.manifest_sha256,
                    "chromosome": {
                        "source_name": chrom,
                        "ucsc_name": f"chr{chrom}",
                    },
                    "validation": {
                        "entropy": {
                            "sample_count": 2,
                            "first_position": 1,
                            "last_position": 3,
                            "first_gap_position": 2,
                            "gap_checks": {"entropy": True},
                            "samples": {
                                "entropy": [
                                    {
                                        "position_1based": 1,
                                        "expected_float32": float(source_values[0]),
                                    },
                                    {
                                        "position_1based": 3,
                                        "expected_float32": float(source_values[1]),
                                    },
                                ]
                            },
                        }
                    },
                }
            )
        )
        chromosome_reports.append(report)

    concatenation_report = tmp_path / "concatenation.json"
    concatenation_report.write_text(
        json.dumps(
            {
                "report_version": 1,
                "valid": True,
                "score_set": "ce11",
                "track": "entropy",
                "value_decimals": 3,
                "inventory_manifest_sha256": contract.manifest_sha256,
                "concatenation_method": "pyBigWig-stream-copy",
            }
        )
    )

    def write_final(path: Path, last_value: float) -> None:
        writer = pyBigWig.open(str(path), "w")
        writer.addHeader(list(chromosome_sizes.items()))
        for ucsc_chrom in chromosome_sizes:
            writer.addEntries(
                [ucsc_chrom, ucsc_chrom],
                [0, 2],
                ends=[1, 3],
                values=[rounded_values[0], last_value],
            )
        writer.close()

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="zoomLevels: 1\n"
        ),
    )
    final = tmp_path / "entropy.bw"
    audit_report = tmp_path / "audit.json"
    write_final(final, rounded_values[1])

    audit_final_track_bigwig(
        final,
        chromosome_reports,
        concatenation_report,
        audit_report,
        manifest,
        selection,
        score_set="ce11",
        track="entropy",
        value_decimals=3,
    )

    audit = json.loads(audit_report.read_text())
    assert audit["validation_stage"] == "post-assembly-audit"
    assert audit["concatenated_sample_check_count"] == 12
    assert audit["concatenated_gap_check_count"] == 6
    assert all(item["gap_absent"] for item in audit["concatenated_chromosome_checks"])

    previous_report = audit_report.read_text()
    corrupt = tmp_path / "corrupt.bw"
    write_final(corrupt, 0.5)
    with pytest.raises(ValueError, match="final entropy differs"):
        audit_final_track_bigwig(
            corrupt,
            chromosome_reports,
            concatenation_report,
            audit_report,
            manifest,
            selection,
            score_set="ce11",
            track="entropy",
            value_decimals=3,
        )
    assert audit_report.read_text() == previous_report


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
  value_decimals: 3
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
    audit: {{mem_mb: 4096, runtime: 30, disk_mb: 1024}}
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
    assert "audit_final_bigwig" in result.stdout
    assert "aggregate_validation" in result.stdout


@pytest.mark.slow
@pytest.mark.parametrize("method", ["wig", "direct"])
def test_real_ucsc_tools_concatenate_full_assembly_headers(
    tmp_path: Path, method: str
) -> None:
    executables = {
        name: shutil.which(name) for name in ("wigToBigWig", "bigWigCat", "bigWigInfo")
    }
    if any(path is None for path in executables.values()):
        pytest.skip("pinned UCSC command-line tools are not installed")
    header_sizes = {"chr1": 3, "chr2": 2}
    inputs = []
    for chrom, position, value in (("1", 1, 0.25), ("2", 2, 0.5)):
        source = tmp_path / f"chr{chrom}.parquet"
        pq.write_table(
            pa.table(
                {
                    "chrom": pa.array([chrom]),
                    "pos": pa.array([position], type=pa.int64()),
                    "ref": pa.array(["A"]),
                    "entropy_calibrated": pa.array([value], type=pa.float32()),
                }
            ),
            source,
        )
        outputs, _ = build_score_type_tracks(
            source,
            tmp_path / f"{method}-chr{chrom}",
            score_type="entropy",
            method=method,
            chromosome=ChromosomeSpec(
                chrom, f"chr{chrom}", header_sizes[f"chr{chrom}"]
            ),
            wig_to_bigwig=str(executables["wigToBigWig"]),
            header_chromosome_sizes=header_sizes,
        )
        inputs.append(outputs["entropy"])

    combined = tmp_path / f"{method}.bw"
    subprocess.run(
        [str(executables["bigWigCat"]), str(combined), *map(str, inputs)],
        check=True,
    )
    subprocess.run(
        [str(executables["bigWigInfo"]), str(combined)],
        check=True,
        capture_output=True,
    )
    validate_bigwig(combined, header_sizes, expected_bases_covered=2)
