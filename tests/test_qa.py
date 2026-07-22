from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import polars as pl
import pytest

from gpn_star_scores.catalog import SCORE_SETS, expected_shards
from gpn_star_scores.inventory import EXPECTED_SCHEMAS, sha256_file
from gpn_star_scores.qa import (
    QA_APPROVAL_ISSUE,
    RELEASE_TAG,
    VIEWER_FOLLOWUP_ISSUE,
    VIEWER_WAIVER_ID,
    VIEWER_WAIVER_ISSUE,
    build_release_record,
    create_release_tag,
    run_dataset_card_examples,
    validate_dataset_viewer_discovery,
    validate_public_hub_for_qa,
    validate_public_release_for_qa,
)
from gpn_star_scores.release import (
    CAPACITY_APPROVAL_ISSUE,
    CAPACITY_BLOCKER,
    PUBLIC_STORAGE_POLICY,
    REPOSITORY_ID,
    dataset_configs,
)
from gpn_star_scores.tracks import TRACKS, ucsc_assembly_name

REPOSITORY_ROOT = Path(__file__).parents[1]
RELEASE_REVISION = "a" * 40
HUB_REVISION = "b" * 40
WORKFLOW_COMMIT = "c" * 40


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_runs_all_dataset_card_polars_examples(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_parquet(
        root / "gpn-star-hg38-v100-200m" / "entropy" / "entropy_chr22.parquet",
        pl.DataFrame(
            {
                "chrom": pl.Series(["22"], dtype=pl.String),
                "pos": pl.Series([20_000_001], dtype=pl.Int64),
                "ref": pl.Series(["A"], dtype=pl.String),
                "entropy_calibrated": pl.Series([0.5], dtype=pl.Float32),
            }
        ),
    )
    _write_parquet(
        root / "ce11" / "entropy" / "entropy_chrI.parquet",
        pl.DataFrame(
            {
                "chrom": pl.Series(["I"], dtype=pl.String),
                "pos": pl.Series([65], dtype=pl.Int64),
                "ref": pl.Series(["A"], dtype=pl.String),
                "entropy_calibrated": pl.Series([0.25], dtype=pl.Float32),
            }
        ),
    )
    _write_parquet(
        root / "dm6" / "llr" / "llr_chr2L.parquet",
        pl.DataFrame(
            {
                "chrom": pl.Series(["2L", "X"], dtype=pl.String),
                "pos": pl.Series([65, 65], dtype=pl.Int64),
                "ref": pl.Series(["A", "C"], dtype=pl.String),
                "alt": pl.Series(["G", "T"], dtype=pl.String),
                "llr_calibrated": pl.Series([0.1, 0.2], dtype=pl.Float32),
                "abs_llr_calibrated": pl.Series([0.3, 0.4], dtype=pl.Float32),
            }
        ),
    )
    _write_parquet(
        root / "gpn-star-hg38-p243-200m" / "llr" / "llr_chr22.parquet",
        pl.DataFrame(
            {
                "chrom": pl.Series(["22"], dtype=pl.String),
                "pos": pl.Series([20_000_001], dtype=pl.Int64),
                "ref": pl.Series(["A"], dtype=pl.String),
                "alt": pl.Series(["G"], dtype=pl.String),
                "llr_calibrated": pl.Series([0.1], dtype=pl.Float32),
                "abs_llr_calibrated": pl.Series([0.2], dtype=pl.Float32),
            }
        ),
    )

    report = run_dataset_card_examples(str(root))

    assert report["valid"] is True
    assert report["credentials_sent"] is False
    assert [item["name"] for item in report["examples"]] == [
        "interval_filter_projection",
        "projected_score_scan",
        "multi_chromosome_scan",
        "variant_join",
    ]
    assert [item["rows"] for item in report["examples"]] == [1, 1, 2, 1]


def test_public_release_qa_combines_anonymous_checks_and_examples(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"valid": True, "credentials_sent": False}

    def examples(root: str) -> dict[str, Any]:
        assert root.endswith(f"@{RELEASE_REVISION}/data")
        return {"valid": True, "credentials_sent": False, "examples": []}

    output = tmp_path / "public-release.json"
    validate_public_release_for_qa(
        tmp_path,
        output,
        revision=RELEASE_REVISION,
        validator=validator,
        example_runner=examples,
        viewer_discovery_validator=lambda repository_id: {
            "valid": repository_id == REPOSITORY_ID,
            "credentials_sent": False,
        },
    )

    report = json.loads(output.read_text())
    assert report["valid"] is True
    assert report["revision"] == RELEASE_REVISION
    assert report["credentials_sent"] is False
    assert calls[0][1]["viewer_required"] is False


def test_public_hub_qa_requires_valid_anonymous_result(tmp_path: Path) -> None:
    output = tmp_path / "public-hub.json"

    def validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["revision"] == HUB_REVISION
        return {"valid": True, "credentials_sent": False}

    validate_public_hub_for_qa(
        tmp_path,
        output,
        revision=HUB_REVISION,
        udc_dir=tmp_path / "udc",
        validator=validator,
    )
    assert json.loads(output.read_text())["valid"] is True


def test_dataset_viewer_discovery_requires_all_configs() -> None:
    splits = {
        "splits": [
            {
                "dataset": REPOSITORY_ID,
                "config": config["config_name"],
                "split": "train",
            }
            for config in dataset_configs()
        ],
        "pending": [],
        "failed": [],
    }
    valid = {"preview": True, "viewer": True, "search": True, "filter": True}

    def opener(url: str, **kwargs: Any) -> io.BytesIO:
        payload = splits if "/splits?" in url else valid
        return io.BytesIO(json.dumps(payload).encode())

    report = validate_dataset_viewer_discovery(opener=opener)

    assert report["valid"] is True
    assert report["split_count"] == 16
    assert report["credentials_sent"] is False


def _schema(score_type: str) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in EXPECTED_SCHEMAS[score_type]
    ]


def _write_release_metadata(tmp_path: Path) -> Path:
    metadata = tmp_path / "metadata"
    manifest_dir = metadata / "manifest"
    manifest_dir.mkdir(parents=True)
    records = []
    for shard in expected_shards():
        content = {
            "coordinate_bounds": {"min": 1, "max": 3},
            "invalid_alt_rows": 0,
            "invalid_ref_rows": 0,
            "llr_group_checks_skipped_for_nulls": False,
            "llr_group_errors": (
                {"inconsistent_ref": 0, "wrong_alt_set": 0, "wrong_size": 0}
                if shard.score_type == "llr"
                else None
            ),
            "non_finite_counts": (
                {"entropy_calibrated": 0}
                if shard.score_type == "entropy"
                else {"llr_calibrated": 0, "abs_llr_calibrated": 0}
            ),
            "null_counts": {
                field.name: 0 for field in EXPECTED_SCHEMAS[shard.score_type]
            },
            "order_violations": 0,
            "out_of_bounds_rows": 0,
            "reference_mismatch_rows": 0,
            "rows_scanned": 3,
            "unexpected_chrom_rows": 0,
        }
        records.append(
            {
                "path": shard.relative_path.as_posix(),
                "score_set": shard.score_set,
                "assembly": shard.assembly,
                "score_type": shard.score_type,
                "chrom": shard.chrom,
                "size": 1,
                "sha256": "d" * 64,
                "valid": True,
                "errors": [],
                "schema": _schema(shard.score_type),
                "parquet": {"num_rows": 3},
                "content": content,
            }
        )
    inventory = {
        "manifest_version": 1,
        "source": {
            "expected_shards": 290,
            "reported_shards": 290,
            "discovered_parquet_files": 290,
            "missing_paths": [],
            "unexpected_paths": [],
            "unreported_paths": [],
            "total_shard_bytes": 290,
        },
        "validation": {
            "valid_shards": 290,
            "invalid_shards": 0,
            "release_ready": False,
            "blockers": [CAPACITY_BLOCKER],
        },
        "references": [
            {
                "assembly": assembly,
                "identity_verified": True,
                "source_sha256": "e" * 64,
                "expected_sha256": "e" * 64,
            }
            for assembly in ("hg38", "ce11", "dm6", "gg6", "tair10", "mm39")
        ],
        "shards": records,
    }
    inventory_path = manifest_dir / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    inventory_sha = sha256_file(inventory_path)
    selection = {
        "status": "selected",
        "selected_candidate": "source",
        "blockers": [],
        "source_inventory": {"valid": True, "manifest_sha256": inventory_sha},
    }
    (manifest_dir / "parquet-layout.json").write_text(json.dumps(selection))
    tracks = [
        {
            "score_set": score_set.name,
            "assembly": score_set.assembly,
            "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
            "track": track,
            "bases_covered": 3,
            "zoom_levels": 1,
        }
        for score_set in SCORE_SETS
        for track in TRACKS
    ]
    bigwig_validation = {
        "report_version": 1,
        "valid": True,
        "track_count": 40,
        "selected_method": "direct",
        "value_decimals": 3,
        "inventory_manifest_sha256": inventory_sha,
        "sample_check_count": 400,
        "gap_check_count": 40,
        "tracks": tracks,
    }
    (manifest_dir / "bigwig-validation.json").write_text(json.dumps(bigwig_validation))
    release_manifest = {
        "release_manifest_version": 1,
        "repository": {
            "id": REPOSITORY_ID,
            "public": True,
            "license": "apache-2.0",
        },
        "source_inventory": {
            "manifest_sha256": inventory_sha,
            "total_shard_bytes": 290,
            "capacity_waiver": {
                "approved": True,
                "public_repository": True,
                "approved_by": "author",
                "approved_at": "2026-07-22",
                "evidence_url": CAPACITY_APPROVAL_ISSUE,
                "public_storage_policy_url": PUBLIC_STORAGE_POLICY,
                "planned_release_bytes": 330,
                "reserved_headroom_bytes": 33,
            },
        },
        "parquet": {
            "file_count": 290,
            "total_bytes": 290,
            "files": [
                {
                    "path": f"data/{record['path']}",
                    "score_set": record["score_set"],
                    "assembly": record["assembly"],
                    "score_type": record["score_type"],
                    "chrom": record["chrom"],
                    "size": record["size"],
                    "sha256": record["sha256"],
                    "rows": record["parquet"]["num_rows"],
                    "coordinate_bounds": record["content"]["coordinate_bounds"],
                }
                for record in records
            ],
        },
        "bigwig": {
            "file_count": 40,
            "total_bytes": 40,
            "value_decimals": 3,
            "files": [
                {
                    **track,
                    "path": (f"bigwig/{track['score_set']}/{track['track']}.bw"),
                    "size": 1,
                    "sha256": "f" * 64,
                }
                for track in tracks
            ],
        },
        "validation": {
            "preflight_passed": True,
            "inventory_data_validation_passed": True,
            "parquet_layout_selected": "source",
            "bigwig_validation_passed": True,
            "expected_parquet_files": 290,
            "expected_bigwig_files": 40,
            "expected_viewer_configs": 16,
        },
        "dataset_configs": dataset_configs(),
    }
    (manifest_dir / "release.json").write_text(json.dumps(release_manifest))
    (metadata / "README.md").write_text("# GPN-Star genome-wide scores\n")
    return metadata


def _write_qa_evidence(tmp_path: Path, metadata: Path) -> dict[str, Path]:
    hub_metadata = tmp_path / "hub-metadata"
    hub_metadata.mkdir()
    hub_readme = hub_metadata / "README.md"
    hub_readme.write_text(
        '\nvariant_chrom = "22"\n'
        'f"{root}/model/llr/llr_chr{variant_chrom}.parquet"\n'
        'pl.col("pos").is_between(variant_start, variant_end)\n'
    )
    public_release = {
        "report_version": 1,
        "valid": True,
        "repository": REPOSITORY_ID,
        "revision": RELEASE_REVISION,
        "public": True,
        "credentials_sent": False,
        "release_validation": {
            "valid": True,
            "repository": REPOSITORY_ID,
            "revision": RELEASE_REVISION,
            "public": True,
            "credentials_sent": False,
            "checksum_file_count": 330,
            "published_artifact_file_count": 330,
            "bigwig_range_count": 40,
            "bigwig_range_checks": [
                {"path": f"track-{index}", "status": 206} for index in range(40)
            ],
            "dataset_card_rendered": True,
            "parquet_query_checks": [
                {
                    "object_bytes": 100,
                    "range_reader_transferred_bytes": 10,
                    "polars_rows": 1,
                    "polars_projection_pushdown": True,
                    "polars_predicate_pushdown": True,
                },
                {
                    "object_bytes": 200,
                    "range_reader_transferred_bytes": 20,
                    "polars_rows": 1,
                    "polars_projection_pushdown": True,
                    "polars_predicate_pushdown": True,
                },
            ],
            "viewer_ready": False,
            "viewer_config_count": 0,
            "viewer_pending": [
                {"config_name": f"pending-{index}"} for index in range(16)
            ],
        },
        "dataset_card_examples": {
            "valid": True,
            "credentials_sent": False,
            "root": f"hf://datasets/{REPOSITORY_ID}@{RELEASE_REVISION}/data",
            "examples": [
                {
                    "name": "interval_filter_projection",
                    "rows": 1,
                    "columns": ["chrom", "pos", "ref", "entropy_calibrated"],
                },
                {
                    "name": "projected_score_scan",
                    "rows": 1,
                    "columns": ["chrom", "pos", "entropy_calibrated"],
                },
                {
                    "name": "multi_chromosome_scan",
                    "rows": 1,
                    "columns": ["chrom", "pos", "ref", "alt", "llr_calibrated"],
                },
                {
                    "name": "variant_join",
                    "rows": 1,
                    "columns": [
                        "chrom",
                        "pos",
                        "ref",
                        "alt",
                        "llr_calibrated",
                        "abs_llr_calibrated",
                    ],
                },
            ],
        },
        "dataset_viewer_discovery": {
            "valid": True,
            "credentials_sent": False,
            "split_count": 16,
            "configs": sorted(config["config_name"] for config in dataset_configs()),
            "preview": True,
            "viewer": True,
        },
    }
    hub_manifest_sha = "1" * 64
    public_hub = {
        "valid": True,
        "repository": REPOSITORY_ID,
        "revision": HUB_REVISION,
        "public": True,
        "credentials_sent": False,
        "file_count": 35,
        "file_checks": [
            {"path": "manifest/ucsc-hub.json", "sha256": hub_manifest_sha},
            {"path": "README.md", "sha256": sha256_file(hub_readme)},
        ],
        "hub_validation": {
            "valid": True,
            "artifact_revision": RELEASE_REVISION,
            "track_count": 40,
            "score_set_count": 8,
            "assembly_count": 6,
            "http_range_count": 40,
            "chromosome_checks": [{} for _ in range(40)],
            "representative_checks": [{} for _ in range(8)],
            "hub_check": {"passed": True},
        },
    }
    release_manifest_sha = sha256_file(metadata / "manifest" / "release.json")
    hub_evidence = {
        "release_manifest_sha256": release_manifest_sha,
        "publication": {
            "public_revision": HUB_REVISION,
            "artifact_revision": RELEASE_REVISION,
            "public_validation_valid": True,
            "credentials_sent_during_validation": False,
            "hub_manifest_sha256": hub_manifest_sha,
        },
        "manual_browser_validation": {
            "status": "passed",
            "failed": [],
            "passed_base_and_zoom": [score_set.name for score_set in SCORE_SETS],
        },
    }
    scf_evidence = {
        "result": "completed",
        "profile": "workflow/profiles/scf/config.yaml",
        "workflow_run_id": "run",
        "jobs": [
            {
                "job_id": "1",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed_seconds": 6,
                "step_max_rss_kib": 476,
                "allocated_cpus": 4,
                "cpu_efficiency_percent": 18.5,
                "memory_usage_percent": 0.01,
            },
            {
                "job_id": "2",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed_seconds": 6,
                "step_max_rss_kib": 500,
                "allocated_cpus": 4,
                "cpu_efficiency_percent": 18.6,
                "memory_usage_percent": 0.01,
            },
        ],
    }
    bigwig_evidence = {
        "status": "complete",
        "production": {
            "valid": True,
            "track_count": 40,
            "final_bytes": 40,
            "inventory_manifest_sha256": sha256_file(
                metadata / "manifest" / "inventory.json"
            ),
        },
        "scheduler_efficiency": {
            "benchmark_jobs": {
                "cpu_efficiency_percent_range": [66.0, 94.0],
                "memory_usage_percent_range": [0.1, 55.0],
            },
            "p243_chromosome_builds": {
                "cpu_efficiency_percent_range": [13.0, 62.0],
                "memory_usage_percent_range": [10.0, 93.0],
            },
            "p243_finalizers": {
                "cpu_efficiency_percent_range": [82.0, 92.0],
                "memory_usage_percent_range": [47.0, 53.0],
            },
            "expanded_audit_pilot": {
                "job_id": 3,
                "elapsed_seconds": 39,
                "peak_rss_mib": 1050.0,
                "cpu_efficiency_percent": 30.0,
                "memory_usage_percent": 26.0,
            },
            "expanded_audit_production": {
                "job_count": 39,
                "elapsed_seconds_range": [40, 265],
                "peak_rss_mib_range": [270.0, 1439.0],
                "cpu_efficiency_percent_range": [5.0, 17.0],
                "memory_usage_percent_range": [6.0, 36.0],
            },
        },
    }
    values = {
        "public_release": public_release,
        "public_hub": public_hub,
        "hub_evidence": hub_evidence,
        "scf_evidence": scf_evidence,
        "bigwig_evidence": bigwig_evidence,
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    lock = tmp_path / "uv.lock"
    profile = tmp_path / "config.yaml"
    lock.write_text("locked\n")
    profile.write_text("executor: slurm\n")
    paths["lock"] = lock
    paths["profile"] = profile
    paths["hub_metadata"] = hub_metadata
    return paths


def _viewer_waiver() -> dict[str, Any]:
    return {
        "id": VIEWER_WAIVER_ID,
        "approved": True,
        "evidence_url": VIEWER_WAIVER_ISSUE,
        "tracked_by": VIEWER_FOLLOWUP_ISSUE,
        "approved_by": "author",
        "approved_at": "2026-07-22",
    }


def test_release_record_reconciles_full_chain_and_keeps_tag_gated(
    tmp_path: Path,
) -> None:
    metadata = _write_release_metadata(tmp_path)
    paths = _write_qa_evidence(tmp_path, metadata)
    output_json = tmp_path / "record" / "release-record.json"
    output_markdown = tmp_path / "record" / "release-record.md"

    build_release_record(
        metadata,
        paths["hub_metadata"],
        paths["public_release"],
        paths["public_hub"],
        paths["scf_evidence"],
        paths["bigwig_evidence"],
        paths["hub_evidence"],
        paths["lock"],
        paths["profile"],
        output_json,
        output_markdown,
        release_revision=RELEASE_REVISION,
        hub_revision=HUB_REVISION,
        workflow_commit=WORKFLOW_COMMIT,
        waivers=[_viewer_waiver()],
        known_limitations=[
            {
                "id": "documented",
                "description": "A documented limitation.",
                "evidence_url": "https://example.org/issue",
            }
        ],
    )

    record = json.loads(output_json.read_text())
    assert record["valid"] is True
    assert record["ready_to_tag"] is False
    assert record["source"]["parquet_file_count"] == 290
    assert record["source"]["bigwig_file_count"] == 40
    assert record["source"]["inventory"]["schema_counts"] == {
        "entropy": 145,
        "llr": 145,
    }
    assert record["public_release"]["dataset_card_example_count"] == 4
    assert record["execution"]["scf_evidence_sha256"] == sha256_file(
        paths["scf_evidence"]
    )
    assert record["execution"]["bigwig_evidence_sha256"] == sha256_file(
        paths["bigwig_evidence"]
    )
    assert record["execution"]["scf_smoke_jobs"][0]["step_max_rss_kib"] == 476
    assert "SCF job `1`" in output_markdown.read_text()
    assert "TAG APPROVAL PENDING" in output_markdown.read_text()


def test_release_record_rejects_schema_drift(tmp_path: Path) -> None:
    metadata = _write_release_metadata(tmp_path)
    inventory_path = metadata / "manifest" / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["shards"][0]["schema"][-1]["type"] = "double"
    inventory_path.write_text(json.dumps(inventory))
    paths = _write_qa_evidence(tmp_path, metadata)

    with pytest.raises(ValueError, match="schema or content"):
        build_release_record(
            metadata,
            paths["hub_metadata"],
            paths["public_release"],
            paths["public_hub"],
            paths["scf_evidence"],
            paths["bigwig_evidence"],
            paths["hub_evidence"],
            paths["lock"],
            paths["profile"],
            tmp_path / "record.json",
            tmp_path / "record.md",
            release_revision=RELEASE_REVISION,
            hub_revision=HUB_REVISION,
            workflow_commit=WORKFLOW_COMMIT,
            waivers=[_viewer_waiver()],
            known_limitations=[],
        )


def test_release_record_rejects_incomplete_slurm_measurements(
    tmp_path: Path,
) -> None:
    metadata = _write_release_metadata(tmp_path)
    paths = _write_qa_evidence(tmp_path, metadata)
    scf = json.loads(paths["scf_evidence"].read_text())
    del scf["jobs"][0]["step_max_rss_kib"]
    paths["scf_evidence"].write_text(json.dumps(scf))

    with pytest.raises(ValueError, match="SCF and Slurm efficiency"):
        build_release_record(
            metadata,
            paths["hub_metadata"],
            paths["public_release"],
            paths["public_hub"],
            paths["scf_evidence"],
            paths["bigwig_evidence"],
            paths["hub_evidence"],
            paths["lock"],
            paths["profile"],
            tmp_path / "record.json",
            tmp_path / "record.md",
            release_revision=RELEASE_REVISION,
            hub_revision=HUB_REVISION,
            workflow_commit=WORKFLOW_COMMIT,
            waivers=[_viewer_waiver()],
            known_limitations=[],
        )


def test_release_record_rejects_unbounded_public_dataset_card(
    tmp_path: Path,
) -> None:
    metadata = _write_release_metadata(tmp_path)
    paths = _write_qa_evidence(tmp_path, metadata)
    (paths["hub_metadata"] / "README.md").write_text(
        'llr = pl.scan_parquet(f"{root}/model/llr/*.parquet")\n'
    )

    with pytest.raises(ValueError, match="unbounded genome-wide join"):
        build_release_record(
            metadata,
            paths["hub_metadata"],
            paths["public_release"],
            paths["public_hub"],
            paths["scf_evidence"],
            paths["bigwig_evidence"],
            paths["hub_evidence"],
            paths["lock"],
            paths["profile"],
            tmp_path / "record.json",
            tmp_path / "record.md",
            release_revision=RELEASE_REVISION,
            hub_revision=HUB_REVISION,
            workflow_commit=WORKFLOW_COMMIT,
            waivers=[_viewer_waiver()],
            known_limitations=[],
        )


def test_release_tag_requires_approval_and_exact_clean_commit(tmp_path: Path) -> None:
    record_path = tmp_path / "release-record.json"
    record_path.write_text(
        json.dumps(
            {
                "release_record_version": 1,
                "valid": True,
                "ready_to_tag": True,
                "tag": RELEASE_TAG,
                "repository": REPOSITORY_ID,
                "workflow_commit": WORKFLOW_COMMIT,
                "tag_approval": {
                    "approved": True,
                    "tag": RELEASE_TAG,
                    "evidence_url": QA_APPROVAL_ISSUE,
                    "approved_by": "author",
                    "approved_at": "2026-07-22",
                    "workflow_commit": WORKFLOW_COMMIT,
                    "release_revision": RELEASE_REVISION,
                    "hub_revision": HUB_REVISION,
                },
                "release_revision": RELEASE_REVISION,
                "hub_revision": HUB_REVISION,
                "source": {
                    "inventory": {"manifest_sha256": "d" * 64},
                    "release_manifest_sha256": "e" * 64,
                },
            }
        )
    )
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output = {
            ("git", "rev-parse", "HEAD"): f"{WORKFLOW_COMMIT}\n",
            ("git", "status", "--porcelain"): "",
            ("git", "tag", "--list", RELEASE_TAG): "",
        }.get(tuple(command), "")
        return subprocess.CompletedProcess(command, 0, output, "")

    create_release_tag(record_path, tmp_path, runner=runner)

    assert commands[-1][:4] == ["git", "tag", "--annotate", RELEASE_TAG]
    assert commands[-1][-1] == WORKFLOW_COMMIT


@pytest.mark.parametrize("reuse_public_hub_report", [False, True])
def test_enabled_qa_workflow_dry_runs_all_stages(
    tmp_path: Path, reuse_public_hub_report: bool
) -> None:
    metadata = tmp_path / "release-metadata"
    hub_metadata = tmp_path / "hub-metadata"
    metadata.mkdir()
    hub_metadata.mkdir()
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    config_path = tmp_path / "qa.yaml"
    config_path.write_text(
        f"""\
qa:
  enabled: true
  release_metadata_root: {metadata}
  hub_metadata_root: {hub_metadata}
  output_root: {tmp_path / "qa"}
  udc_cache_root: {tmp_path / "udc"}
  release_revision: {RELEASE_REVISION}
  hub_revision: {HUB_REVISION}
  public_hub_report: {evidence if reuse_public_hub_report else "null"}
  workflow_commit: {WORKFLOW_COMMIT}
  scf_evidence: {evidence}
  bigwig_evidence: {evidence}
  hub_evidence: {evidence}
  waivers: []
  known_limitations: []
  tag_approval: null
  resources:
    release: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    hub: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    record: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
"""
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            "workflow/Snakefile",
            "--configfile",
            str(config_path),
            "--cores",
            "1",
            "--dry-run",
            "qa",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "qa_public_release" in result.stdout
    assert ("qa_public_hub" in result.stdout) is not reuse_public_hub_report
    assert "qa_release_record" in result.stdout
