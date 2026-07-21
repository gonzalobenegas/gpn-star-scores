from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from gpn_star_scores.scf_smoke import (
    EXPECTED_PACKAGE_VERSIONS,
    _detect_slurm_partition,
    atomic_write_json,
    write_smoke_report,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
PROFILE_PATH = REPOSITORY_ROOT / "workflow/profiles/scf/config.yaml"


def test_runtime_dependencies_are_exactly_pinned() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    assert {
        dependency
        for dependency in dependencies
        if dependency.split("==", maxsplit=1)[0] in EXPECTED_PACKAGE_VERSIONS
    } == {
        f"{package}=={package_version}"
        for package, package_version in EXPECTED_PACKAGE_VERSIONS.items()
        if package != "polars-runtime-32"
    }


def test_scf_profile_encodes_partition_and_execution_policy() -> None:
    with PROFILE_PATH.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)

    assert profile["executor"] == "slurm"
    assert profile["jobs"] == 64
    assert profile["rerun-incomplete"] is True
    assert profile["retries"] == 2
    assert profile["slurm-array-limit"] == 64
    assert profile["latency-wait"] == 60
    assert profile["slurm-logdir"] == "logs/slurm"
    assert profile["slurm-keep-successful-logs"] is True
    assert profile["slurm-efficiency-report"] is True
    assert profile["slurm-efficiency-report-path"] == "logs/slurm-efficiency"

    array_rules = {rule.strip() for rule in profile["slurm-array-jobs"].split(",")}
    assert {
        "validate_source_shard",
        "rewrite_parquet_shard",
        "build_chromosome_bigwig",
    } == array_rules

    four_thread_rules = {
        "scf_smoke_chromosome",
        "inventory_source_shard",
        "validate_source_shard",
        "rewrite_parquet_shard",
        "validate_parquet_shard",
        "benchmark_parquet_shard",
    }
    one_thread_rules = {
        "build_chromosome_bigwig",
        "concatenate_bigwig",
        "aggregate_validation",
        "render_report",
    }
    assert all(profile["set-threads"][rule] == 4 for rule in four_thread_rules)
    assert all(profile["set-threads"][rule] == 1 for rule in one_thread_rules)

    partitions = {
        resources["slurm_partition"] for resources in profile["set-resources"].values()
    }
    assert partitions == {"epurdom", "high"}
    assert not any(
        resource.lower().startswith("gpu")
        for resources in profile["set-resources"].values()
        for resource in resources
    )


def test_portable_resource_policy_starts_heavy_rules_at_four_hours() -> None:
    with (REPOSITORY_ROOT / "workflow/config/config.yaml").open(
        encoding="utf-8"
    ) as handle:
        workflow_config = yaml.safe_load(handle)
    policy = workflow_config["resource_policy"]

    assert policy["heavy_initial"] == {
        "threads": 4,
        "mem_mb": 4096,
        "runtime": 240,
        "disk_mb": 1024,
    }
    assert policy["pilot_tuning"] == {
        "runtime_multiplier": 2.0,
        "runtime_min": 30,
        "runtime_max": 360,
        "memory_multiplier": 1.5,
        "memory_min_mb": 4096,
    }

    assert workflow_config["scf_smoke"]["chromosomes"] == ["chr1", "chr22"]


def test_scf_smoke_report_is_validated_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_PARTITION", "epurdom")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    output_path = tmp_path / "smoke" / "chr1.json"

    write_smoke_report("chr1", output_path)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["chrom"] == "chr1"
    assert report["job_id"] == "12345"
    assert report["packages"] == EXPECTED_PACKAGE_VERSIONS
    assert report["polars_runtime"] == "polars-runtime-32"
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_scf_smoke_rejects_the_wrong_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_PARTITION", "high")

    with pytest.raises(RuntimeError, match="requires partition 'epurdom'"):
        write_smoke_report("chr1", tmp_path / "chr1.json")


def test_scf_partition_falls_back_to_controller_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_PARTITION", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "3341913")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[0] == ["scontrol", "show", "job", "3341913"]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="JobId=3341913 Partition=epurdom\n"
        )

    monkeypatch.setattr("gpn_star_scores.scf_smoke.subprocess.run", fake_run)

    assert _detect_slurm_partition() == "epurdom"


def test_failed_validation_preserves_previous_output(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    output_path.write_text('{"generation": 1}\n', encoding="utf-8")

    def reject(_: object) -> None:
        raise ValueError("incomplete")

    with pytest.raises(ValueError, match="incomplete"):
        atomic_write_json(output_path, {"generation": 2}, reject)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"generation": 1}
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))

    atomic_write_json(output_path, {"generation": 2}, lambda _: None)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"generation": 2}
