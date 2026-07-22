from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest
import yaml

import gpn_star_scores.release as release_module
from gpn_star_scores.catalog import SCORE_SETS, expected_shards
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.release import (
    CAPACITY_APPROVAL_ISSUE,
    CAPACITY_BLOCKER,
    PUBLIC_STORAGE_POLICY,
    REPOSITORY_ID,
    build_release_metadata,
    dataset_configs,
    publish_release,
    validate_existing_release,
    validate_public_release,
)
from gpn_star_scores.tracks import TRACKS, ucsc_assembly_name

REPOSITORY_ROOT = Path(__file__).parents[1]


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_release_inputs(tmp_path: Path) -> dict[str, Path]:
    source_root = tmp_path / "source"
    inventory_records = []
    parquet_content = b"p"
    for shard in expected_shards():
        path = source_root / shard.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(parquet_content)
        inventory_records.append(
            {
                "path": shard.relative_path.as_posix(),
                "score_set": shard.score_set,
                "assembly": shard.assembly,
                "score_type": shard.score_type,
                "chrom": shard.chrom,
                "size": len(parquet_content),
                "sha256": _digest(parquet_content),
                "valid": True,
                "parquet": {"num_rows": 3},
                "content": {"coordinate_bounds": {"min": 1, "max": 3}},
            }
        )
    inventory = {
        "manifest_version": 1,
        "source": {"total_shard_bytes": len(inventory_records)},
        "validation": {"release_ready": True, "blockers": []},
        "shards": inventory_records,
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "selected",
                "selected_candidate": "source",
                "source_inventory": {
                    "valid": True,
                    "manifest_sha256": sha256_file(inventory_path),
                },
            }
        ),
        encoding="utf-8",
    )

    bigwig_root = tmp_path / "bigwig"
    track_records = []
    for score_set in SCORE_SETS:
        for track in TRACKS:
            path = bigwig_root / "final" / score_set.name / f"{track}.bw"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{score_set.name}-{track}".encode())
            track_records.append(
                {
                    "score_set": score_set.name,
                    "assembly": score_set.assembly,
                    "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
                    "track": track,
                    "bases_covered": 3,
                    "zoom_levels": 1,
                }
            )
    bigwig_validation_path = tmp_path / "bigwig-validation.json"
    bigwig_validation_path.write_text(
        json.dumps(
            {
                "report_version": 1,
                "valid": True,
                "track_count": len(track_records),
                "value_decimals": 3,
                "inventory_manifest_sha256": sha256_file(inventory_path),
                "tracks": track_records,
            }
        ),
        encoding="utf-8",
    )
    return {
        "source_root": source_root,
        "bigwig_root": bigwig_root,
        "inventory": inventory_path,
        "selection": selection_path,
        "bigwig_validation": bigwig_validation_path,
    }


def _build_metadata(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    inputs = _write_release_inputs(tmp_path)
    output = tmp_path / "metadata"
    build_release_metadata(
        inputs["source_root"],
        inputs["bigwig_root"],
        inputs["inventory"],
        inputs["selection"],
        inputs["bigwig_validation"],
        output,
    )
    return inputs, output


def test_release_metadata_has_exact_configs_models_and_checksums(
    tmp_path: Path,
) -> None:
    inputs, output = _build_metadata(tmp_path)

    manifest = json.loads((output / "manifest" / "release.json").read_text())
    assert manifest["repository"] == {
        "id": REPOSITORY_ID,
        "repo_type": "dataset",
        "public": True,
        "license": "apache-2.0",
    }
    assert manifest["parquet"]["file_count"] == 290
    assert manifest["bigwig"]["file_count"] == 40
    assert len(manifest["dataset_configs"]) == 16
    assert manifest["dataset_configs"] == dataset_configs()
    assert all(
        config["data_files"][0]["path"].startswith("data/")
        and config["data_files"][0]["path"].endswith("/*.parquet")
        for config in manifest["dataset_configs"]
    )
    first_bigwig = manifest["bigwig"]["files"][0]
    local_bigwig = (
        inputs["bigwig_root"]
        / "final"
        / first_bigwig["score_set"]
        / f"{first_bigwig['track']}.bw"
    )
    assert first_bigwig["sha256"] == sha256_file(local_bigwig)
    assert str(tmp_path) not in json.dumps(manifest)

    readme = (output / "README.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(readme.split("---", maxsplit=2)[1])
    assert len(metadata["configs"]) == 16
    assert metadata["size_categories"] == ["n<1K"]
    assert release_module._dataset_size_category(51_402_120_888) == "10B<n<100B"
    assert all("bigwig" not in str(config) for config in metadata["configs"])
    assert all("ucsc" not in str(config) for config in metadata["configs"])
    assert "songlab/gpn-star-ce11-n135-25m" in readme
    assert "abs_llr_calibrated` is an independently supplied" in readme
    assert "pl.scan_parquet" in readme
    assert "variants.join" in readme
    assert "10.1101/2025.09.21.677619" in readme

    representative = release_module._representative_parquet_records(manifest)
    assert [record["path"] for record in representative] == [
        "data/gg6/entropy/entropy_chr1.parquet",
        "data/gpn-star-hg38-v100-200m/llr/llr_chr22.parquet",
    ]


def test_release_preflight_rejects_blocked_inventory(tmp_path: Path) -> None:
    inputs = _write_release_inputs(tmp_path)
    inventory = json.loads(inputs["inventory"].read_text())
    inventory["validation"] = {
        "release_ready": False,
        "blockers": ["capacity not confirmed"],
    }
    inputs["inventory"].write_text(json.dumps(inventory))
    output = tmp_path / "metadata"

    with pytest.raises(ValueError, match="not release-ready"):
        build_release_metadata(
            inputs["source_root"],
            inputs["bigwig_root"],
            inputs["inventory"],
            inputs["selection"],
            inputs["bigwig_validation"],
            output,
        )
    assert not output.exists()


def test_release_preflight_rejects_output_inside_source_root(tmp_path: Path) -> None:
    inputs = _write_release_inputs(tmp_path)
    output = inputs["source_root"] / "release"

    with pytest.raises(ValueError, match="must not overlap"):
        build_release_metadata(
            inputs["source_root"],
            inputs["bigwig_root"],
            inputs["inventory"],
            inputs["selection"],
            inputs["bigwig_validation"],
            output,
        )
    assert not output.exists()


def test_release_preflight_records_narrow_public_capacity_waiver(
    tmp_path: Path,
) -> None:
    inputs = _write_release_inputs(tmp_path)
    inventory = json.loads(inputs["inventory"].read_text())
    inventory["validation"] = {
        "release_ready": False,
        "blockers": [CAPACITY_BLOCKER],
    }
    inputs["inventory"].write_text(json.dumps(inventory))
    inventory_sha256 = sha256_file(inputs["inventory"])
    selection = json.loads(inputs["selection"].read_text())
    selection["source_inventory"]["manifest_sha256"] = inventory_sha256
    inputs["selection"].write_text(json.dumps(selection))
    bigwig_validation = json.loads(inputs["bigwig_validation"].read_text())
    bigwig_validation["inventory_manifest_sha256"] = inventory_sha256
    inputs["bigwig_validation"].write_text(json.dumps(bigwig_validation))
    planned_bytes = sum(
        path.stat().st_size
        for path in inputs["source_root"].rglob("*")
        if path.is_file()
    )
    planned_bytes += sum(
        path.stat().st_size for path in inputs["bigwig_root"].rglob("*.bw")
    )
    approval = {
        "approved": True,
        "public_repository": True,
        "evidence_url": CAPACITY_APPROVAL_ISSUE,
        "public_storage_policy_url": PUBLIC_STORAGE_POLICY,
        "approved_by": "gonzalobenegas",
        "approved_at": "2026-07-22",
        "planned_release_bytes": planned_bytes,
        "reserved_headroom_bytes": planned_bytes // 10,
    }
    output = tmp_path / "metadata"

    build_release_metadata(
        inputs["source_root"],
        inputs["bigwig_root"],
        inputs["inventory"],
        inputs["selection"],
        inputs["bigwig_validation"],
        output,
        capacity_approval=approval,
    )

    manifest = json.loads((output / "manifest" / "release.json").read_text())
    assert manifest["source_inventory"]["release_ready"] is False
    assert manifest["source_inventory"]["capacity_waiver"] == approval
    assert manifest["validation"]["inventory_data_validation_passed"] is True
    assert manifest["validation"]["capacity_waiver_applied"] is True


class _FakeApi:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.uploads: list[dict[str, object]] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        self.created.append({"repo_id": repo_id, **kwargs})

    def repo_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(private=False, sha="a" * 40)

    def upload_folder(self, **kwargs: object) -> None:
        self.uploads.append(kwargs)


class _FakeResponse:
    def __init__(
        self, status: int, content: bytes, headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self.content = content
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.content


def _fake_parquet_query_checks() -> list[dict[str, object]]:
    return [
        {
            "range_reader_engine": "pyarrow",
            "range_reader_transferred_bytes": 128,
            "range_reader_rows": 1_000,
            "polars_engine": "polars",
            "polars_rows": 1,
            "polars_projection_pushdown": True,
            "polars_predicate_pushdown": True,
            "polars_transfer_bytes_measured": False,
            "polars_optimized_plan": "PROJECT 2/4 COLUMNS\nSELECTION: pos",
        }
        for _ in range(2)
    ]


def test_publication_uploads_data_then_tracks_then_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, metadata = _build_metadata(tmp_path)
    api = _FakeApi()
    validation_calls: list[dict[str, object]] = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        validation_calls.append({"args": args, "kwargs": kwargs})
        return {"valid": True, "credentials_sent": False}

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    report_path = tmp_path / "publication.json"
    publish_release(
        inputs["source_root"],
        inputs["bigwig_root"],
        metadata,
        report_path,
        api=api,
        validator=validator,
        viewer_attempts=1,
        viewer_retry_seconds=0,
    )

    assert api.created == [
        {
            "repo_id": REPOSITORY_ID,
            "repo_type": "dataset",
            "private": False,
            "exist_ok": True,
        }
    ]
    assert len(api.uploads) == 25
    assert all(
        str(upload["path_in_repo"]).startswith("data/") for upload in api.uploads[:16]
    )
    assert all(
        str(upload["path_in_repo"]).startswith("bigwig/")
        for upload in api.uploads[16:24]
    )
    assert api.uploads[-1]["allow_patterns"] == ["README.md", "manifest/**"]
    assert len(validation_calls) == 1
    assert validation_calls[0]["kwargs"]["revision"] == "a" * 40
    report = json.loads(report_path.read_text())
    assert report["valid"] is True
    assert report["final_revision"] == "a" * 40
    assert report["public_validation"]["credentials_sent"] is False


def test_publication_rejects_slurm_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, metadata = _build_metadata(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    with pytest.raises(RuntimeError, match="non-Slurm"):
        publish_release(
            inputs["source_root"],
            inputs["bigwig_root"],
            metadata,
            tmp_path / "publication.json",
            api=_FakeApi(),
        )


def test_enabled_workflow_separates_preflight_from_publication(tmp_path: Path) -> None:
    inputs = _write_release_inputs(tmp_path)
    config_path = tmp_path / "release.yaml"
    config_path.write_text(
        f"""\
release:
  enabled: true
  source_root: {inputs["source_root"]}
  bigwig_root: {inputs["bigwig_root"]}
  inventory_manifest: {inputs["inventory"]}
  parquet_selection: {inputs["selection"]}
  bigwig_validation: {inputs["bigwig_validation"]}
  output_root: {tmp_path / "release"}
  viewer_attempts: 1
  viewer_retry_seconds: 0
  hf_block_size: 1024
  resources:
    preflight: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    publish: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
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
            "publish",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "release_preflight" in result.stdout
    assert "publish" in result.stdout


def test_enabled_workflow_rejects_output_inside_source_root(tmp_path: Path) -> None:
    inputs = _write_release_inputs(tmp_path)
    config_path = tmp_path / "release-overlap.yaml"
    config_path.write_text(
        f"""\
release:
  enabled: true
  source_root: {inputs["source_root"]}
  bigwig_root: {inputs["bigwig_root"]}
  inventory_manifest: {inputs["inventory"]}
  parquet_selection: {inputs["selection"]}
  bigwig_validation: {inputs["bigwig_validation"]}
  output_root: {inputs["source_root"] / "release"}
  resources:
    preflight: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    publish: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
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
            "publish",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must not overlap" in result.stdout + result.stderr


def test_public_validation_checks_all_public_interfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, metadata = _build_metadata(tmp_path)
    manifest = json.loads((metadata / "manifest" / "release.json").read_text())
    siblings = [
        SimpleNamespace(
            rfilename=record["path"],
            size=record["size"],
            lfs={"sha256": record["sha256"]},
        )
        for record in [*manifest["parquet"]["files"], *manifest["bigwig"]["files"]]
    ]

    class PublicApi:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            assert kwargs["token"] is False
            assert kwargs["files_metadata"] is True
            return SimpleNamespace(private=False, sha="b" * 40, siblings=siblings)

    expected_columns = {
        config["config_name"]: (
            ["chrom", "pos", "ref", "entropy_calibrated"]
            if config["score_type"] == "entropy"
            else [
                "chrom",
                "pos",
                "ref",
                "alt",
                "llr_calibrated",
                "abs_llr_calibrated",
            ]
        )
        for config in manifest["dataset_configs"]
    }

    def opener(request: str | Request, **kwargs: object) -> _FakeResponse:
        url = request.full_url if isinstance(request, Request) else request
        if "datasets-server" in url:
            config = parse_qs(urlparse(url).query)["config"][0]
            return _FakeResponse(
                200,
                json.dumps(
                    {
                        "features": [
                            {"name": column} for column in expected_columns[config]
                        ],
                        "rows": [{"row": {}}],
                    }
                ).encode(),
            )
        if url.endswith("/README.md"):
            return _FakeResponse(200, (metadata / "README.md").read_bytes())
        if isinstance(request, Request) and request.get_header("Range"):
            record = next(
                item for item in manifest["bigwig"]["files"] if item["path"] in url
            )
            end = min(63, record["size"] - 1)
            return _FakeResponse(
                206,
                b"x" * (end + 1),
                {"Content-Range": f"bytes 0-{end}/{record['size']}"},
            )
        return _FakeResponse(
            200,
            b"<html>songlab/gpn-star-scores GPN-Star genome-wide scores</html>",
        )

    monkeypatch.setattr(
        release_module,
        "_validate_hf_range_queries",
        lambda *args, **kwargs: _fake_parquet_query_checks(),
    )
    report = validate_public_release(
        metadata,
        repository_id=REPOSITORY_ID,
        revision="b" * 40,
        api=PublicApi(),
        opener=opener,
        viewer_attempts=1,
        viewer_retry_seconds=0,
    )

    assert report["valid"] is True
    assert report["credentials_sent"] is False
    assert report["checksum_file_count"] == 330
    assert report["published_artifact_file_count"] == 330
    assert report["viewer_required"] is False
    assert report["viewer_ready"] is True
    assert report["viewer_config_count"] == 16
    assert report["viewer_pending"] == []
    assert report["bigwig_range_count"] == 40
    assert len(report["parquet_query_checks"]) == 2
    assert all(
        check["range_reader_engine"] == "pyarrow"
        and check["polars_engine"] == "polars"
        and check["polars_transfer_bytes_measured"] is False
        for check in report["parquet_query_checks"]
    )


def test_public_validation_reports_pending_viewer_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, metadata = _build_metadata(tmp_path)
    manifest = json.loads((metadata / "manifest" / "release.json").read_text())
    siblings = [
        SimpleNamespace(
            rfilename=record["path"],
            size=record["size"],
            lfs={"sha256": record["sha256"]},
        )
        for record in [*manifest["parquet"]["files"], *manifest["bigwig"]["files"]]
    ]

    class PublicApi:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="c" * 40, siblings=siblings)

    def opener(request: str | Request, **kwargs: object) -> _FakeResponse:
        url = request.full_url if isinstance(request, Request) else request
        if "datasets-server" in url:
            return _FakeResponse(500, b'{"error":"not ready"}')
        if url.endswith("/README.md"):
            return _FakeResponse(200, (metadata / "README.md").read_bytes())
        if isinstance(request, Request) and request.get_header("Range"):
            record = next(
                item for item in manifest["bigwig"]["files"] if item["path"] in url
            )
            end = min(63, record["size"] - 1)
            return _FakeResponse(
                206,
                b"x" * (end + 1),
                {"Content-Range": f"bytes 0-{end}/{record['size']}"},
            )
        return _FakeResponse(
            200,
            b"<html>songlab/gpn-star-scores GPN-Star genome-wide scores</html>",
        )

    monkeypatch.setattr(
        release_module,
        "_validate_hf_range_queries",
        lambda *args, **kwargs: _fake_parquet_query_checks(),
    )
    report = validate_public_release(
        metadata,
        repository_id=REPOSITORY_ID,
        revision="c" * 40,
        api=PublicApi(),
        opener=opener,
    )

    assert report["valid"] is True
    assert report["checksum_file_count"] == 330
    assert report["bigwig_range_count"] == 40
    assert len(report["parquet_query_checks"]) == 2
    assert report["viewer_required"] is False
    assert report["viewer_ready"] is False
    assert report["viewer_config_count"] == 0
    assert len(report["viewer_pending"]) == 16


def test_public_validation_rejects_unexpected_published_artifact(
    tmp_path: Path,
) -> None:
    _, metadata = _build_metadata(tmp_path)
    manifest = json.loads((metadata / "manifest" / "release.json").read_text())
    siblings = [
        SimpleNamespace(
            rfilename=record["path"],
            size=record["size"],
            lfs={"sha256": record["sha256"]},
        )
        for record in [*manifest["parquet"]["files"], *manifest["bigwig"]["files"]]
    ]
    siblings.append(
        SimpleNamespace(
            rfilename="data/stale/entropy/entropy_chr1.parquet",
            size=1,
            lfs={"sha256": "0" * 64},
        )
    )

    class PublicApi:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="e" * 40, siblings=siblings)

    with pytest.raises(RuntimeError, match="unexpected data artifacts"):
        validate_public_release(
            metadata,
            repository_id=REPOSITORY_ID,
            revision="e" * 40,
            api=PublicApi(),
        )


def test_public_validation_rejects_unrelated_rendered_page(tmp_path: Path) -> None:
    _, metadata = _build_metadata(tmp_path)
    manifest = json.loads((metadata / "manifest" / "release.json").read_text())
    siblings = [
        SimpleNamespace(
            rfilename=record["path"],
            size=record["size"],
            lfs={"sha256": record["sha256"]},
        )
        for record in [*manifest["parquet"]["files"], *manifest["bigwig"]["files"]]
    ]

    class PublicApi:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="f" * 40, siblings=siblings)

    def opener(request: str | Request, **kwargs: object) -> _FakeResponse:
        url = request.full_url if isinstance(request, Request) else request
        if url.endswith("/README.md"):
            return _FakeResponse(200, (metadata / "README.md").read_bytes())
        return _FakeResponse(200, b"<html>unrelated page</html>")

    with pytest.raises(RuntimeError, match="page did not render"):
        validate_public_release(
            metadata,
            repository_id=REPOSITORY_ID,
            revision="f" * 40,
            api=PublicApi(),
            opener=opener,
        )


def test_public_validation_rejects_dataset_card_source_drift(tmp_path: Path) -> None:
    _, metadata = _build_metadata(tmp_path)
    manifest = json.loads((metadata / "manifest" / "release.json").read_text())
    siblings = [
        SimpleNamespace(
            rfilename=record["path"],
            size=record["size"],
            lfs={"sha256": record["sha256"]},
        )
        for record in [*manifest["parquet"]["files"], *manifest["bigwig"]["files"]]
    ]

    class PublicApi:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="1" * 40, siblings=siblings)

    def opener(request: str | Request, **kwargs: object) -> _FakeResponse:
        url = request.full_url if isinstance(request, Request) else request
        if url.endswith("/README.md"):
            return _FakeResponse(200, b"# GPN-Star genome-wide scores\nchanged")
        raise AssertionError("rendered page must not be requested after source drift")

    with pytest.raises(RuntimeError, match="card source is unavailable"):
        validate_public_release(
            metadata,
            repository_id=REPOSITORY_ID,
            revision="1" * 40,
            api=PublicApi(),
            opener=opener,
        )


def test_existing_revision_validation_writes_report(tmp_path: Path) -> None:
    _, metadata = _build_metadata(tmp_path)
    calls: list[dict[str, object]] = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append({"args": args, "kwargs": kwargs})
        return {"valid": True, "viewer_ready": False}

    report_path = tmp_path / "publication.json"
    validate_existing_release(
        metadata,
        report_path,
        revision="d" * 40,
        validator=validator,
    )

    assert calls[0]["kwargs"]["revision"] == "d" * 40
    assert calls[0]["kwargs"]["viewer_required"] is False
    report = json.loads(report_path.read_text())
    assert report["valid"] is True
    assert report["final_revision"] == "d" * 40
    assert report["validation_mode"] == "existing_revision"
    assert report["public_validation"]["viewer_ready"] is False
