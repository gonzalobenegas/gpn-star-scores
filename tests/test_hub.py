from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from gpn_star_scores.catalog import ASSEMBLIES, SCORE_SETS
from gpn_star_scores.hub import (
    HUB_APPROVAL_ISSUE,
    HUB_ASSEMBLY_ORDER,
    HUB_TRACK_DB_DIRECTORY_ORDER,
    NUCLEOTIDE_COLORS,
    browser_launch_links,
    build_track_hub,
    hub_database_name,
    publish_track_hub,
    validate_existing_track_hub_publication,
    validate_public_track_hub,
    validate_track_hub,
)
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.release import REPOSITORY_ID, TRACK_HUB_URL
from gpn_star_scores.tracks import TRACKS, ucsc_assembly_name

REPOSITORY_ROOT = Path(__file__).parents[1]
ARTIFACT_REVISION = "a" * 40


def _release_manifest() -> dict[str, object]:
    files = []
    for score_set in SCORE_SETS:
        for track in TRACKS:
            files.append(
                {
                    "path": f"bigwig/{score_set.name}/{track}.bw",
                    "score_set": score_set.name,
                    "assembly": score_set.assembly,
                    "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
                    "track": track,
                    "size": 100,
                    "sha256": "1" * 64,
                    "bases_covered": 10,
                    "zoom_levels": 2,
                }
            )
    return {
        "release_manifest_version": 1,
        "repository": {
            "id": REPOSITORY_ID,
            "repo_type": "dataset",
            "public": True,
            "license": "apache-2.0",
        },
        "paper": {"title": "paper", "doi": "doi"},
        "source_inventory": {},
        "parquet": {"files": []},
        "bigwig": {
            "file_count": len(files),
            "total_bytes": 100 * len(files),
            "value_decimals": 3,
            "files": files,
        },
        "dataset_configs": [],
        "validation": {
            "bigwig_validation_passed": True,
            "expected_bigwig_files": len(files),
        },
    }


def _build_metadata(tmp_path: Path) -> Path:
    release_manifest = tmp_path / "release.json"
    release_manifest.write_text(json.dumps(_release_manifest()), encoding="utf-8")
    metadata = tmp_path / "metadata"
    build_track_hub(
        release_manifest,
        metadata,
        artifact_revision=ARTIFACT_REVISION,
        contact_email="maintainer@example.org",
    )
    return metadata


def _write_validation_report(metadata: Path, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "valid": True,
                "artifact_revision": ARTIFACT_REVISION,
                "hub_manifest_sha256": sha256_file(
                    metadata / "manifest" / "ucsc-hub.json"
                ),
            }
        )
    )


def test_builds_one_multi_assembly_hub_with_entropy_and_logo(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)

    hub = (metadata / "ucsc" / "hub.txt").read_text()
    genomes = (metadata / "ucsc" / "genomes.txt").read_text()
    assert "genomesFile genomes.txt" in hub
    assert "useOneFile" not in hub
    assert ("genome GCF_000001735.4\ntrackDb araTha1/trackDb.txt") in genomes
    assert [
        line.removeprefix("genome ")
        for line in genomes.splitlines()
        if line.startswith("genome ")
    ] == list(HUB_ASSEMBLY_ORDER)

    hg38 = (metadata / "ucsc" / "hg38" / "trackDb.txt").read_text()
    assert hg38.count("superTrack on show") == 3
    assert hg38.count("container multiWig") == 3
    assert hg38.count("graphTypeDefault bar") == 3
    assert hg38.count("logo on") == 3
    assert hg38.count("bigDataUrl ") == 15
    assert "noInherit" not in hg38
    for color in NUCLEOTIDE_COLORS.values():
        assert hg38.count(f"color {color}") == 3

    for directory in HUB_TRACK_DB_DIRECTORY_ORDER[1:]:
        track_db = (metadata / "ucsc" / directory / "trackDb.txt").read_text()
        assert track_db.count("superTrack on show") == 1
        assert track_db.count("container multiWig") == 1
        assert track_db.count("graphTypeDefault bar") == 1
        assert track_db.count("logo on") == 1
        assert track_db.count("bigDataUrl ") == 5

    assert f"/resolve/{ARTIFACT_REVISION}/bigwig/" in hg38
    assert "llr_A" not in hg38
    readme = (metadata / "README.md").read_text()
    assert TRACK_HUB_URL in readme
    assert "one-dimensional\nentropy signal" in readme
    assert readme.count("Open in UCSC") == 8

    links = browser_launch_links()
    assert len(links) == 8
    for link in links:
        query = parse_qs(urlparse(link["url"]).query)
        assert query["db"] == [link["ucsc_assembly"]]
        assert query["hubUrl"] == [TRACK_HUB_URL]
        assert query["hideTracks"] == ["1"]
        assert query["ignoreCookie"] == ["1"]

    hub_manifest = json.loads((metadata / "manifest" / "ucsc-hub.json").read_text())
    assert hub_manifest["assembly_count"] == 6
    assert hub_manifest["score_set_count"] == 8
    assert hub_manifest["track_count"] == 40
    assert len(hub_manifest["files"]) == 33
    assert all(score_set["browser_url"] for score_set in hub_manifest["score_sets"])
    tair10 = next(
        score_set
        for score_set in hub_manifest["score_sets"]
        if score_set["name"] == "tair10"
    )
    assert tair10["ucsc_assembly"] == "GCF_000001735.4"
    assert "db=GCF_000001735.4" in tair10["browser_url"]
    tair10_tracks = [
        track for track in hub_manifest["tracks"] if track["score_set"] == "tair10"
    ]
    assert {track["source_ucsc_assembly"] for track in tair10_tracks} == {"araTha1"}
    assert {track["ucsc_assembly"] for track in tair10_tracks} == {"GCF_000001735.4"}
    assert {track["url"] for track in hub_manifest["tracks"]} == {
        f"https://huggingface.co/datasets/{REPOSITORY_ID}/resolve/"
        f"{ARTIFACT_REVISION}/bigwig/{score_set.name}/{track}.bw"
        for score_set in SCORE_SETS
        for track in TRACKS
    }


def test_track_descriptions_preserve_score_and_coordinate_semantics(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    track_db = (metadata / "ucsc" / "araTha1" / "trackDb.txt").read_text()
    html_files = {
        path.name: path.read_text()
        for path in (metadata / "ucsc" / "araTha1").glob("*.html")
    }

    assert "genome GCF_000001735.4" in (metadata / "ucsc" / "genomes.txt").read_text()
    assert hub_database_name("tair10") == "GCF_000001735.4"
    assert "GPN-Star TAIR10" in track_db
    assert len(html_files) == 3
    entropy = next(value for name, value in html_files.items() if "Entropy" in name)
    logo = next(value for name, value in html_files.items() if "Logo" in name)
    assert "entropy_calibrated" in entropy
    assert "does not add an unreviewed biological directionality" in entropy
    assert "zero-based,\nhalf-open one-base BigWig intervals" in entropy
    assert "p(base) * (2 - H)" in logo
    assert "abs_llr_calibrated" in logo
    assert "not a raw model\nprobability" in logo


class _FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status = status
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.content


def _score_set_from_url(url: str):
    return next(score_set for score_set in SCORE_SETS if f"/{score_set.name}/" in url)


def _fake_runner(
    command: list[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    executable = Path(command[0]).name
    if executable == "hubCheck":
        return subprocess.CompletedProcess(command, 0, "hub is valid\n", "")
    if executable == "bigWigInfo":
        score_set = _score_set_from_url(command[-1])
        output = "\n".join(
            f"chr{chrom} {index} 1000"
            for index, chrom in enumerate(ASSEMBLIES[score_set.assembly].chromosomes)
        )
        return subprocess.CompletedProcess(command, 0, output + "\n", "")
    if executable == "bigWigSummary":
        bins = int(command[-1])
        return subprocess.CompletedProcess(
            command, 0, "\t".join("0.5" for _ in range(bins)) + "\n", ""
        )
    raise AssertionError(command)


def test_validation_checks_hub_ranges_chromosomes_and_zoom_values(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(request: object, **kwargs: object) -> _FakeResponse:
        assert getattr(request, "get_header")("Range") == "bytes=0-63"
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    report_path = tmp_path / "validation.json"
    markdown_path = tmp_path / "validation.md"
    validate_track_hub(
        metadata,
        report_path,
        markdown_path,
        udc_dir=tmp_path / "udc",
        runner=_fake_runner,
        opener=opener,
    )

    report = json.loads(report_path.read_text())
    assert report["valid"] is True
    assert report["hub_check"]["passed"] is True
    assert report["hub_check"]["settings_spec"].startswith("https://")
    assert report["hub_manifest_sha256"] == sha256_file(
        metadata / "manifest" / "ucsc-hub.json"
    )
    assert report["assembly_count"] == 6
    assert report["score_set_count"] == 8
    assert report["track_count"] == 40
    assert report["http_range_count"] == 40
    assert len(report["chromosome_checks"]) == 40
    assert len(report["representative_checks"]) == 8
    assert all(
        set(check["track_values"]) == set(TRACKS)
        and check["zero_based_half_open"] is True
        for check in report["representative_checks"]
    )
    assert "All 40 BigWig URLs" in markdown_path.read_text()


def test_validation_rejects_cross_track_chromosome_length_mismatch(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(request: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = _fake_runner(command, **kwargs)
        if Path(command[0]).name == "bigWigInfo" and command[-1].endswith("/A.bw"):
            return subprocess.CompletedProcess(
                command,
                0,
                result.stdout.replace(" 0 1000", " 0 999", 1),
                "",
            )
        return result

    with pytest.raises(RuntimeError, match="chromosome sizes differ"):
        validate_track_hub(
            metadata,
            tmp_path / "validation.json",
            tmp_path / "validation.md",
            udc_dir=tmp_path / "udc",
            runner=runner,
            opener=opener,
        )


class _FakeApi:
    def __init__(self) -> None:
        self.commit_kwargs: dict[str, object] | None = None

    def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(private=False, sha=ARTIFACT_REVISION)

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.commit_kwargs = kwargs
        return SimpleNamespace(oid="b" * 40)


def test_public_validation_checks_published_files_and_remote_hub(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(target: object, **kwargs: object) -> _FakeResponse:
        if isinstance(target, str):
            marker = f"/resolve/{ARTIFACT_REVISION}/"
            relative_path = target.split(marker, maxsplit=1)[1]
            return _FakeResponse((metadata / relative_path).read_bytes())
        assert getattr(target, "get_header")("Range") == "bytes=0-63"
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    report = validate_public_track_hub(
        metadata,
        revision=ARTIFACT_REVISION,
        udc_dir=tmp_path / "udc",
        api=_FakeApi(),
        opener=opener,
        runner=_fake_runner,
    )

    assert report["valid"] is True
    assert report["credentials_sent"] is False
    assert report["revision"] == ARTIFACT_REVISION
    assert report["file_count"] == 35
    assert report["hub_validation"]["hub_target"].endswith("/ucsc/hub.txt")


def test_publication_is_one_approval_gated_commit(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    api = _FakeApi()
    validator_calls = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        validator_calls.append((args, kwargs))
        return {"valid": True, "credentials_sent": False}

    report = tmp_path / "publication.json"
    publish_track_hub(
        metadata,
        validation,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        publication_approval={
            "approved": True,
            "evidence_url": HUB_APPROVAL_ISSUE,
            "approved_by": "author",
            "approved_at": "2026-07-22",
            "expected_base_revision": ARTIFACT_REVISION,
        },
        udc_dir=tmp_path / "udc",
        api=api,
        validator=validator,
    )

    assert api.commit_kwargs is not None
    assert api.commit_kwargs["parent_commit"] == ARTIFACT_REVISION
    operations = api.commit_kwargs["operations"]
    paths = {operation.path_in_repo for operation in operations}
    assert "README.md" in paths
    assert "manifest/ucsc-hub.json" in paths
    assert "ucsc/hub.txt" in paths
    assert "ucsc/genomes.txt" in paths
    assert len(validator_calls) == 1
    publication = json.loads(report.read_text())
    assert publication["valid"] is True
    assert publication["status"] == "validated"
    assert publication["base_revision"] == ARTIFACT_REVISION
    assert publication["final_revision"] == "b" * 40
    assert publication["single_commit"] is True


def test_publication_failure_preserves_created_revision_for_recovery(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        raise FileNotFoundError("hubCheck")

    report = tmp_path / "publication.json"
    with pytest.raises(RuntimeError, match=f"hub revision {'b' * 40}"):
        publish_track_hub(
            metadata,
            validation,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval={
                "approved": True,
                "evidence_url": HUB_APPROVAL_ISSUE,
                "approved_by": "author",
                "approved_at": "2026-07-22",
                "expected_base_revision": ARTIFACT_REVISION,
            },
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
            validator=validator,
        )

    publication = json.loads(report.read_text())
    assert publication["valid"] is False
    assert publication["status"] == "published_validation_failed"
    assert publication["final_revision"] == "b" * 40
    assert publication["validation_error_type"] == "FileNotFoundError"


def test_publication_rejects_validator_invalid_result(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        return {"valid": False, "credentials_sent": False}

    report = tmp_path / "publication.json"
    with pytest.raises(RuntimeError, match="published but post-validation failed"):
        publish_track_hub(
            metadata,
            validation,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval={
                "approved": True,
                "evidence_url": HUB_APPROVAL_ISSUE,
                "approved_by": "author",
                "approved_at": "2026-07-22",
                "expected_base_revision": ARTIFACT_REVISION,
            },
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
            validator=validator,
        )

    publication = json.loads(report.read_text())
    assert publication["valid"] is False
    assert publication["status"] == "published_validation_failed"
    assert publication["validation_error_type"] == "RuntimeError"


def test_existing_publication_validation_recovers_without_a_commit(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    calls = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"valid": True, "credentials_sent": False}

    report = tmp_path / "publication.json"
    validate_existing_track_hub_publication(
        metadata,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        final_revision="b" * 40,
        publication_approval={
            "approved": True,
            "evidence_url": HUB_APPROVAL_ISSUE,
            "approved_by": "author",
            "approved_at": "2026-07-22",
            "expected_base_revision": ARTIFACT_REVISION,
        },
        udc_dir=tmp_path / "udc",
        validator=validator,
    )

    assert len(calls) == 1
    publication = json.loads(report.read_text())
    assert publication["valid"] is True
    assert publication["status"] == "validated_existing_publication"
    assert publication["final_revision"] == "b" * 40
    assert len(publication["published_files"]) == 35


def test_existing_publication_rejects_validator_invalid_result(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        return {"valid": False, "credentials_sent": False}

    with pytest.raises(RuntimeError, match="returned an invalid result"):
        validate_existing_track_hub_publication(
            metadata,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            final_revision="b" * 40,
            publication_approval={
                "approved": True,
                "evidence_url": HUB_APPROVAL_ISSUE,
                "approved_by": "author",
                "approved_at": "2026-07-22",
                "expected_base_revision": ARTIFACT_REVISION,
            },
            udc_dir=tmp_path / "udc",
            validator=validator,
        )


def test_publication_rejects_missing_approval_and_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    with pytest.raises(ValueError, match="explicit author approval"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=None,
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
        )

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="non-Slurm"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval={
                "approved": True,
                "evidence_url": HUB_APPROVAL_ISSUE,
                "approved_by": "author",
                "approved_at": "2026-07-22",
                "expected_base_revision": ARTIFACT_REVISION,
            },
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
        )


def test_enabled_workflow_builds_validates_and_separates_publication(
    tmp_path: Path,
) -> None:
    release_manifest = tmp_path / "release.json"
    release_manifest.write_text(json.dumps(_release_manifest()), encoding="utf-8")
    config_path = tmp_path / "hub.yaml"
    config_path.write_text(
        f"""\
hub:
  enabled: true
  release_manifest: {release_manifest}
  artifact_revision: {ARTIFACT_REVISION}
  expected_base_revision: {ARTIFACT_REVISION}
  contact_email: maintainer@example.org
  output_root: {tmp_path / "hub"}
  udc_cache_root: {tmp_path / "udc"}
  publication_approval: null
  resources:
    build: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    validate: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    publish: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
""",
        encoding="utf-8",
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
            "publish_hub",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "build_track_hub_metadata" in result.stdout
    assert "validate_track_hub" in result.stdout
    assert "publish_hub" in result.stdout
