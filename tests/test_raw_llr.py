from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gpn_star_scores.bigwig import BASES, ChromosomeSpec, write_raw_llr_bigwigs
from gpn_star_scores.catalog import SCORE_SETS
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.raw_llr import (
    RAW_LLR_TRACKS,
    aggregate_raw_llr_validation,
    validate_raw_llr_chromosome,
)
from gpn_star_scores.raw_llr_publication import (
    RAW_LLR_APPROVAL_ISSUE,
    publish_raw_llr,
    raw_llr_candidate_sha256,
    validate_existing_raw_llr_publication,
    validate_public_raw_llr,
)
from gpn_star_scores.release import PUBLIC_STORAGE_POLICY, REPOSITORY_ID
from gpn_star_scores.tracks import ucsc_assembly_name

BASE_REVISION = "a" * 40
FINAL_REVISION = "b" * 40


def _write_llr_fixture(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "chrom": pa.array(["1"] * 9),
                "pos": pa.array([1, 1, 1, 2, 2, 2, 5, 5, 5], type=pa.int64()),
                "ref": pa.array(["A"] * 3 + ["C"] * 3 + ["T"] * 3),
                "alt": pa.array(["T", "C", "G", "G", "A", "T", "C", "G", "A"]),
                "llr_calibrated": pa.array(
                    [-0.1, 0.2, 0.3, -0.5, 1.0, 0.0, 0.4, -0.2, 0.8],
                    type=pa.float32(),
                ),
                "abs_llr_calibrated": pa.array([101.0] * 9, type=pa.float32()),
            }
        ),
        path,
        row_group_size=4,
    )


def test_focused_chromosome_validation_covers_only_raw_llr_tracks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "llr.parquet"
    _write_llr_fixture(source)
    outputs = {f"llr_{base}": tmp_path / f"llr_{base}.bw" for base in BASES}
    chromosome = ChromosomeSpec("1", "chr1", 5)
    write_raw_llr_bigwigs(
        [source],
        {base: outputs[f"llr_{base}"] for base in BASES},
        chromosome,
        batch_size=4,
    )

    report = validate_raw_llr_chromosome(
        source,
        outputs,
        chromosome=chromosome,
        expected_position_count=3,
        sample_count=3,
        batch_size=4,
    )

    assert set(report["samples"]) == set(RAW_LLR_TRACKS)
    assert report["sample_count"] == 3
    assert report["first_gap_position"] == 3
    assert report["gap_checks"] == {track: True for track in RAW_LLR_TRACKS}
    assert report["reference_zero_baseline"] is True
    assert report["abs_llr_calibrated_used"] is False
    assert report["sign_counts"]["negative"] > 0
    assert report["sign_counts"]["positive"] > 0
    assert report["sign_counts"]["zero"] >= 3


def test_aggregate_requires_exactly_32_new_track_audits(tmp_path: Path) -> None:
    manifest_sha256 = "a" * 64
    reports = []
    for index, score_set in enumerate(SCORE_SETS, start=1):
        for track in RAW_LLR_TRACKS:
            path = tmp_path / score_set.name / f"{track}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "report_version": 1,
                        "product": "raw_calibrated_llr",
                        "valid": True,
                        "validation_stage": "post-assembly-audit",
                        "score_set": score_set.name,
                        "assembly": score_set.assembly,
                        "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
                        "track": track,
                        "base": track.removeprefix("llr_"),
                        "value_decimals": 3,
                        "reference_zero_baseline": True,
                        "abs_llr_calibrated_used": False,
                        "source_matrix_sign_counts": {
                            "negative": index,
                            "zero": index * 2,
                            "positive": index,
                        },
                        "inventory_manifest_sha256": manifest_sha256,
                        "size": index,
                        "sha256": f"{index:064x}",
                        "summary": {
                            "bases_covered": index,
                            "zoom_levels": 1,
                        },
                        "sample_check_count": 2,
                        "gap_check_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            reports.append(path)

    selection = tmp_path / "track-selection.json"
    selection.write_text(
        json.dumps(
            {
                "report_version": 1,
                "status": "selected",
                "selected_method": "direct",
                "inventory_manifest_sha256": manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "validation.json"
    output_markdown = tmp_path / "validation.md"

    aggregate_raw_llr_validation(
        reports,
        selection,
        output_json,
        output_markdown,
    )

    validation = json.loads(output_json.read_text(encoding="utf-8"))
    assert validation["track_count"] == 32
    assert validation["sample_check_count"] == 64
    assert validation["gap_check_count"] == 32
    assert validation["reference_zero_baseline"] is True
    assert validation["abs_llr_calibrated_used"] is False
    assert validation["source_matrix_sign_counts"] == {
        "negative": sum(range(1, len(SCORE_SETS) + 1)),
        "zero": 2 * sum(range(1, len(SCORE_SETS) + 1)),
        "positive": sum(range(1, len(SCORE_SETS) + 1)),
    }
    assert {
        (record["score_set"], record["track"]) for record in validation["tracks"]
    } == {
        (score_set.name, track) for score_set in SCORE_SETS for track in RAW_LLR_TRACKS
    }
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "Only the 32 post-v1" in markdown
    assert "immutable v1 BigWigs are not revalidated" in markdown


def _write_publication_candidate(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "raw-llr"
    tracks = []
    for index, score_set in enumerate(SCORE_SETS):
        for track in RAW_LLR_TRACKS:
            local = root / "final" / score_set.name / f"{track}.bw"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(bytes([index + 1]) * 80)
            tracks.append(
                {
                    "score_set": score_set.name,
                    "assembly": score_set.assembly,
                    "ucsc_assembly": score_set.assembly,
                    "track": track,
                    "path": f"bigwig/{score_set.name}/{track}.bw",
                    "size": local.stat().st_size,
                    "sha256": sha256_file(local),
                }
            )
    validation = {
        "report_version": 1,
        "product": "raw_calibrated_llr",
        "valid": True,
        "track_count": 32,
        "value_decimals": 3,
        "reference_zero_baseline": True,
        "abs_llr_calibrated_used": False,
        "total_bytes": sum(record["size"] for record in tracks),
        "tracks": tracks,
    }
    validation_path = root / "validation.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    return root, validation_path, validation


class _PublicationApi:
    def __init__(
        self,
        validation: dict,
        *,
        base_paths: tuple[str, ...] = (),
    ) -> None:
        self.validation = validation
        self.base_paths = base_paths
        self.commit_kwargs = None

    def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        if kwargs.get("revision") is None:
            return SimpleNamespace(
                private=False,
                sha=BASE_REVISION,
                siblings=[SimpleNamespace(rfilename=path) for path in self.base_paths],
            )
        siblings = [
            SimpleNamespace(
                rfilename=record["path"],
                size=record["size"],
                lfs={"sha256": record["sha256"]},
            )
            for record in self.validation["tracks"]
        ]
        return SimpleNamespace(
            private=False,
            sha=kwargs["revision"],
            siblings=siblings,
        )

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.commit_kwargs = kwargs
        return SimpleNamespace(oid=FINAL_REVISION)


def test_publication_adds_only_32_approved_files_in_one_commit(
    tmp_path: Path,
) -> None:
    root, validation_path, validation = _write_publication_candidate(tmp_path)
    candidate_sha256 = raw_llr_candidate_sha256(root, validation_path)
    approval = {
        "approved": True,
        "evidence_url": RAW_LLR_APPROVAL_ISSUE,
        "approved_by": "gonzalobenegas",
        "approved_at": "2026-07-23",
        "expected_base_revision": BASE_REVISION,
        "operation": "publish_raw_llr",
        "incremental_bytes": validation["total_bytes"],
        "candidate_sha256": candidate_sha256,
        "storage_policy_url": PUBLIC_STORAGE_POLICY,
    }
    api = _PublicationApi(validation)
    report = tmp_path / "publication.json"
    success = tmp_path / "publication.complete"

    publish_raw_llr(
        root,
        validation_path,
        report,
        expected_base_revision=BASE_REVISION,
        publication_approval=approval,
        success_marker_path=success,
        api=api,
        validator=lambda *args, **kwargs: {
            "valid": True,
            "repository": REPOSITORY_ID,
            "revision": FINAL_REVISION,
            "credentials_sent": False,
            "validation_scope": "new_raw_llr_tracks_only",
        },
    )

    assert api.commit_kwargs["parent_commit"] == BASE_REVISION
    operations = api.commit_kwargs["operations"]
    assert len(operations) == 32
    assert {operation.path_in_repo for operation in operations} == {
        record["path"] for record in validation["tracks"]
    }
    publication = json.loads(report.read_text())
    assert publication["additive_only"] is True
    assert publication["deleted_files"] == []
    assert publication["incremental_bytes"] == validation["total_bytes"]
    assert publication["status"] == "validated"
    assert success.read_text() == f"{FINAL_REVISION}\n"


def test_publication_rejects_any_existing_target_path(tmp_path: Path) -> None:
    root, validation_path, validation = _write_publication_candidate(tmp_path)
    approval = {
        "approved": True,
        "evidence_url": RAW_LLR_APPROVAL_ISSUE,
        "approved_by": "gonzalobenegas",
        "approved_at": "2026-07-23",
        "expected_base_revision": BASE_REVISION,
        "operation": "publish_raw_llr",
        "incremental_bytes": validation["total_bytes"],
        "candidate_sha256": raw_llr_candidate_sha256(root, validation_path),
        "storage_policy_url": PUBLIC_STORAGE_POLICY,
    }
    collision = validation["tracks"][0]["path"]
    api = _PublicationApi(validation, base_paths=(collision,))

    with pytest.raises(RuntimeError, match="would overwrite approved-base paths"):
        publish_raw_llr(
            root,
            validation_path,
            tmp_path / "publication.json",
            expected_base_revision=BASE_REVISION,
            publication_approval=approval,
            success_marker_path=tmp_path / "publication.complete",
            api=api,
        )

    assert api.commit_kwargs is None
    assert not (tmp_path / "publication.json").exists()
    assert not (tmp_path / "publication.complete").exists()


def test_public_validation_checks_only_new_remote_identities_and_ranges(
    tmp_path: Path,
) -> None:
    _, validation_path, validation = _write_publication_candidate(tmp_path)
    requested = []

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 0-63/80"}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"x" * 64

    def opener(request: object, **kwargs: object) -> Response:
        requested.append(getattr(request, "full_url"))
        return Response()

    report = validate_public_raw_llr(
        validation_path,
        revision=FINAL_REVISION,
        api=_PublicationApi(validation),
        opener=opener,
    )

    assert report["valid"] is True
    assert report["validation_scope"] == "new_raw_llr_tracks_only"
    assert report["existing_v1_files_checked"] == 0
    assert report["identity_check_count"] == 32
    assert report["http_range_count"] == 32
    assert len(requested) == 32
    assert all(f"/resolve/{FINAL_REVISION}/bigwig/" in url for url in requested)


def test_failed_post_validation_resumes_without_second_commit(
    tmp_path: Path,
) -> None:
    root, validation_path, validation = _write_publication_candidate(tmp_path)
    approval = {
        "approved": True,
        "evidence_url": RAW_LLR_APPROVAL_ISSUE,
        "approved_by": "gonzalobenegas",
        "approved_at": "2026-07-23",
        "expected_base_revision": BASE_REVISION,
        "operation": "publish_raw_llr",
        "incremental_bytes": validation["total_bytes"],
        "candidate_sha256": raw_llr_candidate_sha256(root, validation_path),
        "storage_policy_url": PUBLIC_STORAGE_POLICY,
    }
    api = _PublicationApi(validation)
    report = tmp_path / "publication.json"
    success = tmp_path / "publication.complete"

    def fail_validation(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError, match="was published but validation failed"):
        publish_raw_llr(
            root,
            validation_path,
            report,
            expected_base_revision=BASE_REVISION,
            publication_approval=approval,
            success_marker_path=success,
            api=api,
            validator=fail_validation,
        )
    assert api.commit_kwargs is not None
    assert not success.exists()

    validate_existing_raw_llr_publication(
        root,
        validation_path,
        report,
        expected_base_revision=BASE_REVISION,
        final_revision=FINAL_REVISION,
        publication_approval=approval,
        success_marker_path=success,
        validator=lambda *args, **kwargs: {
            "valid": True,
            "repository": REPOSITORY_ID,
            "revision": FINAL_REVISION,
            "credentials_sent": False,
        },
    )

    publication = json.loads(report.read_text())
    assert publication["status"] == "validated_existing_publication"
    assert publication["recovered_from_status"] == "published_validation_failed"
    assert success.read_text() == f"{FINAL_REVISION}\n"
