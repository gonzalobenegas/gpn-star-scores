"""Approval-gated publication of the additive raw calibrated-LLR BigWigs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from huggingface_hub import CommitOperationAdd, HfApi

from gpn_star_scores.catalog import SCORE_SETS
from gpn_star_scores.inventory import atomic_write_json, sha256_file
from gpn_star_scores.raw_llr import RAW_LLR_TRACKS
from gpn_star_scores.release import (
    HUGGING_FACE_URL,
    PUBLIC_STORAGE_POLICY,
    REPOSITORY_ID,
)

RAW_LLR_APPROVAL_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/15"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_revision(revision: Any, *, field: str) -> str:
    if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
        raise ValueError(f"{field} must be a lowercase 40-character commit SHA")
    return revision


def _validated_local_records(
    bigwig_root: str | Path,
    validation_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(bigwig_root)
    validation = _read_json(validation_path)
    records = validation.get("tracks")
    if (
        validation.get("report_version") != 1
        or validation.get("product") != "raw_calibrated_llr"
        or validation.get("valid") is not True
        or validation.get("track_count") != 32
        or validation.get("value_decimals") != 3
        or validation.get("reference_zero_baseline") is not True
        or validation.get("abs_llr_calibrated_used") is not False
        or not isinstance(records, list)
        or len(records) != 32
    ):
        raise ValueError("raw-LLR validation does not certify exactly 32 tracks")
    expected = {
        (score_set.name, track): f"bigwig/{score_set.name}/{track}.bw"
        for score_set in SCORE_SETS
        for track in RAW_LLR_TRACKS
    }
    observed: set[tuple[str, str]] = set()
    validated = []
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("raw-LLR validation records must be objects")
        record = dict(raw_record)
        key = (str(record.get("score_set")), str(record.get("track")))
        path_in_repo = expected.get(key)
        local = root / "final" / key[0] / f"{key[1]}.bw"
        if (
            path_in_repo is None
            or key in observed
            or record.get("path") != path_in_repo
            or not isinstance(record.get("size"), int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
            or not isinstance(record.get("sha256"), str)
            or not _SHA256_PATTERN.fullmatch(record["sha256"])
            or not local.is_file()
            or local.stat().st_size != record["size"]
            or sha256_file(local) != record["sha256"]
        ):
            raise ValueError(f"raw-LLR artifact identity differs: {key!r}")
        record["local_path"] = local
        validated.append(record)
        observed.add(key)
    if observed != set(expected):
        raise ValueError("raw-LLR records differ from the exact 32-track catalog")
    validated.sort(key=lambda record: (record["score_set"], record["track"]))
    if sum(record["size"] for record in validated) != validation.get("total_bytes"):
        raise ValueError("raw-LLR validation total_bytes differs")
    return validation, validated


def raw_llr_candidate_sha256(
    bigwig_root: str | Path,
    validation_path: str | Path,
) -> str:
    """Hash the exact path, byte count, and SHA-256 of the 32-file candidate."""

    _, records = _validated_local_records(bigwig_root, validation_path)
    return _candidate_sha256(records)


def _candidate_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(record["sha256"]))
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_publication_approval(
    approval: Mapping[str, Any] | None,
    *,
    expected_base_revision: str,
    incremental_bytes: int,
    candidate_sha256: str,
) -> dict[str, Any]:
    if not isinstance(approval, Mapping):
        raise ValueError("public raw-LLR upload requires explicit author approval")
    if (
        approval.get("approved") is not True
        or approval.get("evidence_url") != RAW_LLR_APPROVAL_ISSUE
        or approval.get("expected_base_revision") != expected_base_revision
        or approval.get("operation") != "publish_raw_llr"
        or approval.get("incremental_bytes") != incremental_bytes
        or approval.get("candidate_sha256") != candidate_sha256
        or approval.get("storage_policy_url") != PUBLIC_STORAGE_POLICY
        or not isinstance(approval.get("approved_by"), str)
        or not approval["approved_by"]
        or not isinstance(approval.get("approved_at"), str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approval["approved_at"])
    ):
        raise ValueError("raw-LLR publication approval is incomplete or mismatched")
    return dict(approval)


def _sibling_lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) else None


def _validate_range_response(
    record: Mapping[str, Any],
    *,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    url = (
        f"{HUGGING_FACE_URL}/datasets/{REPOSITORY_ID}/resolve/"
        f"{record['artifact_revision']}/{record['path']}"
    )
    request = Request(url, headers={"Range": "bytes=0-63"})
    with opener(request, timeout=60) as response:
        status = getattr(response, "status", None)
        content = response.read()
        content_range = response.headers.get("Content-Range")
    if status != 206 or not content or not isinstance(content_range, str):
        raise RuntimeError(f"raw-LLR HTTP range request failed: {record['path']}")
    match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", content_range)
    if (
        match is None
        or int(match.group(2)) != record["size"]
        or len(content) != int(match.group(1)) + 1
    ):
        raise RuntimeError(f"invalid raw-LLR Content-Range: {record['path']}")
    return {
        "path": record["path"],
        "status": status,
        "content_range": content_range,
        "bytes_read": len(content),
    }


def validate_public_raw_llr(
    validation_path: str | Path,
    *,
    revision: str,
    repository_id: str = REPOSITORY_ID,
    api: Any | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Validate identities and byte-range service for only the 32 new tracks."""

    _validate_revision(revision, field="revision")
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"raw-LLR repository must be {REPOSITORY_ID}")
    validation = _read_json(validation_path)
    records = validation.get("tracks")
    if (
        validation.get("valid") is not True
        or validation.get("track_count") != 32
        or not isinstance(records, list)
        or len(records) != 32
    ):
        raise ValueError("raw-LLR validation manifest is incomplete")
    expected = {
        (score_set.name, track): f"bigwig/{score_set.name}/{track}.bw"
        for score_set in SCORE_SETS
        for track in RAW_LLR_TRACKS
    }
    observed = {
        (record.get("score_set"), record.get("track")): record.get("path")
        for record in records
        if isinstance(record, Mapping)
    }
    if observed != expected or len(observed) != len(records):
        raise ValueError("raw-LLR public validation catalog differs")

    public_api = api or HfApi(token=False)
    info = public_api.repo_info(
        repository_id,
        repo_type="dataset",
        revision=revision,
        files_metadata=True,
        token=False,
    )
    if getattr(info, "private", True) or getattr(info, "sha", None) != revision:
        raise RuntimeError("public raw-LLR revision did not resolve exactly")
    siblings = {
        getattr(sibling, "rfilename", None): sibling
        for sibling in getattr(info, "siblings", [])
    }
    identity_checks = []
    range_checks = []
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("raw-LLR validation records must be objects")
        record = dict(raw_record)
        sibling = siblings.get(record.get("path"))
        if (
            sibling is None
            or not isinstance(record.get("size"), int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
            or not isinstance(record.get("sha256"), str)
            or not _SHA256_PATTERN.fullmatch(record["sha256"])
            or getattr(sibling, "size", None) != record.get("size")
            or _sibling_lfs_sha256(sibling) != record.get("sha256")
        ):
            raise RuntimeError(f"public raw-LLR identity differs: {record.get('path')}")
        identity_checks.append(
            {
                "path": record["path"],
                "size": record["size"],
                "sha256": record["sha256"],
            }
        )
        record["artifact_revision"] = revision
        range_checks.append(_validate_range_response(record, opener=opener))
    return {
        "report_version": 1,
        "valid": True,
        "repository": repository_id,
        "revision": revision,
        "public": True,
        "credentials_sent": False,
        "validation_scope": "new_raw_llr_tracks_only",
        "existing_v1_files_checked": 0,
        "identity_check_count": len(identity_checks),
        "identity_checks": identity_checks,
        "http_range_count": len(range_checks),
        "http_range_checks": range_checks,
    }


def publish_raw_llr(
    bigwig_root: str | Path,
    validation_path: str | Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    publication_approval: Mapping[str, Any] | None,
    success_marker_path: str | Path | None = None,
    repository_id: str = REPOSITORY_ID,
    api: Any | None = None,
    validator: Callable[..., dict[str, Any]] = validate_public_raw_llr,
) -> None:
    """Add exactly 32 approved BigWigs in one optimistic public commit."""

    _validate_revision(expected_base_revision, field="expected_base_revision")
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"raw-LLR repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("raw-LLR publication must run outside Slurm")
    validation, records = _validated_local_records(bigwig_root, validation_path)
    candidate_sha256 = _candidate_sha256(records)
    approval = _validated_publication_approval(
        publication_approval,
        expected_base_revision=expected_base_revision,
        incremental_bytes=validation["total_bytes"],
        candidate_sha256=candidate_sha256,
    )
    success_marker = (
        Path(success_marker_path) if success_marker_path is not None else None
    )
    if success_marker is not None:
        if success_marker.resolve() == Path(report_path).resolve():
            raise ValueError("success marker and publication report must differ")
        success_marker.unlink(missing_ok=True)

    authenticated_api = api or HfApi()
    repository = authenticated_api.repo_info(repository_id, repo_type="dataset")
    if getattr(repository, "private", True):
        raise RuntimeError("raw-LLR publication requires the public repository")
    if getattr(repository, "sha", None) != expected_base_revision:
        raise RuntimeError("public repository changed since the approved base revision")
    operations = [
        CommitOperationAdd(
            path_in_repo=record["path"],
            path_or_fileobj=record["local_path"],
        )
        for record in records
    ]
    commit = authenticated_api.create_commit(
        repo_id=repository_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish raw calibrated-LLR BigWig tracks",
        parent_commit=expected_base_revision,
    )
    final_revision = _validate_revision(
        getattr(commit, "oid", None), field="final_revision"
    )
    publication = {
        "report_version": 1,
        "valid": False,
        "status": "published_pending_validation",
        "repository": repository_id,
        "public": True,
        "base_revision": expected_base_revision,
        "final_revision": final_revision,
        "single_commit": True,
        "single_process": True,
        "slurm_job_id": None,
        "additive_only": True,
        "deleted_files": [],
        "published_file_count": len(records),
        "incremental_bytes": validation["total_bytes"],
        "candidate_sha256": candidate_sha256,
        "publication_approval": approval,
        "published_files": [record["path"] for record in records],
    }
    atomic_write_json(Path(report_path), publication)
    try:
        public_validation = validator(
            validation_path,
            revision=final_revision,
            repository_id=repository_id,
        )
        if public_validation.get("valid") is not True:
            raise RuntimeError("public raw-LLR validation returned invalid")
    except BaseException as error:
        publication.update(
            {
                "status": "published_validation_failed",
                "validation_error_type": type(error).__name__,
                "validation_error": str(error),
            }
        )
        atomic_write_json(Path(report_path), publication)
        raise RuntimeError(
            f"raw-LLR revision {final_revision} was published but validation "
            "failed; validate that immutable revision before retrying"
        ) from error
    publication.update(
        {
            "valid": True,
            "status": "validated",
            "public_validation": public_validation,
        }
    )
    atomic_write_json(Path(report_path), publication)
    if success_marker is not None:
        _atomic_write_text(success_marker, f"{final_revision}\n")


def validate_existing_raw_llr_publication(
    bigwig_root: str | Path,
    validation_path: str | Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    final_revision: str,
    publication_approval: Mapping[str, Any] | None,
    success_marker_path: str | Path,
    repository_id: str = REPOSITORY_ID,
    validator: Callable[..., dict[str, Any]] = validate_public_raw_llr,
) -> None:
    """Resume anonymous checks for an already-created raw-LLR commit."""

    _validate_revision(expected_base_revision, field="expected_base_revision")
    _validate_revision(final_revision, field="final_revision")
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"raw-LLR repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("raw-LLR publication validation must run outside Slurm")
    validation, records = _validated_local_records(bigwig_root, validation_path)
    candidate_sha256 = _candidate_sha256(records)
    approval = _validated_publication_approval(
        publication_approval,
        expected_base_revision=expected_base_revision,
        incremental_bytes=validation["total_bytes"],
        candidate_sha256=candidate_sha256,
    )
    publication = _read_json(report_path)
    expected = {
        "report_version": 1,
        "repository": repository_id,
        "public": True,
        "base_revision": expected_base_revision,
        "final_revision": final_revision,
        "single_commit": True,
        "single_process": True,
        "slurm_job_id": None,
        "additive_only": True,
        "deleted_files": [],
        "published_file_count": 32,
        "incremental_bytes": validation["total_bytes"],
        "candidate_sha256": candidate_sha256,
        "publication_approval": approval,
        "published_files": [record["path"] for record in records],
    }
    recoverable = publication.get("status") in {
        "published_pending_validation",
        "published_validation_failed",
        "validated",
        "validated_existing_publication",
    }
    if (
        any(publication.get(field) != value for field, value in expected.items())
        or not recoverable
    ):
        raise ValueError(
            "validate-existing requires the matching raw-LLR publisher report"
        )
    success_marker = Path(success_marker_path)
    if success_marker.resolve() == Path(report_path).resolve():
        raise ValueError("success marker and publication report must differ")
    success_marker.unlink(missing_ok=True)
    recovered_from_status = publication["status"]
    public_validation = validator(
        validation_path,
        revision=final_revision,
        repository_id=repository_id,
    )
    if public_validation.get("valid") is not True:
        raise RuntimeError("existing public raw-LLR validation returned invalid")
    publication.update(
        {
            "valid": True,
            "status": "validated_existing_publication",
            "recovered_from_status": recovered_from_status,
            "public_validation": public_validation,
        }
    )
    publication.pop("validation_error_type", None)
    publication.pop("validation_error", None)
    atomic_write_json(Path(report_path), publication)
    _atomic_write_text(success_marker, f"{final_revision}\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bigwig-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path)
    parser.add_argument("--expected-base-revision", required=True)
    parser.add_argument("--approval-approved", required=True)
    parser.add_argument("--approval-evidence-url", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--approval-expected-base-revision", required=True)
    parser.add_argument("--approval-operation", required=True)
    parser.add_argument("--approval-incremental-bytes", type=int, required=True)
    parser.add_argument("--approval-candidate-sha256", required=True)
    parser.add_argument("--approval-storage-policy-url", required=True)
    parser.add_argument(
        "--validate-existing-revision",
        help="resume validation for this already-created immutable revision",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    approval = {
        "approved": args.approval_approved.lower() == "true",
        "evidence_url": args.approval_evidence_url,
        "approved_by": args.approved_by,
        "approved_at": args.approved_at,
        "expected_base_revision": args.approval_expected_base_revision,
        "operation": args.approval_operation,
        "incremental_bytes": args.approval_incremental_bytes,
        "candidate_sha256": args.approval_candidate_sha256,
        "storage_policy_url": args.approval_storage_policy_url,
    }
    if args.validate_existing_revision:
        if args.success_marker is None:
            raise ValueError("validate-existing requires --success-marker")
        validate_existing_raw_llr_publication(
            args.bigwig_root,
            args.validation,
            args.report,
            expected_base_revision=args.expected_base_revision,
            final_revision=args.validate_existing_revision,
            publication_approval=approval,
            success_marker_path=args.success_marker,
        )
    else:
        publish_raw_llr(
            args.bigwig_root,
            args.validation,
            args.report,
            expected_base_revision=args.expected_base_revision,
            publication_approval=approval,
            success_marker_path=args.success_marker,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
