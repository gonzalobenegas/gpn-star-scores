"""Run end-to-end release QA and build the immutable v1 release record."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import polars as pl

from gpn_star_scores.catalog import EXPECTED_SHARD_COUNT, SCORE_SETS, expected_shards
from gpn_star_scores.hub import validate_public_track_hub
from gpn_star_scores.inventory import EXPECTED_SCHEMAS, sha256_file
from gpn_star_scores.release import (
    CAPACITY_APPROVAL_ISSUE,
    CAPACITY_BLOCKER,
    PUBLIC_STORAGE_POLICY,
    REPOSITORY_ID,
    dataset_configs,
    validate_public_release,
)
from gpn_star_scores.tracks import TRACKS

RELEASE_TAG = "v1.0.0"
QA_APPROVAL_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/2"
VIEWER_WAIVER_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/4"
VIEWER_FOLLOWUP_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/17"
CAPACITY_WAIVER_ID = "hugging-face-numeric-capacity"
VIEWER_WAIVER_ID = "dataset-viewer-readiness"
_BOUNDED_JOIN_MARKERS = (
    'variant_chrom = "22"',
    "/llr/llr_chr{variant_chrom}.parquet",
    'pl.col("pos").is_between(variant_start, variant_end)',
)
_VIEWER_API_URL = "https://datasets-server.huggingface.co"


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _is_sha(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and any(character != "0" for character in value)
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_sha(value: Any, *, field: str, length: int = 40) -> str:
    if not _is_sha(value, length):
        raise ValueError(f"{field} must be a {length}-character hexadecimal SHA")
    return str(value).lower()


def _all_zero(value: Any) -> bool:
    return isinstance(value, Mapping) and all(item == 0 for item in value.values())


def _schema_record(score_type: str) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in EXPECTED_SCHEMAS[score_type]
    ]


def _validate_inventory(
    inventory: Mapping[str, Any], inventory_sha256: str
) -> dict[str, Any]:
    expected = {shard.relative_path.as_posix(): shard for shard in expected_shards()}
    source = inventory.get("source")
    validation = inventory.get("validation")
    records = inventory.get("shards")
    if (
        inventory.get("manifest_version") != 1
        or not isinstance(source, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(records, list)
        or len(records) != EXPECTED_SHARD_COUNT
        or source.get("expected_shards") != EXPECTED_SHARD_COUNT
        or source.get("reported_shards") != EXPECTED_SHARD_COUNT
        or source.get("discovered_parquet_files") != EXPECTED_SHARD_COUNT
        or source.get("missing_paths") != []
        or source.get("unexpected_paths") != []
        or source.get("unreported_paths") != []
        or validation.get("valid_shards") != EXPECTED_SHARD_COUNT
        or validation.get("invalid_shards") != 0
        or validation.get("release_ready") is not False
        or validation.get("blockers") != [CAPACITY_BLOCKER]
    ):
        raise ValueError("inventory does not cover exactly 290 valid source shards")

    observed: dict[str, Mapping[str, Any]] = {}
    total_rows = 0
    total_bytes = 0
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("inventory contains a malformed shard record")
        path = record["path"]
        if path in observed:
            raise ValueError(f"inventory contains duplicate shard {path}")
        observed[path] = record
    if set(observed) != set(expected):
        raise ValueError("inventory shard paths differ from the release catalog")

    schema_counts = {"entropy": 0, "llr": 0}
    for path, shard in expected.items():
        record = observed[path]
        parquet = record.get("parquet")
        content = record.get("content")
        errors = record.get("errors")
        digest = record.get("sha256")
        size = record.get("size")
        rows = parquet.get("num_rows") if isinstance(parquet, Mapping) else None
        common_zero_fields = (
            "invalid_alt_rows",
            "invalid_ref_rows",
            "order_violations",
            "out_of_bounds_rows",
            "reference_mismatch_rows",
            "unexpected_chrom_rows",
        )
        if (
            record.get("valid") is not True
            or errors != []
            or record.get("score_set") != shard.score_set
            or record.get("assembly") != shard.assembly
            or record.get("score_type") != shard.score_type
            or str(record.get("chrom")) != shard.chrom
            or record.get("schema") != _schema_record(shard.score_type)
            or not _is_sha(digest, 64)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows <= 0
            or not isinstance(content, Mapping)
            or content.get("rows_scanned") != rows
            or not _all_zero(content.get("null_counts"))
            or not _all_zero(content.get("non_finite_counts"))
            or any(content.get(field) != 0 for field in common_zero_fields)
            or content.get("llr_group_checks_skipped_for_nulls") is not False
        ):
            raise ValueError(f"invalid schema or content evidence for {path}")
        if shard.score_type == "llr" and not _all_zero(content.get("llr_group_errors")):
            raise ValueError(f"invalid LLR allele-group evidence for {path}")
        if (
            shard.score_type == "entropy"
            and content.get("llr_group_errors") is not None
        ):
            raise ValueError(f"unexpected LLR group evidence for {path}")
        schema_counts[shard.score_type] += 1
        total_rows += rows
        total_bytes += size

    if source.get("total_shard_bytes") != total_bytes:
        raise ValueError("inventory byte total differs from its shard records")
    references = inventory.get("references")
    if (
        not isinstance(references, list)
        or len(references) != 6
        or any(
            not isinstance(reference, Mapping)
            or reference.get("identity_verified") is not True
            or reference.get("source_sha256") != reference.get("expected_sha256")
            or not _is_sha(reference.get("source_sha256"), 64)
            for reference in references
        )
    ):
        raise ValueError("reference FASTA identities are incomplete")
    return {
        "manifest_sha256": inventory_sha256,
        "shard_count": len(records),
        "total_bytes": total_bytes,
        "total_rows": total_rows,
        "schema_counts": schema_counts,
        "all_source_checks_passed": True,
        "reference_count": len(references),
    }


def _validate_release_chain(metadata_root: Path) -> dict[str, Any]:
    manifest_path = metadata_root / "manifest" / "release.json"
    inventory_path = metadata_root / "manifest" / "inventory.json"
    selection_path = metadata_root / "manifest" / "parquet-layout.json"
    bigwig_validation_path = metadata_root / "manifest" / "bigwig-validation.json"
    for path in (
        manifest_path,
        inventory_path,
        selection_path,
        bigwig_validation_path,
        metadata_root / "README.md",
    ):
        if not path.is_file():
            raise ValueError(f"release metadata is missing {path.name}")

    manifest = _read_json(manifest_path)
    inventory = _read_json(inventory_path)
    selection = _read_json(selection_path)
    bigwig_validation = _read_json(bigwig_validation_path)
    inventory_sha256 = sha256_file(inventory_path)
    inventory_summary = _validate_inventory(inventory, inventory_sha256)
    repository = manifest.get("repository")
    if (
        manifest.get("release_manifest_version") != 1
        or not isinstance(repository, Mapping)
        or repository.get("id") != REPOSITORY_ID
        or repository.get("public") is not True
        or repository.get("license") != "apache-2.0"
        or manifest.get("source_inventory", {}).get("manifest_sha256")
        != inventory_sha256
        or manifest.get("validation", {}).get("preflight_passed") is not True
        or manifest.get("validation", {}).get("inventory_data_validation_passed")
        is not True
        or manifest.get("validation", {}).get("parquet_layout_selected") != "source"
        or manifest.get("validation", {}).get("bigwig_validation_passed") is not True
        or manifest.get("validation", {}).get("expected_parquet_files")
        != EXPECTED_SHARD_COUNT
        or manifest.get("validation", {}).get("expected_bigwig_files") != 40
        or manifest.get("validation", {}).get("expected_viewer_configs") != 16
        or manifest.get("dataset_configs") != dataset_configs()
        or manifest.get("source_inventory", {}).get("total_shard_bytes")
        != inventory_summary["total_bytes"]
    ):
        raise ValueError("release manifest does not contain a valid preflight chain")

    inventory_by_path = {
        f"data/{record['path']}": record for record in inventory["shards"]
    }
    parquet = manifest.get("parquet")
    parquet_files = parquet.get("files") if isinstance(parquet, Mapping) else None
    if (
        not isinstance(parquet_files, list)
        or parquet.get("file_count") != EXPECTED_SHARD_COUNT
        or len(parquet_files) != EXPECTED_SHARD_COUNT
        or {record.get("path") for record in parquet_files} != set(inventory_by_path)
    ):
        raise ValueError("release manifest does not cover all source Parquet shards")
    for release_record in parquet_files:
        source = inventory_by_path[release_record["path"]]
        if (
            any(
                release_record.get(field) != source.get(field)
                for field in (
                    "score_set",
                    "assembly",
                    "score_type",
                    "chrom",
                    "size",
                    "sha256",
                )
            )
            or release_record.get("rows") != source.get("parquet", {}).get("num_rows")
            or release_record.get("coordinate_bounds")
            != source.get("content", {}).get("coordinate_bounds")
        ):
            raise ValueError(
                f"release Parquet identity differs for {release_record['path']}"
            )
    if parquet.get("total_bytes") != inventory_summary["total_bytes"]:
        raise ValueError("release Parquet byte total differs from the inventory")

    if (
        selection.get("status") != "selected"
        or selection.get("selected_candidate") != "source"
        or selection.get("blockers") != []
        or selection.get("source_inventory", {}).get("valid") is not True
        or selection.get("source_inventory", {}).get("manifest_sha256")
        != inventory_sha256
    ):
        raise ValueError("Parquet layout selection is not final for this inventory")

    expected_tracks = {
        (score_set.name, track) for score_set in SCORE_SETS for track in TRACKS
    }
    bigwig = manifest.get("bigwig")
    bigwig_files = bigwig.get("files") if isinstance(bigwig, Mapping) else None
    validation_tracks = bigwig_validation.get("tracks")
    if (
        bigwig_validation.get("valid") is not True
        or bigwig_validation.get("inventory_manifest_sha256") != inventory_sha256
        or bigwig_validation.get("track_count") != len(expected_tracks)
        or bigwig_validation.get("selected_method") != "direct"
        or bigwig_validation.get("value_decimals") != 3
        or bigwig.get("value_decimals") != 3
        or not isinstance(validation_tracks, list)
        or {(item.get("score_set"), item.get("track")) for item in validation_tracks}
        != expected_tracks
        or not isinstance(bigwig_files, list)
        or bigwig.get("file_count") != len(expected_tracks)
        or len(bigwig_files) != len(expected_tracks)
        or {(item.get("score_set"), item.get("track")) for item in bigwig_files}
        != expected_tracks
    ):
        raise ValueError("BigWig evidence does not cover all 40 final tracks")
    validation_by_track = {
        (record["score_set"], record["track"]): record for record in validation_tracks
    }
    for release_record in bigwig_files:
        validation_record = validation_by_track[
            (release_record["score_set"], release_record["track"])
        ]
        if any(
            release_record.get(field) != validation_record.get(field)
            for field in (
                "score_set",
                "assembly",
                "ucsc_assembly",
                "track",
                "bases_covered",
                "zoom_levels",
            )
        ):
            raise ValueError(
                f"release BigWig evidence differs for {release_record['path']}"
            )
    if (
        not isinstance(bigwig_validation.get("sample_check_count"), int)
        or bigwig_validation["sample_check_count"] <= 0
        or not isinstance(bigwig_validation.get("gap_check_count"), int)
        or bigwig_validation["gap_check_count"] < 0
        or any(
            not _is_sha(record.get("sha256"), 64)
            or record.get("zoom_levels", 0) < 1
            or record.get("bases_covered", 0) <= 0
            for record in bigwig_files
        )
        or bigwig.get("total_bytes") != sum(record["size"] for record in bigwig_files)
    ):
        raise ValueError("BigWig audit evidence is incomplete")

    capacity_waiver = manifest.get("source_inventory", {}).get("capacity_waiver")
    planned_release_bytes = parquet["total_bytes"] + bigwig["total_bytes"]
    if (
        not isinstance(capacity_waiver, Mapping)
        or capacity_waiver.get("approved") is not True
        or capacity_waiver.get("public_repository") is not True
        or capacity_waiver.get("evidence_url") != CAPACITY_APPROVAL_ISSUE
        or capacity_waiver.get("public_storage_policy_url") != PUBLIC_STORAGE_POLICY
        or capacity_waiver.get("planned_release_bytes") != planned_release_bytes
        or not isinstance(capacity_waiver.get("reserved_headroom_bytes"), int)
        or capacity_waiver["reserved_headroom_bytes"] < 0
        or not isinstance(capacity_waiver.get("approved_by"), str)
        or not capacity_waiver["approved_by"].strip()
        or not isinstance(capacity_waiver.get("approved_at"), str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", capacity_waiver["approved_at"])
    ):
        raise ValueError("release does not record its numeric capacity waiver")
    return {
        "release_manifest_sha256": sha256_file(manifest_path),
        "inventory": inventory_summary,
        "parquet_layout_sha256": sha256_file(selection_path),
        "parquet_file_count": len(parquet_files),
        "bigwig_validation_sha256": sha256_file(bigwig_validation_path),
        "bigwig_file_count": len(bigwig_files),
        "bigwig_sample_check_count": bigwig_validation["sample_check_count"],
        "bigwig_gap_check_count": bigwig_validation["gap_check_count"],
        "dataset_card_sha256": sha256_file(metadata_root / "README.md"),
        "capacity_waiver": dict(capacity_waiver),
    }


@contextmanager
def _without_credentials() -> Any:
    names = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_dataset_card_examples(root: str) -> dict[str, Any]:
    """Execute all four dataset-card Polars examples against ``root``."""

    root = root.rstrip("/")
    with _without_credentials():
        scores = pl.scan_parquet(f"{root}/gpn-star-hg38-v100-200m/entropy/*.parquet")
        region = (
            scores.filter(
                (pl.col("chrom") == "22")
                & pl.col("pos").is_between(20_000_000, 20_001_000)
            )
            .select("chrom", "pos", "ref", "entropy_calibrated")
            .collect()
        )
        projected = (
            pl.scan_parquet(f"{root}/ce11/entropy/*.parquet")
            .select("chrom", "pos", "entropy_calibrated")
            .collect()
        )
        multi_chrom = (
            pl.scan_parquet(f"{root}/dm6/llr/*.parquet")
            .filter(pl.col("chrom").is_in(["2L", "X"]))
            .select("chrom", "pos", "ref", "alt", "llr_calibrated")
            .collect()
        )
        variant_chrom = "22"
        variant_start = 20_000_001
        variant_end = 20_000_001
        variants = pl.DataFrame(
            {
                "chrom": [variant_chrom],
                "pos": [variant_start],
                "ref": ["A"],
                "alt": ["G"],
            }
        ).lazy()
        llr = pl.scan_parquet(
            f"{root}/gpn-star-hg38-p243-200m/llr/llr_chr{variant_chrom}.parquet"
        ).filter(
            (pl.col("chrom") == variant_chrom)
            & pl.col("pos").is_between(variant_start, variant_end)
        )
        annotated = variants.join(
            llr, on=["chrom", "pos", "ref", "alt"], how="left"
        ).collect()

    expected_columns = {
        "interval_filter_projection": [
            "chrom",
            "pos",
            "ref",
            "entropy_calibrated",
        ],
        "projected_score_scan": ["chrom", "pos", "entropy_calibrated"],
        "multi_chromosome_scan": ["chrom", "pos", "ref", "alt", "llr_calibrated"],
    }
    frames = {
        "interval_filter_projection": region,
        "projected_score_scan": projected,
        "multi_chromosome_scan": multi_chrom,
    }
    for name, frame in frames.items():
        if frame.height < 1 or frame.columns != expected_columns[name]:
            raise RuntimeError(f"dataset-card example failed: {name}")
    if annotated.height != 1 or annotated.columns != [
        "chrom",
        "pos",
        "ref",
        "alt",
        "llr_calibrated",
        "abs_llr_calibrated",
    ]:
        raise RuntimeError("dataset-card example failed: variant_join")
    return {
        "valid": True,
        "credentials_sent": False,
        "root": root,
        "examples": [
            {
                "name": name,
                "rows": frame.height,
                "columns": frame.columns,
            }
            for name, frame in frames.items()
        ]
        + [
            {
                "name": "variant_join",
                "rows": annotated.height,
                "columns": annotated.columns,
            }
        ],
    }


def validate_dataset_viewer_discovery(
    repository_id: str = REPOSITORY_ID,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Validate anonymous Viewer discovery and capability endpoints."""

    query = urlencode({"dataset": repository_id})

    def payload(endpoint: str) -> dict[str, Any]:
        with opener(f"{_VIEWER_API_URL}/{endpoint}?{query}", timeout=120) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError(f"Dataset Viewer {endpoint} response is not an object")
        return value

    with _without_credentials():
        splits_payload = payload("splits")
        valid_payload = payload("is-valid")
    splits = splits_payload.get("splits")
    expected = {(config["config_name"], "train") for config in dataset_configs()}
    observed = (
        {
            (record.get("config"), record.get("split"))
            for record in splits
            if isinstance(record, Mapping) and record.get("dataset") == repository_id
        }
        if isinstance(splits, list)
        else set()
    )
    if (
        not isinstance(splits, list)
        or observed != expected
        or len(splits) != len(expected)
        or splits_payload.get("pending") != []
        or splits_payload.get("failed") != []
        or valid_payload.get("preview") is not True
        or valid_payload.get("viewer") is not True
    ):
        raise RuntimeError("Dataset Viewer discovery or preview support is incomplete")
    return {
        "valid": True,
        "credentials_sent": False,
        "split_count": len(splits),
        "configs": sorted(config for config, _ in observed),
        "preview": True,
        "viewer": True,
        "search": valid_payload.get("search"),
        "filter": valid_payload.get("filter"),
        "statistics": valid_payload.get("statistics"),
    }


def validate_public_release_for_qa(
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    revision: str,
    viewer_attempts: int = 1,
    viewer_retry_seconds: float = 0,
    hf_block_size: int = 4_194_304,
    repository_id: str = REPOSITORY_ID,
    validator: Callable[..., dict[str, Any]] = validate_public_release,
    example_runner: Callable[[str], dict[str, Any]] = run_dataset_card_examples,
    viewer_discovery_validator: Callable[[str], dict[str, Any]] = (
        validate_dataset_viewer_discovery
    ),
) -> None:
    """Repeat anonymous release validation and every published Polars example."""

    _require_sha(revision, field="release revision")
    public_validation = validator(
        metadata_root,
        repository_id=repository_id,
        revision=revision,
        viewer_attempts=viewer_attempts,
        viewer_retry_seconds=viewer_retry_seconds,
        viewer_required=False,
        hf_block_size=hf_block_size,
    )
    examples = example_runner(f"hf://datasets/{repository_id}@{revision}/data")
    viewer_discovery = viewer_discovery_validator(repository_id)
    valid = (
        public_validation.get("valid") is True
        and examples.get("valid") is True
        and viewer_discovery.get("valid") is True
    )
    _atomic_write_json(
        Path(report_path),
        {
            "report_version": 1,
            "valid": valid,
            "repository": repository_id,
            "revision": revision,
            "public": True,
            "credentials_sent": False,
            "release_validation": public_validation,
            "dataset_card_examples": examples,
            "dataset_viewer_discovery": viewer_discovery,
        },
    )


def validate_public_hub_for_qa(
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    revision: str,
    udc_dir: str | Path,
    repository_id: str = REPOSITORY_ID,
    validator: Callable[..., dict[str, Any]] = validate_public_track_hub,
) -> None:
    """Repeat complete anonymous validation of the immutable public hub."""

    _require_sha(revision, field="hub revision")
    report = validator(
        metadata_root,
        revision=revision,
        udc_dir=udc_dir,
        repository_id=repository_id,
    )
    if report.get("valid") is not True:
        raise RuntimeError("public hub QA returned an invalid result")
    _atomic_write_json(Path(report_path), report)


def _validated_viewer_waiver(
    waivers: Sequence[Mapping[str, Any]], *, viewer_ready: bool
) -> list[dict[str, Any]]:
    observed = []
    for waiver in waivers:
        if not isinstance(waiver, Mapping) or not isinstance(waiver.get("id"), str):
            raise ValueError("QA waivers must be named objects")
        observed.append(dict(waiver))
    waiver_ids = [item["id"] for item in observed]
    if len(waiver_ids) != len(set(waiver_ids)):
        raise ValueError("QA waiver ids must be unique")
    viewer = [item for item in observed if item["id"] == VIEWER_WAIVER_ID]
    if viewer_ready:
        if viewer:
            raise ValueError(
                "Dataset Viewer waiver is stale because all configs are ready"
            )
        return observed
    if len(viewer) != 1:
        raise ValueError("pending Dataset Viewer configs require one explicit waiver")
    waiver = viewer[0]
    if (
        waiver.get("approved") is not True
        or waiver.get("evidence_url") != VIEWER_WAIVER_ISSUE
        or waiver.get("tracked_by") != VIEWER_FOLLOWUP_ISSUE
        or not isinstance(waiver.get("approved_by"), str)
        or not waiver["approved_by"].strip()
        or not isinstance(waiver.get("approved_at"), str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", waiver["approved_at"])
    ):
        raise ValueError("Dataset Viewer waiver evidence is incomplete")
    return observed


def _validated_tag_approval(
    approval: Mapping[str, Any] | None,
    *,
    workflow_commit: str,
    release_revision: str,
    hub_revision: str,
) -> dict[str, Any] | None:
    if approval is None:
        return None
    if (
        not isinstance(approval, Mapping)
        or approval.get("approved") is not True
        or approval.get("tag") != RELEASE_TAG
        or approval.get("evidence_url") != QA_APPROVAL_ISSUE
        or approval.get("workflow_commit") != workflow_commit
        or approval.get("release_revision") != release_revision
        or approval.get("hub_revision") != hub_revision
        or not isinstance(approval.get("approved_by"), str)
        or not approval["approved_by"].strip()
        or not isinstance(approval.get("approved_at"), str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approval["approved_at"])
    ):
        raise ValueError("v1.0.0 tag approval is incomplete or mismatched")
    return dict(approval)


def _validate_public_release_report(
    report: Mapping[str, Any], release_revision: str
) -> dict[str, Any]:
    validation = report.get("release_validation")
    examples = report.get("dataset_card_examples")
    discovery = report.get("dataset_viewer_discovery")
    query_checks = (
        validation.get("parquet_query_checks")
        if isinstance(validation, Mapping)
        else None
    )
    range_checks = (
        validation.get("bigwig_range_checks")
        if isinstance(validation, Mapping)
        else None
    )
    example_records = (
        examples.get("examples") if isinstance(examples, Mapping) else None
    )
    expected_example_columns = {
        "interval_filter_projection": [
            "chrom",
            "pos",
            "ref",
            "entropy_calibrated",
        ],
        "projected_score_scan": ["chrom", "pos", "entropy_calibrated"],
        "multi_chromosome_scan": [
            "chrom",
            "pos",
            "ref",
            "alt",
            "llr_calibrated",
        ],
        "variant_join": [
            "chrom",
            "pos",
            "ref",
            "alt",
            "llr_calibrated",
            "abs_llr_calibrated",
        ],
    }
    if (
        report.get("valid") is not True
        or report.get("repository") != REPOSITORY_ID
        or report.get("revision") != release_revision
        or report.get("public") is not True
        or report.get("credentials_sent") is not False
        or not isinstance(validation, Mapping)
        or validation.get("valid") is not True
        or validation.get("repository") != REPOSITORY_ID
        or validation.get("revision") != release_revision
        or validation.get("public") is not True
        or validation.get("credentials_sent") is not False
        or validation.get("checksum_file_count") != 330
        or validation.get("published_artifact_file_count") != 330
        or validation.get("bigwig_range_count") != 40
        or validation.get("dataset_card_rendered") is not True
        or not isinstance(query_checks, list)
        or len(query_checks) < 2
        or any(
            check.get("range_reader_transferred_bytes", 0) <= 0
            or check.get("range_reader_transferred_bytes", 0)
            >= check.get("object_bytes", 0)
            or check.get("polars_rows", 0) < 1
            or check.get("polars_projection_pushdown") is not True
            or check.get("polars_predicate_pushdown") is not True
            for check in query_checks
        )
        or not isinstance(range_checks, list)
        or len(range_checks) != 40
        or any(check.get("status") != 206 for check in range_checks)
        or not isinstance(examples, Mapping)
        or examples.get("valid") is not True
        or examples.get("credentials_sent") is not False
        or examples.get("root")
        != f"hf://datasets/{REPOSITORY_ID}@{release_revision}/data"
        or not isinstance(example_records, list)
        or [item.get("name") for item in example_records]
        != [
            "interval_filter_projection",
            "projected_score_scan",
            "multi_chromosome_scan",
            "variant_join",
        ]
        or any(item.get("rows", 0) < 1 for item in example_records)
        or any(
            item.get("columns") != expected_example_columns[item["name"]]
            for item in example_records
        )
        or not isinstance(discovery, Mapping)
        or discovery.get("valid") is not True
        or discovery.get("credentials_sent") is not False
        or discovery.get("split_count") != 16
        or discovery.get("configs")
        != sorted(config["config_name"] for config in dataset_configs())
        or discovery.get("preview") is not True
        or discovery.get("viewer") is not True
        or validation.get("viewer_config_count", 0)
        + len(validation.get("viewer_pending", []))
        != 16
        or (
            validation.get("viewer_ready") is True
            and (
                validation.get("viewer_config_count") != 16
                or validation.get("viewer_pending") != []
            )
        )
    ):
        raise ValueError("anonymous public release QA evidence is incomplete")
    return {
        "viewer_ready": validation.get("viewer_ready") is True,
        "viewer_config_count": validation.get("viewer_config_count"),
        "viewer_pending": validation.get("viewer_pending"),
        "checksum_file_count": validation["checksum_file_count"],
        "bigwig_range_count": validation["bigwig_range_count"],
        "parquet_query_count": len(query_checks),
        "dataset_card_example_count": len(example_records),
    }


def _validate_hub_evidence(
    public_report: Mapping[str, Any],
    committed_evidence: Mapping[str, Any],
    hub_metadata_root: Path,
    *,
    release_revision: str,
    hub_revision: str,
) -> dict[str, Any]:
    validation = public_report.get("hub_validation")
    publication = committed_evidence.get("publication")
    manual = committed_evidence.get("manual_browser_validation")
    manifest_checks = [
        item
        for item in public_report.get("file_checks", [])
        if isinstance(item, Mapping) and item.get("path") == "manifest/ucsc-hub.json"
    ]
    readme_path = hub_metadata_root / "README.md"
    if not readme_path.is_file():
        raise ValueError("hub metadata lacks the public dataset card")
    readme = readme_path.read_text(encoding="utf-8")
    if any(marker not in readme for marker in _BOUNDED_JOIN_MARKERS):
        raise ValueError(
            "public dataset card still contains the unbounded genome-wide join"
        )
    readme_checks = [
        item
        for item in public_report.get("file_checks", [])
        if isinstance(item, Mapping) and item.get("path") == "README.md"
    ]
    expected_score_sets = {score_set.name for score_set in SCORE_SETS}
    if (
        public_report.get("valid") is not True
        or public_report.get("repository") != REPOSITORY_ID
        or public_report.get("revision") != hub_revision
        or public_report.get("public") is not True
        or public_report.get("credentials_sent") is not False
        or public_report.get("file_count") != 35
        or not isinstance(validation, Mapping)
        or validation.get("valid") is not True
        or validation.get("artifact_revision") != release_revision
        or validation.get("track_count") != 40
        or validation.get("score_set_count") != 8
        or validation.get("assembly_count") != 6
        or validation.get("http_range_count") != 40
        or len(validation.get("chromosome_checks", [])) != 40
        or len(validation.get("representative_checks", [])) != 8
        or validation.get("hub_check", {}).get("passed") is not True
        or not isinstance(publication, Mapping)
        or publication.get("public_revision") != hub_revision
        or publication.get("artifact_revision") != release_revision
        or publication.get("public_validation_valid") is not True
        or publication.get("credentials_sent_during_validation") is not False
        or len(manifest_checks) != 1
        or manifest_checks[0].get("sha256") != publication.get("hub_manifest_sha256")
        or len(readme_checks) != 1
        or readme_checks[0].get("sha256") != sha256_file(readme_path)
        or not isinstance(manual, Mapping)
        or manual.get("status") != "passed"
        or manual.get("failed") != []
        or set(manual.get("passed_base_and_zoom", [])) != expected_score_sets
    ):
        raise ValueError("public and manual UCSC hub evidence is incomplete")
    return {
        "metadata_file_count": public_report["file_count"],
        "track_count": validation["track_count"],
        "score_set_count": validation["score_set_count"],
        "hub_manifest_sha256": manifest_checks[0]["sha256"],
        "dataset_card_sha256": readme_checks[0]["sha256"],
        "manual_browser_score_set_count": len(manual["passed_base_and_zoom"]),
    }


def _positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _percentage(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0 <= value <= 100
    )


def _measurement_range(value: Any, *, allow_zero: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    lower, upper = value
    valid = (
        _percentage(lower) and _percentage(upper)
        if allow_zero
        else _positive_number(lower) and _positive_number(upper)
    )
    return valid and lower <= upper


def _validate_execution_evidence(
    scf_evidence: Mapping[str, Any],
    bigwig_evidence: Mapping[str, Any],
    *,
    scf_evidence_sha256: str,
    bigwig_evidence_sha256: str,
) -> dict[str, Any]:
    jobs = scf_evidence.get("jobs")
    scheduler = bigwig_evidence.get("scheduler_efficiency")
    production = bigwig_evidence.get("production")
    if (
        scf_evidence.get("result") != "completed"
        or scf_evidence.get("profile") != "workflow/profiles/scf/config.yaml"
        or not isinstance(jobs, list)
        or len(jobs) != 2
        or any(
            job.get("state") != "COMPLETED"
            or not isinstance(job.get("job_id"), str)
            or not job["job_id"].strip()
            or job.get("exit_code") != "0:0"
            or not _positive_number(job.get("elapsed_seconds"))
            or not _positive_number(job.get("step_max_rss_kib"))
            or not _positive_number(job.get("allocated_cpus"))
            or not _percentage(job.get("cpu_efficiency_percent"))
            or not _percentage(job.get("memory_usage_percent"))
            for job in jobs
        )
        or bigwig_evidence.get("status") != "complete"
        or not isinstance(production, Mapping)
        or production.get("valid") is not True
        or production.get("track_count") != 40
        or not isinstance(scheduler, Mapping)
        or not {
            "benchmark_jobs",
            "p243_chromosome_builds",
            "p243_finalizers",
            "expanded_audit_pilot",
            "expanded_audit_production",
        }.issubset(scheduler)
    ):
        raise ValueError("SCF and Slurm efficiency evidence is incomplete")
    for section_name in (
        "benchmark_jobs",
        "p243_chromosome_builds",
        "p243_finalizers",
    ):
        section = scheduler[section_name]
        if (
            not isinstance(section, Mapping)
            or not _measurement_range(
                section.get("cpu_efficiency_percent_range"), allow_zero=True
            )
            or not _measurement_range(
                section.get("memory_usage_percent_range"), allow_zero=True
            )
        ):
            raise ValueError(f"invalid Slurm efficiency section: {section_name}")
    pilot = scheduler["expanded_audit_pilot"]
    expanded = scheduler["expanded_audit_production"]
    if (
        not isinstance(pilot, Mapping)
        or not _positive_number(pilot.get("job_id"))
        or not _positive_number(pilot.get("elapsed_seconds"))
        or not _positive_number(pilot.get("peak_rss_mib"))
        or not _percentage(pilot.get("cpu_efficiency_percent"))
        or not _percentage(pilot.get("memory_usage_percent"))
        or not isinstance(expanded, Mapping)
        or not _positive_number(expanded.get("job_count"))
        or not _measurement_range(expanded.get("elapsed_seconds_range"))
        or not _measurement_range(expanded.get("peak_rss_mib_range"))
        or not _measurement_range(
            expanded.get("cpu_efficiency_percent_range"), allow_zero=True
        )
        or not _measurement_range(
            expanded.get("memory_usage_percent_range"), allow_zero=True
        )
    ):
        raise ValueError("expanded-audit Slurm efficiency evidence is incomplete")
    return {
        "scf_smoke_run_id": scf_evidence.get("workflow_run_id"),
        "scf_evidence_sha256": scf_evidence_sha256,
        "scf_smoke_jobs": [
            {
                field: job[field]
                for field in (
                    "job_id",
                    "state",
                    "exit_code",
                    "elapsed_seconds",
                    "step_max_rss_kib",
                    "allocated_cpus",
                    "cpu_efficiency_percent",
                    "memory_usage_percent",
                )
            }
            for job in jobs
        ],
        "bigwig_evidence_sha256": bigwig_evidence_sha256,
        "production_track_count": production["track_count"],
        "production_final_bytes": production.get("final_bytes"),
        "inventory_manifest_sha256": production.get("inventory_manifest_sha256"),
        "scheduler_efficiency": dict(scheduler),
    }


def render_release_record(record: Mapping[str, Any]) -> str:
    """Render the human-readable companion to the machine release record."""

    status = "READY TO TAG" if record["ready_to_tag"] else "TAG APPROVAL PENDING"
    viewer = record["public_release"]["viewer_ready"]
    lines = [
        "# GPN-Star v1.0.0 release record",
        "",
        f"Status: **{status}**",
        "",
        "## Immutable identifiers",
        "",
        f"- Source inventory SHA-256: `{record['source']['inventory']['manifest_sha256']}`",
        f"- Release manifest SHA-256: `{record['source']['release_manifest_sha256']}`",
        f"- Hugging Face artifact revision: `{record['release_revision']}`",
        f"- Hugging Face hub revision: `{record['hub_revision']}`",
        f"- GitHub workflow commit: `{record['workflow_commit']}`",
        f"- uv.lock SHA-256: `{record['environment']['uv_lock_sha256']}`",
        f"- SCF profile SHA-256: `{record['environment']['scf_profile_sha256']}`",
        "",
        "## QA result",
        "",
        f"- Source Parquet shards: {record['source']['parquet_file_count']} exact identities",
        f"- Source schemas/content checks: {record['source']['inventory']['all_source_checks_passed']}",
        f"- Final BigWigs: {record['source']['bigwig_file_count']} exact identities",
        f"- Public HTTP BigWig ranges: {record['public_release']['bigwig_range_count']}",
        f"- Dataset-card Polars examples: {record['public_release']['dataset_card_example_count']}",
        f"- Dataset Viewer ready: {viewer}",
        f"- UCSC model groups rendered at base and zoom scales: {record['public_hub']['manual_browser_score_set_count']}",
        "- Credential-free public validation: yes",
        f"- SCF smoke evidence SHA-256: `{record['execution']['scf_evidence_sha256']}`",
        f"- BigWig production evidence SHA-256: `{record['execution']['bigwig_evidence_sha256']}`",
        "",
        "## Known limitations and waivers",
        "",
    ]
    for job in record["execution"]["scf_smoke_jobs"]:
        lines.insert(
            -3,
            (
                f"- SCF job `{job['job_id']}`: {job['elapsed_seconds']} s, "
                f"{job['step_max_rss_kib']} KiB peak RSS, "
                f"{job['cpu_efficiency_percent']:.2f}% CPU efficiency, "
                f"{job['memory_usage_percent']:.2f}% memory usage"
            ),
        )
    for limitation in record["known_limitations"]:
        lines.append(
            f"- `{limitation['id']}`: {limitation['description']} "
            f"({limitation['evidence_url']})"
        )
    for waiver in record["waivers"]:
        lines.append(
            f"- Waiver `{waiver['id']}` approved by `{waiver['approved_by']}` "
            f"on {waiver['approved_at']}: {waiver['evidence_url']}"
        )
    capacity = record["source"]["capacity_waiver"]
    lines.append(
        f"- Waiver `{CAPACITY_WAIVER_ID}` approved by `{capacity['approved_by']}` "
        f"on {capacity['approved_at']}: {capacity['evidence_url']}"
    )
    lines.extend(
        [
            "",
            "## Tag gate",
            "",
            (
                f"Author approval is recorded for `{RELEASE_TAG}`."
                if record["tag_approval"] is not None
                else (
                    f"Do not create or publish `{RELEASE_TAG}` until a GPN-Star "
                    f"author records approval on {QA_APPROVAL_ISSUE}."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_release_record(
    metadata_root: str | Path,
    hub_metadata_root: str | Path,
    public_release_report_path: str | Path,
    public_hub_report_path: str | Path,
    scf_evidence_path: str | Path,
    bigwig_evidence_path: str | Path,
    hub_evidence_path: str | Path,
    uv_lock_path: str | Path,
    scf_profile_path: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
    *,
    release_revision: str,
    hub_revision: str,
    workflow_commit: str,
    waivers: Sequence[Mapping[str, Any]],
    known_limitations: Sequence[Mapping[str, Any]],
    tag_approval: Mapping[str, Any] | None = None,
) -> None:
    """Validate the complete evidence chain and atomically write the v1 record."""

    release_revision = _require_sha(release_revision, field="release revision")
    hub_revision = _require_sha(hub_revision, field="hub revision")
    workflow_commit = _require_sha(workflow_commit, field="workflow commit")
    source = _validate_release_chain(Path(metadata_root))
    public_release = _validate_public_release_report(
        _read_json(public_release_report_path), release_revision
    )
    validated_waivers = _validated_viewer_waiver(
        waivers, viewer_ready=public_release["viewer_ready"]
    )
    public_hub = _validate_hub_evidence(
        _read_json(public_hub_report_path),
        _read_json(hub_evidence_path),
        Path(hub_metadata_root),
        release_revision=release_revision,
        hub_revision=hub_revision,
    )
    hub_evidence = _read_json(hub_evidence_path)
    if hub_evidence.get("release_manifest_sha256") != source["release_manifest_sha256"]:
        raise ValueError("hub evidence references a different release manifest")
    execution = _validate_execution_evidence(
        _read_json(scf_evidence_path),
        _read_json(bigwig_evidence_path),
        scf_evidence_sha256=sha256_file(Path(scf_evidence_path)),
        bigwig_evidence_sha256=sha256_file(Path(bigwig_evidence_path)),
    )
    if (
        execution["inventory_manifest_sha256"] != source["inventory"]["manifest_sha256"]
        or execution["production_final_bytes"]
        != _read_json(Path(metadata_root) / "manifest" / "release.json")["bigwig"][
            "total_bytes"
        ]
    ):
        raise ValueError("production execution evidence references different artifacts")
    limitations = []
    for limitation in known_limitations:
        if (
            not isinstance(limitation, Mapping)
            or not isinstance(limitation.get("id"), str)
            or not isinstance(limitation.get("description"), str)
            or not limitation["description"].strip()
            or not isinstance(limitation.get("evidence_url"), str)
            or not limitation["evidence_url"].startswith("https://")
        ):
            raise ValueError(
                "known limitations require id, description, and HTTPS evidence"
            )
        limitations.append(dict(limitation))
    limitation_ids = [item["id"] for item in limitations]
    if len(limitation_ids) != len(set(limitation_ids)):
        raise ValueError("known limitation ids must be unique")
    approval = _validated_tag_approval(
        tag_approval,
        workflow_commit=workflow_commit,
        release_revision=release_revision,
        hub_revision=hub_revision,
    )
    record = {
        "release_record_version": 1,
        "valid": True,
        "tag": RELEASE_TAG,
        "ready_to_tag": approval is not None,
        "repository": REPOSITORY_ID,
        "release_revision": release_revision,
        "hub_revision": hub_revision,
        "workflow_commit": workflow_commit,
        "source": source,
        "public_release": public_release,
        "public_hub": public_hub,
        "environment": {
            "uv_lock_sha256": sha256_file(Path(uv_lock_path)),
            "scf_profile": "workflow/profiles/scf/config.yaml",
            "scf_profile_sha256": sha256_file(Path(scf_profile_path)),
        },
        "execution": execution,
        "known_limitations": limitations,
        "waivers": validated_waivers,
        "tag_approval": approval,
    }
    _atomic_write_json(Path(output_json), record)
    _atomic_write_text(Path(output_markdown), render_release_record(record))


def create_release_tag(
    release_record_path: str | Path,
    repository_root: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Create the approved local annotated tag at the exact QA workflow commit."""

    record = _read_json(release_record_path)
    if (
        record.get("release_record_version") != 1
        or record.get("valid") is not True
        or record.get("ready_to_tag") is not True
        or record.get("tag") != RELEASE_TAG
        or record.get("repository") != REPOSITORY_ID
        or not isinstance(record.get("tag_approval"), Mapping)
    ):
        raise ValueError("release record is not author-approved for v1.0.0")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "release tagging must run from one intentional non-Slurm process"
        )
    workflow_commit = _require_sha(
        record.get("workflow_commit"), field="workflow commit"
    )
    release_revision = _require_sha(
        record.get("release_revision"), field="release revision"
    )
    hub_revision = _require_sha(record.get("hub_revision"), field="hub revision")
    _validated_tag_approval(
        record["tag_approval"],
        workflow_commit=workflow_commit,
        release_revision=release_revision,
        hub_revision=hub_revision,
    )
    root = Path(repository_root)
    head = runner(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != workflow_commit:
        raise RuntimeError("HEAD differs from the author-approved workflow commit")
    status = runner(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("working tree must be clean before release tagging")
    existing = runner(
        ["git", "tag", "--list", RELEASE_TAG],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if existing:
        target = runner(
            ["git", "rev-list", "-n", "1", RELEASE_TAG],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if target != head:
            raise RuntimeError(f"existing {RELEASE_TAG} points to a different commit")
        return
    annotation = "\n".join(
        [
            "GPN-Star genome-wide score release v1.0.0",
            "",
            f"Workflow commit: {workflow_commit}",
            f"Hugging Face artifact revision: {release_revision}",
            f"Hugging Face hub revision: {hub_revision}",
            (
                "Source inventory SHA-256: "
                f"{record['source']['inventory']['manifest_sha256']}"
            ),
            (
                "Release manifest SHA-256: "
                f"{record['source']['release_manifest_sha256']}"
            ),
            f"Release record SHA-256: {sha256_file(Path(release_record_path))}",
            f"Approval: {record['tag_approval']['evidence_url']}",
        ]
    )
    runner(
        [
            "git",
            "tag",
            "--annotate",
            RELEASE_TAG,
            "--message",
            annotation,
            head,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    hub = commands.add_parser("validate-hub")
    hub.add_argument("--metadata-root", type=Path, required=True)
    hub.add_argument("--report", type=Path, required=True)
    hub.add_argument("--revision", required=True)
    hub.add_argument("--udc-dir", type=Path, required=True)
    tag = commands.add_parser("tag")
    tag.add_argument("--release-record", type=Path, required=True)
    tag.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate-hub":
        validate_public_hub_for_qa(
            args.metadata_root,
            args.report,
            revision=args.revision,
            udc_dir=args.udc_dir,
        )
        return
    if args.command == "tag":
        create_release_tag(args.release_record, args.repository_root)
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
