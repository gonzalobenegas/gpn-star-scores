"""Generate and validate the post-v1 raw calibrated-LLR BigWig extension."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig

from gpn_star_scores.bigwig import (
    BASES,
    ChromosomeSpec,
    iter_raw_llr_track_batches,
    validate_bigwig,
    write_raw_llr_bigwigs,
)
from gpn_star_scores.catalog import (
    ASSEMBLIES,
    SCORE_SETS,
    get_shard_spec,
    score_set_assembly,
)
from gpn_star_scores.inventory import atomic_write_json, sha256_file
from gpn_star_scores.tracks import (
    assembly_chromosome_sizes_from_contract,
    chromosome_spec_from_contract,
    load_track_input_contract,
    stream_concatenate_bigwigs,
    ucsc_assembly_name,
    ucsc_chromosome_name,
)

RAW_LLR_TRACKS = tuple(f"llr_{base}" for base in BASES)
RAW_LLR_BASES = dict(zip(RAW_LLR_TRACKS, BASES, strict=True))
VALUE_DECIMALS = 3


def build_raw_llr_chromosome(
    source_root: str | Path,
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    track_selection_path: str | Path,
    output_paths: Mapping[str, str | Path],
    report_path: str | Path,
    *,
    score_set: str,
    chrom: str,
    batch_size: int = 262_144,
    sample_count: int = 1_024,
) -> None:
    """Build and validate four raw-LLR BigWigs for one chromosome."""

    outputs = _validated_outputs(output_paths)
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    _validate_direct_selection(
        track_selection_path, expected_manifest_sha256=contract.manifest_sha256
    )
    chromosome = chromosome_spec_from_contract(contract, score_set, chrom)
    header_sizes = assembly_chromosome_sizes_from_contract(contract, score_set)
    source, expected_positions = _validated_llr_source(
        Path(source_root), contract.records, score_set, chrom
    )

    generated_by_base = {base: outputs[f"llr_{base}"] for base in BASES}
    stats = write_raw_llr_bigwigs(
        [source],
        generated_by_base,
        chromosome,
        batch_size=batch_size,
        header_chromosome_sizes=header_sizes,
    )
    validation = validate_raw_llr_chromosome(
        source,
        outputs,
        chromosome=chromosome,
        expected_position_count=expected_positions,
        sample_count=sample_count,
        sample_seed=15,
        batch_size=batch_size,
    )
    atomic_write_json(
        Path(report_path),
        {
            "report_version": 1,
            "product": "raw_calibrated_llr",
            "valid": True,
            "method": "direct",
            "score_set": score_set,
            "assembly": score_set_assembly(score_set),
            "ucsc_assembly": ucsc_assembly_name(score_set_assembly(score_set)),
            "chromosome": asdict(chromosome),
            "inventory_manifest_sha256": contract.manifest_sha256,
            "stats": asdict(stats),
            "validation": validation,
        },
    )


def validate_raw_llr_chromosome(
    source_path: str | Path,
    output_paths: Mapping[str, str | Path],
    *,
    chromosome: ChromosomeSpec,
    expected_position_count: int,
    sample_count: int = 1_024,
    sample_seed: int = 0,
    batch_size: int = 262_144,
) -> dict[str, Any]:
    """Compare deterministic raw-LLR source samples with four BigWigs."""

    outputs = _validated_outputs(output_paths)
    if expected_position_count <= 0:
        raise ValueError("expected_position_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected_indices = _sample_indices(
        expected_position_count, sample_count, sample_seed
    )
    samples: dict[str, list[dict[str, Any]]] = {track: [] for track in RAW_LLR_TRACKS}
    sign_counts = {"negative": 0, "zero": 0, "positive": 0}
    observed_positions = 0
    first_position: int | None = None
    last_position: int | None = None
    first_gap: int | None = None

    open_bigwigs = {
        track: pyBigWig.open(str(outputs[track])) for track in RAW_LLR_TRACKS
    }
    if any(bigwig is None for bigwig in open_bigwigs.values()):
        for bigwig in open_bigwigs.values():
            if bigwig is not None:
                bigwig.close()
        raise ValueError("could not open every raw-LLR BigWig")

    try:
        sample_pointer = 0
        for positions, values in iter_raw_llr_track_batches(
            [source_path], chromosome, batch_size=batch_size
        ):
            if first_position is None:
                first_position = int(positions[0])
            if last_position is not None and first_gap is None:
                if int(positions[0]) > last_position + 1:
                    first_gap = last_position + 1
            differences = np.diff(positions)
            if first_gap is None and np.any(differences > 1):
                gap_index = int(np.flatnonzero(differences > 1)[0])
                first_gap = int(positions[gap_index]) + 1

            sign_counts["negative"] += int(np.count_nonzero(values < 0))
            sign_counts["zero"] += int(np.count_nonzero(values == 0))
            sign_counts["positive"] += int(np.count_nonzero(values > 0))
            batch_end = observed_positions + len(positions)
            while (
                sample_pointer < len(selected_indices)
                and selected_indices[sample_pointer] < batch_end
            ):
                local_index = int(selected_indices[sample_pointer] - observed_positions)
                position = int(positions[local_index])
                for track in RAW_LLR_TRACKS:
                    base_index = BASES.index(RAW_LLR_BASES[track])
                    expected = np.float32(values[local_index, base_index])
                    observed = np.float32(
                        open_bigwigs[track].values(
                            chromosome.ucsc_name, position - 1, position
                        )[0]
                    )
                    if not _float32_exact(expected, observed):
                        raise ValueError(
                            f"{track} differs at {chromosome.ucsc_name}:{position}: "
                            f"{observed!r} != {expected!r}"
                        )
                    samples[track].append(
                        {
                            "position_1based": position,
                            "expected_float32": float(expected),
                            "observed_float32": float(observed),
                        }
                    )
                sample_pointer += 1
            observed_positions = batch_end
            last_position = int(positions[-1])

        if observed_positions != expected_position_count:
            raise ValueError(
                f"observed {observed_positions} positions; expected "
                f"{expected_position_count}"
            )
        if sample_pointer != len(selected_indices):
            raise AssertionError("not every requested sample was observed")
        if first_position is None or last_position is None:
            raise ValueError("source contains no positions")

        gap_checks: dict[str, bool] = {}
        if first_gap is not None:
            for track in RAW_LLR_TRACKS:
                gap_value = open_bigwigs[track].values(
                    chromosome.ucsc_name, first_gap - 1, first_gap
                )[0]
                if not np.isnan(gap_value):
                    raise ValueError(
                        f"{track} unexpectedly covers source gap at {first_gap}"
                    )
                gap_checks[track] = True
        return {
            "source_path": str(source_path),
            "source_chrom": chromosome.source_name,
            "ucsc_chrom": chromosome.ucsc_name,
            "chromosome_length": chromosome.length,
            "position_count": observed_positions,
            "first_position": first_position,
            "last_position": last_position,
            "sample_seed": sample_seed,
            "sample_count": len(selected_indices),
            "float32_exact": True,
            "reference_zero_baseline": True,
            "abs_llr_calibrated_used": False,
            "sign_counts": sign_counts,
            "first_gap_position": first_gap,
            "gap_checks": gap_checks,
            "samples": samples,
        }
    finally:
        for bigwig in open_bigwigs.values():
            bigwig.close()


def concatenate_raw_llr_bigwig(
    input_paths: Sequence[str | Path],
    chromosome_report_paths: Sequence[str | Path],
    output_path: str | Path,
    report_path: str | Path,
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    *,
    score_set: str,
    track: str,
    value_decimals: int = VALUE_DECIMALS,
    bigwig_info: str = "bigWigInfo",
) -> None:
    """Repack chromosome artifacts and validate one final raw-LLR track."""

    _require_track(track)
    _require_value_decimals(value_decimals)
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    assembly = score_set_assembly(score_set)
    chromosomes = ASSEMBLIES[assembly].chromosomes
    inputs = [Path(path) for path in input_paths]
    reports = [Path(path) for path in chromosome_report_paths]
    if len(inputs) != len(chromosomes) or len(reports) != len(chromosomes):
        raise ValueError("one input and validation report are required per chromosome")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    temporary = temporary_dir / output.name
    try:
        stream_concatenate_bigwigs(
            inputs,
            temporary,
            assembly_chromosome_sizes_from_contract(contract, score_set),
            [ucsc_chromosome_name(chrom) for chrom in chromosomes],
            value_decimals=value_decimals,
        )
        payload = _final_validation_payload(
            temporary,
            reports,
            inventory_manifest_path,
            parquet_selection_path,
            score_set=score_set,
            track=track,
            value_decimals=value_decimals,
            validation_stage="pre-promotion",
            bigwig_info=bigwig_info,
        )
        os.replace(temporary, output)
        atomic_write_json(Path(report_path), payload)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def audit_final_raw_llr_bigwig(
    bigwig_path: str | Path,
    chromosome_report_paths: Sequence[str | Path],
    concatenation_report_path: str | Path,
    report_path: str | Path,
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    *,
    score_set: str,
    track: str,
    value_decimals: int = VALUE_DECIMALS,
    bigwig_info: str = "bigWigInfo",
) -> None:
    """Audit one existing final raw-LLR BigWig without rebuilding it."""

    _require_track(track)
    concatenation = _read_json(concatenation_report_path)
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    if (
        concatenation.get("report_version") != 1
        or concatenation.get("product") != "raw_calibrated_llr"
        or concatenation.get("valid") is not True
        or concatenation.get("validation_stage") != "pre-promotion"
        or concatenation.get("score_set") != score_set
        or concatenation.get("track") != track
        or concatenation.get("value_decimals") != value_decimals
        or concatenation.get("inventory_manifest_sha256") != contract.manifest_sha256
    ):
        raise ValueError("invalid concatenation report for raw-LLR audit")
    path = Path(bigwig_path)
    if path.stat().st_size != concatenation.get("size") or sha256_file(
        path
    ) != concatenation.get("sha256"):
        raise ValueError("final raw-LLR identity differs from concatenation report")

    payload = _final_validation_payload(
        path,
        chromosome_report_paths,
        inventory_manifest_path,
        parquet_selection_path,
        score_set=score_set,
        track=track,
        value_decimals=value_decimals,
        validation_stage="post-assembly-audit",
        bigwig_info=bigwig_info,
    )
    atomic_write_json(Path(report_path), payload)


def aggregate_raw_llr_validation(
    final_report_paths: Sequence[str | Path],
    track_selection_path: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
) -> None:
    """Require one audited final report for each of the 32 new BigWigs."""

    reports = [_read_json(path) for path in final_report_paths]
    expected = {
        (score_set.name, track) for score_set in SCORE_SETS for track in RAW_LLR_TRACKS
    }
    observed = {(report.get("score_set"), report.get("track")) for report in reports}
    if observed != expected or len(reports) != len(expected):
        raise ValueError("raw-LLR reports do not cover the exact 32-track catalog")
    invalid = [
        f"{report.get('score_set')}/{report.get('track')}"
        for report in reports
        if report.get("report_version") != 1
        or report.get("product") != "raw_calibrated_llr"
        or report.get("valid") is not True
        or report.get("validation_stage") != "post-assembly-audit"
        or report.get("value_decimals") != VALUE_DECIMALS
        or report.get("reference_zero_baseline") is not True
        or report.get("abs_llr_calibrated_used") is not False
    ]
    if invalid:
        raise ValueError(f"invalid raw-LLR audit reports: {invalid!r}")
    manifest_hashes = {
        str(report.get("inventory_manifest_sha256")) for report in reports
    }
    if len(manifest_hashes) != 1:
        raise ValueError("raw-LLR reports use different inventory manifests")
    manifest_sha256 = next(iter(manifest_hashes))
    _validate_direct_selection(
        track_selection_path, expected_manifest_sha256=manifest_sha256
    )

    source_matrix_sign_counts = {"negative": 0, "zero": 0, "positive": 0}
    for score_set in SCORE_SETS:
        score_set_reports = [
            report for report in reports if report["score_set"] == score_set.name
        ]
        expected_sign_counts: dict[str, int] | None = None
        for report in score_set_reports:
            track = str(report["track"])
            summary = report.get("summary")
            sign_counts = report.get("source_matrix_sign_counts")
            if (
                report.get("assembly") != score_set.assembly
                or report.get("ucsc_assembly") != ucsc_assembly_name(score_set.assembly)
                or report.get("base") != RAW_LLR_BASES[track]
                or not isinstance(report.get("size"), int)
                or isinstance(report["size"], bool)
                or report["size"] <= 0
                or not isinstance(report.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", report["sha256"])
                or not isinstance(summary, Mapping)
                or not isinstance(summary.get("bases_covered"), int)
                or summary["bases_covered"] <= 0
                or not isinstance(summary.get("zoom_levels"), int)
                or summary["zoom_levels"] < 1
                or not isinstance(report.get("sample_check_count"), int)
                or report["sample_check_count"] <= 0
                or not isinstance(report.get("gap_check_count"), int)
                or report["gap_check_count"] < 0
                or not _valid_sign_counts(sign_counts)
            ):
                raise ValueError(
                    f"invalid raw-LLR release evidence: {score_set.name}/{track}"
                )
            normalized_sign_counts = {
                name: int(sign_counts[name]) for name in source_matrix_sign_counts
            }
            if expected_sign_counts is None:
                expected_sign_counts = normalized_sign_counts
            elif normalized_sign_counts != expected_sign_counts:
                raise ValueError(
                    f"source sign counts differ between tracks for {score_set.name}"
                )
        assert expected_sign_counts is not None
        for name, value in expected_sign_counts.items():
            source_matrix_sign_counts[name] += value

    tracks = sorted(
        [
            {
                "score_set": report["score_set"],
                "assembly": report["assembly"],
                "ucsc_assembly": report["ucsc_assembly"],
                "track": report["track"],
                "base": RAW_LLR_BASES[report["track"]],
                "path": f"bigwig/{report['score_set']}/{report['track']}.bw",
                "size": report["size"],
                "sha256": report["sha256"],
                "bases_covered": report["summary"]["bases_covered"],
                "zoom_levels": report["summary"]["zoom_levels"],
                "sample_check_count": report["sample_check_count"],
                "gap_check_count": report["gap_check_count"],
            }
            for report in reports
        ],
        key=lambda item: (item["score_set"], item["track"]),
    )
    payload = {
        "report_version": 1,
        "product": "raw_calibrated_llr",
        "valid": True,
        "track_count": len(tracks),
        "selected_method": "direct",
        "value_decimals": VALUE_DECIMALS,
        "reference_zero_baseline": True,
        "abs_llr_calibrated_used": False,
        "source_matrix_sign_counts": source_matrix_sign_counts,
        "inventory_manifest_sha256": manifest_sha256,
        "total_bytes": sum(int(record["size"]) for record in tracks),
        "sample_check_count": sum(
            int(record["sample_check_count"]) for record in tracks
        ),
        "gap_check_count": sum(int(record["gap_check_count"]) for record in tracks),
        "tracks": tracks,
    }
    atomic_write_json(Path(output_json), payload)
    _atomic_write_text(
        Path(output_markdown),
        "\n".join(
            [
                "# Raw calibrated-LLR BigWig validation",
                "",
                "Status: **valid**",
                "",
                f"Validated new tracks: {payload['track_count']}",
                "",
                f"Incremental bytes: {payload['total_bytes']}",
                "",
                f"Validated sampled values: {payload['sample_check_count']}",
                "",
                f"Validated source gaps: {payload['gap_check_count']}",
                "",
                "Only the 32 post-v1 raw calibrated-LLR BigWigs are included. "
                "The immutable v1 BigWigs are not revalidated.",
                "",
            ]
        ),
    )


def _final_validation_payload(
    bigwig_path: str | Path,
    chromosome_report_paths: Sequence[str | Path],
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    *,
    score_set: str,
    track: str,
    value_decimals: int,
    validation_stage: str,
    bigwig_info: str,
) -> dict[str, Any]:
    _require_track(track)
    _require_value_decimals(value_decimals)
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    assembly = score_set_assembly(score_set)
    chromosomes = ASSEMBLIES[assembly].chromosomes
    reports = [_read_json(path) for path in chromosome_report_paths]
    if len(reports) != len(chromosomes):
        raise ValueError("one raw-LLR validation report is required per chromosome")

    expected_sizes = assembly_chromosome_sizes_from_contract(contract, score_set)
    expected_bases = sum(
        _llr_position_count(contract.records, score_set, chrom) for chrom in chromosomes
    )
    path = Path(bigwig_path)
    summary = validate_bigwig(
        path,
        expected_sizes,
        expected_bases_covered=expected_bases,
    )
    info = subprocess.run(
        [bigwig_info, str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    zoom_match = re.search(r"^zoomLevels:\s*(\d+)\s*$", info, re.MULTILINE)
    if zoom_match is None or int(zoom_match.group(1)) < 1:
        raise ValueError("bigWigInfo reports no zoom levels")

    sample_check_count = 0
    gap_check_count = 0
    chromosome_checks = []
    source_matrix_sign_counts = {"negative": 0, "zero": 0, "positive": 0}
    bigwig = pyBigWig.open(str(path))
    if bigwig is None or not bigwig.isBigWig():
        raise ValueError(f"not a readable final BigWig: {path}")
    try:
        for chrom, report in zip(chromosomes, reports, strict=True):
            chromosome_record = report.get("chromosome")
            stats = report.get("stats")
            validation = report.get("validation")
            expected_position_count = _llr_position_count(
                contract.records, score_set, chrom
            )
            sign_counts = (
                validation.get("sign_counts")
                if isinstance(validation, Mapping)
                else None
            )
            if (
                report.get("report_version") != 1
                or report.get("product") != "raw_calibrated_llr"
                or report.get("valid") is not True
                or report.get("method") != "direct"
                or report.get("score_set") != score_set
                or report.get("inventory_manifest_sha256") != contract.manifest_sha256
                or not isinstance(chromosome_record, Mapping)
                or chromosome_record.get("source_name") != chrom
                or chromosome_record.get("ucsc_name") != ucsc_chromosome_name(chrom)
                or not isinstance(stats, Mapping)
                or stats.get("position_count") != expected_position_count
                or not isinstance(validation, Mapping)
                or validation.get("position_count") != expected_position_count
                or validation.get("float32_exact") is not True
                or validation.get("reference_zero_baseline") is not True
                or validation.get("abs_llr_calibrated_used") is not False
                or not _valid_sign_counts(
                    sign_counts,
                    expected_total=expected_position_count * len(BASES),
                )
            ):
                raise ValueError(f"invalid raw-LLR chromosome report for {chrom}")
            samples_by_track = validation.get("samples")
            if not isinstance(samples_by_track, Mapping) or set(
                samples_by_track
            ) != set(RAW_LLR_TRACKS):
                raise ValueError(
                    f"chromosome report has an invalid sample catalog for {chrom}"
                )
            samples = (
                samples_by_track.get(track)
                if isinstance(samples_by_track, Mapping)
                else None  # pragma: no cover - guarded above
            )
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"chromosome report has no {track} samples")
            if validation.get("sample_count") != len(samples):
                raise ValueError(
                    f"sample count differs for {score_set}/{chrom}/{track}"
                )

            positions = [
                int(sample["position_1based"])
                for sample in samples
                if isinstance(sample, Mapping)
            ]
            if len(positions) != len(samples):
                raise ValueError(f"malformed samples for {score_set}/{chrom}/{track}")
            first_position = int(validation["first_position"])
            last_position = int(validation["last_position"])
            if first_position not in positions or last_position not in positions:
                raise ValueError(f"samples omit a chromosome boundary for {chrom}")

            ucsc_chrom = ucsc_chromosome_name(chrom)
            for sample, position in zip(samples, positions, strict=True):
                observed = np.float32(
                    bigwig.values(ucsc_chrom, position - 1, position)[0]
                )
                expected = _round_float32(
                    np.float32(sample["expected_float32"]), value_decimals
                )
                if not _float32_exact(expected, observed):
                    raise ValueError(
                        f"final {track} differs at {ucsc_chrom}:{position}"
                    )
            sample_check_count += len(samples)
            for name in source_matrix_sign_counts:
                source_matrix_sign_counts[name] += int(sign_counts[name])

            first_gap = validation.get("first_gap_position")
            gap_absent: bool | None = None
            if first_gap is not None:
                gap_checks = validation.get("gap_checks")
                if (
                    not isinstance(gap_checks, Mapping)
                    or gap_checks.get(track) is not True
                ):
                    raise ValueError(f"chromosome report lacks {track} gap evidence")
                gap_position = int(first_gap)
                value = bigwig.values(ucsc_chrom, gap_position - 1, gap_position)[0]
                if not np.isnan(value):
                    raise ValueError(
                        f"final {track} covers gap at {ucsc_chrom}:{gap_position}"
                    )
                gap_absent = True
                gap_check_count += 1
            chromosome_checks.append(
                {
                    "chrom": ucsc_chrom,
                    "sample_count": len(samples),
                    "first_position_1based": first_position,
                    "last_position_1based": last_position,
                    "first_gap_position_1based": first_gap,
                    "gap_absent": gap_absent,
                }
            )
    finally:
        bigwig.close()

    return {
        "report_version": 1,
        "product": "raw_calibrated_llr",
        "valid": True,
        "validation_stage": validation_stage,
        "score_set": score_set,
        "assembly": assembly,
        "ucsc_assembly": ucsc_assembly_name(assembly),
        "track": track,
        "base": RAW_LLR_BASES[track],
        "value_decimals": value_decimals,
        "reference_zero_baseline": True,
        "abs_llr_calibrated_used": False,
        "source_matrix_sign_counts": source_matrix_sign_counts,
        "chromosome_method": "direct",
        "concatenation_method": "pyBigWig-stream-copy",
        "input_count": len(reports),
        "inventory_manifest_sha256": contract.manifest_sha256,
        "summary": asdict(summary),
        "bigWigInfo": info,
        "sample_check_count": sample_check_count,
        "gap_check_count": gap_check_count,
        "chromosome_checks": chromosome_checks,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validated_llr_source(
    source_root: Path,
    records: Mapping[str, Mapping[str, Any]],
    score_set: str,
    chrom: str,
) -> tuple[Path, int]:
    spec = get_shard_spec(score_set, "llr", chrom)
    relative = spec.relative_path.as_posix()
    record = records[relative]
    source = source_root / spec.relative_path
    expected_size = record.get("size")
    if (
        not source.is_file()
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or source.stat().st_size != expected_size
    ):
        raise ValueError(f"LLR source identity differs from inventory: {relative}")
    return source, _llr_position_count(records, score_set, chrom)


def _llr_position_count(
    records: Mapping[str, Mapping[str, Any]], score_set: str, chrom: str
) -> int:
    relative = get_shard_spec(score_set, "llr", chrom).relative_path.as_posix()
    parquet = records[relative].get("parquet")
    rows = parquet.get("num_rows") if isinstance(parquet, Mapping) else None
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0 or rows % 3:
        raise ValueError(f"invalid LLR row count for {relative}")
    return rows // 3


def _validate_direct_selection(
    path: str | Path, *, expected_manifest_sha256: str
) -> None:
    selection = _read_json(path)
    if (
        selection.get("report_version") != 1
        or selection.get("status") != "selected"
        or selection.get("selected_method") != "direct"
        or selection.get("inventory_manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError(
            "raw-LLR generation requires the finalized direct-streaming selection "
            "for the same inventory"
        )


def _validated_outputs(
    output_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    if set(output_paths) != set(RAW_LLR_TRACKS):
        raise ValueError(f"raw-LLR outputs must be {RAW_LLR_TRACKS!r}")
    outputs = {track: Path(output_paths[track]) for track in RAW_LLR_TRACKS}
    if len(set(outputs.values())) != len(RAW_LLR_TRACKS):
        raise ValueError("raw-LLR output paths must be distinct")
    return outputs


def _require_track(track: str) -> None:
    if track not in RAW_LLR_TRACKS:
        raise ValueError(f"unknown raw-LLR track: {track!r}")


def _require_value_decimals(value: int) -> None:
    if value != VALUE_DECIMALS:
        raise ValueError(f"raw-LLR value_decimals must be {VALUE_DECIMALS}")


def _sample_indices(total: int, requested: int, seed: int) -> np.ndarray:
    count = min(total, requested)
    if count == total:
        return np.arange(total, dtype=np.int64)
    if count == 1:
        return np.array([0], dtype=np.int64)
    required = {0, total - 1}
    remaining = count - len(required)
    if remaining > 0:
        rng = np.random.default_rng(seed)
        candidates = np.arange(1, total - 1, dtype=np.int64)
        required.update(
            int(value) for value in rng.choice(candidates, remaining, replace=False)
        )
    return np.array(sorted(required), dtype=np.int64)


def _float32_exact(left: np.float32, right: np.float32) -> bool:
    return bool(
        np.asarray([left], dtype=np.float32).view(np.uint32)[0]
        == np.asarray([right], dtype=np.float32).view(np.uint32)[0]
    )


def _valid_sign_counts(
    value: Any,
    *,
    expected_total: int | None = None,
) -> bool:
    names = {"negative", "zero", "positive"}
    if not isinstance(value, Mapping) or set(value) != names:
        return False
    counts = list(value.values())
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in counts
    ):
        return False
    return expected_total is None or sum(counts) == expected_total


def _round_float32(value: np.float32, decimals: int) -> np.float32:
    return np.round(np.asarray([value], dtype=np.float32), decimals=decimals).astype(
        np.float32, copy=False
    )[0]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-chromosome")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--inventory-manifest", type=Path, required=True)
    build.add_argument("--parquet-selection", type=Path, required=True)
    build.add_argument("--track-selection", type=Path, required=True)
    build.add_argument("--score-set", required=True)
    build.add_argument("--chrom", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.add_argument("--batch-size", type=int, default=262_144)
    build.add_argument("--sample-count", type=int, default=1_024)

    concatenate = subparsers.add_parser("concatenate")
    concatenate.add_argument("--inventory-manifest", type=Path, required=True)
    concatenate.add_argument("--parquet-selection", type=Path, required=True)
    concatenate.add_argument("--score-set", required=True)
    concatenate.add_argument("--track", choices=RAW_LLR_TRACKS, required=True)
    concatenate.add_argument("--output", type=Path, required=True)
    concatenate.add_argument("--report", type=Path, required=True)
    concatenate.add_argument("--value-decimals", type=int, default=VALUE_DECIMALS)
    concatenate.add_argument("--inputs", nargs="+", type=Path, required=True)
    concatenate.add_argument(
        "--chromosome-reports", nargs="+", type=Path, required=True
    )

    audit = subparsers.add_parser("audit-final")
    audit.add_argument("--inventory-manifest", type=Path, required=True)
    audit.add_argument("--parquet-selection", type=Path, required=True)
    audit.add_argument("--score-set", required=True)
    audit.add_argument("--track", choices=RAW_LLR_TRACKS, required=True)
    audit.add_argument("--bigwig", type=Path, required=True)
    audit.add_argument("--concatenation-report", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--value-decimals", type=int, default=VALUE_DECIMALS)
    audit.add_argument("--chromosome-reports", nargs="+", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "build-chromosome":
        outputs = {track: args.output_dir / f"{track}.bw" for track in RAW_LLR_TRACKS}
        build_raw_llr_chromosome(
            args.source_root,
            args.inventory_manifest,
            args.parquet_selection,
            args.track_selection,
            outputs,
            args.report,
            score_set=args.score_set,
            chrom=args.chrom,
            batch_size=args.batch_size,
            sample_count=args.sample_count,
        )
    elif args.command == "concatenate":
        concatenate_raw_llr_bigwig(
            args.inputs,
            args.chromosome_reports,
            args.output,
            args.report,
            args.inventory_manifest,
            args.parquet_selection,
            score_set=args.score_set,
            track=args.track,
            value_decimals=args.value_decimals,
        )
    elif args.command == "audit-final":
        audit_final_raw_llr_bigwig(
            args.bigwig,
            args.chromosome_reports,
            args.concatenation_report,
            args.report,
            args.inventory_manifest,
            args.parquet_selection,
            score_set=args.score_set,
            track=args.track,
            value_decimals=args.value_decimals,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
