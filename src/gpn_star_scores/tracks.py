"""Issue #7 BigWig orchestration, benchmarking, and release validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig

from gpn_star_scores.benchmark import (
    BenchmarkMeasurement,
    CandidateSummary,
    measure_command,
    select_bigwig_method,
    summarize_candidate,
)
from gpn_star_scores.bigwig import (
    BASES,
    BigWigWriteStats,
    ChromosomeSpec,
    iter_entropy_track_batches,
    iter_logo_track_batches,
    validate_bigwig,
    write_entropy_bigwig,
    write_entropy_wig,
    write_logo_bigwigs,
    write_logo_wigs,
)
from gpn_star_scores.catalog import (
    ASSEMBLIES,
    EXPECTED_SHARD_COUNT,
    SCORE_SETS,
    expected_shards,
    get_shard_spec,
    score_set_assembly,
)
from gpn_star_scores.inventory import atomic_write_json, sha256_file

METHODS = ("wig", "direct")
TRACKS = ("entropy", *BASES)
UCSC_ASSEMBLY_NAMES = {
    "hg38": "hg38",
    "ce11": "ce11",
    "dm6": "dm6",
    "gg6": "galGal6",
    "mm39": "mm39",
    "tair10": "araTha1",
}


@dataclass(frozen=True)
class TrackInputContract:
    """Validated issue #8 manifest and issue #5 selection records."""

    manifest_sha256: str
    selected_parquet_candidate: str
    records: dict[str, Mapping[str, Any]]


def ucsc_assembly_name(assembly: str) -> str:
    """Map the release assembly key to the UCSC browser assembly name."""

    try:
        return UCSC_ASSEMBLY_NAMES[assembly]
    except KeyError as error:
        raise KeyError(f"unknown release assembly: {assembly}") from error


def ucsc_chromosome_name(source_chromosome: str) -> str:
    """Map an Ensembl-style release chromosome to its UCSC track name."""

    if not source_chromosome:
        raise ValueError("source chromosome must not be empty")
    return f"chr{source_chromosome}"


def load_track_input_contract(
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
) -> TrackInputContract:
    """Require a complete valid inventory and a finalized source-layout choice."""

    manifest_path = Path(inventory_manifest_path)
    selection_path = Path(parquet_selection_path)
    manifest = _load_json(manifest_path)
    selection = _load_json(selection_path)
    records = _inventory_records(manifest)
    manifest_sha256 = sha256_file(manifest_path)

    if selection.get("report_version") != 1:
        raise ValueError("Parquet selection report_version must be 1")
    if selection.get("status") != "selected":
        raise ValueError("Parquet layout selection is not finalized")
    selected_candidate = selection.get("selected_candidate")
    if selected_candidate != "source":
        raise ValueError(
            "issue #7 currently requires issue #5 to select immutable source files; "
            f"got {selected_candidate!r}"
        )
    source_inventory = selection.get("source_inventory")
    if not isinstance(source_inventory, Mapping):
        raise ValueError("Parquet selection lacks source inventory evidence")
    if source_inventory.get("valid") is not True:
        raise ValueError("Parquet selection source inventory is not valid")
    if source_inventory.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Parquet selection references a different inventory manifest")

    return TrackInputContract(manifest_sha256, selected_candidate, records)


def chromosome_spec_from_contract(
    contract: TrackInputContract, score_set: str, chrom: str
) -> ChromosomeSpec:
    """Resolve a chromosome length validated independently in both score shards."""

    lengths = []
    for score_type in ("entropy", "llr"):
        record = _record_for(contract, score_set, score_type, chrom)
        content = record.get("content")
        if not isinstance(content, Mapping):
            raise ValueError(f"inventory record lacks content: {record.get('path')}")
        length = content.get("reference_length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(
                f"inventory record has invalid reference length: {record.get('path')}"
            )
        lengths.append(length)
    if len(set(lengths)) != 1:
        raise ValueError(
            f"entropy and LLR reference lengths disagree for {score_set}/{chrom}"
        )
    return ChromosomeSpec(chrom, ucsc_chromosome_name(chrom), lengths[0])


def assembly_chromosome_sizes_from_contract(
    contract: TrackInputContract, score_set: str
) -> dict[str, int]:
    """Return the ordered full-assembly header required by ``bigWigCat``."""

    assembly = score_set_assembly(score_set)
    return {
        chromosome.ucsc_name: chromosome.length
        for chromosome in (
            chromosome_spec_from_contract(contract, score_set, chrom)
            for chrom in ASSEMBLIES[assembly].chromosomes
        )
    }


def build_score_type_tracks(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    score_type: str,
    method: str,
    chromosome: ChromosomeSpec,
    batch_size: int = 262_144,
    wig_to_bigwig: str = "wigToBigWig",
    header_chromosome_sizes: Mapping[str, int] | None = None,
) -> tuple[dict[str, Path], BigWigWriteStats]:
    """Build one entropy track or four logo tracks with the selected method."""

    if method not in METHODS:
        raise ValueError(f"unknown BigWig method: {method!r}")
    if score_type not in {"entropy", "llr"}:
        raise ValueError(f"unknown score type: {score_type!r}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tracks = ("entropy",) if score_type == "entropy" else BASES
    outputs = {track: output_root / f"{track}.bw" for track in tracks}
    header_sizes = _validated_header_sizes(chromosome, header_chromosome_sizes)

    if method == "direct":
        if score_type == "entropy":
            stats = write_entropy_bigwig(
                [source_path],
                outputs["entropy"],
                chromosome,
                batch_size=batch_size,
                header_chromosome_sizes=header_sizes,
            )
        else:
            stats = write_logo_bigwigs(
                [source_path],
                outputs,
                chromosome,
                batch_size=batch_size,
                header_chromosome_sizes=header_sizes,
            )
        return outputs, stats

    temporary_dir = Path(tempfile.mkdtemp(prefix=".wig-baseline.", dir=output_root))
    try:
        chrom_sizes = temporary_dir / "chrom.sizes"
        chrom_sizes.write_text(
            "".join(f"{name}\t{length}\n" for name, length in header_sizes.items()),
            encoding="utf-8",
        )
        if score_type == "entropy":
            wigs = {"entropy": temporary_dir / "entropy.wig"}
            stats = write_entropy_wig(
                [source_path], wigs["entropy"], chromosome, batch_size=batch_size
            )
        else:
            wigs = {base: temporary_dir / f"{base}.wig" for base in BASES}
            stats = write_logo_wigs(
                [source_path], wigs, chromosome, batch_size=batch_size
            )
        for track in tracks:
            _convert_wig_to_bigwig(
                wigs[track],
                chrom_sizes,
                outputs[track],
                header_sizes,
                expected_bases_covered=stats.position_count,
                executable=wig_to_bigwig,
            )
        return outputs, stats
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def validate_score_type_tracks(
    source_path: str | Path,
    output_paths: Mapping[str, str | Path],
    *,
    score_type: str,
    chromosome: ChromosomeSpec,
    expected_position_count: int,
    sample_count: int = 1_024,
    sample_seed: int = 0,
    batch_size: int = 262_144,
) -> dict[str, Any]:
    """Compare deterministic random source/derived samples with BigWig values."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if expected_position_count <= 0:
        raise ValueError("expected position count must be positive")
    tracks = ("entropy",) if score_type == "entropy" else BASES
    if set(output_paths) != set(tracks):
        raise ValueError(f"outputs for {score_type} must be {tracks!r}")

    selected_indices = _sample_indices(
        expected_position_count, sample_count, sample_seed
    )
    samples: dict[str, list[dict[str, Any]]] = {track: [] for track in tracks}
    observed_positions = 0
    first_position: int | None = None
    last_position: int | None = None
    first_gap: int | None = None

    if score_type == "entropy":
        batches: Iterable[tuple[np.ndarray, np.ndarray]] = (
            (positions, values[:, None])
            for positions, values in iter_entropy_track_batches(
                [source_path], chromosome, batch_size=batch_size
            )
        )
    else:
        batches = iter_logo_track_batches(
            [source_path], chromosome, batch_size=batch_size
        )

    open_bigwigs = {track: pyBigWig.open(str(output_paths[track])) for track in tracks}
    if any(bigwig is None for bigwig in open_bigwigs.values()):
        for bigwig in open_bigwigs.values():
            if bigwig is not None:
                bigwig.close()
        raise ValueError("could not open every generated BigWig")

    try:
        sample_pointer = 0
        for positions, values in batches:
            if first_position is None:
                first_position = int(positions[0])
            if last_position is not None and first_gap is None:
                if int(positions[0]) > last_position + 1:
                    first_gap = last_position + 1
            differences = np.diff(positions)
            if first_gap is None and np.any(differences > 1):
                gap_index = int(np.flatnonzero(differences > 1)[0])
                first_gap = int(positions[gap_index]) + 1

            batch_end = observed_positions + len(positions)
            while (
                sample_pointer < len(selected_indices)
                and selected_indices[sample_pointer] < batch_end
            ):
                local_index = int(selected_indices[sample_pointer] - observed_positions)
                position = int(positions[local_index])
                for track_index, track in enumerate(tracks):
                    expected = np.float32(values[local_index, track_index])
                    observed = open_bigwigs[track].values(
                        chromosome.ucsc_name, position - 1, position
                    )[0]
                    observed_float32 = np.float32(observed)
                    if not _float32_exact(expected, observed_float32):
                        raise ValueError(
                            f"{track} differs at {chromosome.ucsc_name}:{position}: "
                            f"{observed_float32!r} != {expected!r}"
                        )
                    samples[track].append(
                        {
                            "position_1based": position,
                            "expected_float32": float(expected),
                            "observed_float32": float(observed_float32),
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

        gap_checks = {}
        if first_gap is not None:
            for track in tracks:
                gap_value = open_bigwigs[track].values(
                    chromosome.ucsc_name, first_gap - 1, first_gap
                )[0]
                if not np.isnan(gap_value):
                    raise ValueError(
                        f"{track} unexpectedly covers source gap at {first_gap}"
                    )
                gap_checks[track] = True
        return {
            "score_type": score_type,
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
            "first_gap_position": first_gap,
            "gap_checks": gap_checks,
            "samples": samples,
        }
    finally:
        for bigwig in open_bigwigs.values():
            bigwig.close()


def build_chromosome_tracks(
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
    wig_to_bigwig: str = "wigToBigWig",
) -> None:
    """Build and validate all five chromosome restart-unit artifacts."""

    if set(output_paths) != set(TRACKS):
        raise ValueError(f"chromosome outputs must be {TRACKS!r}")
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    method = _load_track_method(
        track_selection_path, expected_manifest_sha256=contract.manifest_sha256
    )
    chromosome = chromosome_spec_from_contract(contract, score_set, chrom)
    header_sizes = assembly_chromosome_sizes_from_contract(contract, score_set)
    root = Path(source_root)
    entropy_path = _validated_source_path(root, contract, score_set, "entropy", chrom)
    llr_path = _validated_source_path(root, contract, score_set, "llr", chrom)
    entropy_record = _record_for(contract, score_set, "entropy", chrom)
    llr_record = _record_for(contract, score_set, "llr", chrom)
    entropy_count = _position_count(entropy_record)
    llr_count = _position_count(llr_record)

    entropy_outputs, entropy_stats = build_score_type_tracks(
        entropy_path,
        Path(output_paths["entropy"]).parent,
        score_type="entropy",
        method=method,
        chromosome=chromosome,
        batch_size=batch_size,
        wig_to_bigwig=wig_to_bigwig,
        header_chromosome_sizes=header_sizes,
    )
    logo_outputs, logo_stats = build_score_type_tracks(
        llr_path,
        Path(output_paths["A"]).parent,
        score_type="llr",
        method=method,
        chromosome=chromosome,
        batch_size=batch_size,
        wig_to_bigwig=wig_to_bigwig,
        header_chromosome_sizes=header_sizes,
    )
    generated = {**entropy_outputs, **logo_outputs}
    expected_outputs = {track: Path(path) for track, path in output_paths.items()}
    if generated != expected_outputs:
        raise ValueError(
            f"generated paths {generated!r} do not match workflow outputs "
            f"{expected_outputs!r}"
        )

    entropy_validation = validate_score_type_tracks(
        entropy_path,
        entropy_outputs,
        score_type="entropy",
        chromosome=chromosome,
        expected_position_count=entropy_count,
        sample_count=sample_count,
        sample_seed=0,
        batch_size=batch_size,
    )
    logo_validation = validate_score_type_tracks(
        llr_path,
        logo_outputs,
        score_type="llr",
        chromosome=chromosome,
        expected_position_count=llr_count,
        sample_count=sample_count,
        sample_seed=1,
        batch_size=batch_size,
    )
    atomic_write_json(
        Path(report_path),
        {
            "report_version": 1,
            "valid": True,
            "method": method,
            "score_set": score_set,
            "assembly": score_set_assembly(score_set),
            "ucsc_assembly": ucsc_assembly_name(score_set_assembly(score_set)),
            "chromosome": asdict(chromosome),
            "inventory_manifest_sha256": contract.manifest_sha256,
            "stats": {
                "entropy": asdict(entropy_stats),
                "llr": asdict(logo_stats),
            },
            "validation": {
                "entropy": entropy_validation,
                "llr": logo_validation,
            },
        },
    )


def benchmark_track_method(
    source_root: str | Path,
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    artifact_root: str | Path,
    report_path: str | Path,
    *,
    case: str,
    score_set: str,
    score_type: str,
    chrom: str,
    method: str,
    repetitions: int = 5,
    sample_count: int = 1_024,
    batch_size: int = 262_144,
) -> None:
    """Run one excluded warm-up and measured repetitions for one pilot case."""

    if repetitions <= 0:
        raise ValueError("benchmark repetitions must be positive")
    if method not in METHODS:
        raise ValueError(f"unknown BigWig method: {method!r}")
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    chromosome = chromosome_spec_from_contract(contract, score_set, chrom)
    header_sizes = assembly_chromosome_sizes_from_contract(contract, score_set)
    source = _validated_source_path(
        Path(source_root), contract, score_set, score_type, chrom
    )
    expected_count = _position_count(
        _record_for(contract, score_set, score_type, chrom)
    )
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=root))

    warmup_dir = run_root / "warmup"
    warmup_command = _build_score_type_command(
        source,
        warmup_dir,
        score_type=score_type,
        method=method,
        chromosome=chromosome,
        batch_size=batch_size,
        header_chromosome_sizes=header_sizes,
    )
    subprocess.run(warmup_command, check=True)
    warmup_outputs = _score_type_output_paths(warmup_dir, score_type)
    warmup_validation = validate_score_type_tracks(
        source,
        warmup_outputs,
        score_type=score_type,
        chromosome=chromosome,
        expected_position_count=expected_count,
        sample_count=sample_count,
        sample_seed=7,
        batch_size=batch_size,
    )

    measurements: list[BenchmarkMeasurement] = []
    validations = []
    for repetition in range(repetitions):
        repetition_dir = run_root / f"repetition-{repetition + 1}"
        outputs = _score_type_output_paths(repetition_dir, score_type)
        command = _build_score_type_command(
            source,
            repetition_dir,
            score_type=score_type,
            method=method,
            chromosome=chromosome,
            batch_size=batch_size,
            header_chromosome_sizes=header_sizes,
        )
        measurement = measure_command(
            method,
            repetition + 1,
            command,
            working_directory=Path.cwd(),
            measurement_directory=run_root / "timing",
            scratch_paths=[repetition_dir],
            final_paths=list(outputs.values()),
            stdout_path=run_root / "logs" / f"repetition-{repetition + 1}.stdout.log",
            stderr_path=run_root / "logs" / f"repetition-{repetition + 1}.stderr.log",
            correct=True,
        )
        validation = validate_score_type_tracks(
            source,
            outputs,
            score_type=score_type,
            chromosome=chromosome,
            expected_position_count=expected_count,
            sample_count=sample_count,
            sample_seed=7 + repetition + 1,
            batch_size=batch_size,
        )
        measurements.append(measurement)
        validations.append(validation)

    summary = summarize_candidate(measurements)
    atomic_write_json(
        Path(report_path),
        {
            "report_version": 1,
            "case": case,
            "score_set": score_set,
            "score_type": score_type,
            "chrom": chrom,
            "method": method,
            "inventory_manifest_sha256": contract.manifest_sha256,
            "execution": _execution_context(),
            "artifact_run": str(run_root),
            "warmups": 1,
            "measured_repetitions": repetitions,
            "warmup_validation": warmup_validation,
            "measurements": [asdict(item) for item in measurements],
            "summary": asdict(summary),
            "validations": validations,
        },
    )


def render_track_benchmark(
    report_paths: Sequence[str | Path],
    output_json: str | Path,
    output_markdown: str | Path,
) -> None:
    """Aggregate case medians, apply the selection rule, and report evidence."""

    reports = [_load_json(path) for path in report_paths]
    if any(report.get("report_version") != 1 for report in reports):
        raise ValueError("benchmark report_version must be 1")
    report_keys = [
        (str(report.get("case")), str(report.get("method"))) for report in reports
    ]
    if len(set(report_keys)) != len(report_keys):
        raise ValueError("benchmark reports repeat a case/method pair")
    methods = {str(report.get("method")) for report in reports}
    if methods != set(METHODS):
        raise ValueError(f"benchmark reports must cover methods {METHODS!r}")
    cases_by_method = {
        method: {
            str(report.get("case"))
            for report in reports
            if report.get("method") == method
        }
        for method in METHODS
    }
    if cases_by_method["wig"] != cases_by_method["direct"]:
        raise ValueError("WIG and direct reports cover different benchmark cases")
    if not cases_by_method["wig"]:
        raise ValueError("benchmark reports contain no cases")
    for case in cases_by_method["wig"]:
        case_reports = [report for report in reports if report.get("case") == case]
        case_inputs = {
            (
                report.get("score_set"),
                report.get("score_type"),
                report.get("chrom"),
            )
            for report in case_reports
        }
        if len(case_reports) != len(METHODS) or len(case_inputs) != 1:
            raise ValueError(f"benchmark methods use different inputs for {case}")

    aggregates = {}
    for method in METHODS:
        summaries = []
        for report in reports:
            if report.get("method") != method:
                continue
            summary = _candidate_summary_from_json(report["summary"])
            if summary.method != method:
                raise ValueError("benchmark summary method does not match its report")
            summaries.append(summary)
        aggregates[method] = CandidateSummary(
            method=method,
            measured_repetitions=sum(
                summary.measured_repetitions for summary in summaries
            ),
            median_wall_seconds=sum(
                summary.median_wall_seconds for summary in summaries
            ),
            peak_rss_bytes=max(summary.peak_rss_bytes for summary in summaries),
            peak_scratch_bytes=max(summary.peak_scratch_bytes for summary in summaries),
            final_bytes=sum(summary.final_bytes for summary in summaries),
            correct=all(summary.correct for summary in summaries),
        )
    decision = select_bigwig_method(aggregates["wig"], aggregates["direct"])
    manifest_hashes = {report.get("inventory_manifest_sha256") for report in reports}
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        raise ValueError("benchmark reports reference inconsistent inventory manifests")

    payload = {
        "report_version": 1,
        "status": "selected",
        "selected_method": decision.selected_method,
        "rationale": decision.reason,
        "inventory_manifest_sha256": next(iter(manifest_hashes)),
        "cases": sorted(cases_by_method["wig"]),
        "aggregation": {
            "wall_time": "sum of per-case median measured wall times",
            "peak_rss": "maximum observed across cases and repetitions",
            "peak_scratch": "maximum observed across cases and repetitions",
            "final_size": "sum of per-case final BigWig sizes",
        },
        "thresholds": {
            "minimum_speedup": 0.20,
            "minimum_scratch_reduction": 0.80,
            "maximum_slowdown_for_scratch_winner": 0.20,
            "inclusive": True,
        },
        "aggregates": {
            method: asdict(summary) for method, summary in aggregates.items()
        },
        "decision": asdict(decision),
        "case_reports": [str(path) for path in report_paths],
    }
    atomic_write_json(Path(output_json), payload)
    _atomic_write_text(Path(output_markdown), _render_benchmark_markdown(payload))


def concatenate_track_bigwig(
    input_paths: Sequence[str | Path],
    chromosome_report_paths: Sequence[str | Path],
    output_path: str | Path,
    report_path: str | Path,
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    *,
    score_set: str,
    track: str,
    bigwig_cat: str = "bigWigCat",
    bigwig_info: str = "bigWigInfo",
) -> None:
    """Concatenate disjoint chromosome artifacts and validate the final track."""

    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    contract = load_track_input_contract(
        inventory_manifest_path, parquet_selection_path
    )
    assembly = score_set_assembly(score_set)
    chromosomes = ASSEMBLIES[assembly].chromosomes
    inputs = [Path(path) for path in input_paths]
    chromosome_reports = [Path(path) for path in chromosome_report_paths]
    if len(inputs) != len(chromosomes) or len(chromosome_reports) != len(chromosomes):
        raise ValueError("one input and validation report are required per chromosome")

    expected_sizes = {}
    expected_bases = 0
    for chrom in chromosomes:
        spec = chromosome_spec_from_contract(contract, score_set, chrom)
        expected_sizes[spec.ucsc_name] = spec.length
        score_type = "entropy" if track == "entropy" else "llr"
        expected_bases += _position_count(
            _record_for(contract, score_set, score_type, chrom)
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    temporary = temporary_dir / output.name
    try:
        subprocess.run(
            [bigwig_cat, str(temporary), *map(str, inputs)],
            check=True,
            text=True,
            capture_output=True,
        )
        summary = validate_bigwig(
            temporary,
            expected_sizes,
            expected_bases_covered=expected_bases,
        )
        info = subprocess.run(
            [bigwig_info, str(temporary)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        zoom_match = re.search(r"^zoomLevels:\s*(\d+)\s*$", info, re.MULTILINE)
        if zoom_match is None or int(zoom_match.group(1)) < 1:
            raise ValueError("bigWigInfo reports no zoom levels")

        concatenated_checks = []
        with pyBigWig.open(str(temporary)) as bigwig:
            for chrom, report_path_item in zip(
                chromosomes, chromosome_reports, strict=True
            ):
                report = _load_json(report_path_item)
                chromosome_record = report.get("chromosome")
                if (
                    report.get("report_version") != 1
                    or report.get("valid") is not True
                    or report.get("score_set") != score_set
                    or report.get("inventory_manifest_sha256")
                    != contract.manifest_sha256
                    or not isinstance(chromosome_record, Mapping)
                    or chromosome_record.get("source_name") != chrom
                    or chromosome_record.get("ucsc_name") != ucsc_chromosome_name(chrom)
                ):
                    raise ValueError(
                        f"invalid chromosome report for {score_set}/{chrom}"
                    )
                validation_key = "entropy" if track == "entropy" else "llr"
                samples = report["validation"][validation_key]["samples"][track]
                if not samples:
                    raise ValueError(f"chromosome report has no {track} samples")
                sample = samples[0]
                position = int(sample["position_1based"])
                ucsc_chrom = ucsc_chromosome_name(chrom)
                observed = np.float32(
                    bigwig.values(ucsc_chrom, position - 1, position)[0]
                )
                expected = np.float32(sample["expected_float32"])
                if not _float32_exact(expected, observed):
                    raise ValueError(
                        f"concatenated {track} differs at {ucsc_chrom}:{position}"
                    )
                concatenated_checks.append(
                    {
                        "chrom": ucsc_chrom,
                        "position_1based": position,
                        "expected_float32": float(expected),
                        "observed_float32": float(observed),
                    }
                )

        os.replace(temporary, output)
        atomic_write_json(
            Path(report_path),
            {
                "report_version": 1,
                "valid": True,
                "score_set": score_set,
                "assembly": assembly,
                "ucsc_assembly": ucsc_assembly_name(assembly),
                "track": track,
                "input_count": len(inputs),
                "inventory_manifest_sha256": contract.manifest_sha256,
                "summary": asdict(summary),
                "bigWigInfo": info,
                "concatenated_sample_checks": concatenated_checks,
            },
        )
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def aggregate_track_validation(
    final_report_paths: Sequence[str | Path],
    benchmark_selection_path: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
) -> None:
    """Require one valid final report for every score-set/track pair."""

    reports = [_load_json(path) for path in final_report_paths]
    expected = {(spec.name, track) for spec in SCORE_SETS for track in TRACKS}
    observed = {(report.get("score_set"), report.get("track")) for report in reports}
    if observed != expected or len(reports) != len(expected):
        raise ValueError("final validation reports do not cover all 40 tracks")
    invalid = [
        f"{report.get('score_set')}/{report.get('track')}"
        for report in reports
        if report.get("valid") is not True
    ]
    if invalid:
        raise ValueError(f"invalid final BigWig reports: {invalid!r}")
    selection = _load_json(benchmark_selection_path)
    if (
        selection.get("status") != "selected"
        or selection.get("selected_method") not in METHODS
    ):
        raise ValueError("BigWig benchmark has no valid selected method")
    manifest_hashes = {report.get("inventory_manifest_sha256") for report in reports}
    if manifest_hashes != {selection.get("inventory_manifest_sha256")}:
        raise ValueError("final reports and benchmark use different inventories")

    payload = {
        "report_version": 1,
        "valid": True,
        "track_count": len(reports),
        "selected_method": selection["selected_method"],
        "inventory_manifest_sha256": next(iter(manifest_hashes)),
        "tracks": sorted(
            [
                {
                    "score_set": report["score_set"],
                    "assembly": report["assembly"],
                    "ucsc_assembly": report["ucsc_assembly"],
                    "track": report["track"],
                    "bases_covered": report["summary"]["bases_covered"],
                    "zoom_levels": report["summary"]["zoom_levels"],
                }
                for report in reports
            ],
            key=lambda item: (item["score_set"], item["track"]),
        ),
    }
    atomic_write_json(Path(output_json), payload)
    lines = [
        "# BigWig validation",
        "",
        "Status: **valid**",
        "",
        f"Selected method: `{payload['selected_method']}`",
        "",
        f"Validated final tracks: {payload['track_count']}",
        "",
        "Every final BigWig opened through pyBigWig and bigWigInfo, reported zoom "
        "levels, matched expected chromosome sizes and covered-base counts, and "
        "preserved sampled source-derived Float32 values after concatenation.",
        "",
    ]
    _atomic_write_text(Path(output_markdown), "\n".join(lines))


def _inventory_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("manifest_version") != 1:
        raise ValueError("inventory manifest_version must be 1")
    source = manifest.get("source")
    validation = manifest.get("validation")
    shards = manifest.get("shards")
    if not isinstance(source, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("inventory manifest lacks source or validation records")
    if not isinstance(shards, list):
        raise ValueError("inventory manifest shards must be a list")
    for field in ("expected_shards", "reported_shards", "discovered_parquet_files"):
        if source.get(field) != EXPECTED_SHARD_COUNT:
            raise ValueError(f"inventory source field {field!r} is incomplete")
    if any(
        source.get(field)
        for field in ("missing_paths", "unexpected_paths", "unreported_paths")
    ):
        raise ValueError("inventory source path accounting is incomplete")
    if (
        validation.get("invalid_shards") != 0
        or validation.get("valid_shards") != EXPECTED_SHARD_COUNT
    ):
        raise ValueError("inventory shard validation is incomplete")
    if len(shards) != EXPECTED_SHARD_COUNT:
        raise ValueError("inventory manifest has the wrong shard count")

    records = {}
    for record in shards:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("inventory manifest contains a malformed shard record")
        path = str(record["path"])
        if path in records:
            raise ValueError(f"inventory manifest repeats {path!r}")
        content = record.get("content")
        try:
            expected = get_shard_spec(
                str(record.get("score_set")),
                str(record.get("score_type")),
                str(record.get("chrom")),
            )
        except KeyError as error:
            raise ValueError(
                f"inventory record has invalid catalog keys: {path}"
            ) from error
        if (
            record.get("valid") is not True
            or record.get("errors") != []
            or not isinstance(content, Mapping)
            or content.get("order_violations") != 0
            or expected.relative_path.as_posix() != path
            or record.get("assembly") != expected.assembly
            or not _is_sha256(record.get("sha256"))
        ):
            raise ValueError(f"inventory record is not valid: {path}")
        records[path] = record
    expected_paths = {shard.relative_path.as_posix() for shard in expected_shards()}
    if set(records) != expected_paths:
        raise ValueError("inventory paths do not match the release catalog")
    return records


def _record_for(
    contract: TrackInputContract, score_set: str, score_type: str, chrom: str
) -> Mapping[str, Any]:
    relative = get_shard_spec(score_set, score_type, chrom).relative_path.as_posix()
    return contract.records[relative]


def _validated_source_path(
    source_root: Path,
    contract: TrackInputContract,
    score_set: str,
    score_type: str,
    chrom: str,
) -> Path:
    spec = get_shard_spec(score_set, score_type, chrom)
    source = source_root / spec.relative_path
    record = _record_for(contract, score_set, score_type, chrom)
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_size = record.get("size")
    if not isinstance(expected_size, int) or source.stat().st_size != expected_size:
        raise ValueError(f"source size differs from inventory: {spec.relative_path}")
    return source


def _position_count(record: Mapping[str, Any]) -> int:
    parquet = record.get("parquet")
    if not isinstance(parquet, Mapping):
        raise ValueError(f"record lacks Parquet metadata: {record.get('path')}")
    rows = parquet.get("num_rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError(f"record has invalid row count: {record.get('path')}")
    if record.get("score_type") == "llr":
        if rows % 3:
            raise ValueError(
                f"LLR row count is not divisible by three: {record.get('path')}"
            )
        return rows // 3
    return rows


def _convert_wig_to_bigwig(
    wig_path: Path,
    chrom_sizes_path: Path,
    output_path: Path,
    header_chromosome_sizes: Mapping[str, int],
    *,
    expected_bases_covered: int,
    executable: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    temporary = temporary_dir / output_path.name
    try:
        subprocess.run(
            [
                executable,
                str(wig_path),
                str(chrom_sizes_path),
                str(temporary),
                "-keepAllChromosomes",
                "-fixedSummaries",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        validate_bigwig(
            temporary,
            header_chromosome_sizes,
            expected_bases_covered=expected_bases_covered,
        )
        os.replace(temporary, output_path)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def _sample_indices(total: int, requested: int, seed: int) -> np.ndarray:
    count = min(total, requested)
    if count == total:
        return np.arange(total, dtype=np.int64)
    if count == 1:
        return np.array([0], dtype=np.int64)
    required = {0, total - 1}
    rng = np.random.default_rng(seed)
    remaining = count - len(required)
    if remaining > 0:
        candidates = np.arange(1, total - 1, dtype=np.int64)
        required.update(
            int(value) for value in rng.choice(candidates, remaining, replace=False)
        )
    return np.array(sorted(required), dtype=np.int64)


def _validated_header_sizes(
    chromosome: ChromosomeSpec,
    chromosome_sizes: Mapping[str, int] | None,
) -> dict[str, int]:
    values = (
        {chromosome.ucsc_name: chromosome.length}
        if chromosome_sizes is None
        else dict(chromosome_sizes)
    )
    if not values:
        raise ValueError("BigWig header chromosome sizes must not be empty")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        for name, length in values.items()
    ):
        raise ValueError("BigWig header contains an invalid chromosome size")
    if values.get(chromosome.ucsc_name) != chromosome.length:
        raise ValueError(
            "BigWig header must contain the active chromosome with its exact length"
        )
    return values


def _parse_header_sizes(
    values: Sequence[str], chromosome: ChromosomeSpec
) -> dict[str, int] | None:
    if not values:
        return None
    result: dict[str, int] = {}
    for value in values:
        try:
            name, length_text = value.rsplit("=", 1)
            length = int(length_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid --header-chrom-size value: {value!r}") from error
        if name in result:
            raise ValueError(f"duplicate BigWig header chromosome: {name!r}")
        result[name] = length
    return _validated_header_sizes(chromosome, result)


def _float32_exact(left: np.float32, right: np.float32) -> bool:
    return bool(
        np.asarray([left], dtype=np.float32).view(np.uint32)[0]
        == np.asarray([right], dtype=np.float32).view(np.uint32)[0]
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _score_type_output_paths(root: Path, score_type: str) -> dict[str, Path]:
    tracks = ("entropy",) if score_type == "entropy" else BASES
    return {track: root / f"{track}.bw" for track in tracks}


def _build_score_type_command(
    source: Path,
    output_dir: Path,
    *,
    score_type: str,
    method: str,
    chromosome: ChromosomeSpec,
    batch_size: int,
    header_chromosome_sizes: Mapping[str, int],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gpn_star_scores.tracks",
        "build-score-type",
        "--source",
        str(source),
        "--output-dir",
        str(output_dir),
        "--score-type",
        score_type,
        "--method",
        method,
        "--source-chrom",
        chromosome.source_name,
        "--ucsc-chrom",
        chromosome.ucsc_name,
        "--chrom-length",
        str(chromosome.length),
        "--batch-size",
        str(batch_size),
    ]
    for name, length in header_chromosome_sizes.items():
        command.extend(("--header-chrom-size", f"{name}={length}"))
    return command


def _load_track_method(
    path: str | Path, *, expected_manifest_sha256: str | None = None
) -> str:
    selection = _load_json(path)
    method = selection.get("selected_method")
    if selection.get("report_version") != 1 or selection.get("status") != "selected":
        raise ValueError("BigWig benchmark selection is not finalized")
    if method not in METHODS:
        raise ValueError(f"invalid selected BigWig method: {method!r}")
    if (
        expected_manifest_sha256 is not None
        and selection.get("inventory_manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("BigWig benchmark selection uses a different inventory")
    return str(method)


def _candidate_summary_from_json(value: Mapping[str, Any]) -> CandidateSummary:
    return CandidateSummary(
        method=str(value["method"]),
        measured_repetitions=int(value["measured_repetitions"]),
        median_wall_seconds=float(value["median_wall_seconds"]),
        peak_rss_bytes=int(value["peak_rss_bytes"]),
        peak_scratch_bytes=int(value["peak_scratch_bytes"]),
        final_bytes=int(value["final_bytes"]),
        correct=bool(value["correct"]),
    )


def _render_benchmark_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# BigWig generation benchmark",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Selected method: `{report['selected_method']}`",
        "",
        str(report["rationale"]),
        "",
        "| Method | Aggregate median wall (s) | Peak RSS (bytes) | "
        "Peak scratch (bytes) | Final bytes | Correct |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for method in METHODS:
        summary = report["aggregates"][method]
        lines.append(
            f"| {method} | {summary['median_wall_seconds']:.6f} | "
            f"{summary['peak_rss_bytes']} | {summary['peak_scratch_bytes']} | "
            f"{summary['final_bytes']} | {summary['correct']} |"
        )
    lines.extend(
        [
            "",
            "Wall time is the sum of per-case medians. Resource peaks are the "
            "maximum observed across cases and repetitions.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _execution_context() -> dict[str, str | None]:
    return {
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_score = subparsers.add_parser("build-score-type")
    build_score.add_argument("--source", type=Path, required=True)
    build_score.add_argument("--output-dir", type=Path, required=True)
    build_score.add_argument("--score-type", choices=("entropy", "llr"), required=True)
    build_score.add_argument("--method", choices=METHODS, required=True)
    build_score.add_argument("--source-chrom", required=True)
    build_score.add_argument("--ucsc-chrom", required=True)
    build_score.add_argument("--chrom-length", type=int, required=True)
    build_score.add_argument("--batch-size", type=int, default=262_144)
    build_score.add_argument("--header-chrom-size", action="append", default=[])

    build_chrom = subparsers.add_parser("build-chromosome")
    build_chrom.add_argument("--source-root", type=Path, required=True)
    build_chrom.add_argument("--inventory-manifest", type=Path, required=True)
    build_chrom.add_argument("--parquet-selection", type=Path, required=True)
    build_chrom.add_argument("--track-selection", type=Path, required=True)
    build_chrom.add_argument("--score-set", required=True)
    build_chrom.add_argument("--chrom", required=True)
    build_chrom.add_argument("--output-dir", type=Path, required=True)
    build_chrom.add_argument("--report", type=Path, required=True)
    build_chrom.add_argument("--batch-size", type=int, default=262_144)
    build_chrom.add_argument("--sample-count", type=int, default=1_024)

    benchmark = subparsers.add_parser("benchmark-method")
    benchmark.add_argument("--source-root", type=Path, required=True)
    benchmark.add_argument("--inventory-manifest", type=Path, required=True)
    benchmark.add_argument("--parquet-selection", type=Path, required=True)
    benchmark.add_argument("--artifact-root", type=Path, required=True)
    benchmark.add_argument("--report", type=Path, required=True)
    benchmark.add_argument("--case", required=True)
    benchmark.add_argument("--score-set", required=True)
    benchmark.add_argument("--score-type", choices=("entropy", "llr"), required=True)
    benchmark.add_argument("--chrom", required=True)
    benchmark.add_argument("--method", choices=METHODS, required=True)
    benchmark.add_argument("--repetitions", type=int, default=5)
    benchmark.add_argument("--sample-count", type=int, default=1_024)
    benchmark.add_argument("--batch-size", type=int, default=262_144)

    concatenate = subparsers.add_parser("concatenate")
    concatenate.add_argument("--inventory-manifest", type=Path, required=True)
    concatenate.add_argument("--parquet-selection", type=Path, required=True)
    concatenate.add_argument("--score-set", required=True)
    concatenate.add_argument("--track", choices=TRACKS, required=True)
    concatenate.add_argument("--output", type=Path, required=True)
    concatenate.add_argument("--report", type=Path, required=True)
    concatenate.add_argument("--inputs", nargs="+", type=Path, required=True)
    concatenate.add_argument(
        "--chromosome-reports", nargs="+", type=Path, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the rule-local command-line entry points."""

    args = _build_parser().parse_args(argv)
    if args.command == "build-score-type":
        chromosome = ChromosomeSpec(
            args.source_chrom, args.ucsc_chrom, args.chrom_length
        )
        build_score_type_tracks(
            args.source,
            args.output_dir,
            score_type=args.score_type,
            method=args.method,
            chromosome=chromosome,
            batch_size=args.batch_size,
            header_chromosome_sizes=_parse_header_sizes(
                args.header_chrom_size, chromosome
            ),
        )
    elif args.command == "build-chromosome":
        output_paths = {track: args.output_dir / f"{track}.bw" for track in TRACKS}
        build_chromosome_tracks(
            args.source_root,
            args.inventory_manifest,
            args.parquet_selection,
            args.track_selection,
            output_paths,
            args.report,
            score_set=args.score_set,
            chrom=args.chrom,
            batch_size=args.batch_size,
            sample_count=args.sample_count,
        )
    elif args.command == "benchmark-method":
        benchmark_track_method(
            args.source_root,
            args.inventory_manifest,
            args.parquet_selection,
            args.artifact_root,
            args.report,
            case=args.case,
            score_set=args.score_set,
            score_type=args.score_type,
            chrom=args.chrom,
            method=args.method,
            repetitions=args.repetitions,
            sample_count=args.sample_count,
            batch_size=args.batch_size,
        )
    elif args.command == "concatenate":
        concatenate_track_bigwig(
            args.inputs,
            args.chromosome_reports,
            args.output,
            args.report,
            args.inventory_manifest,
            args.parquet_selection,
            score_set=args.score_set,
            track=args.track,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
