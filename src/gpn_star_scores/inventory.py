"""Inventory and validate immutable staged GPN-Star Parquet shards."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gpn_star_scores.catalog import (
    ASSEMBLIES,
    BOX_REPORTED_FOLDER_BYTES,
    EXPECTED_SHARD_COUNT,
    AssemblySpec,
    ShardSpec,
    expected_shards,
)

_BASES = np.array(["A", "C", "G", "T"])
_BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

EXPECTED_SCHEMAS: dict[str, pa.Schema] = {
    "entropy": pa.schema(
        [
            pa.field("chrom", pa.string()),
            pa.field("pos", pa.int64()),
            pa.field("ref", pa.string()),
            pa.field("entropy_calibrated", pa.float32()),
        ]
    ),
    "llr": pa.schema(
        [
            pa.field("chrom", pa.string()),
            pa.field("pos", pa.int64()),
            pa.field("ref", pa.string()),
            pa.field("alt", pa.string()),
            pa.field("llr_calibrated", pa.float32()),
            pa.field("abs_llr_calibrated", pa.float32()),
        ]
    ),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash ``path`` without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"hex": value.hex()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _schema_record(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _schema_matches(actual: pa.Schema, expected: pa.Schema) -> bool:
    return actual.names == expected.names and all(
        actual.field(name).type == expected.field(name).type for name in expected.names
    )


def _parquet_metadata(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    metadata = parquet_file.metadata
    row_groups = []
    columns: dict[str, dict[str, Any]] = {}

    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        row_groups.append(
            {
                "index": row_group_index,
                "num_rows": row_group.num_rows,
                "total_byte_size": row_group.total_byte_size,
            }
        )
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            name = column.path_in_schema
            record = columns.setdefault(
                name,
                {
                    "physical_type": column.physical_type,
                    "compression_codecs": set(),
                    "encodings": set(),
                    "statistics": {
                        "row_groups_present": 0,
                        "row_groups_with_min_max": 0,
                        "null_count": 0,
                        "null_count_complete": True,
                        "min": None,
                        "max": None,
                    },
                    "page_index": {
                        "column_index_row_groups": 0,
                        "offset_index_row_groups": 0,
                    },
                },
            )
            record["compression_codecs"].add(column.compression)
            record["encodings"].update(column.encodings)

            statistics = column.statistics
            if statistics is not None:
                stats_record = record["statistics"]
                stats_record["row_groups_present"] += 1
                if statistics.has_min_max:
                    stats_record["row_groups_with_min_max"] += 1
                    minimum = _json_value(statistics.min)
                    maximum = _json_value(statistics.max)
                    if stats_record["min"] is None or minimum < stats_record["min"]:
                        stats_record["min"] = minimum
                    if stats_record["max"] is None or maximum > stats_record["max"]:
                        stats_record["max"] = maximum
                if statistics.null_count is None:
                    stats_record["null_count_complete"] = False
                else:
                    stats_record["null_count"] += statistics.null_count
            else:
                record["statistics"]["null_count_complete"] = False

            if bool(getattr(column, "has_column_index", False)):
                record["page_index"]["column_index_row_groups"] += 1
            if bool(getattr(column, "has_offset_index", False)):
                record["page_index"]["offset_index_row_groups"] += 1

    for record in columns.values():
        record["compression_codecs"] = sorted(record["compression_codecs"])
        record["encodings"] = sorted(record["encodings"])

    return {
        "format_version": metadata.format_version,
        "created_by": metadata.created_by,
        "serialized_size": metadata.serialized_size,
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "row_groups": row_groups,
        "columns": columns,
    }


def _string_values(array: pa.Array) -> np.ndarray:
    return np.asarray(pc.fill_null(array, "").to_numpy(zero_copy_only=False))


def _valid_base_mask(values: np.ndarray) -> np.ndarray:
    return np.isin(values, _BASES)


def _count_non_finite(array: pa.Array) -> int:
    values = pc.fill_null(array, 0).to_numpy(zero_copy_only=False)
    return int(np.count_nonzero(~np.isfinite(values)))


def _validate_complete_llr_groups(
    positions: np.ndarray,
    refs: np.ndarray,
    alts: np.ndarray,
) -> dict[str, int]:
    if positions.size == 0:
        return {"wrong_size": 0, "inconsistent_ref": 0, "wrong_alt_set": 0}

    starts = np.r_[0, np.flatnonzero(positions[1:] != positions[:-1]) + 1]
    ends = np.r_[starts[1:], positions.size]
    sizes = ends - starts
    result = {
        "wrong_size": int(np.count_nonzero(sizes != 3)),
        "inconsistent_ref": 0,
        "wrong_alt_set": 0,
    }

    complete_starts = starts[sizes == 3]
    if complete_starts.size == 0:
        return result

    indices = complete_starts[:, None] + np.arange(3)
    group_refs = refs[indices]
    group_alts = alts[indices]
    same_ref = np.all(group_refs == group_refs[:, :1], axis=1)
    result["inconsistent_ref"] = int(np.count_nonzero(~same_ref))

    ref_codes = np.full(group_refs.shape[0], -1, dtype=np.int8)
    alt_codes = np.full(group_alts.shape, -1, dtype=np.int8)
    for base, code in _BASE_TO_CODE.items():
        ref_codes[group_refs[:, 0] == base] = code
        alt_codes[group_alts == base] = code
    valid_codes = (ref_codes >= 0) & np.all(alt_codes >= 0, axis=1)
    observed_masks = np.zeros(complete_starts.size, dtype=np.uint8)
    for offset in range(3):
        safe_codes = np.maximum(alt_codes[:, offset], 0)
        observed_masks |= (1 << safe_codes).astype(np.uint8)
    expected_masks = (15 ^ (1 << np.maximum(ref_codes, 0))).astype(np.uint8)
    valid_alt_sets = valid_codes & same_ref & (observed_masks == expected_masks)
    result["wrong_alt_set"] = int(np.count_nonzero(~valid_alt_sets))
    return result


def _add_group_counts(total: dict[str, int], addition: Mapping[str, int]) -> None:
    for key, value in addition.items():
        total[key] += value


def _content_validation(
    parquet_file: pq.ParquetFile,
    shard: ShardSpec,
    reference_path: Path,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = np.memmap(reference_path, dtype="S1", mode="r")
    expected_schema = EXPECTED_SCHEMAS[shard.score_type]
    columns = expected_schema.names
    score_columns = [name for name in columns if name.endswith("_calibrated")]
    null_counts = dict.fromkeys(columns, 0)
    non_finite_counts = dict.fromkeys(score_columns, 0)
    unexpected_chrom_rows = 0
    invalid_ref_rows = 0
    invalid_alt_rows = 0
    out_of_bounds_rows = 0
    reference_mismatch_rows = 0
    order_violations = 0
    rows_scanned = 0
    position_min: int | None = None
    position_max: int | None = None
    previous_position: int | None = None

    group_counts = {"wrong_size": 0, "inconsistent_ref": 0, "wrong_alt_set": 0}
    carry_positions = np.array([], dtype=np.int64)
    carry_refs = np.array([], dtype="U1")
    carry_alts = np.array([], dtype="U1")
    skip_oversized_position: int | None = None
    group_checks_disabled = False

    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=columns, use_threads=False
    ):
        rows_scanned += batch.num_rows
        arrays = {name: batch.column(name) for name in columns}
        for name, array in arrays.items():
            null_counts[name] += array.null_count

        chrom_values = _string_values(arrays["chrom"])
        unexpected_chrom_rows += int(np.count_nonzero(chrom_values != shard.chrom))

        pos_array = arrays["pos"]
        pos_null = pos_array.is_null().to_numpy(zero_copy_only=False)
        positions = (
            pc.fill_null(pos_array, 0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        valid_positions = positions[~pos_null]
        if valid_positions.size:
            batch_min = int(valid_positions.min())
            batch_max = int(valid_positions.max())
            position_min = (
                batch_min if position_min is None else min(position_min, batch_min)
            )
            position_max = (
                batch_max if position_max is None else max(position_max, batch_max)
            )
            out_of_bounds_rows += int(
                np.count_nonzero(
                    (valid_positions < 1) | (valid_positions > reference.size)
                )
            )
            if previous_position is not None:
                if shard.score_type == "entropy":
                    order_violations += int(valid_positions[0] <= previous_position)
                else:
                    order_violations += int(valid_positions[0] < previous_position)
            differences = np.diff(valid_positions)
            if shard.score_type == "entropy":
                order_violations += int(np.count_nonzero(differences <= 0))
            else:
                order_violations += int(np.count_nonzero(differences < 0))
            previous_position = int(valid_positions[-1])

        ref_values = _string_values(arrays["ref"])
        valid_refs = _valid_base_mask(ref_values)
        invalid_ref_rows += int(np.count_nonzero(~valid_refs))
        valid_coordinates = (
            ~pos_null & (positions >= 1) & (positions <= reference.size) & valid_refs
        )
        if np.any(valid_coordinates):
            expected_refs = reference[positions[valid_coordinates] - 1]
            observed_refs = ref_values[valid_coordinates].astype("S1")
            reference_mismatch_rows += int(
                np.count_nonzero(expected_refs != observed_refs)
            )

        for name in score_columns:
            non_finite_counts[name] += _count_non_finite(arrays[name])

        if shard.score_type != "llr":
            continue

        alt_values = _string_values(arrays["alt"])
        valid_alts = _valid_base_mask(alt_values)
        invalid_alt_rows += int(np.count_nonzero(~valid_alts))
        if pos_array.null_count or arrays["ref"].null_count or arrays["alt"].null_count:
            group_checks_disabled = True
            carry_positions = np.array([], dtype=np.int64)
            carry_refs = np.array([], dtype="U1")
            carry_alts = np.array([], dtype="U1")
            continue
        if group_checks_disabled:
            continue

        group_positions = positions
        group_refs = ref_values.astype("U1")
        group_alts = alt_values.astype("U1")
        if skip_oversized_position is not None:
            keep = group_positions != skip_oversized_position
            first_kept = int(np.argmax(keep)) if np.any(keep) else group_positions.size
            group_positions = group_positions[first_kept:]
            group_refs = group_refs[first_kept:]
            group_alts = group_alts[first_kept:]
            if first_kept < positions.size:
                skip_oversized_position = None
        if group_positions.size == 0:
            continue

        group_positions = np.concatenate((carry_positions, group_positions))
        group_refs = np.concatenate((carry_refs, group_refs))
        group_alts = np.concatenate((carry_alts, group_alts))
        last_group_start = (
            int(np.flatnonzero(group_positions != group_positions[-1])[-1] + 1)
            if np.any(group_positions != group_positions[-1])
            else 0
        )
        _add_group_counts(
            group_counts,
            _validate_complete_llr_groups(
                group_positions[:last_group_start],
                group_refs[:last_group_start],
                group_alts[:last_group_start],
            ),
        )
        carry_positions = group_positions[last_group_start:]
        carry_refs = group_refs[last_group_start:]
        carry_alts = group_alts[last_group_start:]
        if carry_positions.size > 3:
            group_counts["wrong_size"] += 1
            skip_oversized_position = int(carry_positions[-1])
            carry_positions = np.array([], dtype=np.int64)
            carry_refs = np.array([], dtype="U1")
            carry_alts = np.array([], dtype="U1")

    if shard.score_type == "llr" and not group_checks_disabled:
        _add_group_counts(
            group_counts,
            _validate_complete_llr_groups(carry_positions, carry_refs, carry_alts),
        )

    errors: list[dict[str, Any]] = []

    def add_error(check: str, count: int, message: str) -> None:
        if count:
            errors.append({"check": check, "count": count, "message": message})

    add_error("nulls", sum(null_counts.values()), "required values contain nulls")
    add_error("nonempty", int(rows_scanned == 0), "shard contains no rows")
    add_error(
        "chromosome",
        unexpected_chrom_rows,
        f"rows do not use expected chromosome {shard.chrom!r}",
    )
    add_error(
        "coordinates", out_of_bounds_rows, "positions are outside reference bounds"
    )
    add_error("ref_allele", invalid_ref_rows, "reference alleles are not A/C/G/T")
    add_error("alt_allele", invalid_alt_rows, "alternate alleles are not A/C/G/T")
    add_error(
        "reference_match",
        reference_mismatch_rows,
        "reference alleles disagree with the pinned Ensembl FASTA",
    )
    add_error(
        "non_finite_scores",
        sum(non_finite_counts.values()),
        "score columns contain non-finite values",
    )
    add_error(
        "entropy_order" if shard.score_type == "entropy" else "llr_order",
        order_violations,
        (
            "entropy positions are not strictly increasing and unique"
            if shard.score_type == "entropy"
            else "LLR positions are not nondecreasing"
        ),
    )
    if shard.score_type == "llr" and not group_checks_disabled:
        add_error(
            "llr_group_size",
            group_counts["wrong_size"],
            "LLR positions do not have exactly three rows",
        )
        add_error(
            "llr_group_ref",
            group_counts["inconsistent_ref"],
            "LLR rows at a position disagree on the reference allele",
        )
        add_error(
            "llr_alt_set",
            group_counts["wrong_alt_set"],
            "LLR alternate alleles are not the three unique non-reference bases",
        )

    content = {
        "rows_scanned": rows_scanned,
        "reference_length": int(reference.size),
        "coordinate_bounds": {"min": position_min, "max": position_max},
        "null_counts": null_counts,
        "non_finite_counts": non_finite_counts,
        "unexpected_chrom_rows": unexpected_chrom_rows,
        "invalid_ref_rows": invalid_ref_rows,
        "invalid_alt_rows": invalid_alt_rows,
        "out_of_bounds_rows": out_of_bounds_rows,
        "reference_mismatch_rows": reference_mismatch_rows,
        "order_violations": order_violations,
        "llr_group_errors": group_counts if shard.score_type == "llr" else None,
        "llr_group_checks_skipped_for_nulls": group_checks_disabled,
    }
    return content, errors


def inspect_shard(
    source_path: Path,
    shard: ShardSpec,
    reference_path: Path,
    *,
    batch_size: int = 1_048_576,
) -> dict[str, Any]:
    """Measure physical layout and validate one immutable staged shard."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source_path = Path(source_path)
    reference_path = Path(reference_path)
    record: dict[str, Any] = {
        "path": shard.relative_path.as_posix(),
        "score_set": shard.score_set,
        "assembly": shard.assembly,
        "score_type": shard.score_type,
        "chrom": shard.chrom,
        "size": None,
        "sha256": None,
        "schema": None,
        "parquet": None,
        "content": None,
        "valid": False,
        "errors": [],
    }
    if not source_path.is_file():
        record["errors"].append(
            {"check": "presence", "count": 1, "message": "expected shard is missing"}
        )
        return record
    if not reference_path.is_file():
        record["errors"].append(
            {
                "check": "reference_presence",
                "count": 1,
                "message": "prepared reference chromosome is missing",
            }
        )
        return record

    record["size"] = source_path.stat().st_size
    record["sha256"] = sha256_file(source_path)
    parquet_file = pq.ParquetFile(source_path)
    actual_schema = parquet_file.schema_arrow
    expected_schema = EXPECTED_SCHEMAS[shard.score_type]
    record["schema"] = _schema_record(actual_schema)
    record["parquet"] = _parquet_metadata(parquet_file)

    if not _schema_matches(actual_schema, expected_schema):
        record["errors"].append(
            {
                "check": "schema",
                "count": 1,
                "message": (
                    f"schema is {actual_schema.remove_metadata()}, expected {expected_schema}"
                ),
            }
        )
        return record

    content, errors = _content_validation(
        parquet_file, shard, reference_path, batch_size
    )
    record["content"] = content
    record["errors"].extend(errors)
    if content["rows_scanned"] != record["parquet"]["num_rows"]:
        record["errors"].append(
            {
                "check": "row_count",
                "count": abs(content["rows_scanned"] - record["parquet"]["num_rows"]),
                "message": "scanned row count disagrees with Parquet metadata",
            }
        )
    record["valid"] = not record["errors"]
    return record


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON to a temporary sibling and atomically promote it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def inspect_shard_to_json(
    source_path: Path,
    shard: ShardSpec,
    reference_path: Path,
    output_path: Path,
    *,
    batch_size: int = 1_048_576,
) -> None:
    """Validate one shard and atomically write its result."""

    atomic_write_json(
        output_path,
        inspect_shard(source_path, shard, reference_path, batch_size=batch_size),
    )


def _open_fasta(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii")
    return path.open("rt", encoding="ascii")


def prepare_reference(
    source_fasta: Path,
    output_dir: Path,
    assembly: AssemblySpec,
    expected_sha256: str,
) -> None:
    """Split one exact Ensembl FASTA into atomic, memory-mappable contigs."""

    source_fasta = Path(source_fasta)
    output_dir = Path(output_dir)
    if source_fasta.name != assembly.fasta_filename:
        raise ValueError(
            f"{assembly.name} FASTA must retain the pinned filename "
            f"{assembly.fasta_filename!r}; got {source_fasta.name!r}"
        )
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256.lower()
    ):
        raise ValueError(f"invalid expected SHA-256 for {assembly.name}")
    source_sha256 = sha256_file(source_fasta)
    if source_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"{assembly.name} FASTA SHA-256 is {source_sha256}, "
            f"expected {expected_sha256.lower()}"
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.")
    )
    expected = set(assembly.chromosomes)
    seen: set[str] = set()
    lengths: dict[str, int] = {}
    ignored_contigs = 0
    current_name: str | None = None
    current_handle = None

    try:
        with _open_fasta(source_fasta) as fasta:
            for line in fasta:
                if line.startswith(">"):
                    if current_handle is not None:
                        current_handle.flush()
                        os.fsync(current_handle.fileno())
                        current_handle.close()
                        current_handle = None
                    current_name = line[1:].split(maxsplit=1)[0]
                    if current_name in expected:
                        if current_name in seen:
                            raise ValueError(f"duplicate FASTA record {current_name!r}")
                        seen.add(current_name)
                        lengths[current_name] = 0
                        current_handle = (temporary_dir / f"{current_name}.seq").open(
                            "wb"
                        )
                    else:
                        ignored_contigs += 1
                    continue
                if current_handle is None:
                    if current_name is None and line.strip():
                        raise ValueError(
                            "FASTA sequence appears before its first header"
                        )
                    continue
                sequence = line.strip().upper().encode("ascii")
                current_handle.write(sequence)
                lengths[current_name] += len(sequence)
        if current_handle is not None:
            current_handle.flush()
            os.fsync(current_handle.fileno())
            current_handle.close()
            current_handle = None

        missing = sorted(expected - seen)
        empty = sorted(name for name, length in lengths.items() if length == 0)
        if missing or empty:
            raise ValueError(
                f"invalid {assembly.name} FASTA: missing={missing}, empty={empty}"
            )

        provenance = {
            "assembly": assembly.name,
            "expected_url": assembly.fasta_url,
            "expected_sha256": expected_sha256.lower(),
            "source_filename": source_fasta.name,
            "source_size": source_fasta.stat().st_size,
            "source_sha256": source_sha256,
            "identity_verified": True,
            "ignored_contigs": ignored_contigs,
            "contigs": {name: lengths[name] for name in assembly.chromosomes},
        }
        with (temporary_dir / "provenance.json").open("w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_dir, output_dir)
    except BaseException:
        if current_handle is not None:
            current_handle.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(
    source_root: Path,
    shard_reports: Iterable[Path],
    reference_reports: Iterable[Path],
    *,
    expected_shard_bytes: int | None,
    hugging_face_capacity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate shard reports and release-level blockers."""

    source_root = Path(source_root)
    records = sorted(
        (_read_json(path) for path in shard_reports), key=lambda item: item["path"]
    )
    references = sorted(
        (_read_json(path) for path in reference_reports),
        key=lambda item: item["assembly"],
    )
    reference_assemblies = {reference["assembly"] for reference in references}
    duplicate_reference_reports = len(references) - len(reference_assemblies)
    missing_reference_assemblies = sorted(set(ASSEMBLIES) - reference_assemblies)
    unexpected_reference_assemblies = sorted(reference_assemblies - set(ASSEMBLIES))
    invalid_reference_identities = sorted(
        reference["assembly"]
        for reference in references
        if reference["assembly"] in ASSEMBLIES
        and (
            not reference.get("identity_verified")
            or reference.get("source_sha256") != reference.get("expected_sha256")
            or reference.get("expected_url")
            != ASSEMBLIES[reference["assembly"]].fasta_url
            or reference.get("source_filename")
            != ASSEMBLIES[reference["assembly"]].fasta_filename
        )
    )
    expected_paths = {shard.relative_path.as_posix() for shard in expected_shards()}
    reported_paths = {record["path"] for record in records}
    discovered_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.parquet")
        if path.is_file()
    }
    missing_paths = sorted(expected_paths - discovered_paths)
    unexpected_paths = sorted(discovered_paths - expected_paths)
    unreported_paths = sorted(expected_paths - reported_paths)
    total_bytes = sum(record["size"] or 0 for record in records)
    invalid_records = [record for record in records if not record["valid"]]
    capacity = dict(hugging_face_capacity or {})
    capacity.setdefault("organization", "songlab")
    capacity.setdefault("confirmed", False)
    capacity.setdefault("evidence", None)
    capacity.setdefault("confirmed_by", None)
    capacity.setdefault("confirmed_at", None)
    capacity.setdefault("current_storage_bytes", None)
    capacity.setdefault("planned_release_bytes", None)
    capacity.setdefault("reserved_headroom_bytes", None)
    capacity.setdefault("approved_capacity_bytes", None)
    capacity_numbers = [
        capacity["current_storage_bytes"],
        capacity["planned_release_bytes"],
        capacity["reserved_headroom_bytes"],
        capacity["approved_capacity_bytes"],
    ]
    numeric_capacity_evidence = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in capacity_numbers
    )
    required_capacity_bytes = None
    capacity_sufficient = False
    if numeric_capacity_evidence:
        required_capacity_bytes = sum(capacity_numbers[:3])
        capacity_sufficient = (
            capacity["confirmed"]
            and bool(capacity["evidence"])
            and capacity["approved_capacity_bytes"] >= required_capacity_bytes
        )
    capacity["required_capacity_bytes"] = required_capacity_bytes
    capacity["sufficient"] = capacity_sufficient

    blockers = []
    if len(records) != EXPECTED_SHARD_COUNT:
        blockers.append(
            f"received {len(records)} shard reports; expected {EXPECTED_SHARD_COUNT}"
        )
    if missing_paths:
        blockers.append(f"{len(missing_paths)} expected shards are missing")
    if unexpected_paths:
        blockers.append(f"{len(unexpected_paths)} unexpected Parquet files are present")
    if unreported_paths:
        blockers.append(f"{len(unreported_paths)} expected shards have no report")
    if invalid_records:
        blockers.append(f"{len(invalid_records)} shard validations failed")
    if expected_shard_bytes is not None and (
        not isinstance(expected_shard_bytes, int)
        or isinstance(expected_shard_bytes, bool)
        or expected_shard_bytes < 0
    ):
        raise ValueError("expected_shard_bytes must be a non-negative integer or null")
    if expected_shard_bytes is None:
        blockers.append("author-approved expected shard byte total is not configured")
    elif total_bytes != expected_shard_bytes:
        blockers.append(
            f"shard bytes total {total_bytes:,}; expected {expected_shard_bytes:,}"
        )
    if (
        missing_reference_assemblies
        or unexpected_reference_assemblies
        or duplicate_reference_reports
        or invalid_reference_identities
    ):
        blockers.append(
            "reference reports do not match the expected assemblies: "
            f"missing={missing_reference_assemblies}, "
            f"unexpected={unexpected_reference_assemblies}, "
            f"duplicates={duplicate_reference_reports}, "
            f"invalid_identity={invalid_reference_identities}"
        )
    if not capacity_sufficient:
        blockers.append(
            "Hugging Face organization capacity and numeric release headroom are "
            "not confirmed"
        )

    return {
        "manifest_version": 1,
        "source": {
            "expected_shards": EXPECTED_SHARD_COUNT,
            "reported_shards": len(records),
            "discovered_parquet_files": len(discovered_paths),
            "missing_paths": missing_paths,
            "unexpected_paths": unexpected_paths,
            "unreported_paths": unreported_paths,
            "total_shard_bytes": total_bytes,
            "expected_shard_bytes": expected_shard_bytes,
            "box_reported_folder_bytes": BOX_REPORTED_FOLDER_BYTES,
            "box_folder_minus_shards_bytes": BOX_REPORTED_FOLDER_BYTES - total_bytes,
        },
        "references": references,
        "reference_inventory": {
            "missing_assemblies": missing_reference_assemblies,
            "unexpected_assemblies": unexpected_reference_assemblies,
            "duplicate_reports": duplicate_reference_reports,
            "invalid_identities": invalid_reference_identities,
        },
        "hugging_face_capacity": capacity,
        "validation": {
            "valid_shards": len(records) - len(invalid_records),
            "invalid_shards": len(invalid_records),
            "release_ready": not blockers,
            "blockers": blockers,
        },
        "shards": records,
    }


def render_summary(manifest: Mapping[str, Any]) -> str:
    """Render the human-readable companion to a machine-readable manifest."""

    source = manifest["source"]
    validation = manifest["validation"]
    capacity = manifest["hugging_face_capacity"]
    status = "PASS" if validation["release_ready"] else "BLOCKED"
    lines = [
        "# GPN-Star staged score inventory",
        "",
        f"Overall status: **{status}**",
        "",
        "## Source inventory",
        "",
        f"- Expected shards: {source['expected_shards']:,}",
        f"- Discovered Parquet files: {source['discovered_parquet_files']:,}",
        f"- Reported shard bytes: {source['total_shard_bytes']:,}",
        (
            "- Author-approved expected shard bytes: "
            + (
                f"{source['expected_shard_bytes']:,}"
                if source["expected_shard_bytes"] is not None
                else "not recorded"
            )
        ),
        f"- Box-reported folder bytes: {source['box_reported_folder_bytes']:,}",
        (
            "- Box folder minus current shard bytes: "
            f"{source['box_folder_minus_shards_bytes']:,}"
        ),
        f"- Valid shards: {validation['valid_shards']:,}",
        f"- Invalid shards: {validation['invalid_shards']:,}",
        "",
        "The Box folder byte count can include non-Parquet content and retained "
        "file versions. The manifest's shard total is computed from the 290 "
        "current Parquet objects.",
        "",
        "## Reference FASTAs",
        "",
    ]
    for reference in manifest["references"]:
        lines.append(
            f"- `{reference['assembly']}`: `{reference['source_filename']}`, "
            f"SHA-256 `{reference['source_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Hugging Face capacity",
            "",
            f"- Organization: `{capacity['organization']}`",
            f"- Confirmed: {'yes' if capacity['confirmed'] else 'no'}",
            (
                "- Required capacity: "
                + (
                    f"{capacity['required_capacity_bytes']:,} bytes"
                    if capacity["required_capacity_bytes"] is not None
                    else "not recorded"
                )
            ),
            (
                "- Approved capacity: "
                + (
                    f"{capacity['approved_capacity_bytes']:,} bytes"
                    if capacity["approved_capacity_bytes"] is not None
                    else "not recorded"
                )
            ),
            f"- Evidence: {capacity['evidence'] or 'not recorded'}",
            "",
            "## Blockers",
            "",
        ]
    )
    if validation["blockers"]:
        lines.extend(f"- {blocker}" for blocker in validation["blockers"])
    else:
        lines.append("- None")

    invalid = [record for record in manifest["shards"] if not record["valid"]]
    if invalid:
        lines.extend(["", "## Failed shard checks", ""])
        for record in invalid:
            checks = ", ".join(error["check"] for error in record["errors"])
            lines.append(f"- `{record['path']}`: {checks}")
    return "\n".join(lines) + "\n"


def write_release_outputs(
    source_root: Path,
    shard_reports: Sequence[Path],
    reference_reports: Sequence[Path],
    output_dir: Path,
    *,
    expected_shard_bytes: int | None,
    hugging_face_capacity: Mapping[str, Any] | None = None,
) -> None:
    """Build, validate, and atomically promote the final inventory directory."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.")
    )
    try:
        manifest = build_manifest(
            source_root,
            shard_reports,
            reference_reports,
            expected_shard_bytes=expected_shard_bytes,
            hugging_face_capacity=hugging_face_capacity,
        )
        manifest_path = temporary_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary_path = temporary_dir / "summary.md"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(render_summary(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
