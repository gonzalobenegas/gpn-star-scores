"""Streaming BigWig generation for GPN-Star score tracks.

The public functions in this module intentionally operate on one chromosome at
a time. Chromosomes are the workflow's restart unit, and the resulting files
can be combined after every per-chromosome artifact has been validated.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyBigWig

BASES = ("A", "C", "G", "T")
_BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}


class BigWigValidationError(ValueError):
    """Raised when source rows or a generated BigWig violate an invariant."""


@dataclass(frozen=True)
class ChromosomeSpec:
    """Explicit source-to-UCSC chromosome mapping and target length."""

    source_name: str
    ucsc_name: str
    length: int

    def __post_init__(self) -> None:
        if not self.source_name:
            raise ValueError("source chromosome name must not be empty")
        if not self.ucsc_name:
            raise ValueError("UCSC chromosome name must not be empty")
        if self.length <= 0:
            raise ValueError("chromosome length must be positive")


@dataclass(frozen=True)
class BigWigWriteStats:
    """Small, serializable summary returned by a chromosome writer."""

    source_chrom: str
    ucsc_chrom: str
    position_count: int
    first_position: int
    last_position: int


@dataclass(frozen=True)
class BigWigSummary:
    """Header facts checked by :func:`validate_bigwig`."""

    chromosome_sizes: dict[str, int]
    bases_covered: int
    zoom_levels: int


@dataclass(frozen=True)
class _LogoBatch:
    positions: np.ndarray
    heights: np.ndarray


def calibrated_llr_logo_heights(
    reference: str, alternate_scores: Mapping[str, float]
) -> dict[str, np.float32]:
    """Transform three calibrated LLRs into A/C/G/T information heights.

    The reference nucleotide receives logit zero.  The three supplied
    calibrated LLRs are treated as the alternate logits, followed by a stable
    Float64 softmax and base-2 entropy calculation.  These values are derived
    visualization heights, not raw model probabilities.
    """

    reference_index = _base_index(reference, "reference")
    expected_alternates = set(BASES) - {reference}
    if set(alternate_scores) != expected_alternates:
        raise BigWigValidationError(
            "alternate scores must contain each non-reference base exactly once"
        )

    logits = np.zeros((1, len(BASES)), dtype=np.float64)
    for base, score in alternate_scores.items():
        value = float(score)
        if not np.isfinite(value):
            raise BigWigValidationError("calibrated LLR values must be finite")
        logits[0, _base_index(base, "alternate")] = value

    heights = _logo_heights_from_logits(logits)[0]
    if logits[0, reference_index] != 0.0:  # pragma: no cover - defensive guard
        raise AssertionError("reference logit was not zero")
    return dict(zip(BASES, heights, strict=True))


def iter_entropy_track_batches(
    parquet_paths: Sequence[str | Path],
    chromosome: ChromosomeSpec,
    *,
    batch_size: int = 262_144,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield validated one-based positions and Float32 entropy values."""

    yield from _iter_entropy_batches(
        _validated_input_paths(parquet_paths), chromosome, batch_size=batch_size
    )


def iter_logo_track_batches(
    parquet_paths: Sequence[str | Path],
    chromosome: ChromosomeSpec,
    *,
    batch_size: int = 262_144,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield validated positions and A/C/G/T Float32 logo-height matrices."""

    for batch in _iter_logo_batches(
        _validated_input_paths(parquet_paths), chromosome, batch_size=batch_size
    ):
        yield batch.positions, batch.heights


def write_entropy_bigwig(
    parquet_paths: Sequence[str | Path],
    output_path: str | Path,
    chromosome: ChromosomeSpec,
    *,
    batch_size: int = 262_144,
    header_chromosome_sizes: Mapping[str, int] | None = None,
) -> BigWigWriteStats:
    """Stream one chromosome's calibrated entropy rows into a BigWig.

    Source positions remain one-based in Parquet and are converted explicitly
    to zero-based, one-base-wide BigWig entries at the writer boundary.
    """

    paths = _validated_input_paths(parquet_paths)
    header_sizes = _validated_header_sizes(chromosome, header_chromosome_sizes)
    output = Path(output_path)
    temporary = _temporary_sibling(output)
    writer: Any | None = None
    count = 0
    first_position: int | None = None
    last_position: int | None = None

    try:
        writer = _open_writer(temporary, header_sizes)
        for positions, values in _iter_entropy_batches(
            paths, chromosome, batch_size=batch_size
        ):
            _add_sparse_values(writer, chromosome.ucsc_name, positions, values)
            count += len(positions)
            first_position = (
                int(positions[0]) if first_position is None else first_position
            )
            last_position = int(positions[-1])

        if count == 0:
            raise BigWigValidationError("entropy input contains no rows")
        writer.close()
        writer = None
        validate_bigwig(
            temporary,
            header_sizes,
            expected_bases_covered=count,
        )
        os.replace(temporary, output)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    assert first_position is not None
    assert last_position is not None
    return BigWigWriteStats(
        source_chrom=chromosome.source_name,
        ucsc_chrom=chromosome.ucsc_name,
        position_count=count,
        first_position=first_position,
        last_position=last_position,
    )


def write_logo_bigwigs(
    parquet_paths: Sequence[str | Path],
    output_paths: Mapping[str, str | Path],
    chromosome: ChromosomeSpec,
    *,
    batch_size: int = 262_144,
    header_chromosome_sizes: Mapping[str, int] | None = None,
) -> BigWigWriteStats:
    """Stream one chromosome's LLR rows into four derived logo BigWigs."""

    paths = _validated_input_paths(parquet_paths)
    header_sizes = _validated_header_sizes(chromosome, header_chromosome_sizes)
    outputs = _validated_logo_outputs(output_paths)
    temporary = {base: _temporary_sibling(path) for base, path in outputs.items()}
    writers: dict[str, Any] = {}
    count = 0
    first_position: int | None = None
    last_position: int | None = None

    try:
        for base in BASES:
            writers[base] = _open_writer(temporary[base], header_sizes)
        for logo_batch in _iter_logo_batches(paths, chromosome, batch_size=batch_size):
            for base_index, base in enumerate(BASES):
                _add_sparse_values(
                    writers[base],
                    chromosome.ucsc_name,
                    logo_batch.positions,
                    logo_batch.heights[:, base_index],
                )
            count += len(logo_batch.positions)
            first_position = (
                int(logo_batch.positions[0])
                if first_position is None
                else first_position
            )
            last_position = int(logo_batch.positions[-1])

        if count == 0:
            raise BigWigValidationError("LLR input contains no positions")
        for writer in writers.values():
            writer.close()
        writers.clear()

        for base in BASES:
            validate_bigwig(
                temporary[base],
                header_sizes,
                expected_bases_covered=count,
            )
        for base in BASES:
            os.replace(temporary[base], outputs[base])
    except BaseException:
        for writer in writers.values():
            writer.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    assert first_position is not None
    assert last_position is not None
    return BigWigWriteStats(
        source_chrom=chromosome.source_name,
        ucsc_chrom=chromosome.ucsc_name,
        position_count=count,
        first_position=first_position,
        last_position=last_position,
    )


def write_entropy_wig(
    parquet_paths: Sequence[str | Path],
    output_path: str | Path,
    chromosome: ChromosomeSpec,
    *,
    batch_size: int = 262_144,
) -> BigWigWriteStats:
    """Write the variable-step WIG baseline for one entropy chromosome."""

    paths = _validated_input_paths(parquet_paths)
    output = Path(output_path)
    temporary = _temporary_sibling(output)
    count = 0
    first_position: int | None = None
    last_position: int | None = None

    try:
        with temporary.open("w") as handle:
            _write_wig_header(handle, chromosome.ucsc_name)
            for positions, values in _iter_entropy_batches(
                paths, chromosome, batch_size=batch_size
            ):
                _write_wig_values(handle, positions, values)
                count += len(positions)
                first_position = (
                    int(positions[0]) if first_position is None else first_position
                )
                last_position = int(positions[-1])
        if count == 0:
            raise BigWigValidationError("entropy input contains no rows")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    assert first_position is not None
    assert last_position is not None
    return BigWigWriteStats(
        source_chrom=chromosome.source_name,
        ucsc_chrom=chromosome.ucsc_name,
        position_count=count,
        first_position=first_position,
        last_position=last_position,
    )


def write_logo_wigs(
    parquet_paths: Sequence[str | Path],
    output_paths: Mapping[str, str | Path],
    chromosome: ChromosomeSpec,
    *,
    batch_size: int = 262_144,
) -> BigWigWriteStats:
    """Write four variable-step WIG baselines for one LLR chromosome."""

    paths = _validated_input_paths(parquet_paths)
    outputs = _validated_logo_outputs(output_paths)
    temporary = {base: _temporary_sibling(path) for base, path in outputs.items()}
    handles: dict[str, Any] = {}
    count = 0
    first_position: int | None = None
    last_position: int | None = None

    try:
        for base in BASES:
            handles[base] = temporary[base].open("w")
        for handle in handles.values():
            _write_wig_header(handle, chromosome.ucsc_name)
        for logo_batch in _iter_logo_batches(paths, chromosome, batch_size=batch_size):
            for base_index, base in enumerate(BASES):
                _write_wig_values(
                    handles[base],
                    logo_batch.positions,
                    logo_batch.heights[:, base_index],
                )
            count += len(logo_batch.positions)
            first_position = (
                int(logo_batch.positions[0])
                if first_position is None
                else first_position
            )
            last_position = int(logo_batch.positions[-1])

        if count == 0:
            raise BigWigValidationError("LLR input contains no positions")
        for handle in handles.values():
            handle.close()
        handles.clear()
        for base in BASES:
            os.replace(temporary[base], outputs[base])
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    assert first_position is not None
    assert last_position is not None
    return BigWigWriteStats(
        source_chrom=chromosome.source_name,
        ucsc_chrom=chromosome.ucsc_name,
        position_count=count,
        first_position=first_position,
        last_position=last_position,
    )


def validate_bigwig(
    path: str | Path,
    expected_chromosome_sizes: Mapping[str, int],
    *,
    expected_bases_covered: int | None = None,
    require_zoom_levels: bool = True,
) -> BigWigSummary:
    """Open and validate a generated BigWig's core structural metadata."""

    bigwig = pyBigWig.open(str(path))
    if bigwig is None or not bigwig.isBigWig():
        raise BigWigValidationError(f"not a readable BigWig: {path}")
    try:
        chromosome_sizes = {
            str(key): int(value) for key, value in bigwig.chroms().items()
        }
        expected_sizes = {
            str(key): int(value) for key, value in expected_chromosome_sizes.items()
        }
        if chromosome_sizes != expected_sizes:
            raise BigWigValidationError(
                f"chromosome sizes differ: {chromosome_sizes!r} != {expected_sizes!r}"
            )

        header = bigwig.header()
        bases_covered = int(header["nBasesCovered"])
        zoom_levels = int(header["nLevels"])
        if (
            expected_bases_covered is not None
            and bases_covered != expected_bases_covered
        ):
            raise BigWigValidationError(
                f"covered bases differ: {bases_covered} != {expected_bases_covered}"
            )
        if require_zoom_levels and zoom_levels < 1:
            raise BigWigValidationError("BigWig contains no zoom levels")
        return BigWigSummary(chromosome_sizes, bases_covered, zoom_levels)
    finally:
        bigwig.close()


def _iter_entropy_batches(
    paths: Sequence[Path], chromosome: ChromosomeSpec, *, batch_size: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    previous_position: int | None = None
    for batch in _parquet_batches(
        paths,
        columns=("chrom", "pos", "ref", "entropy_calibrated"),
        batch_size=batch_size,
    ):
        _validate_batch_schema(
            batch,
            string_columns=("chrom", "ref"),
            float_columns=("entropy_calibrated",),
        )
        _validate_source_chromosome(batch.column("chrom"), chromosome.source_name)
        _base_indices(batch.column("ref"), "reference")
        positions = batch.column("pos").to_numpy(zero_copy_only=False)
        values = batch.column("entropy_calibrated").to_numpy(zero_copy_only=False)
        _validate_positions(
            positions,
            chromosome,
            previous_position=previous_position,
            strictly_increasing=True,
        )
        if not np.isfinite(values).all():
            raise BigWigValidationError("entropy values must be finite")
        previous_position = int(positions[-1])
        yield positions, values


def _iter_logo_batches(
    paths: Sequence[Path], chromosome: ChromosomeSpec, *, batch_size: int
) -> Iterable[_LogoBatch]:
    pending_positions = np.empty(0, dtype=np.int64)
    pending_references = np.empty(0, dtype=np.int8)
    pending_alternates = np.empty(0, dtype=np.int8)
    pending_scores = np.empty(0, dtype=np.float32)

    for batch in _parquet_batches(
        paths,
        columns=("chrom", "pos", "ref", "alt", "llr_calibrated"),
        batch_size=batch_size,
    ):
        _validate_batch_schema(
            batch,
            string_columns=("chrom", "ref", "alt"),
            float_columns=("llr_calibrated",),
        )
        _validate_source_chromosome(batch.column("chrom"), chromosome.source_name)
        positions = batch.column("pos").to_numpy(zero_copy_only=False)
        references = _base_indices(batch.column("ref"), "reference")
        alternates = _base_indices(batch.column("alt"), "alternate")
        scores = batch.column("llr_calibrated").to_numpy(zero_copy_only=False)
        if not np.isfinite(scores).all():
            raise BigWigValidationError("calibrated LLR values must be finite")

        positions = np.concatenate((pending_positions, positions))
        references = np.concatenate((pending_references, references))
        alternates = np.concatenate((pending_alternates, alternates))
        scores = np.concatenate((pending_scores, scores))
        _validate_positions(
            positions,
            chromosome,
            previous_position=None,
            strictly_increasing=False,
        )

        changes = np.flatnonzero(positions[:-1] != positions[1:])
        split = int(changes[-1] + 1) if len(changes) else 0
        if split:
            yield _make_logo_batch(
                positions[:split],
                references[:split],
                alternates[:split],
                scores[:split],
            )
        pending_positions = positions[split:]
        pending_references = references[split:]
        pending_alternates = alternates[split:]
        pending_scores = scores[split:]

    if len(pending_positions):
        yield _make_logo_batch(
            pending_positions,
            pending_references,
            pending_alternates,
            pending_scores,
        )


def _make_logo_batch(
    positions: np.ndarray,
    references: np.ndarray,
    alternates: np.ndarray,
    scores: np.ndarray,
) -> _LogoBatch:
    group_starts = np.concatenate(
        ([0], np.flatnonzero(positions[:-1] != positions[1:]) + 1)
    )
    group_sizes = np.diff(np.append(group_starts, len(positions)))
    if not np.all(group_sizes == 3):
        bad_position = int(positions[group_starts[np.flatnonzero(group_sizes != 3)[0]]])
        raise BigWigValidationError(
            f"position {bad_position} does not have exactly three LLR rows"
        )

    grouped_positions = positions.reshape(-1, 3)
    grouped_references = references.reshape(-1, 3)
    grouped_alternates = alternates.reshape(-1, 3)
    grouped_scores = scores.reshape(-1, 3)
    if not np.all(grouped_positions == grouped_positions[:, :1]):
        raise AssertionError("LLR grouping produced mixed positions")
    if not np.all(grouped_references == grouped_references[:, :1]):
        raise BigWigValidationError("reference base differs within an LLR position")

    reference_indices = grouped_references[:, 0]
    alternate_masks = np.bitwise_or.reduce(1 << grouped_alternates, axis=1)
    expected_masks = 0b1111 ^ (1 << reference_indices)
    if not np.array_equal(alternate_masks, expected_masks):
        raise BigWigValidationError(
            "alternate bases must be the three unique non-reference bases"
        )

    logits = np.zeros((len(grouped_positions), len(BASES)), dtype=np.float64)
    row_indices = np.repeat(np.arange(len(grouped_positions)), 3)
    logits[row_indices, grouped_alternates.ravel()] = grouped_scores.ravel()
    heights = _logo_heights_from_logits(logits)
    return _LogoBatch(grouped_positions[:, 0], heights)


def _logo_heights_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizers = np.sum(exponentials, axis=1, keepdims=True)
    probabilities = exponentials / normalizers
    log_probabilities = shifted - np.log(normalizers)
    entropy = -np.sum(probabilities * (log_probabilities / np.log(2.0)), axis=1)
    information_content = np.clip(2.0 - entropy, 0.0, 2.0)
    return (probabilities * information_content[:, None]).astype(np.float32)


def _parquet_batches(
    paths: Sequence[Path], *, columns: Sequence[str], batch_size: int
) -> Iterable[pa.RecordBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for path in paths:
        parquet = pq.ParquetFile(path)
        yield from parquet.iter_batches(batch_size=batch_size, columns=list(columns))


def _validate_batch_schema(
    batch: pa.RecordBatch,
    *,
    string_columns: Sequence[str],
    float_columns: Sequence[str],
) -> None:
    required = (*string_columns, "pos", *float_columns)
    missing = [name for name in required if name not in batch.schema.names]
    if missing:
        raise BigWigValidationError(f"missing required columns: {missing!r}")
    for name in required:
        if batch.column(name).null_count:
            raise BigWigValidationError(f"column {name!r} contains nulls")
    for name in string_columns:
        field_type = batch.schema.field(name).type
        if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
            raise BigWigValidationError(f"column {name!r} must be a string")
    if batch.schema.field("pos").type != pa.int64():
        raise BigWigValidationError("column 'pos' must be Int64")
    for name in float_columns:
        if batch.schema.field(name).type != pa.float32():
            raise BigWigValidationError(f"column {name!r} must be Float32")


def _validate_source_chromosome(chromosomes: pa.Array, expected: str) -> None:
    if pc.any(pc.not_equal(chromosomes, pa.scalar(expected))).as_py():
        raise BigWigValidationError(
            f"input contains a chromosome other than {expected!r}"
        )


def _validate_positions(
    positions: np.ndarray,
    chromosome: ChromosomeSpec,
    *,
    previous_position: int | None,
    strictly_increasing: bool,
) -> None:
    if not len(positions):
        return
    if positions[0] < 1 or positions[-1] > chromosome.length:
        raise BigWigValidationError(f"positions must be within 1..{chromosome.length}")
    differences = np.diff(positions)
    invalid = differences <= 0 if strictly_increasing else differences < 0
    if np.any(invalid):
        ordering = "strictly increasing" if strictly_increasing else "nondecreasing"
        raise BigWigValidationError(f"positions must be {ordering}")
    if previous_position is not None:
        valid = positions[0] > previous_position
        if not valid:
            raise BigWigValidationError("positions must be strictly increasing")


def _base_indices(values: pa.Array, role: str) -> np.ndarray:
    value_set = pa.array(BASES, type=values.type)
    indices = pc.index_in(values, value_set=value_set)
    if indices.null_count:
        raise BigWigValidationError(f"{role} bases must be one of {BASES!r}")
    return indices.to_numpy(zero_copy_only=False).astype(np.int8, copy=False)


def _base_index(base: str, role: str) -> int:
    try:
        return _BASE_TO_INDEX[base]
    except KeyError as error:
        raise BigWigValidationError(f"{role} base must be one of {BASES!r}") from error


def _add_sparse_values(
    writer: Any, chromosome: str, one_based_positions: np.ndarray, values: np.ndarray
) -> None:
    starts = one_based_positions - 1
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(starts) != 1) + 1, [len(starts)])
    )
    pending_starts: list[int] = []
    pending_values: list[float] = []

    def flush_variable_step() -> None:
        if pending_starts:
            writer.addEntries(
                chromosome,
                pending_starts,
                values=pending_values,
                span=1,
            )
            pending_starts.clear()
            pending_values.clear()

    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - begin == 1:
            pending_starts.append(int(starts[begin]))
            pending_values.append(float(values[begin]))
            continue
        flush_variable_step()
        writer.addEntries(
            chromosome,
            int(starts[begin]),
            values=values[begin:end].tolist(),
            span=1,
            step=1,
        )
    flush_variable_step()


def _write_wig_header(handle: Any, chromosome: str) -> None:
    handle.write(f"variableStep chrom={chromosome} span=1\n")


def _write_wig_values(
    handle: Any, one_based_positions: np.ndarray, values: np.ndarray
) -> None:
    # Nine significant decimal digits are sufficient for exact Float32
    # round-tripping.  WIG positions are one-based, unlike BigWig starts.
    np.savetxt(
        handle,
        np.column_stack((one_based_positions, values)),
        fmt=("%d", "%.9g"),
        delimiter="\t",
    )


def _open_writer(path: Path, chromosome_sizes: Mapping[str, int]) -> Any:
    writer = pyBigWig.open(str(path), "w")
    if writer is None:
        raise OSError(f"could not open BigWig for writing: {path}")
    # Omitting maxZooms preserves pyBigWig/libBigWig's default zoom levels.
    try:
        writer.addHeader(list(chromosome_sizes.items()))
    except BaseException:
        writer.close()
        raise
    return writer


def _temporary_sibling(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    return Path(name)


def _validated_input_paths(paths: Sequence[str | Path]) -> list[Path]:
    result = [Path(path) for path in paths]
    if not result:
        raise ValueError("at least one Parquet path is required")
    return result


def _validated_header_sizes(
    chromosome: ChromosomeSpec,
    chromosome_sizes: Mapping[str, int] | None,
) -> dict[str, int]:
    if chromosome_sizes is None:
        return {chromosome.ucsc_name: chromosome.length}
    result: dict[str, int] = {}
    for name, length in chromosome_sizes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("BigWig header chromosome names must be non-empty strings")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(
                "BigWig header chromosome lengths must be positive integers"
            )
        result[name] = length
    if result.get(chromosome.ucsc_name) != chromosome.length:
        raise ValueError(
            "BigWig header must contain the active chromosome with its exact length"
        )
    return result


def _validated_logo_outputs(
    outputs: Mapping[str, str | Path],
) -> dict[str, Path]:
    if set(outputs) != set(BASES):
        raise ValueError(f"logo outputs must have exactly these keys: {BASES!r}")
    result = {base: Path(outputs[base]) for base in BASES}
    if len(set(result.values())) != len(BASES):
        raise ValueError("logo output paths must be distinct")
    return result
