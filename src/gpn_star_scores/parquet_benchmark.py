"""Reproducible Parquet layout rewrites, benchmarks, and selection reports."""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import resource
import sys
import tempfile
import time
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
from statistics import median
from typing import Any, BinaryIO

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemFile

from gpn_star_scores.catalog import EXPECTED_SHARD_COUNT, expected_shards

DICTIONARY_COLUMNS = ("chrom", "ref", "alt")
POSITION_COLUMN = "pos"
DEFAULT_REPETITIONS = 5
DEFAULT_WARMUPS = 1
DEFAULT_SPARSE_KEYS = 1_024
EXACT_VALUE_BATCH_ROWS = 65_536


@dataclass(frozen=True)
class LayoutCandidate:
    """One issue #5 Parquet layout candidate."""

    name: str
    row_group_rows: int | None

    @property
    def rewrites_source(self) -> bool:
        return self.row_group_rows is not None


CANDIDATES: tuple[LayoutCandidate, ...] = (
    LayoutCandidate("source", None),
    LayoutCandidate("zstd-262144", 262_144),
    LayoutCandidate("zstd-1048576", 1_048_576),
)


@dataclass(frozen=True)
class QuerySpec:
    """A deterministic query included in each benchmark run."""

    name: str
    kind: str
    start: int | None = None
    end: int | None = None


class _TransferCounter:
    """Mutable byte counter shared by one Hugging Face file handle."""

    def __init__(self) -> None:
        self.bytes = 0


class _CountingHfFile(HfFileSystemFile):
    """Count HTTP range response bodies fetched through ``HfFileSystem``."""

    def __init__(
        self,
        fs: HfFileSystem,
        path: str,
        *,
        counter: _TransferCounter,
        **kwargs: Any,
    ) -> None:
        self._transfer_counter = counter
        super().__init__(fs, path, **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        content = super()._fetch_range(start, end)
        self._transfer_counter.bytes += len(content)
        return content

    def read(self, length: int = -1) -> bytes:
        # HfFileSystemFile special-cases an unbounded read by opening an
        # uncounted streaming handle. Use the fsspec buffered implementation so
        # every payload still passes through the counted range fetcher.
        return super(HfFileSystemFile, self).read(length)


class _CountingHfFileSystem(HfFileSystem):
    """Open seekable HF files while measuring fetched range-response bytes."""

    cachable = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.counter = _TransferCounter()

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> HfFileSystemFile:
        if mode != "rb":
            raise ValueError("counted Hugging Face access is read-only")
        if block_size == 0:
            raise ValueError("range benchmarks require a seekable buffered file")
        effective_block_size = block_size if block_size is not None else self.block_size
        if effective_block_size is not None:
            kwargs["block_size"] = effective_block_size
        return _CountingHfFile(
            self,
            path,
            mode=mode,
            revision=revision,
            counter=self.counter,
            **kwargs,
        )


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_threads(threads: int) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    pa.set_cpu_count(threads)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize JSON to a temporary sibling and atomically promote it."""

    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(path, text)


def _expected_fields(score_type: str) -> tuple[tuple[str, pa.DataType], ...]:
    common = (("chrom", pa.string()), ("pos", pa.int64()), ("ref", pa.string()))
    if score_type == "entropy":
        return common + (("entropy_calibrated", pa.float32()),)
    if score_type == "llr":
        return common + (
            ("alt", pa.string()),
            ("llr_calibrated", pa.float32()),
            ("abs_llr_calibrated", pa.float32()),
        )
    raise ValueError(f"unknown score type: {score_type!r}")


def validate_score_schema(schema: pa.Schema, score_type: str) -> None:
    """Reject schema drift before a benchmark candidate is written."""

    expected = _expected_fields(score_type)
    observed = tuple((field.name, field.type) for field in schema)
    if observed != expected:
        raise ValueError(f"{score_type} schema is {observed!r}; expected {expected!r}")


def score_type_for_schema(schema: pa.Schema) -> str:
    """Infer entropy or LLR only when the complete canonical schema matches."""

    for score_type in ("entropy", "llr"):
        expected = _expected_fields(score_type)
        observed = tuple((field.name, field.type) for field in schema)
        if observed == expected:
            return score_type
    raise ValueError(f"schema is not a canonical entropy or LLR schema: {schema}")


def _float_array_bits(array: pa.Array) -> np.ndarray:
    values = array.to_numpy(zero_copy_only=False)
    return values.view(np.uint32)


def _arrays_exactly_equal(left: pa.Array, right: pa.Array) -> bool:
    if left.type != right.type or len(left) != len(right):
        return False
    if not left.is_null().equals(right.is_null()):
        return False
    if pa.types.is_float32(left.type):
        valid = np.logical_not(left.is_null().to_numpy(zero_copy_only=False))
        return bool(
            np.array_equal(
                _float_array_bits(left)[valid], _float_array_bits(right)[valid]
            )
        )
    return left.equals(right)


def verify_exact_values(source_path: Path, candidate_path: Path) -> None:
    """Verify identical schema, row order, and logical value bit patterns."""

    source = pq.ParquetFile(source_path)
    candidate = pq.ParquetFile(candidate_path)
    if not source.schema_arrow.equals(candidate.schema_arrow, check_metadata=True):
        raise ValueError("candidate schema or schema metadata differs from source")
    if source.metadata.num_rows != candidate.metadata.num_rows:
        raise ValueError("candidate row count differs from source")

    source_batches = source.iter_batches(batch_size=EXACT_VALUE_BATCH_ROWS)
    candidate_batches = candidate.iter_batches(batch_size=EXACT_VALUE_BATCH_ROWS)
    for batch_index, pair in enumerate(
        zip_longest(source_batches, candidate_batches), start=1
    ):
        left, right = pair
        if left is None or right is None:
            raise ValueError("candidate batch count differs from source")
        if left.num_rows != right.num_rows or left.num_columns != right.num_columns:
            raise ValueError(f"candidate batch {batch_index} shape differs from source")
        for column_index, name in enumerate(source.schema_arrow.names):
            if not _arrays_exactly_equal(
                left.column(column_index), right.column(column_index)
            ):
                raise ValueError(
                    f"candidate values differ in batch {batch_index}, column {name!r}"
                )


def inspect_position_statistics(path: Path) -> dict[str, Any]:
    """Check complete, ordered row-group position min/max statistics."""

    parquet_file = pq.ParquetFile(path)
    score_type = score_type_for_schema(parquet_file.schema_arrow)
    position_index = parquet_file.schema_arrow.get_field_index(POSITION_COLUMN)
    bounds = []
    monotonic = True
    complete = True
    previous_maximum: int | None = None
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = (
            parquet_file.metadata.row_group(row_group_index)
            .column(position_index)
            .statistics
        )
        if statistics is None or not statistics.has_min_max:
            complete = False
            bounds.append({"row_group": row_group_index, "min": None, "max": None})
            continue
        minimum = int(statistics.min)
        maximum = int(statistics.max)
        if minimum < 1 or maximum < minimum:
            monotonic = False
        if previous_maximum is not None:
            if score_type == "entropy" and minimum <= previous_maximum:
                monotonic = False
            if score_type == "llr" and minimum < previous_maximum:
                monotonic = False
        previous_maximum = maximum
        bounds.append({"row_group": row_group_index, "min": minimum, "max": maximum})
    return {
        "usable": complete and monotonic and bool(bounds),
        "complete": complete,
        "monotonic": monotonic,
        "row_group_bounds": bounds,
    }


def _physical_layout(path: Path, row_group_rows: int) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    expected_dictionary = set(DICTIONARY_COLUMNS) & set(parquet_file.schema_arrow.names)
    row_group_sizes: list[int] = []
    dictionary_columns: set[str] = set()
    columns_with_statistics: set[str] = set()
    columns_with_page_index: set[str] = set()
    codecs: set[str] = set()

    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        row_group_sizes.append(row_group.num_rows)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            name = column.path_in_schema
            codecs.add(column.compression)
            if (
                "RLE_DICTIONARY" in column.encodings
                or "PLAIN_DICTIONARY" in column.encodings
            ):
                dictionary_columns.add(name)
            if column.statistics is not None and column.statistics.has_min_max:
                columns_with_statistics.add(name)
            if bool(getattr(column, "has_column_index", False)) and bool(
                getattr(column, "has_offset_index", False)
            ):
                columns_with_page_index.add(name)

    expected_sizes = [row_group_rows] * (metadata.num_rows // row_group_rows)
    remainder = metadata.num_rows % row_group_rows
    if remainder:
        expected_sizes.append(remainder)
    if row_group_sizes != expected_sizes:
        raise ValueError(
            f"candidate row groups are {row_group_sizes!r}; expected {expected_sizes!r}"
        )
    if codecs != {"ZSTD"}:
        raise ValueError(f"candidate codecs are {sorted(codecs)!r}, expected ZSTD")
    if dictionary_columns != expected_dictionary:
        raise ValueError(
            "candidate dictionary columns are "
            f"{sorted(dictionary_columns)!r}; expected {sorted(expected_dictionary)!r}"
        )
    expected_columns = set(parquet_file.schema_arrow.names)
    if columns_with_statistics != expected_columns:
        raise ValueError("candidate is missing column min/max statistics")
    if columns_with_page_index != expected_columns:
        raise ValueError("candidate is missing column or offset page indexes")
    position_statistics = inspect_position_statistics(path)
    if not position_statistics["usable"]:
        raise ValueError("candidate position statistics are not usable")

    return {
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "row_group_rows": row_group_sizes,
        "compression": "ZSTD",
        "compression_level": 3,
        "dictionary_columns": sorted(dictionary_columns),
        "statistics_columns": sorted(columns_with_statistics),
        "page_index_columns": sorted(columns_with_page_index),
        "content_defined_chunking": True,
        "position_statistics": position_statistics,
    }


def rewrite_parquet_candidate(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    case: str,
    score_type: str,
    candidate: LayoutCandidate,
    threads: int = 4,
) -> None:
    """Write, fully validate, and atomically promote one rewrite candidate."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    if not candidate.rewrites_source or candidate.row_group_rows is None:
        raise ValueError("the source candidate is not rewritten")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("refusing to replace the immutable source file")
    _configure_threads(threads)
    source_file = pq.ParquetFile(source_path)
    validate_score_schema(source_file.schema_arrow, score_type)
    source_stat = source_path.stat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    write_start = time.perf_counter()
    try:
        dictionary_columns = [
            name
            for name in DICTIONARY_COLUMNS
            if name in source_file.schema_arrow.names
        ]
        with pq.ParquetWriter(
            temporary_name,
            source_file.schema_arrow,
            compression="zstd",
            compression_level=3,
            use_dictionary=dictionary_columns,
            write_statistics=True,
            write_page_index=True,
            use_content_defined_chunking=True,
        ) as writer:
            for batch in source_file.iter_batches(
                batch_size=candidate.row_group_rows, use_threads=threads > 1
            ):
                writer.write_batch(batch, row_group_size=candidate.row_group_rows)

        write_seconds = time.perf_counter() - write_start
        validation_start = time.perf_counter()
        physical_layout = _physical_layout(
            Path(temporary_name), candidate.row_group_rows
        )
        verify_exact_values(source_path, Path(temporary_name))
        output_sha256 = _sha256_file(Path(temporary_name))
        current_source_stat = source_path.stat()
        if (
            source_stat.st_size != current_source_stat.st_size
            or source_stat.st_mtime_ns != current_source_stat.st_mtime_ns
            or source_stat.st_ino != current_source_stat.st_ino
        ):
            raise RuntimeError("immutable source identity changed during rewrite")
        validation_seconds = time.perf_counter() - validation_start
        output_size = Path(temporary_name).stat().st_size
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    atomic_write_json(
        report_path,
        {
            "report_version": 1,
            "case": case,
            "candidate": asdict(candidate),
            "score_type": score_type,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "source_size_bytes": source_stat.st_size,
            "output_size_bytes": output_size,
            "write_seconds": write_seconds,
            "validation_seconds": validation_seconds,
            "output_sha256": output_sha256,
            "peak_rss_bytes": _peak_rss_bytes(),
            "threads": threads,
            "source_unchanged": True,
            "exact_value_equality": True,
            "physical_layout": physical_layout,
            "environment": environment_record(),
        },
    )


def environment_record() -> dict[str, Any]:
    """Return the pinned runtime information needed to reproduce a run."""

    packages = {}
    for package in ("gpn-star-scores", "polars", "pyarrow", "huggingface-hub"):
        try:
            packages[package] = version(package)
        except Exception:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "generated_at": datetime.now(UTC).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }


def interval_query_specs(position_min: int, position_max: int) -> list[QuerySpec]:
    """Create first, middle, and last 1 kb and 1 Mb interval queries."""

    if position_min < 1 or position_max < position_min:
        raise ValueError("invalid one-based position bounds")
    result = []
    midpoint = position_min + (position_max - position_min) // 2
    for width in (1_000, 1_000_000):
        first_start = position_min
        first_end = min(position_max, first_start + width - 1)
        middle_start = max(position_min, midpoint - width // 2)
        middle_end = min(position_max, middle_start + width - 1)
        middle_start = max(position_min, middle_end - width + 1)
        last_end = position_max
        last_start = max(position_min, last_end - width + 1)
        for location, start, end in (
            ("first", first_start, first_end),
            ("middle", middle_start, middle_end),
            ("last", last_start, last_end),
        ):
            result.append(
                QuerySpec(
                    name=f"interval-{location}-{width}",
                    kind="interval",
                    start=start,
                    end=end,
                )
            )
    return result


def _scan(source: str | BinaryIO) -> pl.LazyFrame:
    return pl.scan_parquet(source, cache=False, rechunk=False, low_memory=True)


def _score_columns(schema: pl.Schema) -> list[str]:
    columns = [name for name in schema.names() if name.endswith("_calibrated")]
    if not columns:
        raise ValueError("Parquet input has no calibrated score columns")
    return columns


def _key_columns(schema: pl.Schema) -> list[str]:
    names = set(schema.names())
    keys = [name for name in ("chrom", "pos", "ref", "alt") if name in names]
    if keys not in (["chrom", "pos", "ref"], ["chrom", "pos", "ref", "alt"]):
        raise ValueError(f"unexpected key columns: {keys!r}")
    return keys


def _full_scan_expressions(schema: pl.Schema) -> list[pl.Expr]:
    expressions = [pl.len().alias("rows")]
    for name, dtype in schema.items():
        column = pl.col(name)
        if dtype == pl.String:
            expressions.append(column.str.len_bytes().sum().alias(f"{name}_bytes"))
        elif dtype.is_numeric():
            expressions.append(column.sum().alias(f"{name}_sum"))
        else:
            expressions.append(column.null_count().alias(f"{name}_nulls"))
    return expressions


def _build_query(
    scan: pl.LazyFrame,
    spec: QuerySpec,
    sparse_keys: pl.DataFrame,
) -> pl.LazyFrame:
    schema = scan.collect_schema()
    if spec.kind == "interval":
        if spec.start is None or spec.end is None:
            raise ValueError("interval query is missing bounds")
        return scan.filter(pl.col(POSITION_COLUMN).is_between(spec.start, spec.end))
    if spec.kind == "projection":
        score_columns = _score_columns(schema)
        return scan.select(
            pl.len().alias("rows"),
            *[pl.col(name).sum().alias(f"{name}_sum") for name in score_columns],
        )
    if spec.kind == "sparse_join":
        return scan.join(sparse_keys.lazy(), on=_key_columns(schema), how="inner")
    if spec.kind == "full_scan":
        return scan.select(*_full_scan_expressions(schema))
    raise ValueError(f"unknown query kind: {spec.kind!r}")


def _matching_row_groups(
    parquet_file: pq.ParquetFile,
    *,
    interval: tuple[int, int] | None = None,
    positions: Sequence[int] = (),
) -> list[int]:
    """Select row groups whose position statistics can match a query."""

    if (interval is None) == (not positions):
        raise ValueError("provide exactly one interval or position sequence")
    position_index = parquet_file.schema_arrow.get_field_index(POSITION_COLUMN)
    if position_index < 0:
        raise ValueError("Parquet input has no position column")
    ordered_positions = sorted(set(positions))
    selected = []
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = (
            parquet_file.metadata.row_group(row_group_index)
            .column(position_index)
            .statistics
        )
        if statistics is None or not statistics.has_min_max:
            return list(range(parquet_file.metadata.num_row_groups))
        minimum = int(statistics.min)
        maximum = int(statistics.max)
        if interval is not None:
            start, end = interval
            matches = maximum >= start and minimum <= end
        else:
            position_index_in_query = bisect_left(ordered_positions, minimum)
            matches = (
                position_index_in_query < len(ordered_positions)
                and ordered_positions[position_index_in_query] <= maximum
            )
        if matches:
            selected.append(row_group_index)
    return selected


def _execute_hf_query(
    source: BinaryIO,
    spec: QuerySpec,
    sparse_keys: pl.DataFrame,
) -> pl.DataFrame:
    """Execute one counted query through seekable Parquet range reads."""

    parquet_file = pq.ParquetFile(source)
    schema_names = parquet_file.schema_arrow.names
    if spec.kind == "projection":
        columns = [name for name in schema_names if name.endswith("_calibrated")]
        if not columns:
            raise ValueError("Parquet input has no calibrated score columns")
        table = parquet_file.read(columns=columns, use_threads=True)
    elif spec.kind == "full_scan":
        table = parquet_file.read(use_threads=True)
    elif spec.kind == "interval":
        if spec.start is None or spec.end is None:
            raise ValueError("interval query is missing bounds")
        row_groups = _matching_row_groups(parquet_file, interval=(spec.start, spec.end))
        table = parquet_file.read_row_groups(row_groups, use_threads=True)
    elif spec.kind == "sparse_join":
        row_groups = _matching_row_groups(
            parquet_file,
            positions=[int(value) for value in sparse_keys[POSITION_COLUMN]],
        )
        table = parquet_file.read_row_groups(row_groups, use_threads=True)
    else:
        raise ValueError(f"unknown query kind: {spec.kind!r}")
    frame = pl.from_arrow(table, rechunk=False)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("expected a Polars DataFrame from the Arrow table")
    return _build_query(frame.lazy(), spec, sparse_keys).collect()


@contextmanager
def _benchmark_source(
    uri: str,
    access: str,
    *,
    hf_token: str | None,
    hf_block_size: int,
    hf_filesystem: _CountingHfFileSystem | None = None,
) -> Iterable[tuple[str | BinaryIO, Callable[[], int | None]]]:
    if access == "local":
        yield uri, lambda: None
        return
    if access != "hf":
        raise ValueError(f"unknown access mode: {access!r}")
    filesystem = hf_filesystem or _CountingHfFileSystem(
        token=hf_token, block_size=hf_block_size
    )
    filesystem.counter = _TransferCounter()
    counter = filesystem.counter
    path = uri.removeprefix("hf://")
    with filesystem.open(path, "rb") as handle:
        yield handle, lambda: counter.bytes


def _source_file_size(
    uri: str,
    access: str,
    *,
    hf_token: str | None,
    hf_filesystem: HfFileSystem | None = None,
) -> int:
    if access == "local":
        return Path(uri).stat().st_size
    filesystem = hf_filesystem or HfFileSystem(token=hf_token)
    return int(filesystem.info(uri.removeprefix("hf://"))["size"])


def _collect_setup_frame(
    uri: str,
    access: str,
    builder: Callable[[pl.LazyFrame], pl.LazyFrame],
    *,
    hf_token: str | None,
    hf_block_size: int,
    hf_filesystem: _CountingHfFileSystem | None,
) -> pl.DataFrame:
    with _benchmark_source(
        uri,
        access,
        hf_token=hf_token,
        hf_block_size=hf_block_size,
        hf_filesystem=hf_filesystem,
    ) as (source, _):
        return builder(_scan(source)).collect()


def _benchmark_setup(
    uri: str,
    access: str,
    *,
    hf_token: str | None,
    hf_block_size: int,
    sparse_key_count: int,
    hf_filesystem: _CountingHfFileSystem | None,
) -> tuple[list[QuerySpec], pl.DataFrame]:
    bounds = _collect_setup_frame(
        uri,
        access,
        lambda scan: scan.select(
            pl.col(POSITION_COLUMN).min().alias("minimum"),
            pl.col(POSITION_COLUMN).max().alias("maximum"),
        ),
        hf_token=hf_token,
        hf_block_size=hf_block_size,
        hf_filesystem=hf_filesystem,
    )
    position_min = int(bounds.item(0, "minimum"))
    position_max = int(bounds.item(0, "maximum"))
    interval_specs = interval_query_specs(position_min, position_max)
    sample_intervals = [
        spec for spec in interval_specs if spec.name.endswith("-1000000")
    ]

    def sparse_builder(scan: pl.LazyFrame) -> pl.LazyFrame:
        schema = scan.collect_schema()
        predicates = [
            pl.col(POSITION_COLUMN).is_between(spec.start, spec.end)
            for spec in sample_intervals
        ]
        predicate = predicates[0]
        for addition in predicates[1:]:
            predicate = predicate | addition
        return (
            scan.filter(predicate)
            .select(_key_columns(schema))
            .unique(maintain_order=True)
            .head(sparse_key_count)
        )

    sparse_keys = _collect_setup_frame(
        uri,
        access,
        sparse_builder,
        hf_token=hf_token,
        hf_block_size=hf_block_size,
        hf_filesystem=hf_filesystem,
    )
    if sparse_keys.is_empty():
        raise ValueError("could not derive deterministic sparse join keys")
    return interval_specs + [
        QuerySpec("projected-score-scan", "projection"),
        QuerySpec("sparse-variant-join", "sparse_join"),
        QuerySpec("full-scan", "full_scan"),
    ], sparse_keys


def benchmark_parquet_candidate(
    uri: str,
    report_path: Path,
    *,
    case: str,
    candidate: str,
    access: str,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
    sparse_key_count: int = DEFAULT_SPARSE_KEYS,
    hf_token: str | None = None,
    hf_block_size: int = 4 * 1024 * 1024,
    threads: int = 4,
) -> None:
    """Benchmark one complete local or staged-HF candidate Parquet shard."""

    if warmups != 1 or repetitions != 5:
        raise ValueError("issue #5 requires exactly one warm-up and five repetitions")
    if sparse_key_count <= 0 or hf_block_size <= 0:
        raise ValueError("sparse_key_count and hf_block_size must be positive")
    _configure_threads(threads)

    hf_filesystem = (
        _CountingHfFileSystem(token=hf_token, block_size=hf_block_size)
        if access == "hf"
        else None
    )

    query_specs, sparse_keys = _benchmark_setup(
        uri,
        access,
        hf_token=hf_token,
        hf_block_size=hf_block_size,
        sparse_key_count=sparse_key_count,
        hf_filesystem=hf_filesystem,
    )
    query_records = []
    for spec in query_specs:
        durations: list[float] = []
        transferred: list[int] = []
        result_rows: list[int] = []
        for repetition in range(warmups + repetitions):
            with _benchmark_source(
                uri,
                access,
                hf_token=hf_token,
                hf_block_size=hf_block_size,
                hf_filesystem=hf_filesystem,
            ) as (source, transferred_bytes):
                start = time.perf_counter()
                result = (
                    _execute_hf_query(source, spec, sparse_keys)
                    if access == "hf"
                    else _build_query(_scan(source), spec, sparse_keys).collect()
                )
                elapsed = time.perf_counter() - start
                measured_bytes = transferred_bytes()
            if access == "hf" and (measured_bytes is None or measured_bytes <= 0):
                raise RuntimeError("HF range transfer byte counter recorded no payload")
            if repetition < warmups:
                continue
            durations.append(elapsed)
            result_rows.append(result.height)
            if measured_bytes is not None:
                transferred.append(measured_bytes)
        if len(set(result_rows)) != 1:
            raise RuntimeError(f"query {spec.name} returned an unstable row count")
        query_records.append(
            {
                **asdict(spec),
                "duration_seconds": durations,
                "median_seconds": median(durations),
                "p95_seconds": _percentile(durations, 0.95),
                "transferred_bytes": transferred or None,
                "median_transferred_bytes": (
                    median(transferred) if transferred else None
                ),
                "p95_transferred_bytes": (
                    _percentile(transferred, 0.95) if transferred else None
                ),
                "result_rows": result_rows,
            }
        )

    position_statistics = (
        inspect_position_statistics(Path(uri)) if access == "local" else None
    )
    atomic_write_json(
        report_path,
        {
            "report_version": 1,
            "case": case,
            "candidate": candidate,
            "access": access,
            "uri": uri,
            "file_size_bytes": _source_file_size(
                uri,
                access,
                hf_token=hf_token,
                hf_filesystem=hf_filesystem,
            ),
            "warmups": warmups,
            "repetitions": repetitions,
            "sparse_key_count": sparse_keys.height,
            "sparse_keys": sparse_keys.to_dicts(),
            "hf_block_size": hf_block_size if access == "hf" else None,
            "queries": query_records,
            "position_statistics": position_statistics,
            "peak_rss_bytes": _peak_rss_bytes(),
            "threads": threads,
            "polars_thread_pool_size": pl.thread_pool_size(),
            "environment": environment_record(),
        },
    )


def validate_hf_polars(
    uri: str,
    report_path: Path,
    *,
    case: str,
    candidate: str,
    hf_token: str | None = None,
    threads: int = 4,
) -> None:
    """Exercise direct lazy ``hf://`` predicate and projection pushdown."""

    if not uri.startswith("hf://"):
        raise ValueError("Polars HF validation requires an hf:// URI")
    _configure_threads(threads)
    storage_options = {"token": hf_token} if hf_token else None
    scan = pl.scan_parquet(uri, storage_options=storage_options, cache=False)
    schema = scan.collect_schema()
    query = scan.filter(pl.col(POSITION_COLUMN) >= 1).select(
        POSITION_COLUMN, *_score_columns(schema)
    )
    plan = query.explain(optimized=True)
    result = query.head(1).collect()
    passed = (
        result.height == 1 and "PROJECT" in plan.upper() and "SELECTION" in plan.upper()
    )
    atomic_write_json(
        report_path,
        {
            "report_version": 1,
            "case": case,
            "candidate": candidate,
            "uri": uri,
            "passed": passed,
            "rows": result.height,
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "optimized_plan": plan,
            "threads": threads,
            "polars_thread_pool_size": pl.thread_pool_size(),
            "environment": environment_record(),
        },
    )
    if not passed:
        raise RuntimeError("direct hf:// Polars predicate/projection check failed")


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _inventory_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Validate issue #8's manifest envelope and index its shard records."""

    if manifest.get("manifest_version") != 1:
        raise ValueError("inventory manifest_version must be 1")
    source = manifest.get("source")
    validation = manifest.get("validation")
    shards = manifest.get("shards")
    if not isinstance(source, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("inventory manifest is missing source or validation records")
    if not isinstance(shards, list):
        raise ValueError("inventory manifest shards must be a list")

    failures = []
    for name in ("expected_shards", "reported_shards", "discovered_parquet_files"):
        if source.get(name) != EXPECTED_SHARD_COUNT:
            failures.append(f"source.{name}")
    for name in ("missing_paths", "unexpected_paths", "unreported_paths"):
        if source.get(name) != []:
            failures.append(f"source.{name}")
    if validation.get("valid_shards") != EXPECTED_SHARD_COUNT:
        failures.append("validation.valid_shards")
    if validation.get("invalid_shards") != 0:
        failures.append("validation.invalid_shards")
    if len(shards) != EXPECTED_SHARD_COUNT:
        failures.append("shards")
    if failures:
        raise ValueError(
            "inventory manifest does not describe a complete valid source inventory: "
            + ", ".join(failures)
        )

    records: dict[str, Mapping[str, Any]] = {}
    for record in shards:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("inventory manifest contains a malformed shard record")
        path = str(record["path"])
        if path in records:
            raise ValueError(f"inventory manifest repeats shard path {path!r}")
        records[path] = record
    expected_paths = {shard.relative_path.as_posix() for shard in expected_shards()}
    if set(records) != expected_paths:
        raise ValueError(
            "inventory manifest shard paths do not match the release catalog"
        )
    invalid_records = [
        path
        for path, record in records.items()
        if record.get("valid") is not True
        or record.get("errors") != []
        or not isinstance(record.get("content"), Mapping)
        or record["content"].get("order_violations") != 0
    ]
    if invalid_records:
        raise ValueError(
            f"inventory manifest contains {len(invalid_records)} invalid shard records"
        )
    return records


def validate_benchmark_source(
    inventory_manifest_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    case: str,
    relative_path: str,
    score_type: str,
) -> None:
    """Tie one immutable benchmark source to its validated inventory record."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe inventory relative path: {relative_path!r}")
    if score_type not in {"entropy", "llr"}:
        raise ValueError(f"unknown score type: {score_type!r}")

    manifest_path = Path(inventory_manifest_path)
    manifest = _load_json(manifest_path)
    records = _inventory_records(manifest)
    normalized_relative_path = relative.as_posix()
    record = records.get(normalized_relative_path)
    if record is None:
        raise ValueError(
            f"benchmark source {normalized_relative_path!r} is absent from inventory"
        )
    if record.get("score_type") != score_type:
        raise ValueError(
            f"inventory score type for {normalized_relative_path!r} is "
            f"{record.get('score_type')!r}, expected {score_type!r}"
        )
    content = record.get("content")
    if (
        record.get("valid") is not True
        or record.get("errors") != []
        or not isinstance(content, Mapping)
        or content.get("order_violations") != 0
    ):
        raise ValueError(
            f"inventory validation did not pass for {normalized_relative_path!r}"
        )

    expected_size = record.get("size")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
    ):
        raise ValueError("inventory shard size is not a non-negative integer")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("inventory shard SHA-256 is malformed")

    source_path = Path(source_path)
    before = source_path.stat()
    observed_sha256 = _sha256_file(source_path)
    after = source_path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError("immutable source identity changed during inventory check")
    if before.st_size != expected_size:
        raise ValueError(
            f"source size {before.st_size} does not match inventory {expected_size}"
        )
    if observed_sha256 != expected_sha256:
        raise ValueError("source SHA-256 does not match the inventory manifest")

    atomic_write_json(
        output_path,
        {
            "report_version": 1,
            "case": case,
            "relative_path": normalized_relative_path,
            "score_type": score_type,
            "source_size_bytes": expected_size,
            "source_sha256": expected_sha256,
            "position_sorted": True,
            "inventory_valid": True,
            "inventory_manifest": str(manifest_path),
            "inventory_manifest_sha256": _sha256_file(manifest_path),
            "inventory_release_ready": manifest["validation"].get("release_ready"),
            "inventory_blockers": manifest["validation"].get("blockers", []),
        },
    )


def _query_durations(
    reports: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    access: str,
    kind: str,
) -> list[float]:
    return [
        float(duration)
        for report in reports
        if report["candidate"] == candidate and report["access"] == access
        for query in report["queries"]
        if query["kind"] == kind
        for duration in query["duration_seconds"]
    ]


def _query_transferred_bytes(
    reports: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    access: str,
    kind: str,
) -> list[int]:
    return [
        int(transferred)
        for report in reports
        if report["candidate"] == candidate and report["access"] == access
        for query in report["queries"]
        if query["kind"] == kind
        for transferred in (query.get("transferred_bytes") or [])
    ]


def _candidate_sizes(
    reports: Sequence[Mapping[str, Any]], candidate: str
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for report in reports:
        if report["candidate"] != candidate:
            continue
        case = str(report["case"])
        size = int(report["file_size_bytes"])
        if case in sizes and sizes[case] != size:
            raise ValueError(f"inconsistent size for {case}/{candidate}")
        sizes[case] = size
    return sizes


def _functional_candidate(
    checks: Mapping[str, Any], candidate: str
) -> tuple[bool, list[str]]:
    record = checks.get(candidate, {})
    required = (
        "position_sorted",
        "position_statistics_usable",
        "dataset_viewer",
        "staged_artifacts_match",
        "hf_polars",
    )
    missing_or_failed = [name for name in required if record.get(name) is not True]
    return not missing_or_failed, missing_or_failed


def _staging_functional_checks(
    path: Path, candidate_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Read only Dataset Viewer evidence from the staging record."""

    raw_checks = _load_json(path)
    unknown = sorted(set(raw_checks) - set(candidate_names))
    if unknown:
        raise ValueError(f"staging checks contain unknown candidates: {unknown}")

    checks = {}
    for candidate in candidate_names:
        raw = raw_checks.get(candidate, {})
        if not isinstance(raw, Mapping):
            raw = {}
        raw_evidence = raw.get("evidence", {})
        evidence = dict(raw_evidence) if isinstance(raw_evidence, Mapping) else {}
        evidence_complete = all(
            isinstance(evidence.get(field), str) and bool(evidence[field].strip())
            for field in ("dataset_viewer_url", "dataset_viewer_checked_at")
        )
        checks[candidate] = {
            "dataset_viewer_observed": raw.get("dataset_viewer") is True
            and evidence_complete,
            "evidence": evidence,
        }
    return checks


def select_layout(
    benchmark_reports: Iterable[Path],
    rewrite_reports: Iterable[Path],
    source_evidence_reports: Iterable[Path],
    staging_checks_path: Path,
    hf_validation_reports: Iterable[Path] = (),
) -> dict[str, Any]:
    """Apply issue #5's declared selection thresholds without discretion."""

    benchmarks = [_load_json(path) for path in benchmark_reports]
    rewrites = [_load_json(path) for path in rewrite_reports]
    source_evidence = [_load_json(path) for path in source_evidence_reports]
    hf_validations = [_load_json(path) for path in hf_validation_reports]
    candidate_names = [candidate.name for candidate in CANDIDATES]
    functional_checks = _staging_functional_checks(staging_checks_path, candidate_names)
    source_sizes = _candidate_sizes(benchmarks, "source")
    cases = sorted(source_sizes)
    blockers = []
    if not cases:
        blockers.append("source benchmark reports are missing")

    evidence_by_case: dict[str, Mapping[str, Any]] = {}
    duplicate_evidence = set()
    for record in source_evidence:
        case = str(record.get("case"))
        if case in evidence_by_case:
            duplicate_evidence.add(case)
        evidence_by_case[case] = record
    missing_evidence = sorted(set(cases) - set(evidence_by_case))
    unexpected_evidence = sorted(set(evidence_by_case) - set(cases))
    invalid_evidence = sorted(
        case
        for case in cases
        if case in evidence_by_case
        and (
            evidence_by_case[case].get("inventory_valid") is not True
            or evidence_by_case[case].get("position_sorted") is not True
            or evidence_by_case[case].get("source_size_bytes") != source_sizes.get(case)
            or not isinstance(evidence_by_case[case].get("source_sha256"), str)
            or len(evidence_by_case[case]["source_sha256"]) != 64
        )
    )
    manifest_sha256s = {
        record.get("inventory_manifest_sha256") for record in source_evidence
    }
    inventory_valid = not (
        missing_evidence
        or unexpected_evidence
        or duplicate_evidence
        or invalid_evidence
        or len(manifest_sha256s) != 1
        or None in manifest_sha256s
    )
    functional_checks["source"]["position_sorted"] = inventory_valid
    functional_checks["source"]["inventory_valid"] = inventory_valid
    functional_checks["source"]["inventory_evidence"] = {
        case: {
            "relative_path": record.get("relative_path"),
            "source_size_bytes": record.get("source_size_bytes"),
            "source_sha256": record.get("source_sha256"),
        }
        for case, record in sorted(evidence_by_case.items())
    }
    if not inventory_valid:
        blockers.append(
            "source inventory evidence is incomplete or inconsistent: "
            f"missing={missing_evidence}, unexpected={unexpected_evidence}, "
            f"duplicates={sorted(duplicate_evidence)}, invalid={invalid_evidence}"
        )
    if functional_checks["source"]["position_sorted"] is not True:
        blockers.append(
            "source positions are not confirmed sorted; layout rewrites preserve row "
            "order and cannot repair this"
        )
    if hf_validations:
        for candidate in candidate_names:
            records = [
                record
                for record in hf_validations
                if record.get("candidate") == candidate
            ]
            passed_cases = {
                str(record["case"])
                for record in records
                if record.get("passed") is True
            }
            functional_checks.setdefault(candidate, {})["hf_polars"] = (
                passed_cases == set(cases) and len(records) == len(cases)
            )

    exact_equality = {"source": True}
    expected_artifact_sha256: dict[str, dict[str, str]] = {
        "source": {
            case: str(evidence_by_case[case]["source_sha256"])
            for case in cases
            if case in evidence_by_case
            and isinstance(evidence_by_case[case].get("source_sha256"), str)
        }
    }
    rewrite_metrics: dict[str, dict[str, float | int | None]] = {
        "source": {"write_seconds": None, "peak_rss_bytes": None}
    }
    for candidate in candidate_names[1:]:
        records = [
            record for record in rewrites if record["candidate"]["name"] == candidate
        ]
        exact_equality[candidate] = (
            bool(records)
            and len(records) == len(cases)
            and {str(record.get("case")) for record in records} == set(cases)
            and all(record.get("exact_value_equality") is True for record in records)
            and all(
                isinstance(record.get("output_sha256"), str)
                and len(record["output_sha256"]) == 64
                for record in records
            )
        )
        expected_artifact_sha256[candidate] = {
            str(record["case"]): str(record["output_sha256"])
            for record in records
            if isinstance(record.get("output_sha256"), str)
        }
        rewrite_metrics[candidate] = {
            "write_seconds": (
                sum(float(record["write_seconds"]) for record in records)
                if records
                else None
            ),
            "peak_rss_bytes": (
                max(int(record["peak_rss_bytes"]) for record in records)
                if records
                else None
            ),
        }
        if exact_equality[candidate]:
            functional_checks.setdefault(candidate, {})["position_sorted"] = (
                functional_checks.get("source", {}).get("position_sorted") is True
            )

    for candidate in candidate_names:
        hf_uris = {
            str(report["case"]): str(report.get("uri"))
            for report in benchmarks
            if report.get("candidate") == candidate and report.get("access") == "hf"
        }
        evidence = functional_checks[candidate]["evidence"]
        staged_artifacts = evidence.get("artifacts", {})
        artifacts_match = (
            isinstance(staged_artifacts, Mapping)
            and set(staged_artifacts) == set(cases)
            and set(expected_artifact_sha256[candidate]) == set(cases)
            and set(hf_uris) == set(cases)
            and all(
                isinstance(staged_artifacts[case], Mapping)
                and staged_artifacts[case].get("sha256")
                == expected_artifact_sha256[candidate][case]
                and staged_artifacts[case].get("uri") == hf_uris[case]
                for case in cases
            )
        )
        functional_checks[candidate]["staged_artifacts_match"] = artifacts_match
        functional_checks[candidate]["dataset_viewer"] = (
            functional_checks[candidate].pop("dataset_viewer_observed") is True
            and artifacts_match
        )

    for candidate in candidate_names:
        local_records = [
            report
            for report in benchmarks
            if report["candidate"] == candidate and report["access"] == "local"
        ]
        if local_records and all(
            report.get("position_statistics") is not None for report in local_records
        ):
            functional_checks.setdefault(candidate, {})[
                "position_statistics_usable"
            ] = len(local_records) == len(cases) and all(
                report["position_statistics"].get("usable") is True
                for report in local_records
            )

    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidate_names:
        sizes = _candidate_sizes(benchmarks, candidate)
        remote_intervals = _query_durations(
            benchmarks, candidate, access="hf", kind="interval"
        )
        remote_interval_bytes = _query_transferred_bytes(
            benchmarks, candidate, access="hf", kind="interval"
        )
        full_scan_by_access = {
            access: _query_durations(
                benchmarks, candidate, access=access, kind="full_scan"
            )
            for access in ("local", "hf")
        }
        functional, failed_checks = _functional_candidate(functional_checks, candidate)
        summaries[candidate] = {
            "cases": sorted(sizes),
            "total_size_bytes": sum(sizes.values())
            if len(sizes) == len(cases)
            else None,
            "remote_interval_median_seconds": (
                median(remote_intervals) if remote_intervals else None
            ),
            "remote_interval_median_transferred_bytes": (
                median(remote_interval_bytes) if remote_interval_bytes else None
            ),
            "full_scan_median_seconds": {
                access: median(values) if values else None
                for access, values in full_scan_by_access.items()
            },
            "functional": functional,
            "failed_functional_checks": failed_checks,
            "exact_value_equality": exact_equality[candidate],
            **rewrite_metrics[candidate],
        }

    for candidate, summary in summaries.items():
        if summary["cases"] != cases:
            blockers.append(f"{candidate} does not cover every benchmark case")
        if summary["remote_interval_median_seconds"] is None:
            blockers.append(f"{candidate} has no remote interval measurements")
        if any(value is None for value in summary["full_scan_median_seconds"].values()):
            blockers.append(f"{candidate} is missing a local or remote full scan")

    source = summaries["source"]
    for candidate in candidate_names[1:]:
        summary = summaries[candidate]
        if (
            summary["total_size_bytes"] is not None
            and source["total_size_bytes"] is not None
        ):
            summary["size_ratio_to_source"] = (
                summary["total_size_bytes"] / source["total_size_bytes"]
            )
        else:
            summary["size_ratio_to_source"] = None
        if (
            summary["remote_interval_median_seconds"] is not None
            and source["remote_interval_median_seconds"]
        ):
            summary["remote_interval_ratio_to_source"] = (
                summary["remote_interval_median_seconds"]
                / source["remote_interval_median_seconds"]
            )
        else:
            summary["remote_interval_ratio_to_source"] = None
        full_scan_ratios = {}
        for access in ("local", "hf"):
            source_time = source["full_scan_median_seconds"][access]
            candidate_time = summary["full_scan_median_seconds"][access]
            full_scan_ratios[access] = (
                candidate_time / source_time
                if source_time not in (None, 0) and candidate_time is not None
                else None
            )
        summary["full_scan_ratio_to_source"] = full_scan_ratios

    selected: str | None = None
    rationale: str
    if blockers:
        rationale = "Selection is blocked until every required measurement is present."
    elif source["functional"]:
        performance_winners = [
            candidate
            for candidate in candidate_names[1:]
            if summaries[candidate]["functional"]
            and summaries[candidate]["exact_value_equality"]
            and summaries[candidate]["size_ratio_to_source"] <= 1.05
            and summaries[candidate]["remote_interval_ratio_to_source"] <= 0.75
            and all(
                ratio <= 1.10
                for ratio in summaries[candidate]["full_scan_ratio_to_source"].values()
            )
        ]
        if performance_winners:
            selected = min(
                performance_winners,
                key=lambda name: summaries[name]["remote_interval_median_seconds"],
            )
            rationale = (
                f"Selected {selected}: it clears the 25% remote-range improvement "
                "threshold, the 5% size limit, and both 10% full-scan limits."
            )
        else:
            selected = "source"
            rationale = (
                "Kept source files unchanged: they pass functional checks and no "
                "rewrite clears every declared performance and size threshold."
            )
    else:
        rewrite_sizes = [
            summaries[name]["total_size_bytes"]
            for name in candidate_names[1:]
            if summaries[name]["total_size_bytes"] is not None
        ]
        smallest_rewrite = min(rewrite_sizes)
        functional_rewrites = [
            candidate
            for candidate in candidate_names[1:]
            if summaries[candidate]["functional"]
            and summaries[candidate]["exact_value_equality"]
            and summaries[candidate]["total_size_bytes"] <= smallest_rewrite * 1.05
            and all(
                ratio <= 1.10
                for ratio in summaries[candidate]["full_scan_ratio_to_source"].values()
            )
        ]
        if functional_rewrites:
            selected = min(
                functional_rewrites,
                key=lambda name: summaries[name]["remote_interval_median_seconds"],
            )
            rationale = (
                f"Selected {selected} because rewriting is functionally required; "
                "it has the fastest remote range median among eligible rewrites."
            )
        else:
            blockers.append(
                "source failed functional checks and no rewrite satisfies all safeguards"
            )
            rationale = (
                "Selection is blocked; author review or a new candidate is required."
            )

    return {
        "report_version": 1,
        "status": "selected" if selected is not None else "blocked",
        "selected_candidate": selected,
        "rationale": rationale,
        "blockers": blockers,
        "aggregation": {
            "size": "sum of complete benchmark shard sizes",
            "remote_range": "median of all measured HF interval-query repetitions",
            "full_scan": (
                "separate medians of all measured repetitions for local and HF access; "
                "both must satisfy the 10% limit"
            ),
            "p95": "inclusive linearly interpolated 95th percentile",
        },
        "thresholds": {
            "minimum_remote_range_improvement": 0.25,
            "maximum_size_increase": 0.05,
            "maximum_full_scan_slowdown": 0.10,
        },
        "source_inventory": {
            "cases": sorted(evidence_by_case),
            "manifest_sha256": (
                next(iter(manifest_sha256s))
                if len(manifest_sha256s) == 1 and None not in manifest_sha256s
                else None
            ),
            "valid": inventory_valid,
        },
        "candidates": summaries,
        "functional_checks": functional_checks,
    }


def render_selection_markdown(selection: Mapping[str, Any]) -> str:
    """Render a dataset-card-ready record of the applied selection rule."""

    lines = [
        "## Parquet layout benchmark",
        "",
        f"Status: **{selection['status']}**",
        "",
        str(selection["rationale"]),
        "",
        "| Candidate | Size (bytes) | Write (s) | Peak RSS (bytes) | "
        "HF range median (s) | HF range bytes | Local full scan (s) | "
        "HF full scan (s) | Functional | Exact values |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for name, summary in selection["candidates"].items():
        full_scan = summary["full_scan_median_seconds"]
        range_median = summary["remote_interval_median_seconds"]
        lines.append(
            f"| {name} | {summary['total_size_bytes'] or 'n/a'} | "
            f"{summary['write_seconds'] if summary['write_seconds'] is not None else 'n/a'} | "
            f"{summary['peak_rss_bytes'] if summary['peak_rss_bytes'] is not None else 'n/a'} | "
            f"{range_median if range_median is not None else 'n/a'} | "
            f"{summary['remote_interval_median_transferred_bytes'] if summary['remote_interval_median_transferred_bytes'] is not None else 'n/a'} | "
            f"{full_scan['local'] if full_scan['local'] is not None else 'n/a'} | "
            f"{full_scan['hf'] if full_scan['hf'] is not None else 'n/a'} | "
            f"{'yes' if summary['functional'] else 'no'} | "
            f"{'yes' if summary['exact_value_equality'] else 'no'} |"
        )
    if selection["blockers"]:
        lines.extend(["", "Blockers:"])
        lines.extend(f"- {blocker}" for blocker in selection["blockers"])
    lines.extend(
        [
            "",
            "The selection used one warm-up and five measured repetitions per "
            "query. Source score precision, row order, chromosome names, and "
            "one-based positions were not changed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_selection_outputs(
    benchmark_reports: Iterable[Path],
    rewrite_reports: Iterable[Path],
    source_evidence_reports: Iterable[Path],
    staging_checks_path: Path,
    json_path: Path,
    markdown_path: Path,
    *,
    hf_validation_reports: Iterable[Path] = (),
) -> None:
    """Apply the selection rule and atomically write both report formats."""

    selection = select_layout(
        benchmark_reports,
        rewrite_reports,
        source_evidence_reports,
        staging_checks_path,
        hf_validation_reports,
    )
    atomic_write_json(json_path, selection)
    _atomic_write_text(markdown_path, render_selection_markdown(selection))
