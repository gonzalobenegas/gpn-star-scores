"""Reproducible command measurement and selection for the BigWig benchmark."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """Metrics from one measured benchmark repetition."""

    method: str
    repetition: int
    command: tuple[str, ...]
    wall_seconds: float
    peak_rss_bytes: int
    peak_scratch_bytes: int
    final_bytes: int
    correct: bool


@dataclass(frozen=True)
class CandidateSummary:
    """Metrics used to compare one BigWig generation method."""

    method: str
    measured_repetitions: int
    median_wall_seconds: float
    peak_rss_bytes: int
    peak_scratch_bytes: int
    final_bytes: int
    correct: bool


@dataclass(frozen=True)
class BenchmarkDecision:
    """Application of issue #7's predeclared selection rule."""

    selected_method: str
    direct_speedup_fraction: float
    direct_scratch_reduction_fraction: float
    direct_slowdown_fraction: float
    reason: str


def measure_command(
    method: str,
    repetition: int,
    command: Sequence[str],
    *,
    working_directory: str | Path,
    measurement_directory: str | Path,
    scratch_paths: Sequence[str | Path],
    final_paths: Sequence[str | Path],
    stdout_path: str | Path,
    stderr_path: str | Path,
    correct: bool,
    poll_interval_seconds: float = 0.25,
    time_executable: str | Path = "/usr/bin/time",
) -> BenchmarkMeasurement:
    """Measure one command with GNU time and sampled scratch usage.

    ``peak_scratch_bytes`` is the largest observed growth above the combined
    pre-command size of ``scratch_paths``, less the final artifact bytes that
    remain after the command.  This reports transient scratch in addition to
    the deliverables rather than counting the deliverables themselves as
    temporary storage.  Paths must be non-overlapping so a file cannot be
    counted twice.  GNU time's maximum resident set size is reported in KiB on
    the Linux SCF hosts and is converted to bytes here.
    """

    if not method:
        raise ValueError("method must not be empty")
    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    if not command:
        raise ValueError("command must not be empty")
    if poll_interval_seconds <= 0:
        raise ValueError("poll interval must be positive")

    scratch = _non_overlapping_paths(scratch_paths)
    finals = [Path(path) for path in final_paths]
    cwd = Path(working_directory)
    measurement_dir = Path(measurement_directory)
    measurement_dir.mkdir(parents=True, exist_ok=True)
    stdout = Path(stdout_path)
    stderr = Path(stderr_path)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    baseline_scratch_bytes = sum(_path_size(path) for path in scratch)
    peak_scratch_bytes = 0

    descriptor, timing_name = tempfile.mkstemp(
        prefix=f".{method}.{repetition}.", suffix=".time", dir=measurement_dir
    )
    os.close(descriptor)
    timing_path = Path(timing_name)
    timed_command = [
        str(time_executable),
        "--format=%e\t%M",
        f"--output={timing_path}",
        "--",
        *map(str, command),
    ]

    try:
        with stdout.open("wb") as stdout_handle, stderr.open("wb") as stderr_handle:
            process = subprocess.Popen(
                timed_command,
                cwd=cwd,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            while process.poll() is None:
                current_size = sum(_path_size(path) for path in scratch)
                peak_scratch_bytes = max(
                    peak_scratch_bytes, current_size - baseline_scratch_bytes
                )
                time.sleep(poll_interval_seconds)
            current_size = sum(_path_size(path) for path in scratch)
            peak_scratch_bytes = max(
                peak_scratch_bytes, current_size - baseline_scratch_bytes
            )
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, timed_command)

        wall_text, peak_rss_kib_text = timing_path.read_text().strip().split("\t")
        final_bytes = sum(_path_size(path) for path in finals)
        transient_scratch_bytes = max(0, peak_scratch_bytes - final_bytes)
        return BenchmarkMeasurement(
            method=method,
            repetition=repetition,
            command=tuple(map(str, command)),
            wall_seconds=float(wall_text),
            peak_rss_bytes=int(peak_rss_kib_text) * 1024,
            peak_scratch_bytes=transient_scratch_bytes,
            final_bytes=final_bytes,
            correct=correct,
        )
    finally:
        timing_path.unlink(missing_ok=True)


def summarize_candidate(
    measurements: Sequence[BenchmarkMeasurement],
) -> CandidateSummary:
    """Aggregate repetitions without hiding peak resource requirements."""

    if not measurements:
        raise ValueError("at least one measurement is required")
    methods = {measurement.method for measurement in measurements}
    if len(methods) != 1:
        raise ValueError("all measurements must describe the same method")
    final_sizes = {measurement.final_bytes for measurement in measurements}
    if len(final_sizes) != 1:
        raise ValueError("final output size changed between repetitions")

    return CandidateSummary(
        method=measurements[0].method,
        measured_repetitions=len(measurements),
        median_wall_seconds=statistics.median(
            measurement.wall_seconds for measurement in measurements
        ),
        peak_rss_bytes=max(measurement.peak_rss_bytes for measurement in measurements),
        peak_scratch_bytes=max(
            measurement.peak_scratch_bytes for measurement in measurements
        ),
        final_bytes=measurements[0].final_bytes,
        correct=all(measurement.correct for measurement in measurements),
    )


def select_bigwig_method(
    upstream_wig: CandidateSummary, direct: CandidateSummary
) -> BenchmarkDecision:
    """Apply the issue's speed/scratch rule without post-hoc interpretation."""

    if upstream_wig.median_wall_seconds <= 0 or direct.median_wall_seconds <= 0:
        raise ValueError("median wall times must be positive")
    if upstream_wig.peak_scratch_bytes < 0 or direct.peak_scratch_bytes < 0:
        raise ValueError("peak scratch sizes must not be negative")
    if not upstream_wig.correct:
        raise ValueError("the upstream WIG baseline must pass correctness checks")

    direct_speedup = (
        upstream_wig.median_wall_seconds - direct.median_wall_seconds
    ) / upstream_wig.median_wall_seconds
    direct_slowdown = max(
        0.0,
        (direct.median_wall_seconds - upstream_wig.median_wall_seconds)
        / upstream_wig.median_wall_seconds,
    )
    scratch_reduction = _reduction_fraction(
        upstream_wig.peak_scratch_bytes, direct.peak_scratch_bytes
    )

    if not direct.correct:
        selected = upstream_wig.method
        reason = "direct output failed correctness validation"
    elif direct_speedup >= 0.20:
        selected = direct.method
        reason = "direct writing is at least 20% faster"
    elif scratch_reduction >= 0.80 and direct_slowdown <= 0.20:
        selected = direct.method
        reason = (
            "direct writing reduces peak scratch by at least 80% and is no more "
            "than 20% slower"
        )
    else:
        selected = upstream_wig.method
        reason = "direct writing does not meet the predeclared selection threshold"

    return BenchmarkDecision(
        selected_method=selected,
        direct_speedup_fraction=direct_speedup,
        direct_scratch_reduction_fraction=scratch_reduction,
        direct_slowdown_fraction=direct_slowdown,
        reason=reason,
    )


def write_benchmark_report(
    path: str | Path,
    measurements: Sequence[BenchmarkMeasurement],
    summaries: Mapping[str, CandidateSummary],
    decision: BenchmarkDecision,
) -> None:
    """Atomically write the machine-readable benchmark evidence."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(
                {
                    "measurements": [asdict(item) for item in measurements],
                    "summaries": {
                        name: asdict(summary) for name, summary in summaries.items()
                    },
                    "decision": asdict(decision),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _reduction_fraction(baseline: int, candidate: int) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else -1.0
    return (baseline - candidate) / baseline


def _non_overlapping_paths(paths: Sequence[str | Path]) -> list[Path]:
    result = [Path(path).resolve() for path in paths]
    for index, path in enumerate(result):
        for other in result[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("scratch paths must not overlap")
    return result


def _path_size(path: Path) -> int:
    try:
        if path.is_symlink():
            return path.lstat().st_size
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(_path_size(child) for child in path.iterdir())
    except FileNotFoundError:
        return 0
    return 0
