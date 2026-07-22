import json
import sys
import textwrap
from pathlib import Path

import pytest

from gpn_star_scores.benchmark import (
    BenchmarkMeasurement,
    CandidateSummary,
    measure_command,
    select_bigwig_method,
    summarize_candidate,
    write_benchmark_report,
)


def _candidate(
    method: str,
    *,
    wall: float,
    scratch: int,
    correct: bool = True,
) -> CandidateSummary:
    return CandidateSummary(
        method=method,
        measured_repetitions=5,
        median_wall_seconds=wall,
        peak_rss_bytes=1_000,
        peak_scratch_bytes=scratch,
        final_bytes=100,
        correct=correct,
    )


@pytest.mark.parametrize(
    ("direct", "selected", "reason"),
    [
        (_candidate("direct", wall=80.0, scratch=90), "direct", "20% faster"),
        (
            _candidate("direct", wall=120.0, scratch=20),
            "direct",
            "reduces peak scratch",
        ),
        (
            _candidate("direct", wall=121.0, scratch=19),
            "wig",
            "does not meet",
        ),
        (
            _candidate("direct", wall=50.0, scratch=10, correct=False),
            "wig",
            "failed correctness",
        ),
    ],
)
def test_select_bigwig_method_applies_predeclared_rule(
    direct: CandidateSummary, selected: str, reason: str
) -> None:
    decision = select_bigwig_method(_candidate("wig", wall=100.0, scratch=100), direct)

    assert decision.selected_method == selected
    assert reason in decision.reason


def test_summarize_candidate_uses_median_and_resource_peaks() -> None:
    measurements = [
        BenchmarkMeasurement("direct", 1, ("cmd",), 3.0, 10, 30, 5, True),
        BenchmarkMeasurement("direct", 2, ("cmd",), 1.0, 40, 20, 5, True),
        BenchmarkMeasurement("direct", 3, ("cmd",), 2.0, 20, 10, 5, True),
    ]

    summary = summarize_candidate(measurements)

    assert summary.median_wall_seconds == 2.0
    assert summary.peak_rss_bytes == 40
    assert summary.peak_scratch_bytes == 30
    assert summary.correct


def test_measure_command_records_resources_and_final_size(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    measurements = tmp_path / "measurements"
    logs = tmp_path / "logs"
    scratch.mkdir()
    final = scratch / "final.bin"

    result = measure_command(
        "direct",
        1,
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """\
                import time
                from pathlib import Path

                temporary = Path("scratch/temporary.bin")
                temporary.write_bytes(b"t" * 101)
                time.sleep(0.1)
                Path("scratch/final.bin").write_bytes(b"x" * 17)
                time.sleep(0.1)
                temporary.unlink()
                """
            ),
        ],
        working_directory=tmp_path,
        measurement_directory=measurements,
        scratch_paths=[scratch],
        final_paths=[final],
        stdout_path=logs / "stdout.log",
        stderr_path=logs / "stderr.log",
        correct=True,
        poll_interval_seconds=0.01,
    )

    assert result.wall_seconds >= 0
    assert result.peak_rss_bytes > 0
    assert result.peak_scratch_bytes >= 101
    assert result.final_bytes == 17


def test_benchmark_report_is_machine_readable(tmp_path: Path) -> None:
    measurement = BenchmarkMeasurement("direct", 1, ("command",), 1.0, 2, 3, 4, True)
    summary = summarize_candidate([measurement])
    decision = select_bigwig_method(
        _candidate("wig", wall=2.0, scratch=20),
        _candidate("direct", wall=1.0, scratch=3),
    )
    output = tmp_path / "benchmark.json"

    write_benchmark_report(output, [measurement], {"direct": summary}, decision)

    report = json.loads(output.read_text())
    assert report["measurements"][0]["command"] == ["command"]
    assert report["decision"]["selected_method"] == "direct"
