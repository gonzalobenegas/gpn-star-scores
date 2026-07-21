"""Berkeley SCF environment smoke report with atomic output publication."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from collections.abc import Callable, Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import polars as pl

EXPECTED_PACKAGE_VERSIONS = {
    "polars": "1.42.1",
    "polars-runtime-32": "1.42.1",
    "pyarrow": "25.0.0",
    "pybigwig": "0.3.25",
    "snakemake": "9.23.1",
    "snakemake-executor-plugin-slurm": "2.7.1",
}
REQUIRED_PARTITION = "epurdom"
RUNTIME_IMPORTS = (
    "polars",
    "pyarrow",
    "pyBigWig",
    "snakemake",
    "snakemake_executor_plugin_slurm",
)


def _detect_slurm_partition() -> str | None:
    """Return the job partition from the environment or Slurm controller."""
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition:
        return partition

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None

    completed = subprocess.run(
        ["scontrol", "show", "job", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(?:^|\s)Partition=(\S+)", completed.stdout)
    return match.group(1) if match else None


def _validate_smoke_report(report: Mapping[str, Any]) -> None:
    """Reject incomplete or internally inconsistent smoke reports."""
    if report.get("schema_version") != 1:
        raise ValueError("unexpected smoke report schema")
    if report.get("partition") != REQUIRED_PARTITION:
        raise ValueError(f"smoke job must run on {REQUIRED_PARTITION}")
    if report.get("packages") != EXPECTED_PACKAGE_VERSIONS:
        raise ValueError("runtime package versions do not match the release lock")
    if report.get("polars_runtime") != "polars-runtime-32":
        raise ValueError("the standard polars-runtime-32 build is required")
    if report.get("polars_sum") != 6:
        raise ValueError("Polars execution check returned the wrong result")


def atomic_write_json(
    output_path: Path,
    payload: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    """Validate JSON in a temporary sibling before atomically replacing output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        with temporary_path.open(encoding="utf-8") as handle:
            validated_payload = json.load(handle)
        validator(validated_payload)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_smoke_report(chrom: str) -> dict[str, Any]:
    """Import pinned runtimes and execute a minimal standard-Polars query."""
    partition = _detect_slurm_partition()
    if partition != REQUIRED_PARTITION:
        raise RuntimeError(
            f"SCF smoke job requires partition {REQUIRED_PARTITION!r}; "
            f"found {partition!r}"
        )

    for module_name in RUNTIME_IMPORTS:
        import_module(module_name)

    installed_versions = {
        package: version(package) for package in EXPECTED_PACKAGE_VERSIONS
    }
    if installed_versions != EXPECTED_PACKAGE_VERSIONS:
        raise RuntimeError(
            "runtime versions differ from the pinned smoke-test versions: "
            f"{installed_versions!r}"
        )

    try:
        version("polars-runtime-64")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "polars-runtime-64 is installed; expected standard runtime-32"
        )

    polars_sum = (
        pl.LazyFrame({"value": [1, 2, 3]})
        .select(pl.col("value").sum())
        .collect()
        .item()
    )

    return {
        "schema_version": 1,
        "chrom": chrom,
        "hostname": platform.node(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": installed_versions,
        "partition": partition,
        "polars_runtime": "polars-runtime-32",
        "polars_sum": polars_sum,
        "python": platform.python_version(),
    }


def write_smoke_report(chrom: str, output_path: Path) -> None:
    """Build, validate, and atomically publish one chromosome smoke report."""
    atomic_write_json(output_path, build_smoke_report(chrom), _validate_smoke_report)


def main() -> None:
    """Run the SCF smoke report command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_smoke_report(args.chrom, args.output)


if __name__ == "__main__":
    main()
