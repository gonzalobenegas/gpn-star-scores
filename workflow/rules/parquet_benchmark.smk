"""Issue #5 Parquet layout benchmark workflow."""

from pathlib import Path
import os
import re

from snakemake.exceptions import WorkflowError

from gpn_star_scores.parquet_benchmark import (
    CANDIDATES,
    benchmark_parquet_candidate,
    rewrite_parquet_candidate,
    validate_benchmark_source,
    validate_hf_polars,
    write_selection_outputs,
)

PARQUET_BENCHMARK_CONFIG = config.get("parquet_benchmark", {})
PARQUET_BENCHMARK_ENABLED = bool(PARQUET_BENCHMARK_CONFIG.get("enabled", False))
PARQUET_BENCHMARK_ROOT = Path(
    PARQUET_BENCHMARK_CONFIG.get("output_root", "results/parquet-benchmark")
)
PARQUET_CANDIDATES = {candidate.name: candidate for candidate in CANDIDATES}
PARQUET_REWRITE_CANDIDATES = {
    name: candidate
    for name, candidate in PARQUET_CANDIDATES.items()
    if candidate.rewrites_source
}


def parquet_benchmark_targets():
    """Return final reports only when the production benchmark is enabled."""

    if not PARQUET_BENCHMARK_ENABLED:
        return []
    return [
        str(PARQUET_BENCHMARK_ROOT / "selection.json"),
        str(PARQUET_BENCHMARK_ROOT / "dataset-card-parquet-benchmark.md"),
    ]


def write_parquet_benchmark_log(path, message):
    """Write a concise rule status log."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n", encoding="utf-8")


if PARQUET_BENCHMARK_ENABLED:
    if not PARQUET_BENCHMARK_CONFIG.get("source_root"):
        raise WorkflowError(
            "parquet_benchmark.source_root is required when the benchmark is enabled"
        )
    PARQUET_SOURCE_ROOT = Path(PARQUET_BENCHMARK_CONFIG["source_root"])
    if PARQUET_BENCHMARK_ROOT.resolve().is_relative_to(PARQUET_SOURCE_ROOT.resolve()):
        raise WorkflowError(
            "parquet_benchmark.output_root must not be inside the immutable "
            "source_root"
        )
    PARQUET_CASES = {
        case["id"]: case for case in PARQUET_BENCHMARK_CONFIG.get("cases", [])
    }
    if len(PARQUET_CASES) != len(PARQUET_BENCHMARK_CONFIG.get("cases", [])):
        raise WorkflowError("parquet_benchmark case IDs must be unique")
    if not PARQUET_CASES:
        raise WorkflowError("parquet_benchmark.cases must not be empty")
    for case_id, case in PARQUET_CASES.items():
        if not re.fullmatch(r"[A-Za-z0-9._-]+", case_id):
            raise WorkflowError(f"unsafe parquet benchmark case ID: {case_id!r}")
        if case.get("score_type") not in {"entropy", "llr"}:
            raise WorkflowError(f"invalid score type for benchmark case {case_id}")
        relative_path = Path(case.get("relative_path", ""))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise WorkflowError(f"unsafe relative path for benchmark case {case_id}")

    PARQUET_CASE_IDS = sorted(PARQUET_CASES)
    PARQUET_CANDIDATE_NAMES = sorted(PARQUET_CANDIDATES)
    PARQUET_REWRITE_NAMES = sorted(PARQUET_REWRITE_CANDIDATES)
    PARQUET_REMOTE_URIS = PARQUET_BENCHMARK_CONFIG.get("remote_uris", {})
    PARQUET_INVENTORY_MANIFEST = PARQUET_BENCHMARK_CONFIG.get("inventory_manifest")
    if not PARQUET_INVENTORY_MANIFEST:
        raise WorkflowError(
            "parquet_benchmark.inventory_manifest is required when enabled"
        )
    PARQUET_STAGING_CHECKS = PARQUET_BENCHMARK_CONFIG.get("staging_checks")
    if not PARQUET_STAGING_CHECKS:
        raise WorkflowError("parquet_benchmark.staging_checks is required when enabled")

    for case_id in PARQUET_CASE_IDS:
        candidate_uris = PARQUET_REMOTE_URIS.get(case_id, {})
        missing_candidates = sorted(set(PARQUET_CANDIDATE_NAMES) - set(candidate_uris))
        if missing_candidates:
            raise WorkflowError(
                f"remote URIs for {case_id} are missing candidates: "
                + ", ".join(missing_candidates)
            )
        for candidate, uri in candidate_uris.items():
            if candidate not in PARQUET_CANDIDATES:
                raise WorkflowError(
                    f"unknown remote candidate {candidate!r} for {case_id}"
                )
            if not str(uri).startswith("hf://"):
                raise WorkflowError(
                    f"remote URI for {case_id}/{candidate} must start with hf://"
                )
            revision_match = re.fullmatch(
                r"hf://datasets/[^/@]+/[^/@]+@([0-9a-f]{40})/.+", str(uri)
            )
            if revision_match is None:
                raise WorkflowError(
                    f"remote URI for {case_id}/{candidate} must pin a 40-character "
                    "Hugging Face commit revision"
                )

    def parquet_benchmark_resource(stage, name):
        value = PARQUET_BENCHMARK_CONFIG.get("resources", {}).get(stage, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(
                f"parquet_benchmark.resources.{stage}.{name} must be a positive "
                "integer based on a representative pilot"
            )
        return value

    def local_candidate_path(case_id, candidate):
        case = PARQUET_CASES[case_id]
        if candidate == "source":
            return PARQUET_SOURCE_ROOT / case["relative_path"]
        return PARQUET_BENCHMARK_ROOT / "candidates" / case_id / f"{candidate}.parquet"

    def benchmark_input(wildcards):
        if wildcards.access == "local":
            return str(local_candidate_path(wildcards.case, wildcards.candidate))
        if wildcards.access == "hf":
            return []
        raise WorkflowError(f"unknown benchmark access mode: {wildcards.access}")

    def benchmark_uri(wildcards):
        if wildcards.access == "local":
            return str(local_candidate_path(wildcards.case, wildcards.candidate))
        return PARQUET_REMOTE_URIS[wildcards.case][wildcards.candidate]

    PARQUET_REWRITE_REPORTS = [
        str(PARQUET_BENCHMARK_ROOT / "rewrite" / case / f"{candidate}.json")
        for case in PARQUET_CASE_IDS
        for candidate in PARQUET_REWRITE_NAMES
    ]
    PARQUET_SOURCE_EVIDENCE_REPORTS = [
        str(PARQUET_BENCHMARK_ROOT / "source-evidence" / f"{case}.json")
        for case in PARQUET_CASE_IDS
    ]
    PARQUET_BENCHMARK_REPORTS = [
        str(PARQUET_BENCHMARK_ROOT / "benchmarks" / access / case / f"{candidate}.json")
        for access in ("local", "hf")
        for case in PARQUET_CASE_IDS
        for candidate in PARQUET_CANDIDATE_NAMES
    ]
    PARQUET_HF_CHECK_REPORTS = [
        str(PARQUET_BENCHMARK_ROOT / "hf-polars" / case / f"{candidate}.json")
        for case in PARQUET_CASE_IDS
        for candidate in PARQUET_CANDIDATE_NAMES
    ]

    wildcard_constraints:
        case="|".join(re.escape(case) for case in PARQUET_CASE_IDS),
        candidate="|".join(
            re.escape(candidate) for candidate in PARQUET_CANDIDATE_NAMES
        ),
        access="local|hf",

    rule validate_source_shard:
        """Tie one immutable source shard to issue #8's inventory manifest."""
        input:
            manifest=str(PARQUET_INVENTORY_MANIFEST),
            source=lambda wildcards: str(local_candidate_path(wildcards.case, "source")),
        output:
            str(PARQUET_BENCHMARK_ROOT / "source-evidence" / "{case}.json"),
        log:
            str(PARQUET_BENCHMARK_ROOT / "logs" / "source-evidence" / "{case}.log"),
        threads: 4
        resources:
            mem_mb=lambda wildcards: parquet_benchmark_resource("benchmark", "mem_mb"),
            runtime=lambda wildcards: parquet_benchmark_resource("benchmark", "runtime"),
            disk_mb=lambda wildcards: parquet_benchmark_resource("benchmark", "disk_mb"),
        run:
            try:
                case = PARQUET_CASES[wildcards.case]
                validate_benchmark_source(
                    Path(input.manifest),
                    Path(input.source),
                    Path(output[0]),
                    case=wildcards.case,
                    relative_path=case["relative_path"],
                    score_type=case["score_type"],
                )
                write_parquet_benchmark_log(
                    log[0], f"validated source inventory for {wildcards.case}"
                )
            except BaseException as error:
                write_parquet_benchmark_log(
                    log[0], f"source inventory validation failed: {error}"
                )
                raise

    rule rewrite_parquet_shard:
        """Write and validate one complete candidate shard before promotion."""
        input:
            source=lambda wildcards: str(local_candidate_path(wildcards.case, "source")),
            evidence=lambda wildcards: str(
                PARQUET_BENCHMARK_ROOT / "source-evidence" / f"{wildcards.case}.json"
            ),
        output:
            parquet=str(
                PARQUET_BENCHMARK_ROOT
                / "candidates"
                / "{case}"
                / "{candidate}.parquet"
            ),
            report=str(
                PARQUET_BENCHMARK_ROOT / "rewrite" / "{case}" / "{candidate}.json"
            ),
        log:
            str(
                PARQUET_BENCHMARK_ROOT
                / "logs"
                / "rewrite"
                / "{case}"
                / "{candidate}.log"
            ),
        wildcard_constraints:
            candidate="|".join(
                re.escape(candidate) for candidate in PARQUET_REWRITE_NAMES
            ),
        threads: 4
        resources:
            mem_mb=lambda wildcards: parquet_benchmark_resource("rewrite", "mem_mb"),
            runtime=lambda wildcards: parquet_benchmark_resource("rewrite", "runtime"),
            disk_mb=lambda wildcards: parquet_benchmark_resource("rewrite", "disk_mb"),
        run:
            try:
                rewrite_parquet_candidate(
                    Path(input.source),
                    Path(output.parquet),
                    Path(output.report),
                    case=wildcards.case,
                    score_type=PARQUET_CASES[wildcards.case]["score_type"],
                    candidate=PARQUET_REWRITE_CANDIDATES[wildcards.candidate],
                    threads=int(threads),
                )
                write_parquet_benchmark_log(
                    log[0], f"rewrote {wildcards.case}/{wildcards.candidate}"
                )
            except BaseException as error:
                write_parquet_benchmark_log(log[0], f"rewrite failed: {error}")
                raise

    rule benchmark_parquet_shard:
        """Run the fixed local or HF query suite on one complete shard."""
        input:
            data=benchmark_input,
            evidence=lambda wildcards: str(
                PARQUET_BENCHMARK_ROOT / "source-evidence" / f"{wildcards.case}.json"
            ),
        output:
            str(
                PARQUET_BENCHMARK_ROOT
                / "benchmarks"
                / "{access}"
                / "{case}"
                / "{candidate}.json"
            ),
        log:
            str(
                PARQUET_BENCHMARK_ROOT
                / "logs"
                / "benchmarks"
                / "{access}"
                / "{case}"
                / "{candidate}.log"
            ),
        threads: 4
        resources:
            mem_mb=lambda wildcards: parquet_benchmark_resource("benchmark", "mem_mb"),
            runtime=lambda wildcards: parquet_benchmark_resource("benchmark", "runtime"),
            disk_mb=lambda wildcards: parquet_benchmark_resource("benchmark", "disk_mb"),
        params:
            uri=benchmark_uri,
        run:
            try:
                benchmark_parquet_candidate(
                    params.uri,
                    Path(output[0]),
                    case=wildcards.case,
                    candidate=wildcards.candidate,
                    access=wildcards.access,
                    sparse_key_count=int(
                        PARQUET_BENCHMARK_CONFIG.get("sparse_key_count", 1024)
                    ),
                    hf_token=os.environ.get("HF_TOKEN"),
                    hf_block_size=int(
                        PARQUET_BENCHMARK_CONFIG.get("hf_block_size", 4194304)
                    ),
                    threads=int(threads),
                )
                write_parquet_benchmark_log(
                    log[0],
                    f"benchmarked {wildcards.case}/{wildcards.candidate} "
                    f"via {wildcards.access}",
                )
            except BaseException as error:
                write_parquet_benchmark_log(log[0], f"benchmark failed: {error}")
                raise

    rule validate_parquet_shard:
        """Verify direct lazy predicate/projection pushdown over hf://."""
        input:
            evidence=lambda wildcards: str(
                PARQUET_BENCHMARK_ROOT / "source-evidence" / f"{wildcards.case}.json"
            ),
        output:
            str(PARQUET_BENCHMARK_ROOT / "hf-polars" / "{case}" / "{candidate}.json"),
        log:
            str(
                PARQUET_BENCHMARK_ROOT
                / "logs"
                / "hf-polars"
                / "{case}"
                / "{candidate}.log"
            ),
        threads: 4
        resources:
            mem_mb=lambda wildcards: parquet_benchmark_resource("benchmark", "mem_mb"),
            runtime=lambda wildcards: parquet_benchmark_resource("benchmark", "runtime"),
            disk_mb=lambda wildcards: parquet_benchmark_resource("benchmark", "disk_mb"),
        params:
            uri=lambda wildcards: PARQUET_REMOTE_URIS[wildcards.case][
                wildcards.candidate
            ],
        run:
            try:
                validate_hf_polars(
                    params.uri,
                    Path(output[0]),
                    case=wildcards.case,
                    candidate=wildcards.candidate,
                    hf_token=os.environ.get("HF_TOKEN"),
                    threads=int(threads),
                )
                write_parquet_benchmark_log(
                    log[0],
                    f"validated hf:// for {wildcards.case}/{wildcards.candidate}",
                )
            except BaseException as error:
                write_parquet_benchmark_log(log[0], f"hf:// check failed: {error}")
                raise

    rule render_report:
        """Apply the declared selection rule and render dataset-card evidence."""
        input:
            benchmarks=PARQUET_BENCHMARK_REPORTS,
            rewrites=PARQUET_REWRITE_REPORTS,
            source_evidence=PARQUET_SOURCE_EVIDENCE_REPORTS,
            hf_polars=PARQUET_HF_CHECK_REPORTS,
            staging=str(PARQUET_STAGING_CHECKS),
        output:
            json=str(PARQUET_BENCHMARK_ROOT / "selection.json"),
            markdown=str(PARQUET_BENCHMARK_ROOT / "dataset-card-parquet-benchmark.md"),
        log:
            str(PARQUET_BENCHMARK_ROOT / "logs" / "selection.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: parquet_benchmark_resource("report", "mem_mb"),
            runtime=lambda wildcards: parquet_benchmark_resource("report", "runtime"),
            disk_mb=lambda wildcards: parquet_benchmark_resource("report", "disk_mb"),
        run:
            try:
                write_selection_outputs(
                    [Path(path) for path in input.benchmarks],
                    [Path(path) for path in input.rewrites],
                    [Path(path) for path in input.source_evidence],
                    Path(input.staging),
                    Path(output.json),
                    Path(output.markdown),
                    hf_validation_reports=[Path(path) for path in input.hf_polars],
                )
                write_parquet_benchmark_log(log[0], "rendered layout selection")
            except BaseException as error:
                write_parquet_benchmark_log(log[0], f"selection failed: {error}")
                raise
