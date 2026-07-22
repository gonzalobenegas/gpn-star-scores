"""Issue #7 BigWig benchmark, generation, and validation workflow."""

from pathlib import Path
import re

from snakemake.exceptions import WorkflowError

from gpn_star_scores.catalog import (
    ASSEMBLIES,
    SCORE_SETS,
    get_shard_spec,
    score_set_assembly,
)
from gpn_star_scores.tracks import (
    METHODS,
    TRACKS,
    aggregate_track_validation,
    render_track_benchmark,
)

BIGWIG_CONFIG = config.get("bigwig", {})
BIGWIG_ENABLED = bool(BIGWIG_CONFIG.get("enabled", False))
BIGWIG_OUTPUT_ROOT = Path(BIGWIG_CONFIG.get("output_root", "results/bigwig"))
BIGWIG_BENCHMARK_CONFIG = BIGWIG_CONFIG.get("benchmark", {})
BIGWIG_BENCHMARK_ENABLED = bool(BIGWIG_BENCHMARK_CONFIG.get("enabled", False))
BIGWIG_BENCHMARK_ROOT = Path(
    BIGWIG_BENCHMARK_CONFIG.get("output_root", str(BIGWIG_OUTPUT_ROOT / "benchmark"))
)


def bigwig_targets():
    """Return final BigWigs and reports only when issue #7 is enabled."""

    if not BIGWIG_ENABLED:
        return []
    targets = [*BIGWIG_FINAL_PATHS]
    targets.extend(
        [
            str(BIGWIG_OUTPUT_ROOT / "validation.json"),
            str(BIGWIG_OUTPUT_ROOT / "validation.md"),
            str(BIGWIG_TRACK_SELECTION_PATH),
        ]
    )
    if BIGWIG_BENCHMARK_ENABLED:
        targets.append(str(BIGWIG_BENCHMARK_ROOT / "selection.md"))
    return targets


if BIGWIG_ENABLED:
    for required_key in ("source_root", "inventory_manifest", "parquet_selection"):
        if not BIGWIG_CONFIG.get(required_key):
            raise WorkflowError(
                f"bigwig.{required_key} is required when BigWig generation is enabled"
            )
    BIGWIG_SOURCE_ROOT = Path(BIGWIG_CONFIG["source_root"])
    BIGWIG_INVENTORY_MANIFEST = Path(BIGWIG_CONFIG["inventory_manifest"])
    BIGWIG_PARQUET_SELECTION = Path(BIGWIG_CONFIG["parquet_selection"])
    if BIGWIG_OUTPUT_ROOT.resolve().is_relative_to(BIGWIG_SOURCE_ROOT.resolve()):
        raise WorkflowError(
            "bigwig.output_root must not be inside the immutable source_root"
        )
    BIGWIG_BATCH_SIZE = int(BIGWIG_CONFIG.get("batch_size", 262_144))
    BIGWIG_SAMPLE_COUNT = int(BIGWIG_CONFIG.get("sample_count", 1_024))
    BIGWIG_VALUE_DECIMALS = BIGWIG_CONFIG.get("value_decimals")
    if BIGWIG_BATCH_SIZE <= 0 or BIGWIG_SAMPLE_COUNT <= 0:
        raise WorkflowError("bigwig batch_size and sample_count must be positive")
    if (
        not isinstance(BIGWIG_VALUE_DECIMALS, int)
        or isinstance(BIGWIG_VALUE_DECIMALS, bool)
        or not 0 <= BIGWIG_VALUE_DECIMALS <= 9
    ):
        raise WorkflowError("bigwig.value_decimals must be an integer from 0 through 9")

    def bigwig_resource(stage, name):
        value = BIGWIG_CONFIG.get("resources", {}).get(stage, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(
                f"bigwig.resources.{stage}.{name} must be a positive integer "
                "based on an SCF pilot"
            )
        return value

    if BIGWIG_BENCHMARK_ENABLED:
        raw_cases = BIGWIG_BENCHMARK_CONFIG.get("cases", [])
        BIGWIG_BENCHMARK_CASES = {case.get("id"): case for case in raw_cases}
        if not raw_cases or len(BIGWIG_BENCHMARK_CASES) != len(raw_cases):
            raise WorkflowError("bigwig benchmark case IDs must be present and unique")
        for case_id, case in BIGWIG_BENCHMARK_CASES.items():
            if not isinstance(case_id, str) or not re.fullmatch(
                r"[A-Za-z0-9._-]+", case_id
            ):
                raise WorkflowError(f"unsafe BigWig benchmark case ID: {case_id!r}")
            try:
                get_shard_spec(
                    case.get("score_set", ""),
                    case.get("score_type", ""),
                    str(case.get("chrom", "")),
                )
            except KeyError as error:
                raise WorkflowError(
                    f"invalid BigWig benchmark case {case_id}: {error}"
                ) from error
        BIGWIG_BENCHMARK_CASE_IDS = sorted(BIGWIG_BENCHMARK_CASES)
        BIGWIG_BENCHMARK_REPETITIONS = int(
            BIGWIG_BENCHMARK_CONFIG.get("repetitions", 5)
        )
        if BIGWIG_BENCHMARK_REPETITIONS <= 0:
            raise WorkflowError("bigwig.benchmark.repetitions must be positive")
        BIGWIG_BENCHMARK_REPORTS = [
            str(BIGWIG_BENCHMARK_ROOT / "reports" / case / f"{method}.json")
            for case in BIGWIG_BENCHMARK_CASE_IDS
            for method in METHODS
        ]
        BIGWIG_TRACK_SELECTION_PATH = BIGWIG_BENCHMARK_ROOT / "selection.json"

        wildcard_constraints:
            bigwig_case="|".join(re.escape(case) for case in BIGWIG_BENCHMARK_CASE_IDS),
            bigwig_method="|".join(re.escape(method) for method in METHODS),

        rule benchmark_bigwig_method:
            """Benchmark one complete chromosome score shard with one method."""
            input:
                source=lambda wildcards: str(
                    BIGWIG_SOURCE_ROOT
                    / get_shard_spec(
                        BIGWIG_BENCHMARK_CASES[wildcards.bigwig_case]["score_set"],
                        BIGWIG_BENCHMARK_CASES[wildcards.bigwig_case]["score_type"],
                        str(BIGWIG_BENCHMARK_CASES[wildcards.bigwig_case]["chrom"]),
                    ).relative_path
                ),
                manifest=str(BIGWIG_INVENTORY_MANIFEST),
                parquet_selection=str(BIGWIG_PARQUET_SELECTION),
            output:
                artifacts=temp(
                    directory(
                        str(
                            BIGWIG_BENCHMARK_ROOT
                            / "artifacts"
                            / "{bigwig_case}"
                            / "{bigwig_method}"
                        )
                    )
                ),
                report=str(
                    BIGWIG_BENCHMARK_ROOT
                    / "reports"
                    / "{bigwig_case}"
                    / "{bigwig_method}.json"
                ),
            log:
                str(
                    BIGWIG_BENCHMARK_ROOT
                    / "logs"
                    / "{bigwig_case}"
                    / "{bigwig_method}.log"
                ),
            conda:
                "../envs/ucsc.yaml"
            threads: 1
            resources:
                mem_mb=lambda wildcards: bigwig_resource("benchmark", "mem_mb"),
                runtime=lambda wildcards: bigwig_resource("benchmark", "runtime"),
                disk_mb=lambda wildcards: bigwig_resource("benchmark", "disk_mb"),
            params:
                score_set=lambda wildcards: BIGWIG_BENCHMARK_CASES[
                    wildcards.bigwig_case
                ]["score_set"],
                score_type=lambda wildcards: BIGWIG_BENCHMARK_CASES[
                    wildcards.bigwig_case
                ]["score_type"],
                chrom=lambda wildcards: str(
                    BIGWIG_BENCHMARK_CASES[wildcards.bigwig_case]["chrom"]
                ),
            shell:
                """
                {PYTHON_EXECUTABLE:q} -m gpn_star_scores.tracks benchmark-method \
                    --source-root {BIGWIG_SOURCE_ROOT:q} \
                    --inventory-manifest {input.manifest:q} \
                    --parquet-selection {input.parquet_selection:q} \
                    --artifact-root {output.artifacts:q} \
                    --report {output.report:q} \
                    --case {wildcards.bigwig_case:q} \
                    --score-set {params.score_set:q} \
                    --score-type {params.score_type:q} \
                    --chrom {params.chrom:q} \
                    --method {wildcards.bigwig_method:q} \
                    --repetitions {BIGWIG_BENCHMARK_REPETITIONS} \
                    --sample-count {BIGWIG_SAMPLE_COUNT} \
                    --batch-size {BIGWIG_BATCH_SIZE} \
                    >{log:q} 2>&1
                """

        rule render_bigwig_report:
            """Aggregate BigWig measurements and apply the declared threshold."""
            input:
                BIGWIG_BENCHMARK_REPORTS,
            output:
                json=str(BIGWIG_BENCHMARK_ROOT / "selection.json"),
                markdown=str(BIGWIG_BENCHMARK_ROOT / "selection.md"),
            log:
                str(BIGWIG_BENCHMARK_ROOT / "logs" / "selection.log"),
            threads: 1
            resources:
                mem_mb=lambda wildcards: bigwig_resource("report", "mem_mb"),
                runtime=lambda wildcards: bigwig_resource("report", "runtime"),
                disk_mb=lambda wildcards: bigwig_resource("report", "disk_mb"),
            run:
                try:
                    render_track_benchmark(
                        [Path(path) for path in input],
                        Path(output.json),
                        Path(output.markdown),
                    )
                    Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                    Path(log[0]).write_text("selected BigWig method\n")
                except BaseException as error:
                    Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                    Path(log[0]).write_text(f"BigWig selection failed: {error}\n")
                    raise

    else:
        if not BIGWIG_CONFIG.get("selection_report"):
            raise WorkflowError(
                "bigwig.selection_report is required when benchmark execution is disabled"
            )
        BIGWIG_TRACK_SELECTION_PATH = Path(BIGWIG_CONFIG["selection_report"])
    BIGWIG_SCORE_SET_NAMES = [spec.name for spec in SCORE_SETS]
    BIGWIG_FINAL_PATHS = [
        str(BIGWIG_OUTPUT_ROOT / "final" / score_set / f"{track}.bw")
        for score_set in BIGWIG_SCORE_SET_NAMES
        for track in TRACKS
    ]
    BIGWIG_CONCATENATION_REPORTS = [
        str(BIGWIG_OUTPUT_ROOT / "final-reports" / score_set / f"{track}.json")
        for score_set in BIGWIG_SCORE_SET_NAMES
        for track in TRACKS
    ]
    BIGWIG_FINAL_REPORTS = [
        str(BIGWIG_OUTPUT_ROOT / "audit-reports" / score_set / f"{track}.json")
        for score_set in BIGWIG_SCORE_SET_NAMES
        for track in TRACKS
    ]
    if not all(
        len(paths) == 40
        for paths in (
            BIGWIG_FINAL_PATHS,
            BIGWIG_CONCATENATION_REPORTS,
            BIGWIG_FINAL_REPORTS,
        )
    ):
        raise AssertionError(
            "release catalog must produce exactly 40 BigWigs and validation reports"
        )

    wildcard_constraints:
        bigwig_score_set="|".join(
            re.escape(score_set) for score_set in BIGWIG_SCORE_SET_NAMES
        ),
        bigwig_track="|".join(re.escape(track) for track in TRACKS),

    def bigwig_source_path(wildcards, score_type):
        return str(
            BIGWIG_SOURCE_ROOT
            / get_shard_spec(
                wildcards.bigwig_score_set, score_type, wildcards.chrom
            ).relative_path
        )

    def chromosome_bigwig_inputs(wildcards):
        assembly = score_set_assembly(wildcards.bigwig_score_set)
        return [
            str(
                BIGWIG_OUTPUT_ROOT
                / "chromosomes"
                / wildcards.bigwig_score_set
                / chrom
                / f"{wildcards.bigwig_track}.bw"
            )
            for chrom in ASSEMBLIES[assembly].chromosomes
        ]

    def chromosome_report_inputs(wildcards):
        assembly = score_set_assembly(wildcards.bigwig_score_set)
        return [
            str(
                BIGWIG_OUTPUT_ROOT
                / "chromosome-reports"
                / wildcards.bigwig_score_set
                / f"{chrom}.json"
            )
            for chrom in ASSEMBLIES[assembly].chromosomes
        ]

    # Final concatenations consume the regenerable BigWigs. Keeping the compact
    # report durable bounds storage without losing chromosome validation evidence.
    rule build_chromosome_bigwig:
        """Build and validate all five tracks for one chromosome restart unit."""
        input:
            entropy=lambda wildcards: bigwig_source_path(wildcards, "entropy"),
            llr=lambda wildcards: bigwig_source_path(wildcards, "llr"),
            manifest=str(BIGWIG_INVENTORY_MANIFEST),
            parquet_selection=str(BIGWIG_PARQUET_SELECTION),
            track_selection=str(BIGWIG_TRACK_SELECTION_PATH),
        output:
            entropy=temp(
                str(
                    BIGWIG_OUTPUT_ROOT
                    / "chromosomes"
                    / "{bigwig_score_set}"
                    / "{chrom}"
                    / "entropy.bw"
                )
            ),
            A=temp(
                str(
                    BIGWIG_OUTPUT_ROOT
                    / "chromosomes"
                    / "{bigwig_score_set}"
                    / "{chrom}"
                    / "A.bw"
                )
            ),
            C=temp(
                str(
                    BIGWIG_OUTPUT_ROOT
                    / "chromosomes"
                    / "{bigwig_score_set}"
                    / "{chrom}"
                    / "C.bw"
                )
            ),
            G=temp(
                str(
                    BIGWIG_OUTPUT_ROOT
                    / "chromosomes"
                    / "{bigwig_score_set}"
                    / "{chrom}"
                    / "G.bw"
                )
            ),
            T=temp(
                str(
                    BIGWIG_OUTPUT_ROOT
                    / "chromosomes"
                    / "{bigwig_score_set}"
                    / "{chrom}"
                    / "T.bw"
                )
            ),
            report=str(
                BIGWIG_OUTPUT_ROOT
                / "chromosome-reports"
                / "{bigwig_score_set}"
                / "{chrom}.json"
            ),
        log:
            str(
                BIGWIG_OUTPUT_ROOT
                / "logs"
                / "chromosomes"
                / "{bigwig_score_set}"
                / "{chrom}.log"
            ),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: bigwig_resource("build_chromosome", "mem_mb"),
            runtime=lambda wildcards: bigwig_resource("build_chromosome", "runtime"),
            disk_mb=lambda wildcards: bigwig_resource("build_chromosome", "disk_mb"),
        params:
            output_dir=lambda wildcards: str(
                BIGWIG_OUTPUT_ROOT
                / "chromosomes"
                / wildcards.bigwig_score_set
                / wildcards.chrom
            ),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.tracks build-chromosome \
                --source-root {BIGWIG_SOURCE_ROOT:q} \
                --inventory-manifest {input.manifest:q} \
                --parquet-selection {input.parquet_selection:q} \
                --track-selection {input.track_selection:q} \
                --score-set {wildcards.bigwig_score_set:q} \
                --chrom {wildcards.chrom:q} \
                --output-dir {params.output_dir:q} \
                --report {output.report:q} \
                --batch-size {BIGWIG_BATCH_SIZE} \
                --sample-count {BIGWIG_SAMPLE_COUNT} \
                >{log:q} 2>&1
            """

    rule concatenate_bigwig:
        """Combine validated disjoint chromosome files into one final track."""
        input:
            bigwigs=chromosome_bigwig_inputs,
            chromosome_reports=chromosome_report_inputs,
            manifest=str(BIGWIG_INVENTORY_MANIFEST),
            parquet_selection=str(BIGWIG_PARQUET_SELECTION),
        output:
            bigwig=str(
                BIGWIG_OUTPUT_ROOT
                / "final"
                / "{bigwig_score_set}"
                / "{bigwig_track}.bw"
            ),
            report=str(
                BIGWIG_OUTPUT_ROOT
                / "final-reports"
                / "{bigwig_score_set}"
                / "{bigwig_track}.json"
            ),
        log:
            str(
                BIGWIG_OUTPUT_ROOT
                / "logs"
                / "final"
                / "{bigwig_score_set}"
                / "{bigwig_track}.log"
            ),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: bigwig_resource("concatenate", "mem_mb"),
            runtime=lambda wildcards: bigwig_resource("concatenate", "runtime"),
            disk_mb=lambda wildcards: bigwig_resource("concatenate", "disk_mb"),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.tracks concatenate \
                --inventory-manifest {input.manifest:q} \
                --parquet-selection {input.parquet_selection:q} \
                --score-set {wildcards.bigwig_score_set:q} \
                --track {wildcards.bigwig_track:q} \
                --value-decimals {BIGWIG_VALUE_DECIMALS} \
                --output {output.bigwig:q} \
                --report {output.report:q} \
                --inputs {input.bigwigs:q} \
                --chromosome-reports {input.chromosome_reports:q} \
                >{log:q} 2>&1
            """

    rule audit_final_bigwig:
        """Recheck all stored random, first/last, and gap samples."""
        input:
            bigwig=str(
                BIGWIG_OUTPUT_ROOT
                / "final"
                / "{bigwig_score_set}"
                / "{bigwig_track}.bw"
            ),
            concatenation_report=str(
                BIGWIG_OUTPUT_ROOT
                / "final-reports"
                / "{bigwig_score_set}"
                / "{bigwig_track}.json"
            ),
            chromosome_reports=chromosome_report_inputs,
            manifest=str(BIGWIG_INVENTORY_MANIFEST),
            parquet_selection=str(BIGWIG_PARQUET_SELECTION),
        output:
            report=str(
                BIGWIG_OUTPUT_ROOT
                / "audit-reports"
                / "{bigwig_score_set}"
                / "{bigwig_track}.json"
            ),
        log:
            str(
                BIGWIG_OUTPUT_ROOT
                / "logs"
                / "audit"
                / "{bigwig_score_set}"
                / "{bigwig_track}.log"
            ),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: bigwig_resource("audit", "mem_mb"),
            runtime=lambda wildcards: bigwig_resource("audit", "runtime"),
            disk_mb=lambda wildcards: bigwig_resource("audit", "disk_mb"),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.tracks audit-final \
                --inventory-manifest {input.manifest:q} \
                --parquet-selection {input.parquet_selection:q} \
                --score-set {wildcards.bigwig_score_set:q} \
                --track {wildcards.bigwig_track:q} \
                --value-decimals {BIGWIG_VALUE_DECIMALS} \
                --bigwig {input.bigwig:q} \
                --concatenation-report {input.concatenation_report:q} \
                --report {output.report:q} \
                --chromosome-reports {input.chromosome_reports:q} \
                >{log:q} 2>&1
            """

    rule aggregate_validation:
        """Require complete, valid evidence for all 40 final tracks."""
        input:
            reports=BIGWIG_FINAL_REPORTS,
            selection=str(BIGWIG_TRACK_SELECTION_PATH),
        output:
            json=str(BIGWIG_OUTPUT_ROOT / "validation.json"),
            markdown=str(BIGWIG_OUTPUT_ROOT / "validation.md"),
        log:
            str(BIGWIG_OUTPUT_ROOT / "logs" / "validation.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: bigwig_resource("aggregate", "mem_mb"),
            runtime=lambda wildcards: bigwig_resource("aggregate", "runtime"),
            disk_mb=lambda wildcards: bigwig_resource("aggregate", "disk_mb"),
        run:
            try:
                aggregate_track_validation(
                    [Path(path) for path in input.reports],
                    Path(input.selection),
                    Path(output.json),
                    Path(output.markdown),
                )
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text("validated all 40 BigWigs\n")
            except BaseException as error:
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text(f"BigWig validation failed: {error}\n")
                raise
