"""Issue #15 raw calibrated-LLR BigWig generation and validation workflow."""

from pathlib import Path
import re

from snakemake.exceptions import WorkflowError

from gpn_star_scores.catalog import (
    ASSEMBLIES,
    SCORE_SETS,
    get_shard_spec,
    score_set_assembly,
)
from gpn_star_scores.raw_llr import (
    RAW_LLR_TRACKS,
    VALUE_DECIMALS,
    aggregate_raw_llr_validation,
)

RAW_LLR_CONFIG = config.get("raw_llr", {})
RAW_LLR_ENABLED = bool(RAW_LLR_CONFIG.get("enabled", False))
RAW_LLR_OUTPUT_ROOT = Path(RAW_LLR_CONFIG.get("output_root", "results/raw-llr"))
RAW_LLR_PUBLICATION_REPORT = RAW_LLR_OUTPUT_ROOT / "publication.json"
RAW_LLR_PUBLICATION_SUCCESS = RAW_LLR_OUTPUT_ROOT / "publication.complete"


def raw_llr_targets():
    """Return only the 32 new BigWigs and their focused validation evidence."""

    if not RAW_LLR_ENABLED:
        return []
    return [
        *RAW_LLR_FINAL_PATHS,
        str(RAW_LLR_OUTPUT_ROOT / "validation.json"),
        str(RAW_LLR_OUTPUT_ROOT / "validation.md"),
    ]


def raw_llr_publication_report():
    """Return the durable approval-gated publication report."""

    return str(RAW_LLR_PUBLICATION_REPORT)


def raw_llr_publication_success_marker():
    """Return the success-only raw-LLR publication target."""

    return str(RAW_LLR_PUBLICATION_SUCCESS)


def raw_llr_approval_value(name):
    """Render one exact approval field for the publication command."""

    approval = RAW_LLR_CONFIG.get("publication_approval")
    if not isinstance(approval, dict):
        return ""
    value = approval.get(name)
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


if RAW_LLR_ENABLED:
    for required_key in (
        "source_root",
        "inventory_manifest",
        "parquet_selection",
        "track_selection",
    ):
        if not RAW_LLR_CONFIG.get(required_key):
            raise WorkflowError(
                f"raw_llr.{required_key} is required when raw LLR generation is enabled"
            )
    RAW_LLR_SOURCE_ROOT = Path(RAW_LLR_CONFIG["source_root"])
    RAW_LLR_INVENTORY_MANIFEST = Path(RAW_LLR_CONFIG["inventory_manifest"])
    RAW_LLR_PARQUET_SELECTION = Path(RAW_LLR_CONFIG["parquet_selection"])
    RAW_LLR_TRACK_SELECTION = Path(RAW_LLR_CONFIG["track_selection"])
    if RAW_LLR_OUTPUT_ROOT.resolve().is_relative_to(RAW_LLR_SOURCE_ROOT.resolve()):
        raise WorkflowError(
            "raw_llr.output_root must not be inside the immutable source_root"
        )
    RAW_LLR_BATCH_SIZE = int(RAW_LLR_CONFIG.get("batch_size", 262_144))
    RAW_LLR_SAMPLE_COUNT = int(RAW_LLR_CONFIG.get("sample_count", 1_024))
    RAW_LLR_VALUE_DECIMALS = RAW_LLR_CONFIG.get("value_decimals", VALUE_DECIMALS)
    if RAW_LLR_BATCH_SIZE <= 0 or RAW_LLR_SAMPLE_COUNT <= 0:
        raise WorkflowError("raw_llr batch_size and sample_count must be positive")
    if RAW_LLR_VALUE_DECIMALS != VALUE_DECIMALS:
        raise WorkflowError(f"raw_llr.value_decimals must be exactly {VALUE_DECIMALS}")

    def raw_llr_resource(stage, name):
        value = RAW_LLR_CONFIG.get("resources", {}).get(stage, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(
                f"raw_llr.resources.{stage}.{name} must be a positive integer "
                "based on the issue #15 SCF pilot"
            )
        return value

    RAW_LLR_SCORE_SET_NAMES = [spec.name for spec in SCORE_SETS]
    RAW_LLR_FINAL_PATHS = [
        str(RAW_LLR_OUTPUT_ROOT / "final" / score_set / f"{track}.bw")
        for score_set in RAW_LLR_SCORE_SET_NAMES
        for track in RAW_LLR_TRACKS
    ]
    RAW_LLR_CONCATENATION_REPORTS = [
        str(RAW_LLR_OUTPUT_ROOT / "final-reports" / score_set / f"{track}.json")
        for score_set in RAW_LLR_SCORE_SET_NAMES
        for track in RAW_LLR_TRACKS
    ]
    RAW_LLR_FINAL_REPORTS = [
        str(RAW_LLR_OUTPUT_ROOT / "audit-reports" / score_set / f"{track}.json")
        for score_set in RAW_LLR_SCORE_SET_NAMES
        for track in RAW_LLR_TRACKS
    ]
    if not all(
        len(paths) == 32
        for paths in (
            RAW_LLR_FINAL_PATHS,
            RAW_LLR_CONCATENATION_REPORTS,
            RAW_LLR_FINAL_REPORTS,
        )
    ):
        raise AssertionError("the raw-LLR catalog must contain exactly 32 tracks")

    wildcard_constraints:
        raw_llr_score_set="|".join(
            re.escape(score_set) for score_set in RAW_LLR_SCORE_SET_NAMES
        ),
        raw_llr_track="|".join(re.escape(track) for track in RAW_LLR_TRACKS),

    def raw_llr_source_path(wildcards):
        return str(
            RAW_LLR_SOURCE_ROOT
            / get_shard_spec(
                wildcards.raw_llr_score_set, "llr", wildcards.chrom
            ).relative_path
        )

    def raw_llr_chromosome_bigwig_inputs(wildcards):
        assembly = score_set_assembly(wildcards.raw_llr_score_set)
        return [
            str(
                RAW_LLR_OUTPUT_ROOT
                / "chromosomes"
                / wildcards.raw_llr_score_set
                / chrom
                / f"{wildcards.raw_llr_track}.bw"
            )
            for chrom in ASSEMBLIES[assembly].chromosomes
        ]

    def raw_llr_chromosome_report_inputs(wildcards):
        assembly = score_set_assembly(wildcards.raw_llr_score_set)
        return [
            str(
                RAW_LLR_OUTPUT_ROOT
                / "chromosome-reports"
                / wildcards.raw_llr_score_set
                / f"{chrom}.json"
            )
            for chrom in ASSEMBLIES[assembly].chromosomes
        ]

    rule build_raw_llr_chromosome:
        """Build and validate four raw-LLR tracks for one chromosome."""
        input:
            source=raw_llr_source_path,
            manifest=str(RAW_LLR_INVENTORY_MANIFEST),
            parquet_selection=str(RAW_LLR_PARQUET_SELECTION),
            track_selection=str(RAW_LLR_TRACK_SELECTION),
        output:
            llr_A=temp(
                str(
                    RAW_LLR_OUTPUT_ROOT
                    / "chromosomes"
                    / "{raw_llr_score_set}"
                    / "{chrom}"
                    / "llr_A.bw"
                )
            ),
            llr_C=temp(
                str(
                    RAW_LLR_OUTPUT_ROOT
                    / "chromosomes"
                    / "{raw_llr_score_set}"
                    / "{chrom}"
                    / "llr_C.bw"
                )
            ),
            llr_G=temp(
                str(
                    RAW_LLR_OUTPUT_ROOT
                    / "chromosomes"
                    / "{raw_llr_score_set}"
                    / "{chrom}"
                    / "llr_G.bw"
                )
            ),
            llr_T=temp(
                str(
                    RAW_LLR_OUTPUT_ROOT
                    / "chromosomes"
                    / "{raw_llr_score_set}"
                    / "{chrom}"
                    / "llr_T.bw"
                )
            ),
            report=str(
                RAW_LLR_OUTPUT_ROOT
                / "chromosome-reports"
                / "{raw_llr_score_set}"
                / "{chrom}.json"
            ),
        log:
            str(
                RAW_LLR_OUTPUT_ROOT
                / "logs"
                / "chromosomes"
                / "{raw_llr_score_set}"
                / "{chrom}.log"
            ),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: raw_llr_resource("build_chromosome", "mem_mb"),
            runtime=lambda wildcards: raw_llr_resource("build_chromosome", "runtime"),
            disk_mb=lambda wildcards: raw_llr_resource("build_chromosome", "disk_mb"),
        params:
            output_dir=lambda wildcards: str(
                RAW_LLR_OUTPUT_ROOT
                / "chromosomes"
                / wildcards.raw_llr_score_set
                / wildcards.chrom
            ),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.raw_llr build-chromosome \
                --source-root {RAW_LLR_SOURCE_ROOT:q} \
                --inventory-manifest {input.manifest:q} \
                --parquet-selection {input.parquet_selection:q} \
                --track-selection {input.track_selection:q} \
                --score-set {wildcards.raw_llr_score_set:q} \
                --chrom {wildcards.chrom:q} \
                --output-dir {params.output_dir:q} \
                --report {output.report:q} \
                --batch-size {RAW_LLR_BATCH_SIZE} \
                --sample-count {RAW_LLR_SAMPLE_COUNT} \
                >{log:q} 2>&1
            """

    rule concatenate_raw_llr_bigwig:
        """Combine raw-LLR chromosome restart units into one final track."""
        input:
            bigwigs=raw_llr_chromosome_bigwig_inputs,
            chromosome_reports=raw_llr_chromosome_report_inputs,
            manifest=str(RAW_LLR_INVENTORY_MANIFEST),
            parquet_selection=str(RAW_LLR_PARQUET_SELECTION),
        output:
            bigwig=str(
                RAW_LLR_OUTPUT_ROOT
                / "final"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.bw"
            ),
            report=str(
                RAW_LLR_OUTPUT_ROOT
                / "final-reports"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.json"
            ),
        log:
            str(
                RAW_LLR_OUTPUT_ROOT
                / "logs"
                / "final"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.log"
            ),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: raw_llr_resource("concatenate", "mem_mb"),
            runtime=lambda wildcards: raw_llr_resource("concatenate", "runtime"),
            disk_mb=lambda wildcards: raw_llr_resource("concatenate", "disk_mb"),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.raw_llr concatenate \
                --inventory-manifest {input.manifest:q} \
                --parquet-selection {input.parquet_selection:q} \
                --score-set {wildcards.raw_llr_score_set:q} \
                --track {wildcards.raw_llr_track:q} \
                --value-decimals {RAW_LLR_VALUE_DECIMALS} \
                --output {output.bigwig:q} \
                --report {output.report:q} \
                --inputs {input.bigwigs:q} \
                --chromosome-reports {input.chromosome_reports:q} \
                >{log:q} 2>&1
            """

    rule audit_final_raw_llr_bigwig:
        """Audit one final raw-LLR artifact without rebuilding chromosomes."""
        input:
            bigwig=str(
                RAW_LLR_OUTPUT_ROOT
                / "final"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.bw"
            ),
            concatenation_report=str(
                RAW_LLR_OUTPUT_ROOT
                / "final-reports"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.json"
            ),
            chromosome_reports=raw_llr_chromosome_report_inputs,
            manifest=str(RAW_LLR_INVENTORY_MANIFEST),
            parquet_selection=str(RAW_LLR_PARQUET_SELECTION),
        output:
            report=str(
                RAW_LLR_OUTPUT_ROOT
                / "audit-reports"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.json"
            ),
        log:
            str(
                RAW_LLR_OUTPUT_ROOT
                / "logs"
                / "audit"
                / "{raw_llr_score_set}"
                / "{raw_llr_track}.log"
            ),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: raw_llr_resource("audit", "mem_mb"),
            runtime=lambda wildcards: raw_llr_resource("audit", "runtime"),
            disk_mb=lambda wildcards: raw_llr_resource("audit", "disk_mb"),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.raw_llr audit-final \
                --inventory-manifest {input.manifest:q} \
                --parquet-selection {input.parquet_selection:q} \
                --score-set {wildcards.raw_llr_score_set:q} \
                --track {wildcards.raw_llr_track:q} \
                --value-decimals {RAW_LLR_VALUE_DECIMALS} \
                --bigwig {input.bigwig:q} \
                --concatenation-report {input.concatenation_report:q} \
                --report {output.report:q} \
                --chromosome-reports {input.chromosome_reports:q} \
                >{log:q} 2>&1
            """

    rule aggregate_raw_llr_validation:
        """Aggregate validation evidence for exactly the 32 new tracks."""
        input:
            reports=RAW_LLR_FINAL_REPORTS,
            track_selection=str(RAW_LLR_TRACK_SELECTION),
        output:
            json=str(RAW_LLR_OUTPUT_ROOT / "validation.json"),
            markdown=str(RAW_LLR_OUTPUT_ROOT / "validation.md"),
        log:
            str(RAW_LLR_OUTPUT_ROOT / "logs" / "validation.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: raw_llr_resource("aggregate", "mem_mb"),
            runtime=lambda wildcards: raw_llr_resource("aggregate", "runtime"),
            disk_mb=lambda wildcards: raw_llr_resource("aggregate", "disk_mb"),
        run:
            try:
                aggregate_raw_llr_validation(
                    input.reports,
                    input.track_selection,
                    output.json,
                    output.markdown,
                )
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text("validated all 32 raw-LLR BigWigs\n")
            except BaseException as error:
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text(f"raw-LLR validation failed: {error}\n")
                raise
