"""Inventory and validate immutable staged score shards."""

from pathlib import Path

from snakemake.exceptions import WorkflowError

from gpn_star_scores.catalog import (
    ASSEMBLIES,
    expected_shards,
    get_shard_spec,
    score_set_assembly,
)
from gpn_star_scores.inventory import (
    inspect_shard_to_json,
    prepare_reference,
    write_release_outputs,
)

INVENTORY_CONFIG = config.get("inventory", {})
INVENTORY_ENABLED = bool(INVENTORY_CONFIG.get("enabled", False))
INVENTORY_OUTPUT_ROOT = Path(INVENTORY_CONFIG.get("output_root", "results/inventory"))


def write_inventory_log(path, message):
    """Write a short rule-local status log."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n", encoding="utf-8")


def inventory_targets():
    """Return inventory targets only when explicitly enabled."""

    if not INVENTORY_ENABLED:
        return []
    return [str(INVENTORY_OUTPUT_ROOT / "release")]


if INVENTORY_ENABLED:
    if not INVENTORY_CONFIG.get("source_root"):
        raise WorkflowError(
            "inventory.source_root is required when inventory is enabled"
        )
    SOURCE_ROOT = Path(INVENTORY_CONFIG["source_root"])
    REFERENCE_CONFIGS = INVENTORY_CONFIG.get("reference_fastas", {})
    missing_reference_config = sorted(set(ASSEMBLIES) - set(REFERENCE_CONFIGS))
    if missing_reference_config:
        raise WorkflowError(
            "inventory.reference_fastas is missing assemblies: "
            + ", ".join(missing_reference_config)
        )
    for assembly, reference_config in REFERENCE_CONFIGS.items():
        if not isinstance(reference_config, dict) or not reference_config.get("path"):
            raise WorkflowError(
                f"inventory.reference_fastas.{assembly}.path is required"
            )
        reference_sha256 = reference_config.get("sha256")
        if (
            not isinstance(reference_sha256, str)
            or len(reference_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in reference_sha256.lower()
            )
        ):
            raise WorkflowError(
                f"inventory.reference_fastas.{assembly}.sha256 must be an "
                "author-approved SHA-256"
            )

    SHARDS = expected_shards()
    SHARD_REPORTS = [
        str(
            INVENTORY_OUTPUT_ROOT
            / "shards"
            / shard.score_set
            / shard.score_type
            / f"{shard.chrom}.json"
        )
        for shard in SHARDS
    ]
    REFERENCE_DIRS = [
        str(INVENTORY_OUTPUT_ROOT / "references" / assembly) for assembly in ASSEMBLIES
    ]

    def inventory_resource(rule, name):
        value = INVENTORY_CONFIG.get("resources", {}).get(rule, {}).get(name)
        if not isinstance(value, int) or value <= 0:
            raise WorkflowError(
                f"inventory.resources.{rule}.{name} must be a positive integer "
                "based on a representative pilot"
            )
        return value

    rule prepare_inventory_reference:
        input:
            lambda wildcards: REFERENCE_CONFIGS[wildcards.assembly]["path"],
        output:
            directory(str(INVENTORY_OUTPUT_ROOT / "references" / "{assembly}")),
        log:
            str(INVENTORY_OUTPUT_ROOT / "logs" / "reference" / "{assembly}.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: inventory_resource("prepare_reference", "mem_mb"),
            runtime=lambda wildcards: inventory_resource("prepare_reference", "runtime"),
            tmp_disk_mb=lambda wildcards: inventory_resource(
                "prepare_reference", "tmp_disk_mb"
            ),
        run:
            try:
                prepare_reference(
                    Path(input[0]),
                    Path(output[0]),
                    ASSEMBLIES[wildcards.assembly],
                    REFERENCE_CONFIGS[wildcards.assembly]["sha256"],
                )
                write_inventory_log(
                    log[0], f"prepared {wildcards.assembly} reference"
                )
            except BaseException as error:
                write_inventory_log(
                    log[0],
                    f"failed to prepare {wildcards.assembly} reference: {error}",
                )
                raise

    rule validate_score_shard:
        input:
            stage=str(SOURCE_ROOT),
            reference=lambda wildcards: str(
                INVENTORY_OUTPUT_ROOT
                / "references"
                / score_set_assembly(wildcards.score_set)
            ),
        output:
            str(
                INVENTORY_OUTPUT_ROOT
                / "shards"
                / "{score_set}"
                / "{score_type}"
                / "{chrom}.json"
            ),
        log:
            str(
                INVENTORY_OUTPUT_ROOT
                / "logs"
                / "shards"
                / "{score_set}"
                / "{score_type}"
                / "{chrom}.log"
            ),
        threads: 1
        resources:
            mem_mb=lambda wildcards: inventory_resource("validate_shard", "mem_mb"),
            runtime=lambda wildcards: inventory_resource("validate_shard", "runtime"),
            tmp_disk_mb=lambda wildcards: inventory_resource(
                "validate_shard", "tmp_disk_mb"
            ),
        params:
            source=lambda wildcards: str(
                SOURCE_ROOT
                / get_shard_spec(
                    wildcards.score_set, wildcards.score_type, wildcards.chrom
                ).relative_path
            ),
        run:
            shard = get_shard_spec(
                wildcards.score_set, wildcards.score_type, wildcards.chrom
            )
            try:
                inspect_shard_to_json(
                    Path(params.source),
                    shard,
                    Path(input.reference) / f"{wildcards.chrom}.seq",
                    Path(output[0]),
                    batch_size=int(INVENTORY_CONFIG.get("batch_size", 1_048_576)),
                )
                write_inventory_log(log[0], f"validated {shard.relative_path}")
            except BaseException as error:
                write_inventory_log(
                    log[0], f"failed to validate {shard.relative_path}: {error}"
                )
                raise

    rule inventory_manifest:
        input:
            shards=SHARD_REPORTS,
            references=REFERENCE_DIRS,
        output:
            directory(str(INVENTORY_OUTPUT_ROOT / "release")),
        log:
            str(INVENTORY_OUTPUT_ROOT / "logs" / "manifest.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: inventory_resource("manifest", "mem_mb"),
            runtime=lambda wildcards: inventory_resource("manifest", "runtime"),
            tmp_disk_mb=lambda wildcards: inventory_resource("manifest", "tmp_disk_mb"),
        run:
            reference_reports = [
                Path(path) / "provenance.json" for path in input.references
            ]
            try:
                write_release_outputs(
                    SOURCE_ROOT,
                    [Path(path) for path in input.shards],
                    reference_reports,
                    Path(output[0]),
                    expected_shard_bytes=INVENTORY_CONFIG.get(
                        "expected_shard_bytes"
                    ),
                    hugging_face_capacity=INVENTORY_CONFIG.get(
                        "hugging_face_capacity"
                    ),
                )
                write_inventory_log(log[0], "aggregated inventory manifest")
            except BaseException as error:
                write_inventory_log(
                    log[0], f"failed to aggregate inventory manifest: {error}"
                )
                raise
