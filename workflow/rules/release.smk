"""Issue #4 public Hugging Face release workflow."""

from pathlib import Path

from snakemake.exceptions import WorkflowError

from gpn_star_scores.catalog import SCORE_SETS, SCORE_TYPES, expected_shards
from gpn_star_scores.release import build_release_metadata, publish_release
from gpn_star_scores.tracks import TRACKS

RELEASE_CONFIG = config.get("release", {})
RELEASE_ENABLED = bool(RELEASE_CONFIG.get("enabled", False))
RELEASE_OUTPUT_ROOT = Path(RELEASE_CONFIG.get("output_root", "results/release"))
RELEASE_METADATA_ROOT = RELEASE_OUTPUT_ROOT / "metadata"
RELEASE_PUBLICATION_REPORT = RELEASE_OUTPUT_ROOT / "publication.json"


def release_preflight_targets():
    """Return local publication metadata only when issue #4 is enabled."""

    return [str(RELEASE_METADATA_ROOT)] if RELEASE_ENABLED else []


def release_publication_report():
    """Return the report written by the explicit remote publication target."""

    return str(RELEASE_PUBLICATION_REPORT)


if RELEASE_ENABLED:
    for required_key in (
        "source_root",
        "bigwig_root",
        "inventory_manifest",
        "parquet_selection",
        "bigwig_validation",
    ):
        if not RELEASE_CONFIG.get(required_key):
            raise WorkflowError(
                f"release.{required_key} is required when publication is enabled"
            )
    RELEASE_SOURCE_ROOT = Path(RELEASE_CONFIG["source_root"])
    RELEASE_BIGWIG_ROOT = Path(RELEASE_CONFIG["bigwig_root"])
    RELEASE_INVENTORY_MANIFEST = Path(RELEASE_CONFIG["inventory_manifest"])
    RELEASE_PARQUET_SELECTION = Path(RELEASE_CONFIG["parquet_selection"])
    RELEASE_BIGWIG_VALIDATION = Path(RELEASE_CONFIG["bigwig_validation"])
    if (
        RELEASE_SOURCE_ROOT.resolve() == RELEASE_OUTPUT_ROOT.resolve()
        or RELEASE_SOURCE_ROOT.resolve() in RELEASE_OUTPUT_ROOT.resolve().parents
        or RELEASE_OUTPUT_ROOT.resolve() in RELEASE_SOURCE_ROOT.resolve().parents
    ):
        raise WorkflowError(
            "release.output_root must not overlap the immutable release.source_root"
        )
    RELEASE_VIEWER_ATTEMPTS = int(RELEASE_CONFIG.get("viewer_attempts", 1))
    RELEASE_VIEWER_RETRY_SECONDS = float(RELEASE_CONFIG.get("viewer_retry_seconds", 0))
    RELEASE_VIEWER_REQUIRED = bool(RELEASE_CONFIG.get("viewer_required", False))
    RELEASE_HF_BLOCK_SIZE = int(RELEASE_CONFIG.get("hf_block_size", 4_194_304))
    RELEASE_CAPACITY_APPROVAL = RELEASE_CONFIG.get("capacity_approval")
    if (
        RELEASE_VIEWER_ATTEMPTS <= 0
        or RELEASE_VIEWER_RETRY_SECONDS < 0
        or RELEASE_HF_BLOCK_SIZE <= 0
    ):
        raise WorkflowError("release public-validation settings are invalid")

    def release_resource(stage, name):
        value = RELEASE_CONFIG.get("resources", {}).get(stage, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(
                f"release.resources.{stage}.{name} must be a positive integer"
            )
        return value

    RELEASE_SOURCE_FILES = [
        str(RELEASE_SOURCE_ROOT / shard.relative_path) for shard in expected_shards()
    ]
    RELEASE_BIGWIG_FILES = [
        str(RELEASE_BIGWIG_ROOT / "final" / score_set.name / f"{track}.bw")
        for score_set in SCORE_SETS
        for track in TRACKS
    ]

    rule release_preflight:
        """Require all local release gates and build checksummed metadata."""
        input:
            parquets=RELEASE_SOURCE_FILES,
            bigwigs=RELEASE_BIGWIG_FILES,
            inventory=str(RELEASE_INVENTORY_MANIFEST),
            parquet_selection=str(RELEASE_PARQUET_SELECTION),
            bigwig_validation=str(RELEASE_BIGWIG_VALIDATION),
        output:
            directory(str(RELEASE_METADATA_ROOT)),
        log:
            str(RELEASE_OUTPUT_ROOT / "logs" / "preflight.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: release_resource("preflight", "mem_mb"),
            runtime=lambda wildcards: release_resource("preflight", "runtime"),
            disk_mb=lambda wildcards: release_resource("preflight", "disk_mb"),
        run:
            try:
                build_release_metadata(
                    RELEASE_SOURCE_ROOT,
                    RELEASE_BIGWIG_ROOT,
                    input.inventory,
                    input.parquet_selection,
                    input.bigwig_validation,
                    output[0],
                    capacity_approval=RELEASE_CAPACITY_APPROVAL,
                )
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text("release preflight passed\n")
            except BaseException as error:
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text(f"release preflight failed: {error}\n")
                raise


def run_release_publication(output_path, log_path):
    """Execute the explicit remote side effect or reject disabled publication."""

    if not RELEASE_ENABLED:
        raise WorkflowError(
            "release.enabled must be true with production paths before publishing"
        )
    try:
        publish_release(
            RELEASE_SOURCE_ROOT,
            RELEASE_BIGWIG_ROOT,
            RELEASE_METADATA_ROOT,
            output_path,
            viewer_attempts=RELEASE_VIEWER_ATTEMPTS,
            viewer_retry_seconds=RELEASE_VIEWER_RETRY_SECONDS,
            viewer_required=RELEASE_VIEWER_REQUIRED,
            hf_block_size=RELEASE_HF_BLOCK_SIZE,
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("public Hugging Face release validated\n")
    except BaseException as error:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"public Hugging Face release failed: {error}\n")
        raise
