"""Issue #6 multi-assembly UCSC track-hub workflow."""

from pathlib import Path

from snakemake.exceptions import WorkflowError

from gpn_star_scores.hub import build_track_hub

HUB_CONFIG = config.get("hub", {})
HUB_ENABLED = bool(HUB_CONFIG.get("enabled", False))
HUB_OUTPUT_ROOT = Path(HUB_CONFIG.get("output_root", "results/hub"))
HUB_METADATA_ROOT = HUB_OUTPUT_ROOT / "metadata"
HUB_VALIDATION_JSON = HUB_OUTPUT_ROOT / "validation.json"
HUB_VALIDATION_MARKDOWN = HUB_OUTPUT_ROOT / "validation.md"
HUB_PUBLICATION_REPORT = HUB_OUTPUT_ROOT / "publication.json"
HUB_PUBLICATION_SUCCESS = HUB_OUTPUT_ROOT / "publication.complete"


def hub_targets():
    """Return the validated hub reports only when issue #6 is enabled."""

    if not HUB_ENABLED:
        return []
    return [str(HUB_VALIDATION_JSON), str(HUB_VALIDATION_MARKDOWN)]


def hub_publication_report():
    """Return the durable report from the explicit public hub update."""

    return str(HUB_PUBLICATION_REPORT)


def hub_publication_success_marker():
    """Return the success-only target for the explicit public hub update."""

    return str(HUB_PUBLICATION_SUCCESS)


def hub_approval_value(name):
    """Render one approval field for the shell-based publication command."""

    approval = HUB_CONFIG.get("publication_approval")
    if not isinstance(approval, dict):
        return ""
    value = approval.get(name)
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


if HUB_ENABLED:
    for required_key in (
        "release_manifest",
        "artifact_revision",
        "expected_base_revision",
        "contact_email",
        "udc_cache_root",
    ):
        if not HUB_CONFIG.get(required_key):
            raise WorkflowError(
                f"hub.{required_key} is required when the hub is enabled"
            )
    HUB_RELEASE_MANIFEST = Path(HUB_CONFIG["release_manifest"])
    HUB_ARTIFACT_REVISION = str(HUB_CONFIG["artifact_revision"])
    HUB_EXPECTED_BASE_REVISION = str(HUB_CONFIG["expected_base_revision"])
    HUB_CONTACT_EMAIL = str(HUB_CONFIG["contact_email"])
    HUB_UDC_CACHE_ROOT = Path(HUB_CONFIG["udc_cache_root"])
    HUB_PUBLICATION_APPROVAL = HUB_CONFIG.get("publication_approval")

    def hub_resource(stage, name):
        value = HUB_CONFIG.get("resources", {}).get(stage, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(
                f"hub.resources.{stage}.{name} must be a positive integer"
            )
        return value

    rule build_track_hub_metadata:
        """Render the hub, descriptions, manifest, and linked dataset card."""
        input:
            manifest=str(HUB_RELEASE_MANIFEST),
        output:
            directory(str(HUB_METADATA_ROOT)),
        log:
            str(HUB_OUTPUT_ROOT / "logs" / "build.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: hub_resource("build", "mem_mb"),
            runtime=lambda wildcards: hub_resource("build", "runtime"),
            disk_mb=lambda wildcards: hub_resource("build", "disk_mb"),
        run:
            try:
                build_track_hub(
                    input.manifest,
                    output[0],
                    artifact_revision=HUB_ARTIFACT_REVISION,
                    contact_email=HUB_CONTACT_EMAIL,
                )
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text("built UCSC track hub\n")
            except BaseException as error:
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text(f"UCSC track-hub build failed: {error}\n")
                raise

    rule validate_track_hub:
        """Validate local hub controls and all pinned public BigWigs."""
        input:
            metadata=directory(str(HUB_METADATA_ROOT)),
        output:
            json=str(HUB_VALIDATION_JSON),
            markdown=str(HUB_VALIDATION_MARKDOWN),
        log:
            str(HUB_OUTPUT_ROOT / "logs" / "validation.log"),
        conda:
            "../envs/ucsc.yaml"
        threads: 1
        resources:
            mem_mb=lambda wildcards: hub_resource("validate", "mem_mb"),
            runtime=lambda wildcards: hub_resource("validate", "runtime"),
            disk_mb=lambda wildcards: hub_resource("validate", "disk_mb"),
        shell:
            """
            {PYTHON_EXECUTABLE:q} -m gpn_star_scores.hub validate \
                --metadata-root {input.metadata:q} \
                --report {output.json:q} \
                --markdown {output.markdown:q} \
                --udc-dir {HUB_UDC_CACHE_ROOT:q} \
                >{log:q} 2>&1
            """
