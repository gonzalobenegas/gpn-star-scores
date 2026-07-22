"""Issue #2 end-to-end release QA and release-record workflow."""

from pathlib import Path

from snakemake.exceptions import WorkflowError

from gpn_star_scores.qa import (
    build_release_record,
    validate_public_release_for_qa,
)

QA_CONFIG = config.get("qa", {})
QA_ENABLED = bool(QA_CONFIG.get("enabled", False))
QA_OUTPUT_ROOT = Path(QA_CONFIG.get("output_root", "results/qa"))
QA_PUBLIC_RELEASE_REPORT = QA_OUTPUT_ROOT / "public-release.json"
QA_REUSED_PUBLIC_HUB_REPORT = QA_CONFIG.get("public_hub_report")
QA_PUBLIC_HUB_REPORT = Path(
    QA_REUSED_PUBLIC_HUB_REPORT or QA_OUTPUT_ROOT / "public-hub.json"
)
QA_RELEASE_RECORD_JSON = QA_OUTPUT_ROOT / "release-record.json"
QA_RELEASE_RECORD_MARKDOWN = QA_OUTPUT_ROOT / "release-record.md"


def qa_targets():
    """Return final issue #2 evidence only when QA is explicitly enabled."""

    if not QA_ENABLED:
        return []
    return [str(QA_RELEASE_RECORD_JSON), str(QA_RELEASE_RECORD_MARKDOWN)]


if QA_ENABLED:
    for required_key in (
        "release_metadata_root",
        "hub_metadata_root",
        "release_revision",
        "hub_revision",
        "workflow_commit",
        "udc_cache_root",
        "scf_evidence",
        "bigwig_evidence",
        "hub_evidence",
    ):
        if not QA_CONFIG.get(required_key):
            raise WorkflowError(f"qa.{required_key} is required when QA is enabled")
    QA_RELEASE_METADATA_ROOT = Path(QA_CONFIG["release_metadata_root"])
    QA_HUB_METADATA_ROOT = Path(QA_CONFIG["hub_metadata_root"])
    QA_RELEASE_REVISION = str(QA_CONFIG["release_revision"])
    QA_HUB_REVISION = str(QA_CONFIG["hub_revision"])
    QA_WORKFLOW_COMMIT = str(QA_CONFIG["workflow_commit"])
    QA_UDC_CACHE_ROOT = Path(QA_CONFIG["udc_cache_root"])
    QA_SCF_EVIDENCE = Path(QA_CONFIG["scf_evidence"])
    QA_BIGWIG_EVIDENCE = Path(QA_CONFIG["bigwig_evidence"])
    QA_HUB_EVIDENCE = Path(QA_CONFIG["hub_evidence"])
    QA_VIEWER_ATTEMPTS = int(QA_CONFIG.get("viewer_attempts", 1))
    QA_VIEWER_RETRY_SECONDS = float(QA_CONFIG.get("viewer_retry_seconds", 0))
    QA_HF_BLOCK_SIZE = int(QA_CONFIG.get("hf_block_size", 4_194_304))
    QA_WAIVERS = QA_CONFIG.get("waivers", [])
    QA_KNOWN_LIMITATIONS = QA_CONFIG.get("known_limitations", [])
    QA_TAG_APPROVAL = QA_CONFIG.get("tag_approval")
    if (
        QA_VIEWER_ATTEMPTS <= 0
        or QA_VIEWER_RETRY_SECONDS < 0
        or QA_HF_BLOCK_SIZE <= 0
        or not isinstance(QA_WAIVERS, list)
        or not isinstance(QA_KNOWN_LIMITATIONS, list)
    ):
        raise WorkflowError(
            "qa public-validation or release-record settings are invalid"
        )

    def qa_resource(stage, name):
        value = QA_CONFIG.get("resources", {}).get(stage, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(
                f"qa.resources.{stage}.{name} must be a positive integer"
            )
        return value

    rule qa_public_release:
        """Repeat anonymous artifact checks and all dataset-card examples."""
        input:
            metadata=directory(str(QA_RELEASE_METADATA_ROOT)),
        output:
            str(QA_PUBLIC_RELEASE_REPORT),
        log:
            str(QA_OUTPUT_ROOT / "logs" / "public-release.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: qa_resource("release", "mem_mb"),
            runtime=lambda wildcards: qa_resource("release", "runtime"),
            disk_mb=lambda wildcards: qa_resource("release", "disk_mb"),
        run:
            try:
                validate_public_release_for_qa(
                    input.metadata,
                    output[0],
                    revision=QA_RELEASE_REVISION,
                    viewer_attempts=QA_VIEWER_ATTEMPTS,
                    viewer_retry_seconds=QA_VIEWER_RETRY_SECONDS,
                    hf_block_size=QA_HF_BLOCK_SIZE,
                )
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text("public release QA passed\n")
            except BaseException as error:
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text(f"public release QA failed: {error}\n")
                raise

    if not QA_REUSED_PUBLIC_HUB_REPORT:
        rule qa_public_hub:
            """Repeat anonymous hub, range, header, base, and zoom checks."""
            input:
                metadata=directory(str(QA_HUB_METADATA_ROOT)),
            output:
                str(QA_PUBLIC_HUB_REPORT),
            log:
                str(QA_OUTPUT_ROOT / "logs" / "public-hub.log"),
            conda:
                "../envs/ucsc.yaml"
            threads: 1
            resources:
                mem_mb=lambda wildcards: qa_resource("hub", "mem_mb"),
                runtime=lambda wildcards: qa_resource("hub", "runtime"),
                disk_mb=lambda wildcards: qa_resource("hub", "disk_mb"),
            shell:
                """
                {PYTHON_EXECUTABLE:q} -m gpn_star_scores.qa validate-hub \
                    --metadata-root {input.metadata:q} \
                    --report {output:q} \
                    --revision {QA_HUB_REVISION:q} \
                    --udc-dir {QA_UDC_CACHE_ROOT:q} \
                    >{log:q} 2>&1
                """

    rule qa_release_record:
        """Validate the complete evidence chain and write the v1 release record."""
        input:
            metadata=directory(str(QA_RELEASE_METADATA_ROOT)),
            hub_metadata=directory(str(QA_HUB_METADATA_ROOT)),
            public_release=str(QA_PUBLIC_RELEASE_REPORT),
            public_hub=str(QA_PUBLIC_HUB_REPORT),
            scf=str(QA_SCF_EVIDENCE),
            bigwig=str(QA_BIGWIG_EVIDENCE),
            hub=str(QA_HUB_EVIDENCE),
            lock="uv.lock",
            profile="workflow/profiles/scf/config.yaml",
        output:
            json=str(QA_RELEASE_RECORD_JSON),
            markdown=str(QA_RELEASE_RECORD_MARKDOWN),
        log:
            str(QA_OUTPUT_ROOT / "logs" / "release-record.log"),
        threads: 1
        resources:
            mem_mb=lambda wildcards: qa_resource("record", "mem_mb"),
            runtime=lambda wildcards: qa_resource("record", "runtime"),
            disk_mb=lambda wildcards: qa_resource("record", "disk_mb"),
        run:
            try:
                build_release_record(
                    input.metadata,
                    input.hub_metadata,
                    input.public_release,
                    input.public_hub,
                    input.scf,
                    input.bigwig,
                    input.hub,
                    input.lock,
                    input.profile,
                    output.json,
                    output.markdown,
                    release_revision=QA_RELEASE_REVISION,
                    hub_revision=QA_HUB_REVISION,
                    workflow_commit=QA_WORKFLOW_COMMIT,
                    waivers=QA_WAIVERS,
                    known_limitations=QA_KNOWN_LIMITATIONS,
                    tag_approval=QA_TAG_APPROVAL,
                )
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text("release record written\n")
            except BaseException as error:
                Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(log[0]).write_text(f"release record failed: {error}\n")
                raise
