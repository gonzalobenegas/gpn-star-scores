"""Shared workflow rules.

Implementation issues add thin DAG definitions here or in focused rule files.
Non-trivial Python logic belongs in ``src/gpn_star_scores/``.
"""


rule scf_smoke_chromosome:
    """Verify the locked Python environment on one epurdom job."""
    output:
        SCF_SMOKE_REPORT,
    log:
        "logs/scf-smoke/{chrom}.log",
    threads: 4
    resources:
        mem_mb=4096,
        runtime=30,
        disk_mb=1024,
        tmpdir=lambda wildcards: str(
            SCRATCH_ROOT / "tmp" / "scf-smoke" / wildcards.chrom
        ),
    shell:
        """
        {PYTHON_EXECUTABLE:q} -m gpn_star_scores.scf_smoke \
            --chrom {wildcards.chrom:q} \
            --output {output:q} \
            >{log:q} 2>&1
        """
