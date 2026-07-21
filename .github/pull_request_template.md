## Summary

<!-- What changed, why, and who or what is affected? -->

Closes #

## Scope

### Included

-

### Non-goals

-

## Acceptance criteria

<!-- Map every applicable issue criterion to concrete evidence. -->

| Acceptance criterion | Evidence |
| --- | --- |
|  |  |

## Scientific and data semantics

<!-- Record assumptions and any changes to schemas, dtypes, score meanings,
chromosome names, coordinate conventions, or assembly mappings. State what
requires author validation. Write "Not applicable" when appropriate. -->

-

## Validation

<!-- Include exact commands and outcomes. Do not mark checks that were not run. -->

- [ ] `uv lock --check` passes after dependency changes, if applicable.
- [ ] `uv run --locked pre-commit run --all-files` passes.
- [ ] The fast pytest suite passes with no network or production data.
- [ ] Snakemake dry-run passes, if applicable.
- [ ] Representative SCF pilot passes, if applicable.
- [ ] Interrupted or incomplete outputs rerun safely, if applicable.

```text
# Commands and results
```

## Resources and reproducibility

<!-- Record relevant package/environment versions, benchmark inputs, Slurm job
IDs, wall time, peak RSS, and efficiency evidence. Write "Not applicable" for
documentation-only changes. -->

-

## Outputs and external effects

- [ ] Staged source data was not modified.
- [ ] No credentials, signed URLs, full-scale data, or transient logs are committed.
- [ ] Generated outputs were written, validated, and promoted atomically, if applicable.
- [ ] Any remote upload or visibility change is described below and explicitly authorized.

External systems changed and current publication status:

-

## Risks and recovery

<!-- Failure modes, compatibility concerns, rollback, and known limitations. -->

-

## Reviewer focus

<!-- Point the independent reviewer to the highest-risk assumptions and code. -->

-
