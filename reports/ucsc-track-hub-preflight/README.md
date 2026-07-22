# UCSC track-hub production preflight

This directory records the read-only issue #6 preflight run on 2026-07-22.
The generated hub used the public issue #4 release manifest and pinned every
BigWig URL to immutable Hugging Face revision
`5c799b2ec6aa089f0caa8294ae72adb4510f81ae`.

The run passed all automated checks:

- UCSC `hubCheck -checkSettings` against the current official HTTPS trackDb
  specification;
- anonymous HTTP range reads for all 40 BigWigs;
- exact expected chromosome names in all 40 BigWig headers, with consistent
  chromosome lengths across the five tracks in each score set; and
- direct `bigWigSummary` queries for entropy and A/C/G/T at one covered base
  and a 10-bin surrounding window for each of the eight score sets.

The cold-cache validation completed successfully in 7 minutes 6.27 seconds
with 118,892 KiB peak RSS. Metadata generation completed in 2.07 seconds with
117,060 KiB peak RSS. The generated metadata tree was 79,228 bytes, and the
validation UDC cache was 3,271,437 bytes. Workflow requests retain substantial
margin over these measurements.

[`summary.json`](summary.json) contains the machine-readable counts,
representative loci and observed one-base values. The source release-manifest
SHA-256 was
`94c3298ead5c4d9044548a63d25791f03c45bf6b7254a00d74a7ef6867ce2c94`.

The author approved publication, and one metadata-only commit created public
revision `6671186db8e07c2e87d8f2eb8496c7be5d5b1c7e`. Anonymous validation of
that immutable revision passed for all 35 metadata files and all 40 BigWigs.

Live UCSC image rendering passed at base and zoomed-out scales for seven of
eight model groups. UCSC returns HTTP 500 for `db=araTha1` even without this
hub. The author therefore approved a second commit using the older hub's
`hub_2660163_GCF_000001735.4` identifier; public revision
`8ea5b82c19a61691629f9084b805758a6a0ba1c9` passed complete anonymous
automated validation.

Fresh-session rendering then showed that the `hub_2660163_` prefix is a
session-generated alias: it falls back to hg38 when this hub is connected
alone. The stable GenArk database `GCF_000001735.4` renders the intended
TAIR10.1 assembly directly. The author approved the final correction, which
published revision `2cb55ca6ceb4bddbe4314d2edd0fe370b200fde8`.

Anonymous validation of the final revision passed byte-for-byte for all 35
metadata files, all 40 HTTP range and chromosome-header checks, and all eight
representative base and zoom queries. It sent no credentials. Manual UCSC
rendering passed at base and zoomed-out scales for all eight model groups,
including TAIR10.1 through the stable `GCF_000001735.4` database.
