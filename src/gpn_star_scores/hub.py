"""Build, validate, and intentionally publish the multi-assembly UCSC hub."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from huggingface_hub import CommitOperationAdd, HfApi

from gpn_star_scores.catalog import ASSEMBLIES, SCORE_SETS, ScoreSetSpec
from gpn_star_scores.inventory import atomic_write_json, sha256_file
from gpn_star_scores.raw_llr import RAW_LLR_BASES, RAW_LLR_TRACKS
from gpn_star_scores.release import (
    HUGGING_FACE_URL,
    PAPER_DOI,
    PAPER_TITLE,
    REPOSITORY_ID,
    TRACK_HUB_URL,
    render_dataset_card,
)
from gpn_star_scores.tracks import TRACKS, ucsc_assembly_name

HUB_APPROVAL_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/6"
HUB_QA_APPROVAL_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/2"
HUB_RELEASE_ASSEMBLY_ORDER = ("hg38", "ce11", "dm6", "gg6", "tair10", "mm39")
HUB_DATABASE_NAMES = {
    assembly: ucsc_assembly_name(assembly) for assembly in HUB_RELEASE_ASSEMBLY_ORDER
}
HUB_DATABASE_NAMES["tair10"] = "GCF_000001735.4"
HUB_TRACK_DB_DIRECTORIES = {
    assembly: ucsc_assembly_name(assembly) for assembly in HUB_RELEASE_ASSEMBLY_ORDER
}
HUB_ASSEMBLY_ORDER = tuple(
    HUB_DATABASE_NAMES[assembly] for assembly in HUB_RELEASE_ASSEMBLY_ORDER
)
HUB_TRACK_DB_DIRECTORY_ORDER = tuple(
    HUB_TRACK_DB_DIRECTORIES[assembly] for assembly in HUB_RELEASE_ASSEMBLY_ORDER
)
HUB_SETTINGS_SPEC = "https://genome.ucsc.edu/goldenPath/help/trackDb/trackDbHub.html"
NUCLEOTIDE_COLORS = {
    "A": "0,128,0",
    "C": "0,0,255",
    "G": "255,166,0",
    "T": "255,0,0",
}
RAW_LLR_POSITIVE_COLOR = "0,0,255"
RAW_LLR_NEGATIVE_COLOR = "255,0,0"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_revision(revision: str, *, field: str = "revision") -> str:
    if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
        raise ValueError(f"{field} must be a lowercase 40-character commit SHA")
    return revision


def _track_symbol(score_set: str, suffix: str = "") -> str:
    words = [word for word in re.split(r"[^A-Za-z0-9]+", score_set) if word]
    if words and words[0].lower() == "gpn":
        words = words[1:]
    symbol = "gpn" + "".join(word[:1].upper() + word[1:] for word in words)
    symbol += suffix
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol):
        raise ValueError(f"cannot create a UCSC track symbol from {score_set!r}")
    return symbol


def _score_set_label(score_set: ScoreSetSpec) -> str:
    if score_set.assembly == "hg38":
        model = score_set.name.removeprefix("gpn-star-hg38-").removesuffix("-200m")
        return f"GPN-Star ({model[0].upper()})"
    return "GPN-Star"


def _artifact_url(path: str, revision: str) -> str:
    return f"{HUGGING_FACE_URL}/datasets/{REPOSITORY_ID}/resolve/{revision}/{path}"


def hub_database_name(assembly: str) -> str:
    """Map one release assembly to the working UCSC browser database."""

    try:
        return HUB_DATABASE_NAMES[assembly]
    except KeyError as error:
        raise KeyError(f"unknown release assembly: {assembly}") from error


def browser_launch_links(*, include_raw_llr: bool = False) -> list[dict[str, str]]:
    """Return one focused UCSC launch link for each model group."""

    links = []
    for score_set in SCORE_SETS:
        group = _track_symbol(score_set.name)
        settings = {
            "db": hub_database_name(score_set.assembly),
            "hubUrl": TRACK_HUB_URL,
            "hideTracks": "1",
            "ignoreCookie": "1",
            group: "show",
            f"{group}Entropy": "dense",
            f"{group}Logo": "full",
        }
        if include_raw_llr:
            settings[f"{group}RawLlr"] = "full"
        query = urlencode(settings)
        links.append(
            {
                "score_set": score_set.name,
                "ucsc_assembly": hub_database_name(score_set.assembly),
                "url": f"https://genome.ucsc.edu/cgi-bin/hgTracks?{query}",
            }
        )
    return links


def dataset_card_launch_links() -> list[dict[str, str]]:
    """Return dynamic-default README links that show one complete model."""

    links = []
    for score_set in SCORE_SETS:
        group = _track_symbol(score_set.name)
        settings = {
            "db": hub_database_name(score_set.assembly),
            "hubUrl": TRACK_HUB_URL,
            "ignoreCookie": "1",
            group: "show",
            f"{group}Entropy": "full",
            f"{group}Logo": "full",
            f"{group}RawLlr": "full",
        }
        for sibling in SCORE_SETS:
            if (
                sibling.assembly == score_set.assembly
                and sibling.name != score_set.name
            ):
                settings[_track_symbol(sibling.name)] = "hide"
        query = urlencode(settings)
        links.append(
            {
                "score_set": score_set.name,
                "ucsc_assembly": hub_database_name(score_set.assembly),
                "url": f"https://genome.ucsc.edu/cgi-bin/hgTracks?{query}",
            }
        )
    return links


def raw_llr_validation_links() -> list[dict[str, str]]:
    """Return raw-only UCSC links for focused issue-15 rendering checks."""

    links = []
    for score_set in SCORE_SETS:
        group = _track_symbol(score_set.name)
        settings = {
            "db": hub_database_name(score_set.assembly),
            "hubUrl": TRACK_HUB_URL,
            "hideTracks": "1",
            "ignoreCookie": "1",
            group: "show",
            f"{group}Entropy": "hide",
            f"{group}Logo": "hide",
            f"{group}RawLlr": "full",
        }
        links.append(
            {
                "score_set": score_set.name,
                "ucsc_assembly": hub_database_name(score_set.assembly),
                "url": (
                    f"https://genome.ucsc.edu/cgi-bin/hgTracks?{urlencode(settings)}"
                ),
            }
        )
    return links


def _validated_bigwig_records(
    release_manifest: Mapping[str, Any], artifact_revision: str
) -> dict[tuple[str, str], dict[str, Any]]:
    _validate_revision(artifact_revision, field="artifact_revision")
    if release_manifest.get("release_manifest_version") != 1:
        raise ValueError("release manifest version must be 1")
    repository = release_manifest.get("repository")
    if not isinstance(repository, Mapping) or repository != {
        "id": REPOSITORY_ID,
        "repo_type": "dataset",
        "public": True,
        "license": "apache-2.0",
    }:
        raise ValueError("release manifest repository identity differs")
    validation = release_manifest.get("validation")
    if not isinstance(validation, Mapping) or (
        validation.get("bigwig_validation_passed") is not True
        or validation.get("expected_bigwig_files") != 40
    ):
        raise ValueError("release manifest lacks complete BigWig validation")
    bigwig = release_manifest.get("bigwig")
    records = bigwig.get("files") if isinstance(bigwig, Mapping) else None
    if not isinstance(records, list) or bigwig.get("file_count") != 40:
        raise ValueError("release manifest must contain exactly 40 BigWigs")

    expected = {
        (score_set.name, track): {
            "assembly": score_set.assembly,
            "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
            "path": f"bigwig/{score_set.name}/{track}.bw",
        }
        for score_set in SCORE_SETS
        for track in TRACKS
    }
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("BigWig release records must be JSON objects")
        record = dict(raw_record)
        key = (str(record.get("score_set")), str(record.get("track")))
        contract = expected.get(key)
        if (
            contract is None
            or key in observed
            or any(record.get(field) != value for field, value in contract.items())
            or not isinstance(record.get("size"), int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            or not isinstance(record.get("bases_covered"), int)
            or record["bases_covered"] <= 0
            or not isinstance(record.get("zoom_levels"), int)
            or record["zoom_levels"] < 1
        ):
            raise ValueError(f"invalid BigWig release record: {key!r}")
        record["url"] = _artifact_url(record["path"], artifact_revision)
        observed[key] = record
    if set(observed) != set(expected):
        raise ValueError("release manifest does not cover the exact 40-track catalog")
    return observed


def _validated_raw_llr_records(
    raw_llr_validation: Mapping[str, Any], artifact_revision: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate the exact 32-track extension without touching v1 records."""

    _validate_revision(artifact_revision, field="raw_llr_artifact_revision")
    if (
        raw_llr_validation.get("report_version") != 1
        or raw_llr_validation.get("product") != "raw_calibrated_llr"
        or raw_llr_validation.get("valid") is not True
        or raw_llr_validation.get("track_count") != 32
        or raw_llr_validation.get("value_decimals") != 3
        or raw_llr_validation.get("reference_zero_baseline") is not True
        or raw_llr_validation.get("abs_llr_calibrated_used") is not False
    ):
        raise ValueError("raw-LLR validation does not certify the 32-track extension")
    raw_records = raw_llr_validation.get("tracks")
    if not isinstance(raw_records, list) or len(raw_records) != 32:
        raise ValueError("raw-LLR validation must contain exactly 32 records")

    expected = {
        (score_set.name, track): {
            "assembly": score_set.assembly,
            "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
            "base": RAW_LLR_BASES[track],
            "path": f"bigwig/{score_set.name}/{track}.bw",
        }
        for score_set in SCORE_SETS
        for track in RAW_LLR_TRACKS
    }
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("raw-LLR track records must be JSON objects")
        record = dict(raw_record)
        key = (str(record.get("score_set")), str(record.get("track")))
        contract = expected.get(key)
        if (
            contract is None
            or key in observed
            or any(record.get(field) != value for field, value in contract.items())
            or not isinstance(record.get("size"), int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            or not isinstance(record.get("bases_covered"), int)
            or record["bases_covered"] <= 0
            or not isinstance(record.get("zoom_levels"), int)
            or record["zoom_levels"] < 1
        ):
            raise ValueError(f"invalid raw-LLR release record: {key!r}")
        record["artifact_revision"] = artifact_revision
        record["url"] = _artifact_url(record["path"], artifact_revision)
        observed[key] = record
    if set(observed) != set(expected):
        raise ValueError("raw-LLR validation does not cover the exact 32-track catalog")
    return observed


def _extended_release_manifest(
    release_manifest: Mapping[str, Any],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    raw_llr_records: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    artifact_revision: str,
    raw_llr_artifact_revision: str,
) -> dict[str, Any]:
    """Combine trusted v1 identities and focused raw-LLR evidence."""

    extended = json.loads(json.dumps(release_manifest))
    if not isinstance(extended, dict):
        raise AssertionError("JSON release manifest did not remain an object")
    old_files = []
    for key, record in sorted(records.items()):
        old_files.append(
            {
                field: record[field]
                for field in (
                    "path",
                    "score_set",
                    "assembly",
                    "ucsc_assembly",
                    "track",
                    "size",
                    "sha256",
                    "bases_covered",
                    "zoom_levels",
                )
            }
            | {"artifact_revision": artifact_revision}
        )
    raw_files = []
    for key, record in sorted(raw_llr_records.items()):
        raw_files.append(
            {
                field: record[field]
                for field in (
                    "path",
                    "score_set",
                    "assembly",
                    "ucsc_assembly",
                    "track",
                    "size",
                    "sha256",
                    "bases_covered",
                    "zoom_levels",
                )
            }
            | {"artifact_revision": raw_llr_artifact_revision}
        )
    files = [*old_files, *raw_files]
    bigwig = extended.get("bigwig")
    validation = extended.get("validation")
    if not isinstance(bigwig, dict) or not isinstance(validation, dict):
        raise ValueError("source release manifest lacks mutable BigWig metadata")
    extended["release_manifest_version"] = 2
    bigwig.update(
        {
            "file_count": len(files),
            "total_bytes": sum(int(record["size"]) for record in files),
            "files": files,
            "artifact_revisions": {
                "v1": {
                    "revision": artifact_revision,
                    "track_count": len(old_files),
                    "validation_source": "trusted_v1_release_manifest",
                    "revalidated": False,
                },
                "raw_calibrated_llr": {
                    "revision": raw_llr_artifact_revision,
                    "track_count": len(raw_files),
                    "validation_source": "manifest/raw-llr-validation.json",
                    "revalidated": True,
                },
            },
        }
    )
    validation.update(
        {
            "expected_bigwig_files": len(files),
            "bigwig_validation_passed": True,
            "bigwig_validation_scope": "new_raw_llr_tracks_only",
            "existing_v1_bigwigs_revalidated": False,
            "raw_llr_validation_passed": True,
        }
    )
    return extended


def _render_hub(contact_email: str) -> str:
    return "\n".join(
        [
            "hub GPNStarScores",
            "shortLabel GPN-Star",
            "longLabel GPN-Star genome-wide scores",
            "genomesFile genomes.txt",
            f"email {contact_email}",
            "descriptionUrl description.html",
            "",
        ]
    )


def _render_genomes() -> str:
    lines: list[str] = []
    for assembly in HUB_RELEASE_ASSEMBLY_ORDER:
        lines.extend(
            [
                f"genome {hub_database_name(assembly)}",
                f"trackDb {HUB_TRACK_DB_DIRECTORIES[assembly]}/trackDb.txt",
                "",
            ]
        )
    return "\n".join(lines)


def _render_track_db(
    score_sets: Sequence[ScoreSetSpec],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    artifact_revision: str,
    *,
    raw_llr_records: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    raw_llr_artifact_revision: str | None = None,
) -> str:
    if (raw_llr_records is None) != (raw_llr_artifact_revision is None):
        raise ValueError(
            "raw-LLR records and artifact revision must be supplied together"
        )
    lines: list[str] = []
    for priority, score_set in enumerate(score_sets, start=1):
        group = _track_symbol(score_set.name)
        entropy = f"{group}Entropy"
        logo = f"{group}Logo"
        raw_llr = f"{group}RawLlr"
        label = _score_set_label(score_set)
        lines.extend(
            [
                f"track {group}",
                "superTrack on show",
                f"shortLabel {label}",
                f"longLabel {label} model score tracks",
                f"priority {priority}",
                f"dataVersion {artifact_revision}",
                "",
                f"track {entropy}",
                f"parent {group}",
                "type bigWig",
                "shortLabel Entropy",
                f"longLabel {label} entropy",
                "visibility dense",
                "autoScale on",
                "graphTypeDefault bar",
                "maxHeightPixels 100:40:16",
                "windowingFunction mean",
                f"bigDataUrl {records[(score_set.name, 'entropy')]['url']}",
                f"dataVersion {artifact_revision}",
                "",
                f"track {logo}",
                f"parent {group}",
                "container multiWig",
                "type bigWig 0 2",
                "shortLabel Sequence logo",
                f"longLabel {label} LLR-derived sequence logo",
                "aggregate stacked",
                "showSubtrackColorOnUi on",
                "autoScale off",
                "viewLimits 0:2",
                "viewLimitsMax 0:2",
                "maxHeightPixels 100:50:16",
                "logo on",
                "visibility full",
                f"dataVersion {artifact_revision}",
                "",
            ]
        )
        for base_priority, base in enumerate(NUCLEOTIDE_COLORS, start=1):
            lines.extend(
                [
                    f"    track {logo}{base}",
                    f"    parent {logo}",
                    "    type bigWig 0 2",
                    f"    shortLabel {base}",
                    f"    longLabel {label} {base} logo height",
                    f"    priority {base_priority}",
                    f"    bigDataUrl {records[(score_set.name, base)]['url']}",
                    f"    color {NUCLEOTIDE_COLORS[base]}",
                    "",
                ]
            )
        if raw_llr_records is not None:
            lines.extend(
                [
                    f"track {raw_llr}",
                    f"parent {group}",
                    "compositeTrack on",
                    "type bigWig",
                    "shortLabel LLR",
                    f"longLabel {label} LLR by allele",
                    "visibility full",
                    "autoScale group",
                    "alwaysZero on",
                    "yLineOnOff on",
                    "yLineMark 0",
                    "graphTypeDefault bar",
                    "maxHeightPixels 100:40:16",
                    "windowingFunction mean+whiskers",
                    f"dataVersion {raw_llr_artifact_revision}",
                    "",
                ]
            )
            for base_priority, track in enumerate(RAW_LLR_TRACKS, start=1):
                base = RAW_LLR_BASES[track]
                lines.extend(
                    [
                        f"    track {raw_llr}{base}",
                        f"    parent {raw_llr} on",
                        "    type bigWig",
                        f"    shortLabel LLR {base}",
                        f"    longLabel {label} {base} LLR",
                        f"    priority {base_priority}",
                        "    visibility full",
                        f"    color {RAW_LLR_POSITIVE_COLOR}",
                        f"    altColor {RAW_LLR_NEGATIVE_COLOR}",
                        (
                            "    bigDataUrl "
                            f"{raw_llr_records[(score_set.name, track)]['url']}"
                        ),
                        f"    dataVersion {raw_llr_artifact_revision}",
                        "",
                    ]
                )
    return "\n".join(lines)


def _render_hub_description(
    artifact_revision: str, raw_llr_artifact_revision: str | None
) -> str:
    repository_url = f"{HUGGING_FACE_URL}/datasets/{REPOSITORY_ID}"
    raw_llr_text = ""
    if raw_llr_artifact_revision is not None:
        raw_llr_text = f"""
<p>Each model group also includes a full LLR composite with
separate A/C/G/T rows. Positive values are blue, negative values are red, and
the zero baseline is always visible. These additive tracks are pinned to
artifact revision <code>{raw_llr_artifact_revision}</code>.</p>"""
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>GPN-Star genome-wide scores</title></head>
<body>
<h1>GPN-Star genome-wide scores</h1>
<p>This hub exposes entropy and LLR-derived sequence-logo
tracks for eight GPN-Star score sets across six UCSC assemblies.</p>
<p>The original entropy and sequence-logo data are pinned to Hugging Face
artifact revision
<code>{artifact_revision}</code>. Parquet remains the canonical score product;
BigWig values are three-decimal Float32 visualization values.</p>
{raw_llr_text}
<p><a href="{repository_url}">Dataset, checksums, schemas, and usage</a></p>
<p>Please cite: {html.escape(PAPER_TITLE)}.
<a href="https://doi.org/{PAPER_DOI}">doi:{PAPER_DOI}</a>.</p>
</body>
</html>
"""


def _render_group_description(
    score_set: ScoreSetSpec,
    artifact_revision: str,
    raw_llr_artifact_revision: str | None,
) -> str:
    label = _score_set_label(score_set)
    model_url = f"{HUGGING_FACE_URL}/{score_set.model_id}"
    revision_text = (
        "The entropy and sequence-logo browser tracks are pinned to artifact "
        f"revision <code>{artifact_revision}</code>. The LLR "
        "tracks are pinned to artifact revision "
        f"<code>{raw_llr_artifact_revision}</code>."
        if raw_llr_artifact_revision is not None
        else (
            "These browser tracks are pinned to artifact revision "
            f"<code>{artifact_revision}</code>."
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(label)}</title></head>
<body>
<h1>{html.escape(label)}</h1>
<p>{html.escape(score_set.model_description.capitalize())}:
<a href="{model_url}"><code>{html.escape(score_set.model_id)}</code></a>.</p>
<p>Source Parquet uses one-based positions and supplied assembly chromosome
names. These browser tracks use UCSC chromosome names and zero-based,
half-open one-base intervals. {revision_text}</p>
</body>
</html>
"""


def _render_entropy_description(score_set: ScoreSetSpec, artifact_revision: str) -> str:
    label = _score_set_label(score_set)
    model_url = f"{HUGGING_FACE_URL}/{score_set.model_id}"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(label)} entropy</title></head>
<body>
<h1>{html.escape(label)} entropy</h1>
<p>Model: <a href="{model_url}"><code>{html.escape(score_set.model_id)}</code></a>
({html.escape(score_set.model_description)}).</p>
<p>This quantitative track contains GPN-Star entropy values. It is a Float32
browser view rounded to three decimals; Parquet is the canonical full-precision
score product. The track does not add an unreviewed biological directionality
for high or low values.</p>
<p>One-based Parquet positions were converted explicitly to zero-based,
half-open one-base BigWig intervals with UCSC chromosome names. Artifact
revision: <code>{artifact_revision}</code>.</p>
</body>
</html>
"""


def _render_logo_description(score_set: ScoreSetSpec, artifact_revision: str) -> str:
    label = _score_set_label(score_set)
    model_url = f"{HUGGING_FACE_URL}/{score_set.model_id}"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(label)} sequence logo</title></head>
<body>
<h1>{html.escape(label)} LLR-derived sequence logo</h1>
<p>Model: <a href="{model_url}"><code>{html.escape(score_set.model_id)}</code></a>
({html.escape(score_set.model_description)}).</p>
<p>This stacked A/C/G/T view is a visualization transform, not a model
probability. At each position the reference nucleotide receives logit zero and
the three alternate nucleotides receive their independently supplied
LLR values. A stable Float64 softmax gives <code>p(base)</code>; with base-2
entropy <code>H</code>, each displayed height is <code>p(base) * (2 - H)</code>.
Final BigWig values are Float32 rounded to three decimals.</p>
<p>A is green, C blue, G orange, and T red. The height is only the stated
derived visualization value; use the canonical Parquet files for supplied
scores and their full precision.</p>
<p>One-based Parquet positions were converted explicitly to zero-based,
half-open one-base BigWig intervals with UCSC chromosome names. Artifact
revision: <code>{artifact_revision}</code>.</p>
</body>
</html>
"""


def _render_raw_llr_description(score_set: ScoreSetSpec, artifact_revision: str) -> str:
    label = _score_set_label(score_set)
    model_url = f"{HUGGING_FACE_URL}/{score_set.model_id}"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(label)} LLR</title></head>
<body>
<h1>{html.escape(label)} LLR</h1>
<p>Model: <a href="{model_url}"><code>{html.escape(score_set.model_id)}</code></a>
({html.escape(score_set.model_description)}).</p>
<p>This CADD-inspired composite presents one A, C, G, and T BigWig row. At
each covered position the reference allele is assigned the explicit zero
baseline and each alternate allele retains its independently supplied
LLR value. Positive LLR is blue and negative LLR is red. The four rows share
automatic scaling and display the zero line.</p>
<p>Values are Float32 rounded to three decimals for browser visualization;
Parquet remains the canonical full-precision score product.</p>
<p>One-based Parquet positions were converted explicitly to zero-based,
half-open one-base BigWig intervals with UCSC chromosome names. Artifact
revision: <code>{artifact_revision}</code>.</p>
</body>
</html>
"""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_promote_directory(temporary: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists():
        backup = output.parent / f".{output.name}.old-{uuid.uuid4().hex}"
        os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except BaseException:
        if backup is not None:
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_track_hub(
    release_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    artifact_revision: str,
    contact_email: str,
    raw_llr_validation_path: str | Path | None = None,
    raw_llr_artifact_revision: str | None = None,
    source_revision: str | None = None,
    public_metadata_revision: str | None = None,
) -> None:
    """Build the complete hub and updated dataset card at a temporary sibling."""

    if not _EMAIL_PATTERN.fullmatch(contact_email):
        raise ValueError("contact_email must be a valid single email address")
    if (raw_llr_validation_path is None) != (raw_llr_artifact_revision is None):
        raise ValueError(
            "raw_llr_validation_path and raw_llr_artifact_revision are both required"
        )
    release_manifest_file = Path(release_manifest_path).resolve()
    raw_llr_validation_file = (
        Path(raw_llr_validation_path).resolve()
        if raw_llr_validation_path is not None
        else None
    )
    output = Path(output_dir)
    resolved_output = output.resolve()
    source_manifests = [release_manifest_file]
    if raw_llr_validation_file is not None:
        source_manifests.append(raw_llr_validation_file)
    if any(
        source == resolved_output or resolved_output in source.parents
        for source in source_manifests
    ):
        raise ValueError("output_dir must not contain a source manifest")
    release_manifest = _read_json(release_manifest_file)
    records = _validated_bigwig_records(release_manifest, artifact_revision)
    raw_llr_validation = (
        _read_json(raw_llr_validation_file)
        if raw_llr_validation_file is not None
        else None
    )
    raw_llr_records = (
        _validated_raw_llr_records(
            raw_llr_validation,
            raw_llr_artifact_revision,
        )
        if raw_llr_validation is not None and raw_llr_artifact_revision is not None
        else None
    )
    extended_release_manifest = (
        _extended_release_manifest(
            release_manifest,
            records,
            raw_llr_records,
            artifact_revision=artifact_revision,
            raw_llr_artifact_revision=raw_llr_artifact_revision,
        )
        if raw_llr_records is not None and raw_llr_artifact_revision is not None
        else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        ucsc_root = temporary / "ucsc"
        _write_text(ucsc_root / "hub.txt", _render_hub(contact_email))
        _write_text(ucsc_root / "genomes.txt", _render_genomes())
        _write_text(
            ucsc_root / "description.html",
            _render_hub_description(artifact_revision, raw_llr_artifact_revision),
        )
        for release_assembly in HUB_RELEASE_ASSEMBLY_ORDER:
            score_sets = [
                score_set
                for score_set in SCORE_SETS
                if score_set.assembly == release_assembly
            ]
            if not score_sets:
                raise AssertionError(f"no score sets for {release_assembly}")
            assembly_root = ucsc_root / HUB_TRACK_DB_DIRECTORIES[release_assembly]
            _write_text(
                assembly_root / "trackDb.txt",
                _render_track_db(
                    score_sets,
                    records,
                    artifact_revision,
                    raw_llr_records=raw_llr_records,
                    raw_llr_artifact_revision=raw_llr_artifact_revision,
                ),
            )
            for score_set in score_sets:
                group = _track_symbol(score_set.name)
                _write_text(
                    assembly_root / f"{group}.html",
                    _render_group_description(
                        score_set,
                        artifact_revision,
                        raw_llr_artifact_revision,
                    ),
                )
                _write_text(
                    assembly_root / f"{group}Entropy.html",
                    _render_entropy_description(score_set, artifact_revision),
                )
                _write_text(
                    assembly_root / f"{group}Logo.html",
                    _render_logo_description(score_set, artifact_revision),
                )
                if raw_llr_artifact_revision is not None:
                    _write_text(
                        assembly_root / f"{group}RawLlr.html",
                        _render_raw_llr_description(
                            score_set, raw_llr_artifact_revision
                        ),
                    )

        launch_links = browser_launch_links(include_raw_llr=raw_llr_records is not None)
        card_launch_links = (
            dataset_card_launch_links() if raw_llr_records is not None else launch_links
        )
        validation_links = (
            {link["score_set"]: link for link in raw_llr_validation_links()}
            if raw_llr_records is not None
            else {}
        )
        _write_text(
            temporary / "README.md",
            render_dataset_card(
                extended_release_manifest or release_manifest,
                hub_launch_links=card_launch_links,
                raw_llr_validation=raw_llr_validation,
                source_revision=source_revision,
                public_metadata_revision=public_metadata_revision,
            ),
        )
        hub_files = [
            {
                "path": path.relative_to(temporary).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(ucsc_root.rglob("*"))
            if path.is_file()
        ]
        track_urls = [
            {
                "score_set": score_set,
                "track": track,
                "assembly": record["assembly"],
                "source_ucsc_assembly": record["ucsc_assembly"],
                "ucsc_assembly": hub_database_name(record["assembly"]),
                "path": record["path"],
                "url": record["url"],
                "size": record["size"],
                "sha256": record["sha256"],
                "bases_covered": record["bases_covered"],
                "zoom_levels": record["zoom_levels"],
                "artifact_revision": artifact_revision,
            }
            for (score_set, track), record in sorted(records.items())
        ]
        if raw_llr_records is not None:
            track_urls.extend(
                {
                    "score_set": score_set,
                    "track": track,
                    "assembly": record["assembly"],
                    "source_ucsc_assembly": record["ucsc_assembly"],
                    "ucsc_assembly": hub_database_name(record["assembly"]),
                    "path": record["path"],
                    "url": record["url"],
                    "size": record["size"],
                    "sha256": record["sha256"],
                    "bases_covered": record["bases_covered"],
                    "zoom_levels": record["zoom_levels"],
                    "artifact_revision": raw_llr_artifact_revision,
                }
                for (score_set, track), record in sorted(raw_llr_records.items())
            )
        hub_manifest = {
            "hub_manifest_version": 2 if raw_llr_records is not None else 1,
            "repository": REPOSITORY_ID,
            "hub_url": TRACK_HUB_URL,
            "artifact_revision": artifact_revision,
            "raw_llr_artifact_revision": raw_llr_artifact_revision,
            "assembly_count": len(HUB_ASSEMBLY_ORDER),
            "assemblies": list(HUB_ASSEMBLY_ORDER),
            "score_set_count": len(SCORE_SETS),
            "track_count": len(track_urls),
            "score_sets": [
                {
                    "name": score_set.name,
                    "assembly": score_set.assembly,
                    "ucsc_assembly": hub_database_name(score_set.assembly),
                    "model_id": score_set.model_id,
                    "browser_url": next(
                        link["url"]
                        for link in launch_links
                        if link["score_set"] == score_set.name
                    ),
                    **(
                        {
                            "raw_llr_validation_url": validation_links[score_set.name][
                                "url"
                            ]
                        }
                        if raw_llr_records is not None
                        else {}
                    ),
                }
                for score_set in SCORE_SETS
            ],
            "tracks": track_urls,
            "validation_scope_tracks": (
                [
                    f"{score_set.name}/{track}"
                    for score_set in SCORE_SETS
                    for track in RAW_LLR_TRACKS
                ]
                if raw_llr_records is not None
                else [
                    f"{score_set.name}/{track}"
                    for score_set in SCORE_SETS
                    for track in TRACKS
                ]
            ),
            "files": hub_files,
        }
        manifest_root = temporary / "manifest"
        manifest_root.mkdir()
        if raw_llr_validation_file is not None:
            shutil.copyfile(
                raw_llr_validation_file,
                manifest_root / "raw-llr-validation.json",
            )
        if extended_release_manifest is not None:
            atomic_write_json(
                manifest_root / "release.json",
                extended_release_manifest,
            )
        atomic_write_json(manifest_root / "ucsc-hub.json", hub_manifest)
        _validate_local_metadata(temporary)
        _atomic_promote_directory(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_extended_release_manifest(
    path: Path,
    hub_manifest: Mapping[str, Any],
) -> None:
    if not path.is_file():
        raise ValueError("raw-LLR hub lacks the updated release manifest")
    release = _read_json(path)
    repository = release.get("repository")
    bigwig = release.get("bigwig")
    validation = release.get("validation")
    files = bigwig.get("files") if isinstance(bigwig, Mapping) else None
    hub_tracks = hub_manifest.get("tracks")
    if (
        release.get("release_manifest_version") != 2
        or not isinstance(repository, Mapping)
        or repository.get("id") != REPOSITORY_ID
        or not isinstance(bigwig, Mapping)
        or bigwig.get("file_count") != 72
        or bigwig.get("value_decimals") != 3
        or not isinstance(files, list)
        or len(files) != 72
        or not isinstance(validation, Mapping)
        or validation.get("expected_bigwig_files") != 72
        or validation.get("bigwig_validation_passed") is not True
        or validation.get("bigwig_validation_scope") != "new_raw_llr_tracks_only"
        or validation.get("existing_v1_bigwigs_revalidated") is not False
        or validation.get("raw_llr_validation_passed") is not True
        or not isinstance(hub_tracks, list)
        or len(hub_tracks) != 72
    ):
        raise ValueError("updated release manifest contract differs")
    expected = {(record["score_set"], record["track"]): record for record in hub_tracks}
    observed = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("updated release BigWig records must be objects")
        key = (record.get("score_set"), record.get("track"))
        hub_record = expected.get(key)
        if (
            hub_record is None
            or key in observed
            or record.get("path") != hub_record.get("path")
            or record.get("assembly") != hub_record.get("assembly")
            or record.get("ucsc_assembly") != hub_record.get("source_ucsc_assembly")
            or record.get("size") != hub_record.get("size")
            or record.get("sha256") != hub_record.get("sha256")
            or record.get("bases_covered") != hub_record.get("bases_covered")
            or record.get("zoom_levels") != hub_record.get("zoom_levels")
            or record.get("artifact_revision") != hub_record.get("artifact_revision")
        ):
            raise ValueError(f"updated release identity differs: {key!r}")
        observed.add(key)
    if observed != set(expected):
        raise ValueError("updated release manifest catalog differs")
    if bigwig.get("total_bytes") != sum(int(record["size"]) for record in files):
        raise ValueError("updated release manifest byte total differs")
    provenance = bigwig.get("artifact_revisions")
    expected_provenance = {
        "v1": {
            "revision": hub_manifest["artifact_revision"],
            "track_count": 40,
            "validation_source": "trusted_v1_release_manifest",
            "revalidated": False,
        },
        "raw_calibrated_llr": {
            "revision": hub_manifest["raw_llr_artifact_revision"],
            "track_count": 32,
            "validation_source": "manifest/raw-llr-validation.json",
            "revalidated": True,
        },
    }
    if provenance != expected_provenance:
        raise ValueError("updated release revision provenance differs")


def _validate_local_metadata(metadata_root: Path) -> dict[str, Any]:
    manifest = _read_json(metadata_root / "manifest" / "ucsc-hub.json")
    manifest_version = manifest.get("hub_manifest_version")
    raw_llr_enabled = manifest_version == 2
    expected_track_count = 72 if raw_llr_enabled else 40
    if (
        manifest_version not in {1, 2}
        or manifest.get("repository") != REPOSITORY_ID
        or manifest.get("hub_url") != TRACK_HUB_URL
        or manifest.get("assembly_count") != len(HUB_ASSEMBLY_ORDER)
        or manifest.get("assemblies") != list(HUB_ASSEMBLY_ORDER)
        or manifest.get("score_set_count") != len(SCORE_SETS)
        or manifest.get("track_count") != expected_track_count
    ):
        raise ValueError("invalid UCSC hub manifest contract")
    artifact_revision = _validate_revision(
        manifest.get("artifact_revision"), field="artifact_revision"
    )
    raw_llr_artifact_revision = manifest.get("raw_llr_artifact_revision")
    if raw_llr_enabled:
        raw_llr_artifact_revision = _validate_revision(
            raw_llr_artifact_revision,
            field="raw_llr_artifact_revision",
        )
        raw_validation_path = metadata_root / "manifest" / "raw-llr-validation.json"
        if not raw_validation_path.is_file():
            raise ValueError("raw-LLR hub lacks its focused validation manifest")
        _validated_raw_llr_records(
            _read_json(raw_validation_path),
            raw_llr_artifact_revision,
        )
        _validate_extended_release_manifest(
            metadata_root / "manifest" / "release.json",
            manifest,
        )
    elif raw_llr_artifact_revision is not None:
        raise ValueError("legacy hub manifest cannot name a raw-LLR revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("UCSC hub manifest lacks file identities")
    expected_paths = {"ucsc/hub.txt", "ucsc/genomes.txt", "ucsc/description.html"}
    expected_paths.update(
        f"ucsc/{directory}/trackDb.txt" for directory in HUB_TRACK_DB_DIRECTORY_ORDER
    )
    for score_set in SCORE_SETS:
        group = _track_symbol(score_set.name)
        assembly = HUB_TRACK_DB_DIRECTORIES[score_set.assembly]
        expected_paths.update(
            {
                f"ucsc/{assembly}/{group}.html",
                f"ucsc/{assembly}/{group}Entropy.html",
                f"ucsc/{assembly}/{group}Logo.html",
            }
        )
        if raw_llr_enabled:
            expected_paths.add(f"ucsc/{assembly}/{group}RawLlr.html")
    observed_paths = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("UCSC hub file identities must be objects")
        relative_path = record.get("path")
        if (
            not isinstance(relative_path, str)
            or relative_path in observed_paths
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError("invalid UCSC hub file path")
        path = metadata_root / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"UCSC hub file identity differs: {relative_path}")
        observed_paths.add(relative_path)
    if expected_paths != observed_paths:
        raise ValueError("UCSC hub control and description file set differs")
    genomes = (metadata_root / "ucsc" / "genomes.txt").read_text(encoding="utf-8")
    if genomes != _render_genomes():
        raise ValueError("UCSC genomes.txt differs from the browser database mapping")

    score_sets = manifest.get("score_sets")
    launch_links = {
        link["score_set"]: link
        for link in browser_launch_links(include_raw_llr=raw_llr_enabled)
    }
    validation_links = {link["score_set"]: link for link in raw_llr_validation_links()}
    if not isinstance(score_sets, list) or len(score_sets) != len(SCORE_SETS):
        raise ValueError("UCSC hub manifest must contain the exact score-set catalog")
    for score_set, record in zip(SCORE_SETS, score_sets, strict=True):
        if not isinstance(record, Mapping) or any(
            record.get(field) != expected
            for field, expected in {
                "name": score_set.name,
                "assembly": score_set.assembly,
                "ucsc_assembly": hub_database_name(score_set.assembly),
                "model_id": score_set.model_id,
                "browser_url": launch_links[score_set.name]["url"],
                **(
                    {"raw_llr_validation_url": validation_links[score_set.name]["url"]}
                    if raw_llr_enabled
                    else {}
                ),
            }.items()
        ):
            raise ValueError(f"invalid UCSC score-set record: {score_set.name}")
        if not raw_llr_enabled and "raw_llr_validation_url" in record:
            raise ValueError("legacy UCSC score-set record names raw-LLR validation")

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != expected_track_count:
        raise ValueError(
            f"UCSC hub manifest must contain {expected_track_count} track URLs"
        )
    track_urls = []
    observed_tracks = set()
    for track in tracks:
        if not isinstance(track, Mapping):
            raise ValueError("UCSC hub track records must be objects")
        url = track.get("url")
        assembly = track.get("assembly")
        track_name = track.get("track")
        score_set = track.get("score_set")
        revision = track.get("artifact_revision")
        expected_revision = (
            raw_llr_artifact_revision
            if track_name in RAW_LLR_TRACKS
            else artifact_revision
        )
        if (
            not isinstance(url, str)
            or revision != expected_revision
            or f"/resolve/{expected_revision}/bigwig/" not in url
            or not isinstance(assembly, str)
            or assembly not in HUB_DATABASE_NAMES
            or track.get("ucsc_assembly") != HUB_DATABASE_NAMES[assembly]
            or track.get("source_ucsc_assembly") != HUB_TRACK_DB_DIRECTORIES[assembly]
            or not isinstance(score_set, str)
            or not isinstance(track_name, str)
            or (score_set, track_name) in observed_tracks
        ):
            raise ValueError("invalid UCSC BigWig URL or assembly identity")
        observed_tracks.add((score_set, track_name))
        track_urls.append(url)
    expected_tracks = {
        (score_set.name, track)
        for score_set in SCORE_SETS
        for track in (*TRACKS, *(RAW_LLR_TRACKS if raw_llr_enabled else ()))
    }
    if observed_tracks != expected_tracks:
        raise ValueError("UCSC hub track records differ from the expected catalog")
    if len(set(track_urls)) != expected_track_count:
        raise ValueError("UCSC hub BigWig URLs must be unique")
    track_db_text = "\n".join(
        (metadata_root / "ucsc" / directory / "trackDb.txt").read_text(encoding="utf-8")
        for directory in HUB_TRACK_DB_DIRECTORY_ORDER
    )
    if any(track_db_text.count(url) != 1 for url in track_urls):
        raise ValueError("each pinned BigWig URL must appear once in trackDb")
    if track_db_text.count("container multiWig") != len(SCORE_SETS):
        raise ValueError("each score set must define one multiWig")
    if track_db_text.count("logo on") != len(SCORE_SETS):
        raise ValueError("each score set must enable sequence-logo rendering")
    if track_db_text.count("visibility dense") != len(SCORE_SETS):
        raise ValueError("each entropy track must default to dense")
    expected_full_count = len(SCORE_SETS) * (
        1 + (1 + len(RAW_LLR_TRACKS) if raw_llr_enabled else 0)
    )
    if track_db_text.count("visibility full") != expected_full_count:
        raise ValueError("logo and LLR tracks must default to full")
    if raw_llr_enabled:
        if track_db_text.count("compositeTrack on") != len(SCORE_SETS):
            raise ValueError("each score set must define one raw-LLR composite")
        signed_color_pair = (
            f"    color {RAW_LLR_POSITIVE_COLOR}\n    altColor {RAW_LLR_NEGATIVE_COLOR}"
        )
        if track_db_text.count(signed_color_pair) != 32:
            raise ValueError("raw-LLR signed colors differ")
        if track_db_text.count("autoScale group") != len(SCORE_SETS):
            raise ValueError("raw-LLR group scaling differs")
    readme = (metadata_root / "README.md").read_text(encoding="utf-8")
    if TRACK_HUB_URL not in readme:
        raise ValueError("dataset card does not link the public UCSC hub")
    readme_links = (
        dataset_card_launch_links() if raw_llr_enabled else launch_links.values()
    )
    if any(link["url"] not in readme for link in readme_links):
        raise ValueError("dataset card lacks a model-specific UCSC launch link")
    return manifest


def _run_command(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    result = runner(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def _parse_chromosomes(output: str) -> dict[str, int]:
    chromosomes: dict[str, int] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(chr[A-Za-z0-9_.-]+)\s+\d+\s+(\d+)\s*", line)
        if match:
            chromosomes[match.group(1)] = int(match.group(2))
    if not chromosomes:
        raise RuntimeError("bigWigInfo -chroms returned no chromosome sizes")
    return chromosomes


def _summary_values(
    executable: str,
    url: str,
    chrom: str,
    start: int,
    end: int,
    bins: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[float | None]:
    result = _run_command(
        [
            executable,
            "-type=mean",
            url,
            chrom,
            str(start),
            str(end),
            str(bins),
        ],
        runner=runner,
    )
    values: list[float | None] = []
    for token in result.stdout.split():
        if token.lower() in {"n/a", "nan"}:
            values.append(None)
        else:
            value = float(token)
            if not math.isfinite(value):
                raise RuntimeError("bigWigSummary returned a non-finite value")
            values.append(value)
    if len(values) != bins:
        raise RuntimeError(
            f"bigWigSummary returned {len(values)} bins, expected {bins}"
        )
    return values


def _find_covered_locus(
    executable: str,
    url: str,
    chromosomes: Mapping[str, int],
    chromosome_order: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[str, int]:
    for chrom in chromosome_order:
        end = chromosomes[chrom]
        start = 0
        while end - start > 1:
            width = end - start
            bins = min(128, width)
            values = _summary_values(
                executable,
                url,
                chrom,
                start,
                end,
                bins,
                runner=runner,
            )
            index = next(
                (i for i, value in enumerate(values) if value is not None), None
            )
            if index is None:
                break
            next_start = start + (index * width) // bins
            next_end = start + ((index + 1) * width) // bins
            start, end = next_start, max(next_start + 1, next_end)
        if end - start == 1:
            value = _summary_values(
                executable,
                url,
                chrom,
                start,
                end,
                1,
                runner=runner,
            )[0]
            if value is not None:
                return chrom, start
    raise RuntimeError("could not locate a covered representative BigWig base")


def _check_http_range(
    record: Mapping[str, Any],
    *,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    request = Request(record["url"], headers={"Range": "bytes=0-63"})
    with opener(request, timeout=60) as response:
        status = getattr(response, "status", None)
        content = response.read()
        content_range = response.headers.get("Content-Range")
    if status != 206 or not content or not isinstance(content_range, str):
        raise RuntimeError(f"BigWig HTTP range request failed: {record['path']}")
    match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", content_range)
    if (
        match is None
        or int(match.group(2)) != record["size"]
        or len(content) != int(match.group(1)) + 1
    ):
        raise RuntimeError(f"invalid BigWig Content-Range: {record['path']}")
    return {
        "path": record["path"],
        "status": status,
        "content_range": content_range,
        "bytes_read": len(content),
    }


def _validate_track_hub(
    metadata_root: Path,
    *,
    hub_target: str,
    metadata_only: bool,
    hub_check: str,
    bigwig_info: str,
    bigwig_summary: str,
    udc_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    opener: Callable[..., Any],
) -> dict[str, Any]:
    manifest = _validate_local_metadata(metadata_root)
    udc_dir.mkdir(parents=True, exist_ok=True)
    hub_check_result = _run_command(
        [
            hub_check,
            "-noTracks",
            "-checkSettings",
            f"-version={HUB_SETTINGS_SPEC}",
            f"-udcDir={udc_dir}",
            hub_target,
        ],
        runner=runner,
    )
    common_report = {
        "report_version": 1,
        "valid": True,
        "repository": REPOSITORY_ID,
        "artifact_revision": manifest["artifact_revision"],
        "raw_llr_artifact_revision": manifest.get("raw_llr_artifact_revision"),
        "hub_manifest_sha256": sha256_file(
            metadata_root / "manifest" / "ucsc-hub.json"
        ),
        "hub_target": hub_target,
        "hub_check": {
            "passed": True,
            "remote_tracks_checked": False,
            "check_settings": True,
            "settings_spec": HUB_SETTINGS_SPEC,
            "stdout": hub_check_result.stdout,
            "stderr": hub_check_result.stderr,
        },
        "assembly_count": len(HUB_ASSEMBLY_ORDER),
        "score_set_count": len(SCORE_SETS),
    }
    if metadata_only:
        return {
            **common_report,
            "track_count": 0,
            "validation_scope": "hub_metadata_only",
            "existing_v1_bigwigs_revalidated": False,
            "existing_raw_llr_bigwigs_revalidated": False,
            "prior_artifact_validation_reused": True,
            "http_range_count": 0,
            "http_range_checks": [],
            "chromosome_checks": [],
            "representative_checks": [],
        }

    validation_scope = manifest.get("validation_scope_tracks")
    if not isinstance(validation_scope, list) or not all(
        isinstance(item, str) for item in validation_scope
    ):
        raise ValueError("hub manifest lacks an explicit validation scope")
    scope_keys = {
        tuple(item.split("/", maxsplit=1))
        for item in validation_scope
        if item.count("/") == 1
    }
    tracks = [
        record
        for record in manifest["tracks"]
        if (record["score_set"], record["track"]) in scope_keys
    ]
    if len(tracks) != len(scope_keys):
        raise ValueError("hub validation scope does not resolve to unique tracks")
    raw_llr_scope = manifest["hub_manifest_version"] == 2
    expected_scope_count = 32 if raw_llr_scope else 40
    if len(tracks) != expected_scope_count:
        raise ValueError(
            f"hub validation scope must contain {expected_scope_count} tracks"
        )
    range_checks = [_check_http_range(record, opener=opener) for record in tracks]
    chromosome_checks = []
    chromosomes_by_score_set: dict[str, dict[str, int]] = {}
    for record in tracks:
        result = _run_command([bigwig_info, "-chroms", record["url"]], runner=runner)
        chromosomes = _parse_chromosomes(result.stdout)
        expected = {
            f"chr{chrom}"
            for chrom in ASSEMBLIES[
                next(
                    score_set.assembly
                    for score_set in SCORE_SETS
                    if score_set.name == record["score_set"]
                )
            ].chromosomes
        }
        if set(chromosomes) != expected:
            raise RuntimeError(
                f"BigWig chromosome names differ for {record['score_set']}/"
                f"{record['track']}"
            )
        prior_chromosomes = chromosomes_by_score_set.get(record["score_set"])
        if prior_chromosomes is not None and chromosomes != prior_chromosomes:
            raise RuntimeError(
                f"BigWig chromosome sizes differ for {record['score_set']}/"
                f"{record['track']}"
            )
        chromosomes_by_score_set.setdefault(record["score_set"], chromosomes)
        chromosome_checks.append(
            {
                "score_set": record["score_set"],
                "track": record["track"],
                "chromosome_count": len(chromosomes),
                "names_match": True,
                "sizes_match_score_set": True,
            }
        )

    records_by_key = {
        (record["score_set"], record["track"]): record for record in tracks
    }
    representative_checks = []
    for score_set in SCORE_SETS:
        representative_track = RAW_LLR_TRACKS[0] if raw_llr_scope else "entropy"
        representative = records_by_key[(score_set.name, representative_track)]
        chromosome_sizes = chromosomes_by_score_set[score_set.name]
        chromosome_order = [
            f"chr{chrom}" for chrom in ASSEMBLIES[score_set.assembly].chromosomes
        ]
        chrom, start = _find_covered_locus(
            bigwig_summary,
            representative["url"],
            chromosome_sizes,
            chromosome_order,
            runner=runner,
        )
        track_values = {}
        zoom_values = {}
        zoom_start = max(0, start - 500)
        zoom_end = min(chromosome_sizes[chrom], start + 501)
        zoom_bins = min(10, zoom_end - zoom_start)
        tracks_for_score_set = RAW_LLR_TRACKS if raw_llr_scope else TRACKS
        for track in tracks_for_score_set:
            record = records_by_key[(score_set.name, track)]
            value = _summary_values(
                bigwig_summary,
                record["url"],
                chrom,
                start,
                start + 1,
                1,
                runner=runner,
            )[0]
            if value is None:
                raise RuntimeError(
                    f"representative base is missing from {score_set.name}/{track}"
                )
            zoom = _summary_values(
                bigwig_summary,
                record["url"],
                chrom,
                zoom_start,
                zoom_end,
                zoom_bins,
                runner=runner,
            )
            if not any(item is not None for item in zoom):
                raise RuntimeError(
                    f"representative zoom is empty for {score_set.name}/{track}"
                )
            track_values[track] = value
            zoom_values[track] = zoom
        representative_checks.append(
            {
                "score_set": score_set.name,
                "assembly": score_set.assembly,
                "ucsc_assembly": hub_database_name(score_set.assembly),
                "chrom": chrom,
                "start": start,
                "end": start + 1,
                "zero_based_half_open": True,
                "track_values": track_values,
                "zoom_region": {
                    "start": zoom_start,
                    "end": zoom_end,
                    "bins": zoom_bins,
                    "values": zoom_values,
                },
            }
        )

    return {
        **common_report,
        "track_count": len(tracks),
        "validation_scope": (
            "new_raw_llr_tracks_only" if raw_llr_scope else "legacy_v1_tracks"
        ),
        "existing_v1_bigwigs_revalidated": False if raw_llr_scope else True,
        "existing_raw_llr_bigwigs_revalidated": raw_llr_scope,
        "prior_artifact_validation_reused": raw_llr_scope,
        "http_range_count": len(range_checks),
        "http_range_checks": range_checks,
        "chromosome_checks": chromosome_checks,
        "representative_checks": representative_checks,
    }


def _render_validation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# UCSC track-hub validation",
        "",
        f"Status: **{'valid' if report.get('valid') else 'invalid'}**",
        "",
        f"Artifact revision: `{report['artifact_revision']}`",
        "",
        "`hubCheck -noTracks -checkSettings`: "
        f"{'passed' if report['hub_check']['passed'] else 'failed'}",
        "",
    ]
    if report.get("validation_scope") == "hub_metadata_only":
        lines.extend(
            [
                "Validation was intentionally limited to hub metadata. No BigWig "
                "ranges, headers, bases, or zoom summaries were requested; completed "
                "artifact evidence was reused.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "| Score set | UCSC assembly | Representative locus | Tracks queried |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for check in report["representative_checks"]:
        lines.append(
            f"| `{check['score_set']}` | `{check['ucsc_assembly']}` | "
            f"`{check['chrom']}:{check['start']}-{check['end']}` | "
            f"{len(check['track_values'])} |"
        )
    lines.extend(
        [
            "",
            f"All {report['http_range_count']} BigWig URLs returned valid HTTP "
            "byte ranges. Every track header had the exact release chromosome set, "
            "and direct one-base plus zoom-window summaries were non-empty.",
            "",
        ]
    )
    if report.get("validation_scope") == "new_raw_llr_tracks_only":
        lines.extend(
            [
                f"Raw-LLR artifact revision: `{report['raw_llr_artifact_revision']}`",
                "",
                "Validation covered only the 32 additive raw calibrated-LLR "
                "BigWigs. The 40 immutable v1 BigWigs were not revalidated.",
                "",
            ]
        )
    return "\n".join(lines)


def validate_track_hub(
    metadata_root: str | Path,
    report_path: str | Path,
    markdown_path: str | Path,
    *,
    udc_dir: str | Path,
    metadata_only: bool = False,
    hub_target: str | None = None,
    hub_check: str = "hubCheck",
    bigwig_info: str = "bigWigInfo",
    bigwig_summary: str = "bigWigSummary",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    opener: Callable[..., Any] = urlopen,
) -> None:
    """Validate hub metadata and, unless disabled, its configured BigWig scope."""

    metadata = Path(metadata_root)
    target = hub_target or str((metadata / "ucsc" / "hub.txt").resolve())
    report = _validate_track_hub(
        metadata,
        hub_target=target,
        metadata_only=metadata_only,
        hub_check=hub_check,
        bigwig_info=bigwig_info,
        bigwig_summary=bigwig_summary,
        udc_dir=Path(udc_dir),
        runner=runner,
        opener=opener,
    )
    atomic_write_json(Path(report_path), report)
    _atomic_write_text(Path(markdown_path), _render_validation_markdown(report))


def _validated_publication_approval(
    approval: Mapping[str, Any] | None,
    expected_base_revision: str,
    *,
    operation: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    if operation not in {"publish_hub", "publish_dataset_card"}:
        raise ValueError(f"unknown publication approval operation: {operation}")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
        raise ValueError("publication candidate must have an exact SHA-256")
    if not isinstance(approval, Mapping):
        raise ValueError("public hub update requires explicit author approval")
    if (
        approval.get("approved") is not True
        or not isinstance(approval.get("evidence_url"), str)
        or not re.fullmatch(
            r"https://github\.com/gonzalobenegas/gpn-star-scores/issues/\d+",
            approval["evidence_url"],
        )
        or approval.get("expected_base_revision") != expected_base_revision
        or approval.get("operation") != operation
        or approval.get("candidate_sha256") != candidate_sha256
        or not isinstance(approval.get("approved_by"), str)
        or not approval["approved_by"]
        or not isinstance(approval.get("approved_at"), str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approval["approved_at"])
    ):
        raise ValueError("public hub approval evidence is incomplete or mismatched")
    return dict(approval)


def _publication_files(metadata_root: Path) -> list[Path]:
    files = [
        metadata_root / "README.md",
        *sorted(
            path
            for path in (metadata_root / "manifest").glob("*.json")
            if path.is_file()
        ),
        *sorted(path for path in (metadata_root / "ucsc").rglob("*") if path.is_file()),
    ]
    if any(not path.is_file() for path in files):
        raise ValueError("hub publication metadata is incomplete")
    return files


def publication_candidate_sha256(metadata_root: str | Path) -> str:
    """Hash every path and byte identity submitted by full-hub publication."""

    metadata = Path(metadata_root)
    digest = hashlib.sha256()
    for path in _publication_files(metadata):
        relative_path = path.relative_to(metadata).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_recovery_publication(
    metadata_root: Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    final_revision: str,
    publication_approval: Mapping[str, Any],
    repository_id: str,
    metadata_only: bool,
) -> dict[str, Any]:
    pending = _read_json(report_path)
    expected_files = [
        path.relative_to(metadata_root).as_posix()
        for path in _publication_files(metadata_root)
    ]
    expected = {
        "report_version": 1,
        "repository": repository_id,
        "public": True,
        "base_revision": expected_base_revision,
        "final_revision": final_revision,
        "single_commit": True,
        "single_process": True,
        "slurm_job_id": None,
        "metadata_only": metadata_only,
        "publication_approval": dict(publication_approval),
        "published_files": expected_files,
    }
    status = pending.get("status")
    is_pending = status in {
        "published_pending_validation",
        "published_validation_failed",
    }
    is_validated = status in {"validated", "validated_existing_publication"}
    public_validation = pending.get("public_validation")
    valid_public_identity = (
        isinstance(public_validation, Mapping)
        and public_validation.get("valid") is True
        and public_validation.get("repository") == repository_id
        and public_validation.get("revision") == final_revision
        and public_validation.get("credentials_sent") is False
    )
    if (
        any(pending.get(field) != value for field, value in expected.items())
        or (is_pending and pending.get("valid") is not False)
        or (
            is_validated
            and (pending.get("valid") is not True or not valid_public_identity)
        )
        or not (is_pending or is_validated)
    ):
        raise ValueError(
            "validate-existing requires the matching publisher-created recovery report"
        )
    return pending


def validate_public_track_hub(
    metadata_root: str | Path,
    *,
    revision: str,
    udc_dir: str | Path,
    metadata_only: bool = False,
    repository_id: str = REPOSITORY_ID,
    api: Any | None = None,
    opener: Callable[..., Any] = urlopen,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Validate one immutable public hub revision without credentials."""

    _validate_revision(revision)
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"hub repository must be {REPOSITORY_ID}")
    metadata = Path(metadata_root)
    public_api = api or HfApi(token=False)
    info = public_api.repo_info(
        repository_id,
        repo_type="dataset",
        revision=revision,
        token=False,
    )
    if getattr(info, "private", True) or getattr(info, "sha", None) != revision:
        raise RuntimeError("public Hugging Face hub revision did not resolve exactly")

    file_checks = []
    for path in _publication_files(metadata):
        relative_path = path.relative_to(metadata).as_posix()
        url = f"{HUGGING_FACE_URL}/datasets/{repository_id}/resolve/{revision}/{relative_path}"
        with opener(url, timeout=60) as response:
            content = response.read()
        if content != path.read_bytes():
            raise RuntimeError(f"published hub file identity differs: {relative_path}")
        file_checks.append(
            {
                "path": relative_path,
                "size": len(content),
                "sha256": sha256_file(path),
            }
        )

    remote_hub_url = (
        f"{HUGGING_FACE_URL}/datasets/{repository_id}/resolve/{revision}/ucsc/hub.txt"
    )
    validation = _validate_track_hub(
        metadata,
        hub_target=remote_hub_url,
        metadata_only=metadata_only,
        hub_check="hubCheck",
        bigwig_info="bigWigInfo",
        bigwig_summary="bigWigSummary",
        udc_dir=Path(udc_dir),
        runner=runner,
        opener=opener,
    )
    return {
        "report_version": 1,
        "valid": validation.get("valid") is True,
        "repository": repository_id,
        "revision": revision,
        "public": True,
        "credentials_sent": False,
        "hub_url": remote_hub_url,
        "file_count": len(file_checks),
        "file_checks": file_checks,
        "hub_validation": validation,
    }


def _public_metadata_bytes(
    repository_id: str,
    revision: str,
    relative_path: str,
    *,
    opener: Callable[..., Any],
) -> bytes:
    encoded_path = quote(relative_path, safe="/")
    encoded_revision = quote(revision, safe="")
    url = (
        f"{HUGGING_FACE_URL}/datasets/{repository_id}/resolve/"
        f"{encoded_revision}/{encoded_path}"
    )
    with opener(url, timeout=60) as response:
        status = getattr(response, "status", 200)
        content = response.read()
    if status != 200:
        raise RuntimeError(f"public metadata returned HTTP {status}: {relative_path}")
    return content


def validate_public_dataset_card(
    metadata_root: str | Path,
    *,
    revision: str,
    repository_id: str = REPOSITORY_ID,
    api: Any | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Validate only the public dataset card and its unchanged hub manifest."""

    _validate_revision(revision)
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"hub repository must be {REPOSITORY_ID}")
    metadata = Path(metadata_root)
    manifest = _validate_local_metadata(metadata)
    public_api = api or HfApi(token=False)
    info = public_api.repo_info(
        repository_id,
        repo_type="dataset",
        revision=revision,
        token=False,
    )
    if getattr(info, "private", True) or getattr(info, "sha", None) != revision:
        raise RuntimeError("public dataset-card revision did not resolve exactly")

    checks = []
    for relative_path in ("README.md", "manifest/ucsc-hub.json"):
        local = metadata / relative_path
        content = _public_metadata_bytes(
            repository_id,
            revision,
            relative_path,
            opener=opener,
        )
        if content != local.read_bytes():
            raise RuntimeError(f"published metadata identity differs: {relative_path}")
        checks.append(
            {
                "path": relative_path,
                "size": len(content),
                "sha256": sha256_file(local),
            }
        )

    with opener(f"{HUGGING_FACE_URL}/datasets/{repository_id}", timeout=60) as response:
        page_status = getattr(response, "status", 200)
        page = response.read().decode("utf-8", errors="replace")
    if (
        page_status != 200
        or repository_id not in page
        or "GPN-Star genome-wide scores" not in page
    ):
        raise RuntimeError("public dataset-card page did not render")
    return {
        "report_version": 1,
        "valid": True,
        "repository": repository_id,
        "revision": revision,
        "public": True,
        "credentials_sent": False,
        "artifact_revision": manifest["artifact_revision"],
        "raw_llr_artifact_revision": manifest.get("raw_llr_artifact_revision"),
        "file_checks": checks,
        "dataset_card_rendered": True,
        "bigwig_checks_performed": 0,
    }


def validate_existing_track_hub_publication(
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    final_revision: str,
    publication_approval: Mapping[str, Any] | None,
    udc_dir: str | Path,
    success_marker_path: str | Path,
    metadata_only: bool = False,
    repository_id: str = REPOSITORY_ID,
    validator: Callable[..., dict[str, Any]] = validate_public_track_hub,
) -> None:
    """Recover or repeat validation for an already-published hub commit."""

    _validate_revision(expected_base_revision, field="expected_base_revision")
    _validate_revision(final_revision, field="final_revision")
    metadata = Path(metadata_root)
    _validate_local_metadata(metadata)
    approval = _validated_publication_approval(
        publication_approval,
        expected_base_revision,
        operation="publish_hub",
        candidate_sha256=publication_candidate_sha256(metadata),
    )
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"hub repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "hub publication validation must run from one non-Slurm process"
        )
    publication = _validated_recovery_publication(
        metadata,
        report_path,
        expected_base_revision=expected_base_revision,
        final_revision=final_revision,
        publication_approval=approval,
        repository_id=repository_id,
        metadata_only=metadata_only,
    )
    success_marker = Path(success_marker_path)
    if success_marker.resolve() == Path(report_path).resolve():
        raise ValueError("success marker and publication report must differ")
    success_marker.unlink(missing_ok=True)
    recovered_from_status = publication["status"]
    public_validation = validator(
        metadata,
        revision=final_revision,
        udc_dir=udc_dir,
        repository_id=repository_id,
        metadata_only=metadata_only,
    )
    if public_validation.get("valid") is not True:
        raise RuntimeError("existing public hub validation returned an invalid result")
    publication.update(
        {
            "valid": True,
            "status": "validated_existing_publication",
            "recovered_from_status": recovered_from_status,
            "public_validation": public_validation,
        }
    )
    publication.pop("validation_error_type", None)
    publication.pop("validation_error", None)
    atomic_write_json(Path(report_path), publication)
    _atomic_write_text(success_marker, f"{final_revision}\n")


def validate_existing_dataset_card_publication(
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    final_revision: str,
    publication_approval: Mapping[str, Any] | None,
    success_marker_path: str | Path,
    repository_id: str = REPOSITORY_ID,
    opener: Callable[..., Any] = urlopen,
    validator: Callable[..., dict[str, Any]] = validate_public_dataset_card,
) -> None:
    """Recover validation for an already-published README-only commit."""

    _validate_revision(expected_base_revision, field="expected_base_revision")
    _validate_revision(final_revision, field="final_revision")
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"hub repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("dataset-card validation must run outside Slurm")
    metadata = Path(metadata_root)
    _validate_local_metadata(metadata)
    approval = _validated_publication_approval(
        publication_approval,
        expected_base_revision,
        operation="publish_dataset_card",
        candidate_sha256=sha256_file(metadata / "README.md"),
    )
    publication = _read_json(report_path)
    expected = {
        "repository": repository_id,
        "public": True,
        "base_revision": expected_base_revision,
        "final_revision": final_revision,
        "single_commit": True,
        "single_process": True,
        "slurm_job_id": None,
        "publication_approval": approval,
        "published_files": ["README.md"],
    }
    recoverable = publication.get("status") in {
        "published_pending_validation",
        "published_validation_failed",
        "validated",
        "validated_existing_publication",
    }
    if (
        publication.get("report_version") != 1
        or any(publication.get(field) != value for field, value in expected.items())
        or not recoverable
    ):
        raise ValueError(
            "validate-existing-card requires its matching publisher report"
        )
    success_marker = Path(success_marker_path)
    if success_marker.resolve() == Path(report_path).resolve():
        raise ValueError("success marker and publication report must differ")
    success_marker.unlink(missing_ok=True)
    recovered_from_status = publication["status"]
    public_validation = validator(
        metadata,
        revision=final_revision,
        repository_id=repository_id,
        opener=opener,
    )
    if public_validation.get("valid") is not True:
        raise RuntimeError("existing public dataset-card validation returned invalid")
    publication.update(
        {
            "valid": True,
            "status": "validated_existing_publication",
            "recovered_from_status": recovered_from_status,
            "public_validation": public_validation,
        }
    )
    publication.pop("validation_error_type", None)
    publication.pop("validation_error", None)
    atomic_write_json(Path(report_path), publication)
    _atomic_write_text(success_marker, f"{final_revision}\n")


def publish_track_hub(
    metadata_root: str | Path,
    validation_report_path: str | Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    publication_approval: Mapping[str, Any] | None,
    udc_dir: str | Path,
    metadata_only: bool = False,
    success_marker_path: str | Path | None = None,
    repository_id: str = REPOSITORY_ID,
    api: Any | None = None,
    validator: Callable[..., dict[str, Any]] = validate_public_track_hub,
) -> None:
    """Atomically add the approved hub and README to the public dataset."""

    _validate_revision(expected_base_revision, field="expected_base_revision")
    metadata = Path(metadata_root)
    manifest = _validate_local_metadata(metadata)
    approval = _validated_publication_approval(
        publication_approval,
        expected_base_revision,
        operation="publish_hub",
        candidate_sha256=publication_candidate_sha256(metadata),
    )
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"hub repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("hub publication must run from one non-Slurm process")
    success_marker = (
        Path(success_marker_path) if success_marker_path is not None else None
    )
    if success_marker is not None:
        if success_marker.resolve() == Path(report_path).resolve():
            raise ValueError("success marker and publication report must differ")
        success_marker.unlink(missing_ok=True)
    validation = _read_json(validation_report_path)
    expected_validation_scope = (
        "hub_metadata_only"
        if metadata_only
        else (
            "new_raw_llr_tracks_only"
            if manifest["hub_manifest_version"] == 2
            else "legacy_v1_tracks"
        )
    )
    if (
        validation.get("valid") is not True
        or validation.get("artifact_revision") != manifest["artifact_revision"]
        or validation.get("raw_llr_artifact_revision")
        != manifest.get("raw_llr_artifact_revision")
        or validation.get("hub_manifest_sha256")
        != sha256_file(metadata / "manifest" / "ucsc-hub.json")
        or validation.get("validation_scope") != expected_validation_scope
    ):
        raise ValueError("local hub validation does not match the rendered hub")

    authenticated_api = api or HfApi()
    repository = authenticated_api.repo_info(repository_id, repo_type="dataset")
    if getattr(repository, "private", True):
        raise RuntimeError("hub publication requires the existing public repository")
    if getattr(repository, "sha", None) != expected_base_revision:
        raise RuntimeError("public repository changed since the approved base revision")

    files = _publication_files(metadata)
    operations = [
        CommitOperationAdd(
            path_in_repo=path.relative_to(metadata).as_posix(),
            path_or_fileobj=path,
        )
        for path in files
    ]
    commit = authenticated_api.create_commit(
        repo_id=repository_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish validated multi-assembly UCSC track hub",
        parent_commit=expected_base_revision,
    )
    final_revision = getattr(commit, "oid", None)
    _validate_revision(final_revision, field="final_revision")
    published_files = [path.relative_to(metadata).as_posix() for path in files]
    publication = {
        "report_version": 1,
        "valid": False,
        "status": "published_pending_validation",
        "repository": repository_id,
        "public": True,
        "base_revision": expected_base_revision,
        "final_revision": final_revision,
        "single_commit": True,
        "single_process": True,
        "slurm_job_id": None,
        "metadata_only": metadata_only,
        "publication_approval": approval,
        "published_files": published_files,
    }
    atomic_write_json(Path(report_path), publication)
    try:
        public_validation = validator(
            metadata,
            revision=final_revision,
            udc_dir=udc_dir,
            repository_id=repository_id,
            metadata_only=metadata_only,
        )
        if public_validation.get("valid") is not True:
            raise RuntimeError("public hub validation returned an invalid result")
    except BaseException as error:
        publication.update(
            {
                "status": "published_validation_failed",
                "validation_error_type": type(error).__name__,
                "validation_error": str(error),
            }
        )
        atomic_write_json(Path(report_path), publication)
        raise RuntimeError(
            f"hub revision {final_revision} was published but post-validation "
            "failed; resume with validate-existing"
        ) from error
    publication.update(
        {
            "valid": public_validation.get("valid") is True,
            "status": "validated",
            "public_validation": public_validation,
        }
    )
    atomic_write_json(Path(report_path), publication)
    if success_marker is not None:
        _atomic_write_text(success_marker, f"{final_revision}\n")


def publish_dataset_card(
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    expected_base_revision: str,
    publication_approval: Mapping[str, Any] | None,
    success_marker_path: str | Path | None = None,
    repository_id: str = REPOSITORY_ID,
    api: Any | None = None,
    opener: Callable[..., Any] = urlopen,
    validator: Callable[..., dict[str, Any]] = validate_public_dataset_card,
) -> None:
    """Publish and validate only a generated dataset-card correction."""

    _validate_revision(expected_base_revision, field="expected_base_revision")
    metadata = Path(metadata_root)
    _validate_local_metadata(metadata)
    readme = metadata / "README.md"
    approval = _validated_publication_approval(
        publication_approval,
        expected_base_revision,
        operation="publish_dataset_card",
        candidate_sha256=sha256_file(readme),
    )
    if repository_id != REPOSITORY_ID:
        raise ValueError(f"hub repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("dataset-card publication must run outside Slurm")
    success_marker = (
        Path(success_marker_path) if success_marker_path is not None else None
    )
    if success_marker is not None:
        if success_marker.resolve() == Path(report_path).resolve():
            raise ValueError("success marker and publication report must differ")
        success_marker.unlink(missing_ok=True)

    authenticated_api = api or HfApi()
    repository = authenticated_api.repo_info(repository_id, repo_type="dataset")
    if getattr(repository, "private", True):
        raise RuntimeError("dataset-card publication requires a public repository")
    if getattr(repository, "sha", None) != expected_base_revision:
        raise RuntimeError("public repository changed since the approved base revision")
    remote_manifest = _public_metadata_bytes(
        repository_id,
        expected_base_revision,
        "manifest/ucsc-hub.json",
        opener=opener,
    )
    local_manifest = metadata / "manifest" / "ucsc-hub.json"
    if remote_manifest != local_manifest.read_bytes():
        raise RuntimeError(
            "public hub manifest differs from the dataset-card candidate"
        )

    commit = authenticated_api.create_commit(
        repo_id=repository_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme)
        ],
        commit_message="Update GPN-Star dataset card",
        parent_commit=expected_base_revision,
    )
    final_revision = getattr(commit, "oid", None)
    _validate_revision(final_revision, field="final_revision")
    publication = {
        "report_version": 1,
        "valid": False,
        "status": "published_pending_validation",
        "repository": repository_id,
        "public": True,
        "base_revision": expected_base_revision,
        "final_revision": final_revision,
        "single_commit": True,
        "single_process": True,
        "slurm_job_id": None,
        "publication_approval": approval,
        "published_files": ["README.md"],
    }
    atomic_write_json(Path(report_path), publication)
    try:
        public_validation = validator(
            metadata,
            revision=final_revision,
            repository_id=repository_id,
            opener=opener,
        )
        if public_validation.get("valid") is not True:
            raise RuntimeError("public dataset-card validation returned invalid")
    except BaseException as error:
        publication.update(
            {
                "status": "published_validation_failed",
                "validation_error_type": type(error).__name__,
                "validation_error": str(error),
            }
        )
        atomic_write_json(Path(report_path), publication)
        raise RuntimeError(
            f"dataset-card revision {final_revision} was published but validation "
            "failed; validate that exact revision before retrying publication"
        ) from error
    publication.update(
        {
            "valid": True,
            "status": "validated",
            "public_validation": public_validation,
        }
    )
    atomic_write_json(Path(report_path), publication)
    if success_marker is not None:
        _atomic_write_text(success_marker, f"{final_revision}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--release-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--artifact-revision", required=True)
    build.add_argument("--raw-llr-validation", type=Path)
    build.add_argument("--raw-llr-artifact-revision")
    build.add_argument("--source-revision")
    build.add_argument("--public-metadata-revision")
    build.add_argument("--contact-email", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--metadata-root", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--markdown", type=Path, required=True)
    validate.add_argument("--udc-dir", type=Path, required=True)
    validate.add_argument("--metadata-only", action="store_true")

    publish = commands.add_parser("publish")
    publish.add_argument("--metadata-root", type=Path, required=True)
    publish.add_argument("--validation-report", type=Path, required=True)
    publish.add_argument("--report", type=Path, required=True)
    publish.add_argument("--success-marker", type=Path)
    publish.add_argument("--expected-base-revision", required=True)
    publish.add_argument("--approval-approved", required=True)
    publish.add_argument("--approval-evidence-url", required=True)
    publish.add_argument("--approved-by", required=True)
    publish.add_argument("--approved-at", required=True)
    publish.add_argument("--approval-expected-base-revision", required=True)
    publish.add_argument("--approval-operation", required=True)
    publish.add_argument("--approval-candidate-sha256", required=True)
    publish.add_argument("--udc-dir", type=Path, required=True)
    publish.add_argument("--metadata-only", action="store_true")

    card = commands.add_parser("publish-card")
    card.add_argument("--metadata-root", type=Path, required=True)
    card.add_argument("--report", type=Path, required=True)
    card.add_argument("--success-marker", type=Path)
    card.add_argument("--expected-base-revision", required=True)
    card.add_argument("--approval-approved", required=True)
    card.add_argument("--approval-evidence-url", required=True)
    card.add_argument("--approved-by", required=True)
    card.add_argument("--approved-at", required=True)
    card.add_argument("--approval-expected-base-revision", required=True)
    card.add_argument("--approval-operation", required=True)
    card.add_argument("--approval-candidate-sha256", required=True)

    existing = commands.add_parser("validate-existing")
    existing.add_argument("--metadata-root", type=Path, required=True)
    existing.add_argument("--report", type=Path, required=True)
    existing.add_argument("--success-marker", type=Path, required=True)
    existing.add_argument("--expected-base-revision", required=True)
    existing.add_argument("--final-revision", required=True)
    existing.add_argument("--approval-approved", required=True)
    existing.add_argument("--approval-evidence-url", required=True)
    existing.add_argument("--approved-by", required=True)
    existing.add_argument("--approved-at", required=True)
    existing.add_argument("--approval-expected-base-revision", required=True)
    existing.add_argument("--approval-operation", required=True)
    existing.add_argument("--approval-candidate-sha256", required=True)
    existing.add_argument("--udc-dir", type=Path, required=True)
    existing.add_argument("--metadata-only", action="store_true")

    existing_card = commands.add_parser("validate-existing-card")
    existing_card.add_argument("--metadata-root", type=Path, required=True)
    existing_card.add_argument("--report", type=Path, required=True)
    existing_card.add_argument("--success-marker", type=Path, required=True)
    existing_card.add_argument("--expected-base-revision", required=True)
    existing_card.add_argument("--final-revision", required=True)
    existing_card.add_argument("--approval-approved", required=True)
    existing_card.add_argument("--approval-evidence-url", required=True)
    existing_card.add_argument("--approved-by", required=True)
    existing_card.add_argument("--approved-at", required=True)
    existing_card.add_argument("--approval-expected-base-revision", required=True)
    existing_card.add_argument("--approval-operation", required=True)
    existing_card.add_argument("--approval-candidate-sha256", required=True)
    return parser


def _approval_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "approved": args.approval_approved.lower() == "true",
        "evidence_url": args.approval_evidence_url,
        "approved_by": args.approved_by,
        "approved_at": args.approved_at,
        "expected_base_revision": args.approval_expected_base_revision,
        "operation": args.approval_operation,
        "candidate_sha256": args.approval_candidate_sha256,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        build_track_hub(
            args.release_manifest,
            args.output_dir,
            artifact_revision=args.artifact_revision,
            contact_email=args.contact_email,
            raw_llr_validation_path=args.raw_llr_validation,
            raw_llr_artifact_revision=args.raw_llr_artifact_revision,
            source_revision=args.source_revision,
            public_metadata_revision=args.public_metadata_revision,
        )
        return
    if args.command == "validate":
        validate_track_hub(
            args.metadata_root,
            args.report,
            args.markdown,
            udc_dir=args.udc_dir,
            metadata_only=args.metadata_only,
        )
        return
    if args.command == "publish":
        publish_track_hub(
            args.metadata_root,
            args.validation_report,
            args.report,
            expected_base_revision=args.expected_base_revision,
            publication_approval=_approval_from_args(args),
            udc_dir=args.udc_dir,
            metadata_only=args.metadata_only,
            success_marker_path=args.success_marker,
        )
        return
    if args.command == "publish-card":
        publish_dataset_card(
            args.metadata_root,
            args.report,
            expected_base_revision=args.expected_base_revision,
            publication_approval=_approval_from_args(args),
            success_marker_path=args.success_marker,
        )
        return
    if args.command == "validate-existing":
        validate_existing_track_hub_publication(
            args.metadata_root,
            args.report,
            expected_base_revision=args.expected_base_revision,
            final_revision=args.final_revision,
            publication_approval=_approval_from_args(args),
            udc_dir=args.udc_dir,
            metadata_only=args.metadata_only,
            success_marker_path=args.success_marker,
        )
        return
    if args.command == "validate-existing-card":
        validate_existing_dataset_card_publication(
            args.metadata_root,
            args.report,
            expected_base_revision=args.expected_base_revision,
            final_revision=args.final_revision,
            publication_approval=_approval_from_args(args),
            success_marker_path=args.success_marker,
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
