from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from gpn_star_scores.catalog import ASSEMBLIES, SCORE_SETS
from gpn_star_scores.hub import (
    HUB_APPROVAL_ISSUE,
    HUB_ASSEMBLY_ORDER,
    HUB_QA_APPROVAL_ISSUE,
    HUB_TRACK_DB_DIRECTORY_ORDER,
    NUCLEOTIDE_COLORS,
    RAW_LLR_NEGATIVE_COLOR,
    RAW_LLR_POSITIVE_COLOR,
    browser_launch_links,
    build_track_hub,
    dataset_card_launch_links,
    hub_database_name,
    publication_candidate_sha256,
    publish_dataset_card,
    publish_track_hub,
    raw_llr_validation_links,
    validate_existing_dataset_card_publication,
    validate_existing_track_hub_publication,
    validate_public_dataset_card,
    validate_public_track_hub,
    validate_track_hub,
)
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.raw_llr import RAW_LLR_BASES, RAW_LLR_TRACKS
from gpn_star_scores.release import (
    DEFAULT_DATASET_CONFIG,
    REPOSITORY_ID,
    TRACK_HUB_URL,
    dataset_configs,
)
from gpn_star_scores.tracks import TRACKS, ucsc_assembly_name

REPOSITORY_ROOT = Path(__file__).parents[1]
ARTIFACT_REVISION = "a" * 40
RAW_LLR_ARTIFACT_REVISION = "c" * 40
SOURCE_REVISION = "d" * 40
PUBLIC_METADATA_REVISION = "e" * 40


def _release_manifest() -> dict[str, object]:
    files = []
    for score_set in SCORE_SETS:
        for track in TRACKS:
            files.append(
                {
                    "path": f"bigwig/{score_set.name}/{track}.bw",
                    "score_set": score_set.name,
                    "assembly": score_set.assembly,
                    "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
                    "track": track,
                    "size": 100,
                    "sha256": "1" * 64,
                    "bases_covered": 10,
                    "zoom_levels": 2,
                }
            )
    return {
        "release_manifest_version": 1,
        "repository": {
            "id": REPOSITORY_ID,
            "repo_type": "dataset",
            "public": True,
            "license": "apache-2.0",
        },
        "paper": {"title": "paper", "doi": "doi"},
        "source_inventory": {},
        "parquet": {"files": [{"rows": 3}]},
        "bigwig": {
            "file_count": len(files),
            "total_bytes": 100 * len(files),
            "value_decimals": 3,
            "files": files,
        },
        "dataset_configs": dataset_configs(),
        "validation": {
            "bigwig_validation_passed": True,
            "expected_bigwig_files": len(files),
        },
    }


def _raw_llr_validation() -> dict[str, object]:
    tracks = []
    for score_set in SCORE_SETS:
        for track in RAW_LLR_TRACKS:
            tracks.append(
                {
                    "score_set": score_set.name,
                    "assembly": score_set.assembly,
                    "ucsc_assembly": ucsc_assembly_name(score_set.assembly),
                    "track": track,
                    "base": RAW_LLR_BASES[track],
                    "path": f"bigwig/{score_set.name}/{track}.bw",
                    "size": 50,
                    "sha256": "2" * 64,
                    "bases_covered": 10,
                    "zoom_levels": 2,
                }
            )
    return {
        "report_version": 1,
        "product": "raw_calibrated_llr",
        "valid": True,
        "track_count": len(tracks),
        "value_decimals": 3,
        "reference_zero_baseline": True,
        "abs_llr_calibrated_used": False,
        "tracks": tracks,
    }


def _build_metadata(tmp_path: Path, *, include_raw_llr: bool = False) -> Path:
    release_manifest = tmp_path / "release.json"
    release_manifest.write_text(json.dumps(_release_manifest()), encoding="utf-8")
    raw_llr_validation = None
    if include_raw_llr:
        raw_llr_validation = tmp_path / "raw-llr-validation.json"
        raw_llr_validation.write_text(
            json.dumps(_raw_llr_validation()), encoding="utf-8"
        )
    metadata = tmp_path / "metadata"
    build_track_hub(
        release_manifest,
        metadata,
        artifact_revision=ARTIFACT_REVISION,
        contact_email="maintainer@example.org",
        raw_llr_validation_path=raw_llr_validation,
        raw_llr_artifact_revision=(
            RAW_LLR_ARTIFACT_REVISION if include_raw_llr else None
        ),
        source_revision=SOURCE_REVISION if include_raw_llr else None,
        public_metadata_revision=(
            PUBLIC_METADATA_REVISION if include_raw_llr else None
        ),
    )
    return metadata


def _write_validation_report(metadata: Path, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "valid": True,
                "artifact_revision": ARTIFACT_REVISION,
                "validation_scope": "legacy_v1_tracks",
                "hub_manifest_sha256": sha256_file(
                    metadata / "manifest" / "ucsc-hub.json"
                ),
            }
        )
    )


def _publication_approval(
    metadata: Path,
    evidence_url: str = HUB_APPROVAL_ISSUE,
    operation: str = "publish_hub",
) -> dict[str, object]:
    candidate = (
        publication_candidate_sha256(metadata)
        if operation == "publish_hub"
        else sha256_file(metadata / "README.md")
    )
    return {
        "approved": True,
        "evidence_url": evidence_url,
        "approved_by": "author",
        "approved_at": "2026-07-22",
        "expected_base_revision": ARTIFACT_REVISION,
        "operation": operation,
        "candidate_sha256": candidate,
    }


def _successful_public_validation() -> dict[str, object]:
    return {
        "valid": True,
        "repository": REPOSITORY_ID,
        "revision": "b" * 40,
        "credentials_sent": False,
    }


def _write_pending_publication_report(metadata: Path, path: Path) -> None:
    publication_files = [
        metadata / "README.md",
        metadata / "manifest" / "ucsc-hub.json",
        *sorted(item for item in (metadata / "ucsc").rglob("*") if item.is_file()),
    ]
    path.write_text(
        json.dumps(
            {
                "report_version": 1,
                "valid": False,
                "status": "published_validation_failed",
                "repository": REPOSITORY_ID,
                "public": True,
                "base_revision": ARTIFACT_REVISION,
                "final_revision": "b" * 40,
                "single_commit": True,
                "single_process": True,
                "slurm_job_id": None,
                "metadata_only": False,
                "publication_approval": _publication_approval(metadata),
                "published_files": [
                    item.relative_to(metadata).as_posix() for item in publication_files
                ],
                "validation_error_type": "FileNotFoundError",
                "validation_error": "hubCheck",
            }
        ),
        encoding="utf-8",
    )


def test_launch_links_preserve_ucsc_defaults_and_show_one_model() -> None:
    links = dataset_card_launch_links()
    assert len(links) == len(SCORE_SETS)
    for link in links:
        query = parse_qs(urlparse(link["url"]).query)
        assert query["ignoreCookie"] == ["1"]
        assert "hideTracks" not in query
        assert "position" not in query

        shown_groups = [key for key, value in query.items() if value == ["show"]]
        assert len(shown_groups) == 1
        group = shown_groups[0]
        assert query[f"{group}Entropy"] == ["full"]
        assert query[f"{group}Logo"] == ["full"]
        assert query[f"{group}RawLlr"] == ["full"]

        hidden_groups = [key for key, value in query.items() if value == ["hide"]]
        expected_hidden = 2 if link["ucsc_assembly"] == "hg38" else 0
        assert len(hidden_groups) == expected_hidden


def test_builds_one_multi_assembly_hub_with_entropy_and_logo(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)

    hub = (metadata / "ucsc" / "hub.txt").read_text()
    genomes = (metadata / "ucsc" / "genomes.txt").read_text()
    assert "genomesFile genomes.txt" in hub
    assert "useOneFile" not in hub
    assert ("genome GCF_000001735.4\ntrackDb araTha1/trackDb.txt") in genomes
    assert [
        line.removeprefix("genome ")
        for line in genomes.splitlines()
        if line.startswith("genome ")
    ] == list(HUB_ASSEMBLY_ORDER)

    hg38 = (metadata / "ucsc" / "hg38" / "trackDb.txt").read_text()
    assert hg38.count("superTrack on show") == 3
    assert hg38.count("container multiWig") == 3
    assert hg38.count("graphTypeDefault bar") == 3
    assert hg38.count("logo on") == 3
    assert hg38.count("visibility dense") == 3
    assert hg38.count("visibility full") == 3
    assert hg38.count("bigDataUrl ") == 15
    assert "shortLabel GPN-Star (V)" in hg38
    assert "shortLabel GPN-Star (M)" in hg38
    assert "shortLabel GPN-Star (P)" in hg38
    assert "noInherit" not in hg38
    for color in NUCLEOTIDE_COLORS.values():
        assert hg38.count(f"color {color}") == 3

    for directory in HUB_TRACK_DB_DIRECTORY_ORDER[1:]:
        track_db = (metadata / "ucsc" / directory / "trackDb.txt").read_text()
        assert track_db.count("superTrack on show") == 1
        assert track_db.count("container multiWig") == 1
        assert track_db.count("graphTypeDefault bar") == 1
        assert track_db.count("logo on") == 1
        assert track_db.count("bigDataUrl ") == 5
        assert "shortLabel GPN-Star\n" in track_db

    assert f"/resolve/{ARTIFACT_REVISION}/bigwig/" in hg38
    assert "llr_A" not in hg38
    readme = (metadata / "README.md").read_text()
    assert TRACK_HUB_URL in readme
    assert "one-dimensional\nentropy signal" in readme
    assert readme.count("Open in UCSC") == 8

    links = browser_launch_links()
    assert len(links) == 8
    for link in links:
        query = parse_qs(urlparse(link["url"]).query)
        assert query["db"] == [link["ucsc_assembly"]]
        assert query["hubUrl"] == [TRACK_HUB_URL]
        assert query["hideTracks"] == ["1"]
        assert query["ignoreCookie"] == ["1"]
        group = next(key for key, value in query.items() if value == ["show"])
        assert query[f"{group}Entropy"] == ["dense"]
        assert query[f"{group}Logo"] == ["full"]

    hub_manifest = json.loads((metadata / "manifest" / "ucsc-hub.json").read_text())
    assert hub_manifest["assembly_count"] == 6
    assert hub_manifest["score_set_count"] == 8
    assert hub_manifest["track_count"] == 40
    assert len(hub_manifest["files"]) == 33
    assert all(score_set["browser_url"] for score_set in hub_manifest["score_sets"])
    tair10 = next(
        score_set
        for score_set in hub_manifest["score_sets"]
        if score_set["name"] == "tair10"
    )
    assert tair10["ucsc_assembly"] == "GCF_000001735.4"
    assert "db=GCF_000001735.4" in tair10["browser_url"]
    tair10_tracks = [
        track for track in hub_manifest["tracks"] if track["score_set"] == "tair10"
    ]
    assert {track["source_ucsc_assembly"] for track in tair10_tracks} == {"araTha1"}
    assert {track["ucsc_assembly"] for track in tair10_tracks} == {"GCF_000001735.4"}
    assert {track["url"] for track in hub_manifest["tracks"]} == {
        f"https://huggingface.co/datasets/{REPOSITORY_ID}/resolve/"
        f"{ARTIFACT_REVISION}/bigwig/{score_set.name}/{track}.bw"
        for score_set in SCORE_SETS
        for track in TRACKS
    }


def test_builds_cadd_inspired_signed_raw_llr_extension(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path, include_raw_llr=True)

    hg38 = (metadata / "ucsc" / "hg38" / "trackDb.txt").read_text()
    assert hg38.count("compositeTrack on") == 3
    assert hg38.count("visibility dense") == 3
    assert hg38.count("visibility full") == 18
    assert hg38.count("autoScale group") == 3
    assert hg38.count("alwaysZero on") == 3
    assert hg38.count("yLineMark 0") == 3
    assert hg38.count("windowingFunction mean+whiskers") == 3
    assert "mouseOverFunction" not in hg38
    assert hg38.count(f"color {RAW_LLR_POSITIVE_COLOR}") == 15
    assert hg38.count(f"altColor {RAW_LLR_NEGATIVE_COLOR}") == 12
    assert hg38.count("bigDataUrl ") == 27
    assert f"/resolve/{ARTIFACT_REVISION}/bigwig/" in hg38
    assert f"/resolve/{RAW_LLR_ARTIFACT_REVISION}/bigwig/" in hg38

    hub_manifest = json.loads((metadata / "manifest" / "ucsc-hub.json").read_text())
    assert hub_manifest["hub_manifest_version"] == 2
    assert hub_manifest["track_count"] == 72
    assert hub_manifest["raw_llr_artifact_revision"] == RAW_LLR_ARTIFACT_REVISION
    validation_links = {link["score_set"]: link for link in raw_llr_validation_links()}
    for score_set in hub_manifest["score_sets"]:
        url = score_set["raw_llr_validation_url"]
        assert url == validation_links[score_set["name"]]["url"]
        query = parse_qs(urlparse(url).query)
        group = next(key for key, value in query.items() if value == ["show"])
        assert query[f"{group}Entropy"] == ["hide"]
        assert query[f"{group}Logo"] == ["hide"]
        assert query[f"{group}RawLlr"] == ["full"]
    assert len(hub_manifest["validation_scope_tracks"]) == 32
    assert all(
        any(track in item for track in RAW_LLR_TRACKS)
        for item in hub_manifest["validation_scope_tracks"]
    )
    assert (
        metadata / "manifest" / "raw-llr-validation.json"
    ).read_text() == json.dumps(_raw_llr_validation())
    release_manifest = json.loads((metadata / "manifest" / "release.json").read_text())
    assert release_manifest["release_manifest_version"] == 2
    assert release_manifest["bigwig"]["file_count"] == 72
    assert release_manifest["bigwig"]["total_bytes"] == (40 * 100) + (32 * 50)
    assert release_manifest["validation"]["bigwig_validation_scope"] == (
        "new_raw_llr_tracks_only"
    )
    assert release_manifest["validation"]["existing_v1_bigwigs_revalidated"] is False
    provenance = release_manifest["bigwig"]["artifact_revisions"]
    assert provenance["v1"] == {
        "revision": ARTIFACT_REVISION,
        "track_count": 40,
        "validation_source": "trusted_v1_release_manifest",
        "revalidated": False,
    }
    assert provenance["raw_calibrated_llr"] == {
        "revision": RAW_LLR_ARTIFACT_REVISION,
        "track_count": 32,
        "validation_source": "manifest/raw-llr-validation.json",
        "revalidated": True,
    }
    release_records = {
        (record["score_set"], record["track"]): record
        for record in release_manifest["bigwig"]["files"]
    }
    assert all(
        release_records[(score_set.name, track)]["artifact_revision"]
        == ARTIFACT_REVISION
        for score_set in SCORE_SETS
        for track in TRACKS
    )
    assert all(
        release_records[(score_set.name, track)]["artifact_revision"]
        == RAW_LLR_ARTIFACT_REVISION
        for score_set in SCORE_SETS
        for track in RAW_LLR_TRACKS
    )

    tair10 = metadata / "ucsc" / "araTha1"
    raw_description = next(tair10.glob("*RawLlr.html")).read_text()
    assert "reference allele is assigned the explicit zero" in raw_description
    assert "Positive LLR is blue and negative LLR is red" in raw_description
    readme = (metadata / "README.md").read_text()
    assert "four signed A/C/G/T LLR tracks" in readme
    assert (
        """**`llr_calibrated`:** More negative = more constrained or larger effect.

**`abs_llr_calibrated`:** Magnitude of the variant's effect relative to a neutral substitution.
Positive = larger effect than neutral; negative = smaller. Useful when the direction of effect
is not relevant.

**Example:**
```
chrom     pos ref alt  llr_calibrated  abs_llr_calibrated
   21 5010065   T   A          -1.774               1.774
   21 5010065   T   C          -1.550               1.550
   21 5010065   T   G          -1.670               1.670
```"""
        in readme
    )
    assert (
        """**Interpretation:** ~1.0 = neutral; <1.0 = constrained; the lower the more constrained.

**Example:**
```
chrom     pos ref  entropy_calibrated
   21 5010065   T               0.486
   21 5010066   A               0.644
   21 5010067   A               0.591
```"""
        in readme
    )
    assert "72 BigWigs" in readme
    assert ARTIFACT_REVISION in readme
    assert RAW_LLR_ARTIFACT_REVISION in readme
    assert SOURCE_REVISION in readme
    assert PUBLIC_METADATA_REVISION in readme
    assert "@@" not in readme
    assert "Which product to use" not in readme
    assert "curated session" not in readme.lower()
    assert "screenshot" not in readme.lower()
    assert "Local review candidate" not in readme
    metadata_frontmatter = readme.split("---", maxsplit=2)[1]
    assert metadata_frontmatter.count("default: true") == 1
    assert (
        f"config_name: {DEFAULT_DATASET_CONFIG}\n"
        "    data_files:\n"
        "      - split: train\n"
        "        path: data/gpn-star-hg38-m447-200m/llr/*.parquet\n"
        "    default: true"
    ) in metadata_frontmatter
    species_order = (
        "gpn-star-hg38-v100-200m",
        "gpn-star-hg38-m447-200m",
        "gpn-star-hg38-p243-200m",
        "mm39",
        "gg6",
        "dm6",
        "ce11",
        "tair10",
    )
    table_offsets = [
        readme.index(f"| `{score_set}` |", readme.index("### Score sets"))
        for score_set in species_order
    ]
    assert table_offsets == sorted(table_offsets)
    hub_description = (metadata / "ucsc" / "description.html").read_text()
    assert "original entropy and sequence-logo data" in hub_description
    group_description = (
        metadata / "ucsc" / "hg38" / "gpnStarHg38V100200m.html"
    ).read_text()
    assert "entropy and sequence-logo browser tracks" in group_description
    assert ARTIFACT_REVISION in group_description
    assert RAW_LLR_ARTIFACT_REVISION in group_description
    description_files = [
        metadata / "ucsc" / "description.html",
        *sorted((metadata / "ucsc").glob("*/*.html")),
    ]
    user_facing_metadata = [
        *(description_file.read_text() for description_file in description_files),
        *(
            track_db.read_text()
            for track_db in sorted((metadata / "ucsc").glob("*/trackDb.txt"))
        ),
    ]
    for text in user_facing_metadata:
        assert re.search(r"\b(?:raw|calibrated)\b", text, re.IGNORECASE) is None


def test_build_rejects_output_that_contains_source_release_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release" / "metadata"
    release_manifest = output / "manifest" / "release.json"
    release_manifest.parent.mkdir(parents=True)
    original = json.dumps(_release_manifest())
    release_manifest.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain"):
        build_track_hub(
            release_manifest,
            output,
            artifact_revision=ARTIFACT_REVISION,
            contact_email="maintainer@example.org",
        )

    assert release_manifest.read_text(encoding="utf-8") == original


def test_track_descriptions_preserve_score_and_coordinate_semantics(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    track_db = (metadata / "ucsc" / "araTha1" / "trackDb.txt").read_text()
    html_files = {
        path.name: path.read_text()
        for path in (metadata / "ucsc" / "araTha1").glob("*.html")
    }

    assert "genome GCF_000001735.4" in (metadata / "ucsc" / "genomes.txt").read_text()
    assert hub_database_name("tair10") == "GCF_000001735.4"
    assert "shortLabel GPN-Star\n" in track_db
    assert len(html_files) == 3
    entropy = next(value for name, value in html_files.items() if "Entropy" in name)
    logo = next(value for name, value in html_files.items() if "Logo" in name)
    assert "GPN-Star entropy values" in entropy
    assert "does not add an unreviewed biological directionality" in entropy
    assert "zero-based,\nhalf-open one-base BigWig intervals" in entropy
    assert "p(base) * (2 - H)" in logo
    assert "not a model\nprobability" in logo


class _FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status = status
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.content


def _score_set_from_url(url: str):
    return next(score_set for score_set in SCORE_SETS if f"/{score_set.name}/" in url)


def _fake_runner(
    command: list[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    executable = Path(command[0]).name
    if executable == "hubCheck":
        return subprocess.CompletedProcess(command, 0, "hub is valid\n", "")
    if executable == "bigWigInfo":
        score_set = _score_set_from_url(command[-1])
        output = "\n".join(
            f"chr{chrom} {index} 1000"
            for index, chrom in enumerate(ASSEMBLIES[score_set.assembly].chromosomes)
        )
        return subprocess.CompletedProcess(command, 0, output + "\n", "")
    if executable == "bigWigSummary":
        bins = int(command[-1])
        return subprocess.CompletedProcess(
            command, 0, "\t".join("0.5" for _ in range(bins)) + "\n", ""
        )
    raise AssertionError(command)


def test_validation_checks_hub_ranges_chromosomes_and_zoom_values(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(request: object, **kwargs: object) -> _FakeResponse:
        assert getattr(request, "get_header")("Range") == "bytes=0-63"
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    report_path = tmp_path / "validation.json"
    markdown_path = tmp_path / "validation.md"
    validate_track_hub(
        metadata,
        report_path,
        markdown_path,
        udc_dir=tmp_path / "udc",
        runner=_fake_runner,
        opener=opener,
    )

    report = json.loads(report_path.read_text())
    assert report["valid"] is True
    assert report["hub_check"]["passed"] is True
    assert report["hub_check"]["settings_spec"].startswith("https://")
    assert report["hub_manifest_sha256"] == sha256_file(
        metadata / "manifest" / "ucsc-hub.json"
    )
    assert report["assembly_count"] == 6
    assert report["score_set_count"] == 8
    assert report["track_count"] == 40
    assert report["http_range_count"] == 40
    assert len(report["chromosome_checks"]) == 40
    assert len(report["representative_checks"]) == 8
    assert all(
        set(check["track_values"]) == set(TRACKS)
        and check["zero_based_half_open"] is True
        for check in report["representative_checks"]
    )
    assert "All 40 BigWig URLs" in markdown_path.read_text()


def test_extended_hub_validation_does_not_repeat_v1_bigwig_checks(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path, include_raw_llr=True)
    requested_urls = []

    def opener(request: object, **kwargs: object) -> _FakeResponse:
        url = getattr(request, "full_url")
        requested_urls.append(url)
        assert f"/resolve/{RAW_LLR_ARTIFACT_REVISION}/" in url
        assert any(url.endswith(f"/{track}.bw") for track in RAW_LLR_TRACKS)
        return _FakeResponse(
            b"x" * 50,
            status=206,
            headers={"Content-Range": "bytes 0-49/50"},
        )

    report_path = tmp_path / "validation.json"
    markdown_path = tmp_path / "validation.md"
    validate_track_hub(
        metadata,
        report_path,
        markdown_path,
        udc_dir=tmp_path / "udc",
        runner=_fake_runner,
        opener=opener,
    )

    report = json.loads(report_path.read_text())
    assert report["validation_scope"] == "new_raw_llr_tracks_only"
    assert report["existing_v1_bigwigs_revalidated"] is False
    assert report["hub_check"]["remote_tracks_checked"] is False
    assert report["track_count"] == 32
    assert report["http_range_count"] == 32
    assert len(requested_urls) == 32
    assert all(
        set(check["track_values"]) == set(RAW_LLR_TRACKS)
        for check in report["representative_checks"]
    )
    markdown = markdown_path.read_text()
    assert "All 32 BigWig URLs" in markdown
    assert "40 immutable v1 BigWigs were not revalidated" in markdown


def test_metadata_only_validation_skips_every_bigwig_request(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path, include_raw_llr=True)

    def opener(*args: object, **kwargs: object) -> _FakeResponse:
        raise AssertionError("metadata-only validation must not request a BigWig")

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert Path(command[0]).name == "hubCheck"
        assert "-noTracks" in command
        return subprocess.CompletedProcess(command, 0, "hub is valid\n", "")

    report_path = tmp_path / "validation.json"
    markdown_path = tmp_path / "validation.md"
    validate_track_hub(
        metadata,
        report_path,
        markdown_path,
        udc_dir=tmp_path / "udc",
        metadata_only=True,
        runner=runner,
        opener=opener,
    )

    report = json.loads(report_path.read_text())
    assert report["valid"] is True
    assert report["validation_scope"] == "hub_metadata_only"
    assert report["track_count"] == 0
    assert report["http_range_count"] == 0
    assert report["chromosome_checks"] == []
    assert report["representative_checks"] == []
    assert report["existing_v1_bigwigs_revalidated"] is False
    assert report["existing_raw_llr_bigwigs_revalidated"] is False
    assert report["prior_artifact_validation_reused"] is True
    markdown = markdown_path.read_text()
    assert (
        "No BigWig ranges, headers, bases, or zoom summaries were requested" in markdown
    )


def test_validation_rejects_cross_track_chromosome_length_mismatch(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(request: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = _fake_runner(command, **kwargs)
        if Path(command[0]).name == "bigWigInfo" and command[-1].endswith("/A.bw"):
            return subprocess.CompletedProcess(
                command,
                0,
                result.stdout.replace(" 0 1000", " 0 999", 1),
                "",
            )
        return result

    with pytest.raises(RuntimeError, match="chromosome sizes differ"):
        validate_track_hub(
            metadata,
            tmp_path / "validation.json",
            tmp_path / "validation.md",
            udc_dir=tmp_path / "udc",
            runner=runner,
            opener=opener,
        )


class _FakeApi:
    def __init__(self) -> None:
        self.commit_kwargs: dict[str, object] | None = None

    def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(private=False, sha=ARTIFACT_REVISION)

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.commit_kwargs = kwargs
        return SimpleNamespace(oid="b" * 40)


def test_public_validation_checks_published_files_and_remote_hub(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(target: object, **kwargs: object) -> _FakeResponse:
        if isinstance(target, str):
            marker = f"/resolve/{ARTIFACT_REVISION}/"
            relative_path = target.split(marker, maxsplit=1)[1]
            return _FakeResponse((metadata / relative_path).read_bytes())
        assert getattr(target, "get_header")("Range") == "bytes=0-63"
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    report = validate_public_track_hub(
        metadata,
        revision=ARTIFACT_REVISION,
        udc_dir=tmp_path / "udc",
        api=_FakeApi(),
        opener=opener,
        runner=_fake_runner,
    )

    assert report["valid"] is True
    assert report["credentials_sent"] is False
    assert report["revision"] == ARTIFACT_REVISION
    assert report["file_count"] == 35
    assert report["hub_validation"]["hub_target"].endswith("/ucsc/hub.txt")


def test_public_metadata_only_validation_never_requests_a_bigwig(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path, include_raw_llr=True)

    def opener(target: object, **kwargs: object) -> _FakeResponse:
        assert isinstance(target, str), "BigWig requests use Request objects"
        marker = f"/resolve/{ARTIFACT_REVISION}/"
        relative_path = target.split(marker, maxsplit=1)[1]
        return _FakeResponse((metadata / relative_path).read_bytes())

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert Path(command[0]).name == "hubCheck"
        assert "-noTracks" in command
        return subprocess.CompletedProcess(command, 0, "hub is valid\n", "")

    report = validate_public_track_hub(
        metadata,
        revision=ARTIFACT_REVISION,
        udc_dir=tmp_path / "udc",
        metadata_only=True,
        api=_FakeApi(),
        opener=opener,
        runner=runner,
    )

    assert report["valid"] is True
    assert report["file_count"] == 45
    assert report["hub_validation"]["validation_scope"] == "hub_metadata_only"
    assert report["hub_validation"]["http_range_count"] == 0


def test_public_dataset_card_validation_skips_bigwigs(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)

    def opener(target: str, **kwargs: object) -> _FakeResponse:
        marker = f"/resolve/{ARTIFACT_REVISION}/"
        if marker in target:
            relative_path = target.split(marker, maxsplit=1)[1]
            return _FakeResponse((metadata / relative_path).read_bytes())
        return _FakeResponse(
            f"<html>{REPOSITORY_ID} GPN-Star genome-wide scores</html>".encode()
        )

    report = validate_public_dataset_card(
        metadata,
        revision=ARTIFACT_REVISION,
        api=_FakeApi(),
        opener=opener,
    )

    assert report["valid"] is True
    assert report["bigwig_checks_performed"] == 0
    assert [item["path"] for item in report["file_checks"]] == [
        "README.md",
        "manifest/ucsc-hub.json",
    ]


def test_dataset_card_publication_commits_only_readme(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    api = _FakeApi()
    report = tmp_path / "dataset-card-publication.json"
    success = tmp_path / "dataset-card-publication.complete"

    def opener(target: str, **kwargs: object) -> _FakeResponse:
        assert target.endswith("/manifest/ucsc-hub.json")
        return _FakeResponse((metadata / "manifest" / "ucsc-hub.json").read_bytes())

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "valid": True,
            "repository": REPOSITORY_ID,
            "revision": "b" * 40,
            "credentials_sent": False,
            "bigwig_checks_performed": 0,
        }

    publish_dataset_card(
        metadata,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        publication_approval=_publication_approval(
            metadata,
            evidence_url=HUB_QA_APPROVAL_ISSUE,
            operation="publish_dataset_card",
        ),
        success_marker_path=success,
        api=api,
        opener=opener,
        validator=validator,
    )

    assert api.commit_kwargs is not None
    assert api.commit_kwargs["parent_commit"] == ARTIFACT_REVISION
    operations = api.commit_kwargs["operations"]
    assert [operation.path_in_repo for operation in operations] == ["README.md"]
    assert json.loads(report.read_text())["status"] == "validated"
    assert success.read_text() == f"{'b' * 40}\n"


def test_publication_is_one_approval_gated_commit(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    api = _FakeApi()
    validator_calls = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        validator_calls.append((args, kwargs))
        return _successful_public_validation()

    report = tmp_path / "publication.json"
    success_marker = tmp_path / "publication.complete"
    approval = _publication_approval(metadata)
    publish_track_hub(
        metadata,
        validation,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        publication_approval=approval,
        udc_dir=tmp_path / "udc",
        success_marker_path=success_marker,
        api=api,
        validator=validator,
    )

    assert api.commit_kwargs is not None
    assert api.commit_kwargs["parent_commit"] == ARTIFACT_REVISION
    operations = api.commit_kwargs["operations"]
    paths = {operation.path_in_repo for operation in operations}
    assert "README.md" in paths
    assert "manifest/ucsc-hub.json" in paths
    assert "ucsc/hub.txt" in paths
    assert "ucsc/genomes.txt" in paths
    assert len(validator_calls) == 1
    publication = json.loads(report.read_text())
    assert publication["valid"] is True
    assert publication["status"] == "validated"
    assert publication["base_revision"] == ARTIFACT_REVISION
    assert publication["final_revision"] == "b" * 40
    assert publication["single_commit"] is True
    assert success_marker.read_text() == f"{'b' * 40}\n"


def test_metadata_only_publication_requires_and_preserves_its_scope(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path, include_raw_llr=True)
    validation = tmp_path / "validation.json"
    validation_markdown = tmp_path / "validation.md"

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert Path(command[0]).name == "hubCheck"
        return subprocess.CompletedProcess(command, 0, "hub is valid\n", "")

    validate_track_hub(
        metadata,
        validation,
        validation_markdown,
        udc_dir=tmp_path / "udc",
        metadata_only=True,
        runner=runner,
        opener=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("BigWig request")
        ),
    )
    with pytest.raises(ValueError, match="local hub validation does not match"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "wrong-scope-publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=_publication_approval(metadata),
            udc_dir=tmp_path / "udc",
            metadata_only=False,
            api=_FakeApi(),
        )
    validator_calls: list[dict[str, object]] = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        validator_calls.append(kwargs)
        return _successful_public_validation()

    report_path = tmp_path / "publication.json"
    publish_track_hub(
        metadata,
        validation,
        report_path,
        expected_base_revision=ARTIFACT_REVISION,
        publication_approval=_publication_approval(metadata),
        udc_dir=tmp_path / "udc",
        metadata_only=True,
        api=_FakeApi(),
        validator=validator,
    )

    assert validator_calls[0]["metadata_only"] is True
    publication = json.loads(report_path.read_text())
    assert publication["metadata_only"] is True


def test_readme_approval_cannot_publish_the_full_hub(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)

    with pytest.raises(ValueError, match="incomplete or mismatched"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=_publication_approval(
                metadata,
                evidence_url=HUB_QA_APPROVAL_ISSUE,
                operation="publish_dataset_card",
            ),
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
        )


def test_full_hub_approval_rejects_post_approval_readme_change(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    approval = _publication_approval(metadata)
    with (metadata / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\npost-approval change\n")

    with pytest.raises(ValueError, match="incomplete or mismatched"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=approval,
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
        )


def test_dataset_card_publication_recovers_without_second_commit(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    approval = _publication_approval(
        metadata,
        evidence_url=HUB_QA_APPROVAL_ISSUE,
        operation="publish_dataset_card",
    )
    report = tmp_path / "dataset-card-publication.json"
    success = tmp_path / "dataset-card-publication.complete"
    api = _FakeApi()

    def opener(target: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse((metadata / "manifest" / "ucsc-hub.json").read_bytes())

    def failed_validator(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("render pending")

    with pytest.raises(RuntimeError, match="was published but validation failed"):
        publish_dataset_card(
            metadata,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=approval,
            success_marker_path=success,
            api=api,
            opener=opener,
            validator=failed_validator,
        )
    assert api.commit_kwargs is not None
    assert not success.exists()

    def successful_validator(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "valid": True,
            "repository": REPOSITORY_ID,
            "revision": "b" * 40,
            "credentials_sent": False,
            "bigwig_checks_performed": 0,
        }

    validate_existing_dataset_card_publication(
        metadata,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        final_revision="b" * 40,
        publication_approval=approval,
        success_marker_path=success,
        validator=successful_validator,
    )

    recovered = json.loads(report.read_text())
    assert recovered["status"] == "validated_existing_publication"
    assert recovered["recovered_from_status"] == "published_validation_failed"
    assert success.read_text() == f"{'b' * 40}\n"


def test_publication_failure_preserves_created_revision_for_recovery(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        raise FileNotFoundError("hubCheck")

    report = tmp_path / "publication.json"
    success_marker = tmp_path / "publication.complete"
    success_marker.write_text("stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"hub revision {'b' * 40}"):
        publish_track_hub(
            metadata,
            validation,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=_publication_approval(metadata),
            udc_dir=tmp_path / "udc",
            success_marker_path=success_marker,
            api=_FakeApi(),
            validator=validator,
        )

    publication = json.loads(report.read_text())
    assert publication["valid"] is False
    assert publication["status"] == "published_validation_failed"
    assert publication["final_revision"] == "b" * 40
    assert publication["validation_error_type"] == "FileNotFoundError"
    assert not success_marker.exists()


def test_publication_rejects_validator_invalid_result(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        return {"valid": False, "credentials_sent": False}

    report = tmp_path / "publication.json"
    with pytest.raises(RuntimeError, match="published but post-validation failed"):
        publish_track_hub(
            metadata,
            validation,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=_publication_approval(metadata),
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
            validator=validator,
        )

    publication = json.loads(report.read_text())
    assert publication["valid"] is False
    assert publication["status"] == "published_validation_failed"
    assert publication["validation_error_type"] == "RuntimeError"


def test_existing_publication_validation_recovers_without_a_commit(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    calls = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return _successful_public_validation()

    report = tmp_path / "publication.json"
    success_marker = tmp_path / "publication.complete"
    _write_pending_publication_report(metadata, report)
    validate_existing_track_hub_publication(
        metadata,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        final_revision="b" * 40,
        publication_approval=_publication_approval(metadata),
        udc_dir=tmp_path / "udc",
        success_marker_path=success_marker,
        validator=validator,
    )

    assert len(calls) == 1
    publication = json.loads(report.read_text())
    assert publication["valid"] is True
    assert publication["status"] == "validated_existing_publication"
    assert publication["final_revision"] == "b" * 40
    assert len(publication["published_files"]) == 35
    assert publication["recovered_from_status"] == "published_validation_failed"
    assert "validation_error" not in publication
    assert success_marker.read_text() == f"{'b' * 40}\n"


def test_existing_publication_rejects_validator_invalid_result(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    report = tmp_path / "publication.json"
    success_marker = tmp_path / "publication.complete"
    _write_pending_publication_report(metadata, report)

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        return {"valid": False, "credentials_sent": False}

    with pytest.raises(RuntimeError, match="returned an invalid result"):
        validate_existing_track_hub_publication(
            metadata,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            final_revision="b" * 40,
            publication_approval=_publication_approval(metadata),
            udc_dir=tmp_path / "udc",
            success_marker_path=success_marker,
            validator=validator,
        )
    assert not success_marker.exists()


def test_existing_publication_requires_matching_publisher_report(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    report = tmp_path / "publication.json"
    _write_pending_publication_report(metadata, report)
    pending = json.loads(report.read_text())
    pending["base_revision"] = "c" * 40
    report.write_text(json.dumps(pending), encoding="utf-8")

    with pytest.raises(ValueError, match="publisher-created recovery report"):
        validate_existing_track_hub_publication(
            metadata,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            final_revision="b" * 40,
            publication_approval=_publication_approval(metadata),
            udc_dir=tmp_path / "udc",
            success_marker_path=tmp_path / "publication.complete",
            validator=lambda *args, **kwargs: {"valid": True},
        )


def test_existing_publication_restores_marker_from_validated_report(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    report = tmp_path / "publication.json"
    success_marker = tmp_path / "publication.complete"
    publish_track_hub(
        metadata,
        validation,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        publication_approval=_publication_approval(metadata),
        udc_dir=tmp_path / "udc",
        success_marker_path=success_marker,
        api=_FakeApi(),
        validator=lambda *args, **kwargs: _successful_public_validation(),
    )
    success_marker.unlink()
    calls = []

    def validator(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return _successful_public_validation()

    validate_existing_track_hub_publication(
        metadata,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        final_revision="b" * 40,
        publication_approval=_publication_approval(metadata),
        udc_dir=tmp_path / "udc",
        success_marker_path=success_marker,
        validator=validator,
    )

    publication = json.loads(report.read_text())
    assert len(calls) == 1
    assert publication["status"] == "validated_existing_publication"
    assert publication["recovered_from_status"] == "validated"
    assert success_marker.read_text() == f"{'b' * 40}\n"


def test_validated_recovery_rejects_modified_metadata(tmp_path: Path) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    report = tmp_path / "publication.json"
    success_marker = tmp_path / "publication.complete"
    approval = _publication_approval(metadata)
    publish_track_hub(
        metadata,
        validation,
        report,
        expected_base_revision=ARTIFACT_REVISION,
        publication_approval=approval,
        udc_dir=tmp_path / "udc",
        success_marker_path=success_marker,
        api=_FakeApi(),
        validator=lambda *args, **kwargs: _successful_public_validation(),
    )
    remote_files = {
        path.relative_to(metadata).as_posix(): path.read_bytes()
        for path in [
            metadata / "README.md",
            metadata / "manifest" / "ucsc-hub.json",
            *sorted(item for item in (metadata / "ucsc").rglob("*") if item.is_file()),
        ]
    }
    success_marker.unlink()
    with (metadata / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\nChanged after publication.\n")

    class FinalRevisionApi(_FakeApi):
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="b" * 40)

    def opener(target: object, **kwargs: object) -> _FakeResponse:
        if isinstance(target, str):
            marker = f"/resolve/{'b' * 40}/"
            relative_path = target.split(marker, maxsplit=1)[1]
            return _FakeResponse(remote_files[relative_path])
        return _FakeResponse(
            b"x" * 64,
            status=206,
            headers={"Content-Range": "bytes 0-63/100"},
        )

    def validator(
        metadata_root: str | Path,
        *,
        revision: str,
        udc_dir: str | Path,
        repository_id: str,
    ) -> dict[str, object]:
        return validate_public_track_hub(
            metadata_root,
            revision=revision,
            udc_dir=udc_dir,
            repository_id=repository_id,
            api=FinalRevisionApi(),
            opener=opener,
            runner=_fake_runner,
        )

    with pytest.raises(ValueError, match="incomplete or mismatched"):
        validate_existing_track_hub_publication(
            metadata,
            report,
            expected_base_revision=ARTIFACT_REVISION,
            final_revision="b" * 40,
            publication_approval=approval,
            udc_dir=tmp_path / "udc",
            success_marker_path=success_marker,
            validator=validator,
        )
    assert not success_marker.exists()


def test_publication_rejects_missing_approval_and_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _build_metadata(tmp_path)
    validation = tmp_path / "validation.json"
    _write_validation_report(metadata, validation)
    with pytest.raises(ValueError, match="explicit author approval"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval=None,
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
        )

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="non-Slurm"):
        publish_track_hub(
            metadata,
            validation,
            tmp_path / "publication.json",
            expected_base_revision=ARTIFACT_REVISION,
            publication_approval={
                "approved": True,
                "evidence_url": HUB_APPROVAL_ISSUE,
                "approved_by": "author",
                "approved_at": "2026-07-22",
                "expected_base_revision": ARTIFACT_REVISION,
                "operation": "publish_hub",
                "candidate_sha256": publication_candidate_sha256(metadata),
            },
            udc_dir=tmp_path / "udc",
            api=_FakeApi(),
        )


@pytest.mark.parametrize("target", ["publish_hub", "publish_dataset_card"])
def test_enabled_workflow_builds_validates_and_separates_publication(
    tmp_path: Path, target: str
) -> None:
    release_manifest = tmp_path / "release.json"
    release_manifest.write_text(json.dumps(_release_manifest()), encoding="utf-8")
    config_path = tmp_path / "hub.yaml"
    config_path.write_text(
        f"""\
hub:
  enabled: true
  metadata_only_update: true
  release_manifest: {release_manifest}
  artifact_revision: {ARTIFACT_REVISION}
  expected_base_revision: {ARTIFACT_REVISION}
  contact_email: maintainer@example.org
  output_root: {tmp_path / "hub"}
  udc_cache_root: {tmp_path / "udc"}
  publication_approval: null
  resources:
    build: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    validate: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
    publish: {{mem_mb: 1024, runtime: 30, disk_mb: 1024}}
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            "workflow/Snakefile",
            "--configfile",
            str(config_path),
            "--cores",
            "1",
            "--dry-run",
            "--printshellcmds",
            target,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "build_track_hub_metadata" in result.stdout
    assert ("\nrule validate_track_hub:" in result.stdout) is (target == "publish_hub")
    assert f"\nrule {target}:" in result.stdout
    assert ("--metadata-only" in result.stdout) is (target == "publish_hub")
    expected_marker = (
        "publication.complete"
        if target == "publish_hub"
        else "dataset-card-publication.complete"
    )
    assert expected_marker in result.stdout
