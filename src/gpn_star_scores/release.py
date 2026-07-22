"""Build, publish, and validate the public Hugging Face score release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import polars as pl
from huggingface_hub import HfApi

from gpn_star_scores.catalog import (
    EXPECTED_SHARD_COUNT,
    SCORE_SETS,
    SCORE_TYPES,
    expected_shards,
)
from gpn_star_scores.inventory import sha256_file
from gpn_star_scores.parquet_benchmark import (
    QuerySpec,
    _benchmark_source,
    _CountingHfFileSystem,
    _execute_hf_query,
)
from gpn_star_scores.tracks import TRACKS

REPOSITORY_ID = "songlab/gpn-star-scores"
PAPER_DOI = "10.1101/2025.09.21.677619"
PAPER_TITLE = (
    "Predicting functional constraints across evolutionary timescales with "
    "phylogeny-informed genomic language models"
)
VIEWER_URL = "https://datasets-server.huggingface.co/first-rows"
HUGGING_FACE_URL = "https://huggingface.co"
CAPACITY_BLOCKER = (
    "Hugging Face organization capacity and numeric release headroom are not confirmed"
)
CAPACITY_APPROVAL_ISSUE = "https://github.com/gonzalobenegas/gpn-star-scores/issues/4"
PUBLIC_STORAGE_POLICY = "https://huggingface.co/docs/hub/storage-limits"


def dataset_configs() -> list[dict[str, Any]]:
    """Return the 16 explicit Dataset Viewer configurations."""

    return [
        {
            "config_name": f"{score_set.name}-{score_type}",
            "score_set": score_set.name,
            "score_type": score_type,
            "data_files": [
                {
                    "split": "train",
                    "path": f"data/{score_set.name}/{score_type}/*.parquet",
                }
            ],
        }
        for score_set in SCORE_SETS
        for score_type in SCORE_TYPES
    ]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _validated_inventory(
    source_root: Path,
    manifest: Mapping[str, Any],
    *,
    allow_capacity_waiver: bool,
) -> list[dict[str, Any]]:
    if manifest.get("manifest_version") != 1:
        raise ValueError("inventory manifest_version must be 1")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("inventory manifest lacks validation evidence")
    capacity_only = validation.get("blockers") == [CAPACITY_BLOCKER]
    if validation.get("release_ready") is not True and not (
        allow_capacity_waiver and capacity_only
    ):
        raise ValueError("inventory manifest is not release-ready")
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != EXPECTED_SHARD_COUNT:
        raise ValueError("inventory manifest must contain all 290 shard records")

    expected = {shard.relative_path.as_posix(): shard for shard in expected_shards()}
    observed = {
        record.get("path"): record for record in records if isinstance(record, Mapping)
    }
    if set(observed) != set(expected) or len(observed) != len(records):
        raise ValueError("inventory shard paths do not exactly match the catalog")

    release_records = []
    for relative_path, shard in expected.items():
        record = observed[relative_path]
        size = record.get("size")
        digest = record.get("sha256")
        if (
            record.get("valid") is not True
            or record.get("score_set") != shard.score_set
            or record.get("assembly") != shard.assembly
            or record.get("score_type") != shard.score_type
            or str(record.get("chrom")) != shard.chrom
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not _is_sha256(digest)
        ):
            raise ValueError(f"invalid inventory record for {relative_path}")
        source_path = source_root / relative_path
        if not source_path.is_file() or source_path.stat().st_size != size:
            raise ValueError(f"staged source identity differs for {relative_path}")
        parquet = record.get("parquet")
        if not isinstance(parquet, Mapping) or not isinstance(
            parquet.get("num_rows"), int
        ):
            raise ValueError(f"inventory record lacks row count for {relative_path}")
        content = record.get("content")
        coordinate_bounds = (
            content.get("coordinate_bounds") if isinstance(content, Mapping) else None
        )
        if (
            not isinstance(coordinate_bounds, Mapping)
            or not isinstance(coordinate_bounds.get("min"), int)
            or not isinstance(coordinate_bounds.get("max"), int)
            or coordinate_bounds["min"] < 1
            or coordinate_bounds["max"] < coordinate_bounds["min"]
        ):
            raise ValueError(f"inventory record lacks bounds for {relative_path}")
        release_records.append(
            {
                "path": f"data/{relative_path}",
                "score_set": shard.score_set,
                "assembly": shard.assembly,
                "score_type": shard.score_type,
                "chrom": shard.chrom,
                "size": size,
                "sha256": digest.lower(),
                "rows": parquet["num_rows"],
                "coordinate_bounds": dict(coordinate_bounds),
            }
        )
    return sorted(release_records, key=lambda item: item["path"])


def _validated_capacity_approval(
    approval: Mapping[str, Any] | None,
    *,
    required: bool,
    planned_release_bytes: int,
) -> dict[str, Any] | None:
    if not required:
        if approval is not None:
            raise ValueError(
                "capacity approval is only valid for the capacity-only blocker"
            )
        return None
    if not isinstance(approval, Mapping):
        raise ValueError("capacity-only inventory blocker requires explicit approval")
    reserved_headroom = approval.get("reserved_headroom_bytes")
    if (
        approval.get("approved") is not True
        or approval.get("public_repository") is not True
        or approval.get("evidence_url") != CAPACITY_APPROVAL_ISSUE
        or approval.get("public_storage_policy_url") != PUBLIC_STORAGE_POLICY
        or approval.get("planned_release_bytes") != planned_release_bytes
        or not isinstance(reserved_headroom, int)
        or isinstance(reserved_headroom, bool)
        or reserved_headroom < 0
        or not isinstance(approval.get("approved_by"), str)
        or not approval["approved_by"].strip()
        or not isinstance(approval.get("approved_at"), str)
        or not approval["approved_at"].strip()
    ):
        raise ValueError("invalid public-storage capacity approval")
    return dict(approval)


def _validate_parquet_selection(
    selection: Mapping[str, Any], inventory_sha256: str
) -> None:
    source_inventory = selection.get("source_inventory")
    if (
        selection.get("status") != "selected"
        or selection.get("selected_candidate") != "source"
        or not isinstance(source_inventory, Mapping)
        or source_inventory.get("valid") is not True
        or source_inventory.get("manifest_sha256") != inventory_sha256
    ):
        raise ValueError(
            "Parquet selection must choose the validated source layout for this inventory"
        )


def _validated_bigwigs(
    bigwig_root: Path,
    validation: Mapping[str, Any],
    inventory_sha256: str,
) -> list[dict[str, Any]]:
    expected = {(score_set.name, track) for score_set in SCORE_SETS for track in TRACKS}
    tracks = validation.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("BigWig validation lacks track records")
    observed = {
        (record.get("score_set"), record.get("track")): record
        for record in tracks
        if isinstance(record, Mapping)
    }
    if (
        validation.get("report_version") != 1
        or validation.get("valid") is not True
        or validation.get("track_count") != len(expected)
        or validation.get("inventory_manifest_sha256") != inventory_sha256
        or set(observed) != expected
        or len(observed) != len(tracks)
    ):
        raise ValueError("BigWig validation does not cover all 40 final tracks")

    release_records = []
    for score_set, track in sorted(expected):
        validation_record = observed[(score_set, track)]
        path = bigwig_root / "final" / score_set / f"{track}.bw"
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing final BigWig: {score_set}/{track}.bw")
        if (
            validation_record.get("zoom_levels", 0) < 1
            or validation_record.get("score_set") != score_set
            or validation_record.get("track") != track
        ):
            raise ValueError(f"invalid final BigWig evidence for {score_set}/{track}")
        release_records.append(
            {
                "path": f"bigwig/{score_set}/{track}.bw",
                "score_set": score_set,
                "assembly": validation_record["assembly"],
                "ucsc_assembly": validation_record["ucsc_assembly"],
                "track": track,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "bases_covered": validation_record["bases_covered"],
                "zoom_levels": validation_record["zoom_levels"],
            }
        )
    return release_records


def render_dataset_card(release_manifest: Mapping[str, Any]) -> str:
    """Render the public dataset card and its explicit data-file globs."""

    configs = release_manifest["dataset_configs"]
    frontmatter = [
        "---",
        "license: apache-2.0",
        "pretty_name: GPN-Star genome-wide scores",
        "tags:",
        "  - biology",
        "  - genomics",
        "  - variant-effect-prediction",
        "  - polars",
        "size_categories:",
        "  - 100B<n<1T",
        "configs:",
    ]
    for config in configs:
        frontmatter.extend(
            [
                f"  - config_name: {config['config_name']}",
                "    data_files:",
                "      - split: train",
                f"        path: {config['data_files'][0]['path']}",
            ]
        )
    frontmatter.append("---")

    model_rows = []
    for score_set in SCORE_SETS:
        model_rows.append(
            "| "
            f"`{score_set.name}` | `{score_set.assembly}` | "
            f"[{score_set.model_id}](https://huggingface.co/{score_set.model_id}) | "
            f"{score_set.model_description} |"
        )

    body = f"""

# GPN-Star genome-wide scores

Genome-wide calibrated GPN-Star scores for eight model/assembly combinations,
plus browser-ready BigWig tracks. The original Parquet layout is retained
unchanged because the predeclared local and Hugging Face range-query benchmark
did not justify a rewrite.

## Models and assemblies

| Score set | Parquet assembly | Model | Description |
| --- | --- | --- | --- |
{chr(10).join(model_rows)}

Parquet chromosome names are the supplied assembly names and `pos` is a
one-based position. BigWig files use zero-based, half-open coordinates and UCSC
chromosome names. Browser assembly aliases are `hg38`, `ce11`, `dm6`,
`galGal6` for Parquet assembly `gg6`, `araTha1` for `tair10`, and `mm39`.

## Schemas and interpretation

Each `*-entropy` configuration has:

| Column | Type | Meaning |
| --- | --- | --- |
| `chrom` | String | Supplied assembly chromosome name |
| `pos` | Int64 | One-based genomic position |
| `ref` | String | Reference nucleotide |
| `entropy_calibrated` | Float32 | Calibrated entropy score |

Each `*-llr` configuration has the same keys plus `alt: String`,
`llr_calibrated: Float32`, and `abs_llr_calibrated: Float32`.
`abs_llr_calibrated` is an independently supplied calibrated score; it is not
derived from `llr_calibrated`.

The BigWig `entropy` track contains `entropy_calibrated`. The A/C/G/T tracks
are calibrated-LLR-derived visualization heights, not raw model probabilities:
the reference logit is zero, alternate logits are `llr_calibrated`, a stable
Float64 softmax produces base weights, and each height is
`p(base) * (2 - H)` for base-2 entropy `H`. Final visualization values are
stored to three decimal places; Parquet remains the canonical full-precision
product.

## Repository layout

```text
data/<score-set>/<entropy|llr>/*.parquet
bigwig/<score-set>/{{entropy,A,C,G,T}}.bw
manifest/
ucsc/  # added and validated by the track-hub release workflow
README.md
```

SHA-256 checksums, byte sizes, row counts, and validation provenance are in
[`manifest/release.json`](manifest/release.json). Pin production analyses to
the immutable Hugging Face commit SHA recorded by the publication report.

## Polars examples

The examples use the public `main` revision so they are executable as written.
Replace `main` with a release commit SHA for a reproducible analysis.

### Interval filter with projection

```python
import polars as pl

root = "hf://datasets/{REPOSITORY_ID}@main/data"
scores = pl.scan_parquet(
    f"{{root}}/gpn-star-hg38-v100-200m/entropy/*.parquet"
)
region = (
    scores
    .filter(
        (pl.col("chrom") == "22")
        & pl.col("pos").is_between(20_000_000, 20_001_000)
    )
    .select("chrom", "pos", "ref", "entropy_calibrated")
    .collect()
)
```

### Projected score scan

```python
projected = (
    pl.scan_parquet(f"{{root}}/ce11/entropy/*.parquet")
    .select("chrom", "pos", "entropy_calibrated")
    .collect()
)
```

### Multi-chromosome scan

```python
multi_chrom = (
    pl.scan_parquet(f"{{root}}/dm6/llr/*.parquet")
    .filter(pl.col("chrom").is_in(["2L", "X"]))
    .select("chrom", "pos", "ref", "alt", "llr_calibrated")
    .collect()
)
```

### Join a user variant table

```python
variants = pl.DataFrame(
    {{
        "chrom": ["22"],
        "pos": [20_000_001],
        "ref": ["A"],
        "alt": ["G"],
    }}
).lazy()
llr = pl.scan_parquet(
    f"{{root}}/gpn-star-hg38-p243-200m/llr/*.parquet"
)
annotated = variants.join(
    llr,
    on=["chrom", "pos", "ref", "alt"],
    how="left",
).collect()
```

Lazy scans preserve predicate and projection pushdown. The release validation
also measures representative public `hf://` interval reads and requires fewer
transferred bytes than the corresponding chromosome object size.

## License and citation

The dataset is released under Apache-2.0. Please cite:

Ye C, Benegas G, Albors C, Li JC, Prillo S, Fields PD, Clarke B, Song YS.
[{PAPER_TITLE}](https://doi.org/{PAPER_DOI}). bioRxiv (2025).
doi: `{PAPER_DOI}`.

```bibtex
@article{{ye2025predicting,
  title={{{PAPER_TITLE}}},
  author={{Ye, Chengzhong and Benegas, Gonzalo and Albors, Carlos and Li,
    Jianan Canal and Prillo, Sebastian and Fields, Peter D and Clarke, Brian
    and Song, Yun S}},
  journal={{bioRxiv}},
  year={{2025}},
  doi={{{PAPER_DOI}}}
}}
```
"""
    return "\n".join(frontmatter) + body


def build_release_metadata(
    source_root: str | Path,
    bigwig_root: str | Path,
    inventory_manifest_path: str | Path,
    parquet_selection_path: str | Path,
    bigwig_validation_path: str | Path,
    output_dir: str | Path,
    *,
    repository_id: str = REPOSITORY_ID,
    capacity_approval: Mapping[str, Any] | None = None,
) -> None:
    """Validate local release gates and atomically build upload metadata."""

    if repository_id != REPOSITORY_ID:
        raise ValueError(f"release repository must be {REPOSITORY_ID}")
    source_root = Path(source_root)
    bigwig_root = Path(bigwig_root)
    inventory_path = Path(inventory_manifest_path)
    selection_path = Path(parquet_selection_path)
    bigwig_validation_path = Path(bigwig_validation_path)
    output = Path(output_dir)

    inventory = _read_json(inventory_path)
    inventory_sha256 = sha256_file(inventory_path)
    inventory_validation = inventory.get("validation", {})
    capacity_waiver_required = inventory_validation.get("blockers") == [
        CAPACITY_BLOCKER
    ]
    parquet_records = _validated_inventory(
        source_root,
        inventory,
        allow_capacity_waiver=capacity_approval is not None,
    )
    selection = _read_json(selection_path)
    _validate_parquet_selection(selection, inventory_sha256)
    bigwig_validation = _read_json(bigwig_validation_path)
    bigwig_records = _validated_bigwigs(
        bigwig_root, bigwig_validation, inventory_sha256
    )
    planned_release_bytes = sum(record["size"] for record in parquet_records) + sum(
        record["size"] for record in bigwig_records
    )
    validated_capacity_approval = _validated_capacity_approval(
        capacity_approval,
        required=capacity_waiver_required,
        planned_release_bytes=planned_release_bytes,
    )

    release_manifest = {
        "release_manifest_version": 1,
        "repository": {
            "id": repository_id,
            "repo_type": "dataset",
            "public": True,
            "license": "apache-2.0",
        },
        "paper": {"title": PAPER_TITLE, "doi": PAPER_DOI},
        "source_inventory": {
            "manifest_sha256": inventory_sha256,
            "release_ready": inventory_validation.get("release_ready") is True,
            "total_shard_bytes": inventory["source"]["total_shard_bytes"],
            "capacity_waiver": validated_capacity_approval,
        },
        "parquet": {
            "selected_layout": "source",
            "file_count": len(parquet_records),
            "total_bytes": sum(record["size"] for record in parquet_records),
            "files": parquet_records,
        },
        "bigwig": {
            "file_count": len(bigwig_records),
            "total_bytes": sum(record["size"] for record in bigwig_records),
            "value_decimals": bigwig_validation["value_decimals"],
            "files": bigwig_records,
        },
        "dataset_configs": dataset_configs(),
        "validation": {
            "preflight_passed": True,
            "inventory_data_validation_passed": True,
            "inventory_release_ready": inventory_validation.get("release_ready")
            is True,
            "capacity_waiver_applied": validated_capacity_approval is not None,
            "parquet_layout_selected": "source",
            "bigwig_validation_passed": True,
            "expected_viewer_configs": 16,
            "expected_parquet_files": EXPECTED_SHARD_COUNT,
            "expected_bigwig_files": len(SCORE_SETS) * len(TRACKS),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest_dir = temporary / "manifest"
        manifest_dir.mkdir()
        _atomic_write_json(manifest_dir / "release.json", release_manifest)
        shutil.copyfile(inventory_path, manifest_dir / "inventory.json")
        shutil.copyfile(selection_path, manifest_dir / "parquet-layout.json")
        shutil.copyfile(bigwig_validation_path, manifest_dir / "bigwig-validation.json")
        _atomic_write_text(
            temporary / "README.md", render_dataset_card(release_manifest)
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _upload_release_folders(
    api: Any,
    source_root: Path,
    bigwig_root: Path,
    metadata_root: Path,
    *,
    repository_id: str,
) -> list[dict[str, str]]:
    uploads = []
    for score_set in SCORE_SETS:
        for score_type in SCORE_TYPES:
            path_in_repo = f"data/{score_set.name}/{score_type}"
            api.upload_folder(
                repo_id=repository_id,
                repo_type="dataset",
                folder_path=source_root / score_set.name / score_type,
                path_in_repo=path_in_repo,
                allow_patterns=["*.parquet"],
                commit_message=f"Upload {score_set.name} {score_type} shards",
            )
            uploads.append({"kind": "parquet", "path": path_in_repo})
    for score_set in SCORE_SETS:
        path_in_repo = f"bigwig/{score_set.name}"
        api.upload_folder(
            repo_id=repository_id,
            repo_type="dataset",
            folder_path=bigwig_root / "final" / score_set.name,
            path_in_repo=path_in_repo,
            allow_patterns=["*.bw"],
            commit_message=f"Upload {score_set.name} BigWig tracks",
        )
        uploads.append({"kind": "bigwig", "path": path_in_repo})
    api.upload_folder(
        repo_id=repository_id,
        repo_type="dataset",
        folder_path=metadata_root,
        path_in_repo="",
        allow_patterns=["README.md", "manifest/**"],
        commit_message="Publish release metadata and dataset card",
    )
    uploads.append({"kind": "metadata", "path": "/"})
    return uploads


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    return int(status if status is not None else response.getcode())


def _read_url(
    request: str | Request,
    *,
    opener: Callable[..., Any] = urlopen,
    timeout: float = 120,
) -> tuple[int, Mapping[str, str], bytes]:
    with opener(request, timeout=timeout) as response:
        return _response_status(response), response.headers, response.read()


def _viewer_payload(
    repository_id: str,
    config_name: str,
    *,
    opener: Callable[..., Any],
    attempts: int,
    retry_seconds: float,
) -> dict[str, Any]:
    query = urlencode(
        {"dataset": repository_id, "config": config_name, "split": "train"}
    )
    url = f"{VIEWER_URL}?{query}"
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            status, _, content = _read_url(url, opener=opener)
            if status != 200:
                raise RuntimeError(f"Viewer returned HTTP {status}")
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Viewer response is not an object")
            if not isinstance(payload.get("features"), list) or not isinstance(
                payload.get("rows"), list
            ):
                raise ValueError("Viewer response is not ready")
            if not payload["rows"]:
                raise ValueError("Viewer preview is empty")
            return payload
        except (HTTPError, URLError, RuntimeError, ValueError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(retry_seconds)
    raise RuntimeError(
        f"Dataset Viewer did not become ready for {config_name}: {last_error}"
    )


@contextmanager
def _without_hugging_face_credentials() -> Any:
    names = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _representative_parquet_records(
    release_manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    records = release_manifest["parquet"]["files"]
    keys = (
        ("gg6", "entropy", "1"),
        ("gpn-star-hg38-v100-200m", "llr", "22"),
    )
    indexed = {
        (record["score_set"], record["score_type"], record["chrom"]): record
        for record in records
    }
    try:
        return [indexed[key] for key in keys]
    except KeyError as error:
        raise ValueError(
            f"release manifest lacks representative shard {error}"
        ) from error


def _validate_hf_range_queries(
    repository_id: str,
    revision: str,
    release_manifest: Mapping[str, Any],
    *,
    block_size: int,
) -> list[dict[str, Any]]:
    results = []
    for record in _representative_parquet_records(release_manifest):
        uri = f"hf://datasets/{repository_id}@{revision}/{record['path']}"
        filesystem = _CountingHfFileSystem(token=False, block_size=block_size)
        remote_path = uri.removeprefix("hf://")
        size = int(filesystem.info(remote_path)["size"])
        start = int(record["coordinate_bounds"]["min"])
        end = min(start + 999, int(record["coordinate_bounds"]["max"]))
        query = QuerySpec("release-first-1kb", "interval", start=start, end=end)
        with _benchmark_source(
            uri,
            "hf",
            hf_token=None,
            hf_block_size=block_size,
            hf_filesystem=filesystem,
        ) as (source, transferred_bytes):
            frame = _execute_hf_query(source, query, pl.DataFrame())
            transferred = transferred_bytes()
        if (
            frame.height < 1
            or transferred is None
            or transferred <= 0
            or transferred >= size
        ):
            raise RuntimeError(f"remote interval query did not use ranges for {uri}")

        scan = pl.scan_parquet(uri, cache=False)
        schema = scan.collect_schema()
        score_columns = [
            name for name in schema.names() if name.endswith("_calibrated")
        ]
        lazy_query = scan.filter(
            (pl.col("chrom") == record["chrom"]) & pl.col("pos").is_between(start, end)
        ).select("pos", *score_columns)
        plan = lazy_query.explain(optimized=True)
        direct = lazy_query.head(1).collect()
        if (
            direct.height != 1
            or "PROJECT" not in plan.upper()
            or "SELECTION" not in plan.upper()
        ):
            raise RuntimeError(f"direct Polars pushdown check failed for {uri}")
        results.append(
            {
                "uri": uri,
                "object_bytes": size,
                "transferred_bytes": transferred,
                "rows": frame.height,
                "direct_polars": True,
                "optimized_plan": plan,
            }
        )
    return results


def _sibling_lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, Mapping):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def validate_public_release(
    metadata_root: str | Path,
    *,
    repository_id: str,
    revision: str,
    api: Any | None = None,
    opener: Callable[..., Any] = urlopen,
    viewer_attempts: int = 1,
    viewer_retry_seconds: float = 0,
    viewer_required: bool = False,
    hf_block_size: int = 4_194_304,
) -> dict[str, Any]:
    """Validate the final immutable revision without sending credentials."""

    if repository_id != REPOSITORY_ID:
        raise ValueError(f"release repository must be {REPOSITORY_ID}")
    if viewer_attempts <= 0 or viewer_retry_seconds < 0 or hf_block_size <= 0:
        raise ValueError("invalid public-validation retry or block-size settings")
    release_manifest = _read_json(Path(metadata_root) / "manifest" / "release.json")
    public_api = api or HfApi(token=False)

    with _without_hugging_face_credentials():
        info = public_api.repo_info(
            repository_id,
            repo_type="dataset",
            revision=revision,
            files_metadata=True,
            token=False,
        )
        if getattr(info, "private", None) is not False:
            raise RuntimeError("Hugging Face release repository is not public")
        if getattr(info, "sha", None) != revision:
            raise RuntimeError("Hugging Face revision did not resolve exactly")
        siblings = {
            sibling.rfilename: sibling for sibling in getattr(info, "siblings", [])
        }
        expected_files = [
            *release_manifest["parquet"]["files"],
            *release_manifest["bigwig"]["files"],
        ]
        checksum_checks = []
        for record in expected_files:
            sibling = siblings.get(record["path"])
            if sibling is None:
                raise RuntimeError(f"published file is missing: {record['path']}")
            remote_size = getattr(sibling, "size", None)
            remote_sha256 = _sibling_lfs_sha256(sibling)
            if remote_size != record["size"] or remote_sha256 != record["sha256"]:
                raise RuntimeError(f"published identity differs: {record['path']}")
            checksum_checks.append(record["path"])

        quoted_revision = quote(revision, safe="")
        readme_url = (
            f"{HUGGING_FACE_URL}/datasets/{repository_id}/resolve/"
            f"{quoted_revision}/README.md"
        )
        readme_status, _, readme_content = _read_url(readme_url, opener=opener)
        readme = readme_content.decode("utf-8")
        if readme_status != 200 or "# GPN-Star genome-wide scores" not in readme:
            raise RuntimeError("public dataset card source is unavailable")
        page_status, _, page_content = _read_url(
            f"{HUGGING_FACE_URL}/datasets/{repository_id}", opener=opener
        )
        if page_status != 200 or not page_content:
            raise RuntimeError("public dataset-card page did not render")

        range_checks = []
        for record in release_manifest["bigwig"]["files"]:
            end = min(63, record["size"] - 1)
            encoded_path = quote(record["path"], safe="/")
            request = Request(
                f"{HUGGING_FACE_URL}/datasets/{repository_id}/resolve/"
                f"{quoted_revision}/{encoded_path}",
                headers={"Range": f"bytes=0-{end}"},
            )
            status, headers, content = _read_url(request, opener=opener)
            content_range = headers.get("Content-Range")
            if (
                status != 206
                or len(content) != end + 1
                or not content_range
                or not content_range.startswith(f"bytes 0-{end}/")
            ):
                raise RuntimeError(f"BigWig range request failed: {record['path']}")
            range_checks.append(
                {"path": record["path"], "status": status, "bytes": len(content)}
            )

        polars_checks = _validate_hf_range_queries(
            repository_id,
            revision,
            release_manifest,
            block_size=hf_block_size,
        )

        viewer_checks = []
        viewer_pending = []
        for config in release_manifest["dataset_configs"]:
            try:
                payload = _viewer_payload(
                    repository_id,
                    config["config_name"],
                    opener=opener,
                    attempts=viewer_attempts,
                    retry_seconds=viewer_retry_seconds,
                )
                expected_columns = (
                    ["chrom", "pos", "ref", "entropy_calibrated"]
                    if config["score_type"] == "entropy"
                    else [
                        "chrom",
                        "pos",
                        "ref",
                        "alt",
                        "llr_calibrated",
                        "abs_llr_calibrated",
                    ]
                )
                observed_columns = [
                    feature.get("name") for feature in payload["features"]
                ]
                if observed_columns != expected_columns or not payload.get("rows"):
                    raise RuntimeError(
                        "Dataset Viewer schema differs for "
                        f"{config['config_name']}"
                    )
                viewer_checks.append(
                    {
                        "config_name": config["config_name"],
                        "columns": observed_columns,
                        "preview_rows": len(payload["rows"]),
                    }
                )
            except RuntimeError as error:
                if viewer_required:
                    raise
                viewer_pending.append(
                    {
                        "config_name": config["config_name"],
                        "error": str(error),
                    }
                )

    return {
        "report_version": 1,
        "valid": True,
        "repository": repository_id,
        "public": True,
        "revision": revision,
        "credentials_sent": False,
        "checksum_file_count": len(checksum_checks),
        "viewer_required": viewer_required,
        "viewer_ready": not viewer_pending,
        "viewer_config_count": len(viewer_checks),
        "viewer_checks": viewer_checks,
        "viewer_pending": viewer_pending,
        "dataset_card_rendered": True,
        "bigwig_range_count": len(range_checks),
        "bigwig_range_checks": range_checks,
        "polars_range_checks": polars_checks,
    }


def publish_release(
    source_root: str | Path,
    bigwig_root: str | Path,
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    repository_id: str = REPOSITORY_ID,
    viewer_attempts: int = 1,
    viewer_retry_seconds: float = 0,
    viewer_required: bool = False,
    hf_block_size: int = 4_194_304,
    api: Any | None = None,
    validator: Callable[..., dict[str, Any]] = validate_public_release,
) -> None:
    """Upload from one non-Slurm process and validate the final public commit."""

    if repository_id != REPOSITORY_ID:
        raise ValueError(f"release repository must be {REPOSITORY_ID}")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "publication must run from one intentional non-Slurm process"
        )
    source_root = Path(source_root)
    bigwig_root = Path(bigwig_root)
    metadata_root = Path(metadata_root)
    release_manifest = _read_json(metadata_root / "manifest" / "release.json")
    if release_manifest.get("validation", {}).get("preflight_passed") is not True:
        raise ValueError("release metadata did not pass local preflight")

    authenticated_api = api or HfApi()
    authenticated_api.create_repo(
        repository_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    repository = authenticated_api.repo_info(repository_id, repo_type="dataset")
    if getattr(repository, "private", False):
        authenticated_api.update_repo_visibility(
            repository_id, repo_type="dataset", private=False
        )

    uploads = _upload_release_folders(
        authenticated_api,
        source_root,
        bigwig_root,
        metadata_root,
        repository_id=repository_id,
    )
    final_info = authenticated_api.repo_info(repository_id, repo_type="dataset")
    final_revision = getattr(final_info, "sha", None)
    if not isinstance(final_revision, str) or len(final_revision) != 40:
        raise RuntimeError("Hugging Face did not return a final commit SHA")
    public_validation = validator(
        metadata_root,
        repository_id=repository_id,
        revision=final_revision,
        viewer_attempts=viewer_attempts,
        viewer_retry_seconds=viewer_retry_seconds,
        viewer_required=viewer_required,
        hf_block_size=hf_block_size,
    )
    _atomic_write_json(
        Path(report_path),
        {
            "report_version": 1,
            "valid": public_validation.get("valid") is True,
            "repository": repository_id,
            "public": True,
            "final_revision": final_revision,
            "single_process": True,
            "slurm_job_id": None,
            "upload_method": "huggingface_hub.HfApi.upload_folder",
            "uploads": uploads,
            "public_validation": public_validation,
        },
    )


def validate_existing_release(
    metadata_root: str | Path,
    report_path: str | Path,
    *,
    revision: str,
    repository_id: str = REPOSITORY_ID,
    viewer_attempts: int = 1,
    viewer_retry_seconds: float = 0,
    viewer_required: bool = False,
    hf_block_size: int = 4_194_304,
    validator: Callable[..., dict[str, Any]] = validate_public_release,
) -> None:
    """Validate an already-published immutable revision without re-uploading."""

    if len(revision) != 40:
        raise ValueError("revision must be a 40-character commit SHA")
    public_validation = validator(
        metadata_root,
        repository_id=repository_id,
        revision=revision,
        viewer_attempts=viewer_attempts,
        viewer_retry_seconds=viewer_retry_seconds,
        viewer_required=viewer_required,
        hf_block_size=hf_block_size,
    )
    _atomic_write_json(
        Path(report_path),
        {
            "report_version": 1,
            "valid": public_validation.get("valid") is True,
            "repository": repository_id,
            "public": True,
            "final_revision": revision,
            "validation_mode": "existing_revision",
            "public_validation": public_validation,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--source-root", type=Path, required=True)
    preflight.add_argument("--bigwig-root", type=Path, required=True)
    preflight.add_argument("--inventory-manifest", type=Path, required=True)
    preflight.add_argument("--parquet-selection", type=Path, required=True)
    preflight.add_argument("--bigwig-validation", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument("--capacity-approval-json", type=Path)

    publish = commands.add_parser("publish")
    publish.add_argument("--source-root", type=Path, required=True)
    publish.add_argument("--bigwig-root", type=Path, required=True)
    publish.add_argument("--metadata-root", type=Path, required=True)
    publish.add_argument("--report", type=Path, required=True)
    publish.add_argument("--viewer-attempts", type=int, default=1)
    publish.add_argument("--viewer-retry-seconds", type=float, default=0)
    publish.add_argument("--require-viewer", action="store_true")
    publish.add_argument("--hf-block-size", type=int, default=4_194_304)

    validate = commands.add_parser("validate-existing")
    validate.add_argument("--metadata-root", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--revision", required=True)
    validate.add_argument("--viewer-attempts", type=int, default=1)
    validate.add_argument("--viewer-retry-seconds", type=float, default=0)
    validate.add_argument("--require-viewer", action="store_true")
    validate.add_argument("--hf-block-size", type=int, default=4_194_304)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        build_release_metadata(
            args.source_root,
            args.bigwig_root,
            args.inventory_manifest,
            args.parquet_selection,
            args.bigwig_validation,
            args.output_dir,
            capacity_approval=(
                _read_json(args.capacity_approval_json)
                if args.capacity_approval_json
                else None
            ),
        )
        return
    if args.command == "publish":
        publish_release(
            args.source_root,
            args.bigwig_root,
            args.metadata_root,
            args.report,
            viewer_attempts=args.viewer_attempts,
            viewer_retry_seconds=args.viewer_retry_seconds,
            viewer_required=args.require_viewer,
            hf_block_size=args.hf_block_size,
        )
        return
    if args.command == "validate-existing":
        validate_existing_release(
            args.metadata_root,
            args.report,
            revision=args.revision,
            viewer_attempts=args.viewer_attempts,
            viewer_retry_seconds=args.viewer_retry_seconds,
            viewer_required=args.require_viewer,
            hf_block_size=args.hf_block_size,
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
