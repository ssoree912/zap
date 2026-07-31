# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict CPU-only comparison of native-head and shared-mask artifacts.

Run this only after the shared-control extraction has completed.  The tool
hard-joins all samples on ``(dataset, row_index, sample_id)``, verifies the
unchanged paths, and compares native H2O against shared all-prefill using
actual decoded responses and downstream task scores.

The native head-wise Future mask/metrics and the shared Future mask/metrics
have different definitions.  They are deliberately not compared here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260727-003-eval1400-shared-control"
TOOL_SCHEMA_VERSION = 1
NATIVE_SCHEMA_VERSION = 2
SHARED_SCHEMA_VERSION = 3
SHARED_POLICY_VERSION = "shared-control-v1"
FUTURE_TRAJECTORY = "[last prompt query, y_1, ..., y_(T-1)]"

ZAP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE_ARTIFACT = (
    ZAP_ROOT / "artifacts/rebuttal_eval1400_q2_llava15_total0p2"
)
DEFAULT_SHARED_ARTIFACT = (
    ZAP_ROOT
    / "artifacts/rebuttal_eval1400_shared_control_llava15_total0p2"
)
DEFAULT_DATASET_COUNTS = {
    "gqa": 200,
    "textvqa": 200,
    "docvqa": 200,
    "chartqa": 200,
    "coco_caption": 200,
    "nocaps": 200,
    "textcaps": 200,
}

Identity = tuple[str, int, str]

INVARIANT_FIELDS = (
    "dataset",
    "sample_id",
    "row_index",
    "context",
    "semantic_question",
    "references",
    "max_new_tokens",
    "answer_steps",
    "hit_max_new_tokens",
    "processed_image_sha256",
    "prompt_len",
    "n_text",
    "n_visual",
    "requested_total_keep_ratio",
    "actual_total_keep_ratio",
    "actual_visual_keep_ratio",
    "question_token_count",
    "student_prompt_tail_token_count",
    "qvik_checkpoint",
    "same_processed_image_for_prefill_and_future",
    "full_cache_manual_matches_generate",
    "future_trajectory",
)
INVARIANT_SELECTOR_MAP = {
    "full_cache_generate": "full_cache_generate",
    "full_cache_manual": "full_cache_manual",
    "question_prefill": "question_shared",
    "qvik": "qvik_shared",
}
NATIVE_SELECTORS = {
    "full_cache_generate",
    "full_cache_manual",
    "h2o_prefill",
    "question_prefill",
    "qvik",
    "future_oracle",
}
SHARED_SELECTORS = {
    "full_cache_generate",
    "full_cache_manual",
    "all_prefill_shared",
    "question_shared",
    "qvik_shared",
}
NON_FUTURE_VECTOR_PAIR_MAP = {
    "qvik_vs_h2o": "qvik_shared__all_prefill_shared",
    "qvik_vs_question": "qvik_shared__question_shared",
}
NATIVE_MASK_NAMES = (
    "h2o_keep",
    "question_keep",
    "qvik_keep",
    "future_keep",
)
SHARED_MASK_NAMES = (
    "all_prefill_shared_keep",
    "question_shared_keep",
    "qvik_shared_keep",
    "future_shared_keep",
)
VECTOR_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Validated manifest/result/mask index for one artifact."""

    root: Path
    records: dict[Identity, dict[str, Any]]
    result_paths: dict[Identity, Path]
    mask_paths: dict[Identity, Path]
    manifest_sha256: dict[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare completed eval-1400 native H2O and shared all-prefill "
            "artifacts without loading a model or using a GPU."
        )
    )
    parser.add_argument(
        "--native-artifact",
        type=Path,
        default=DEFAULT_NATIVE_ARTIFACT,
    )
    parser.add_argument(
        "--shared-artifact",
        type=Path,
        default=DEFAULT_SHARED_ARTIFACT,
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_mask(mask: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(mask, dtype=bool)
    digest = hashlib.sha256(
        f"shape={tuple(contiguous.shape)};dtype=bool;".encode()
    )
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _identity(payload: Mapping[str, Any], source: Path) -> Identity:
    missing = [
        key
        for key in ("dataset", "row_index", "sample_id")
        if key not in payload
    ]
    if missing:
        raise ValueError(f"{source}: identity fields missing={missing}")
    try:
        row_index = int(payload["row_index"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source}: invalid row_index={payload['row_index']!r}"
        ) from error
    return (
        str(payload["dataset"]),
        row_index,
        str(payload["sample_id"]),
    )


def _inside(root: Path, path: Path, *, source: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{source}: path escapes artifact root {root}: {path}"
        ) from error
    return resolved


def _mask_path(
    artifact_root: Path,
    record: Mapping[str, Any],
    result_path: Path,
) -> Path:
    value = record.get("mask_file")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{result_path}: missing nonempty mask_file")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{result_path}: mask_file must be relative")
    resolved = _inside(
        artifact_root,
        artifact_root / relative,
        source=result_path,
    )
    if not resolved.is_file():
        raise ValueError(f"{result_path}: missing mask file {resolved}")
    if resolved.suffix != ".npz":
        raise ValueError(f"{result_path}: mask_file is not NPZ: {resolved}")
    return resolved


def _expected_counts(
    dataset_counts: Mapping[str, int],
) -> tuple[dict[str, int], int]:
    counts = {str(name): int(count) for name, count in dataset_counts.items()}
    if not counts:
        raise ValueError("At least one dataset is required")
    invalid = {name: count for name, count in counts.items() if count < 1}
    if invalid:
        raise ValueError(f"Dataset counts must be positive: {invalid}")
    return counts, sum(counts.values())


def _load_snapshot(
    artifact: Path,
    *,
    dataset_counts: Mapping[str, int],
    label: str,
) -> ArtifactSnapshot:
    counts, expected_total = _expected_counts(dataset_counts)
    try:
        root = artifact.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} artifact does not exist: {artifact}") from error
    if not root.is_dir():
        raise ValueError(f"{label} artifact is not a directory: {root}")

    manifest_dir = root / "manifests"
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    expected_manifest_names = {f"{name}.json" for name in counts}
    actual_manifest_names = {path.name for path in manifest_paths}
    if actual_manifest_names != expected_manifest_names:
        raise ValueError(
            f"{label}: manifest files mismatch; "
            f"missing={sorted(expected_manifest_names - actual_manifest_names)}, "
            f"extra={sorted(actual_manifest_names - expected_manifest_names)}"
        )

    manifest_sha256: dict[str, str] = {}
    manifest_identities: set[Identity] = set()
    manifest_samples: dict[Identity, dict[str, Any]] = {}
    for dataset, expected_count in counts.items():
        path = manifest_dir / f"{dataset}.json"
        payload = _load_json(path)
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"{path}: samples must be a list")
        if payload.get("sample_count") != expected_count:
            raise ValueError(
                f"{path}: sample_count={payload.get('sample_count')}, "
                f"expected={expected_count}"
            )
        if payload.get("datasets") != {dataset: expected_count}:
            raise ValueError(
                f"{path}: datasets={payload.get('datasets')!r}, "
                f"expected={{{dataset!r}: {expected_count}}}"
            )
        if len(samples) != expected_count:
            raise ValueError(
                f"{path}: len(samples)={len(samples)}, "
                f"expected={expected_count}"
            )
        local_identities: set[Identity] = set()
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"{path}: every sample must be an object")
            identity = _identity(sample, path)
            if identity[0] != dataset:
                raise ValueError(
                    f"{path}: sample belongs to dataset={identity[0]!r}"
                )
            if identity in local_identities:
                raise ValueError(f"{path}: duplicate identity={identity!r}")
            local_identities.add(identity)
            manifest_samples[identity] = sample
        overlap = manifest_identities & local_identities
        if overlap:
            raise ValueError(
                f"{label}: identities repeated across manifests: "
                f"{sorted(overlap)[:3]!r}"
            )
        manifest_identities.update(local_identities)
        manifest_sha256[dataset] = _sha256_file(path)

    samples_dir = root / "samples"
    structured_results = set(samples_dir.glob("*/*.json"))
    recursive_results = set(samples_dir.rglob("*.json"))
    if structured_results != recursive_results:
        unexpected = sorted(recursive_results - structured_results)
        raise ValueError(
            f"{label}: result JSON outside samples/<dataset>/: "
            f"{unexpected[:3]!r}"
        )
    result_paths = sorted(structured_results)
    if len(result_paths) != expected_total:
        raise ValueError(
            f"{label}: found {len(result_paths)} result JSON files, "
            f"expected exactly {expected_total}"
        )

    failure_dir = root / "failures"
    failure_markers = (
        sorted(failure_dir.rglob("*.txt")) if failure_dir.exists() else []
    )
    if failure_markers:
        raise ValueError(
            f"{label}: completed comparison forbids failure markers; "
            f"found={len(failure_markers)}, first={failure_markers[0]}"
        )

    records: dict[Identity, dict[str, Any]] = {}
    indexed_result_paths: dict[Identity, Path] = {}
    mask_paths: dict[Identity, Path] = {}
    dataset_result_counts = {name: 0 for name in counts}
    for unresolved_path in result_paths:
        path = _inside(root, unresolved_path, source=unresolved_path)
        record = _load_json(path)
        identity = _identity(record, path)
        dataset = identity[0]
        if dataset not in counts:
            raise ValueError(f"{label}: unexpected dataset={dataset!r} in {path}")
        if path.parent.name != dataset:
            raise ValueError(
                f"{path}: parent dataset={path.parent.name!r}, "
                f"record dataset={dataset!r}"
            )
        if identity in records:
            raise ValueError(f"{label}: duplicate result identity={identity!r}")
        if identity not in manifest_samples:
            raise ValueError(
                f"{label}: result identity absent from manifest={identity!r}"
            )
        manifest_sample = manifest_samples[identity]
        for field in ("context", "references", "max_new_tokens"):
            if record.get(field) != manifest_sample.get(field):
                raise ValueError(
                    f"{path}: record/manifest field differs={field!r}"
                )
        if (
            "question_span" in manifest_sample
            and record.get("semantic_question")
            != manifest_sample["question_span"]
        ):
            raise ValueError(
                f"{path}: semantic_question differs from manifest "
                "question_span"
            )
        records[identity] = record
        indexed_result_paths[identity] = path
        mask_paths[identity] = _mask_path(root, record, path)
        dataset_result_counts[dataset] += 1

    if dataset_result_counts != counts:
        raise ValueError(
            f"{label}: per-dataset result counts={dataset_result_counts}, "
            f"expected={counts}"
        )
    record_identities = set(records)
    if record_identities != manifest_identities:
        raise ValueError(
            f"{label}: manifest/result identity mismatch; "
            f"manifest_only={sorted(manifest_identities - record_identities)[:3]}, "
            f"result_only={sorted(record_identities - manifest_identities)[:3]}"
        )

    masks_dir = root / "masks"
    structured_masks = {
        path.resolve() for path in masks_dir.glob("*/*.npz")
    }
    recursive_masks = {path.resolve() for path in masks_dir.rglob("*.npz")}
    if structured_masks != recursive_masks:
        unexpected = sorted(recursive_masks - structured_masks)
        raise ValueError(
            f"{label}: mask NPZ outside masks/<dataset>/: "
            f"{unexpected[:3]!r}"
        )
    referenced_masks = set(mask_paths.values())
    if structured_masks != referenced_masks:
        raise ValueError(
            f"{label}: result/mask pairing mismatch; "
            f"unreferenced={len(structured_masks - referenced_masks)}, "
            f"missing={len(referenced_masks - structured_masks)}"
        )
    if len(referenced_masks) != expected_total:
        raise ValueError(
            f"{label}: {len(referenced_masks)} unique masks, "
            f"expected={expected_total}"
        )

    return ArtifactSnapshot(
        root=root,
        records=records,
        result_paths=indexed_result_paths,
        mask_paths=mask_paths,
        manifest_sha256=manifest_sha256,
    )


def _npz_scalar(
    packed: Mapping[str, Any],
    key: str,
    path: Path,
) -> Any:
    if key not in packed:
        raise ValueError(f"{path}: missing NPZ field={key!r}")
    value = np.asarray(packed[key])
    if value.shape != ():
        raise ValueError(f"{path}: NPZ field={key!r} is not scalar")
    return value.item()


def _unpack_mask(
    packed: Mapping[str, Any],
    key: str,
    *,
    n_visual: int,
    expected_ndim: int,
    path: Path,
) -> np.ndarray:
    if key not in packed:
        raise ValueError(f"{path}: missing packed mask={key!r}")
    value = np.asarray(packed[key])
    if value.dtype != np.uint8 or value.ndim != expected_ndim:
        raise ValueError(
            f"{path}: {key} must be uint8 with ndim={expected_ndim}; "
            f"got dtype={value.dtype}, shape={value.shape}"
        )
    expected_width = math.ceil(n_visual / 8)
    if value.shape[-1] != expected_width:
        raise ValueError(
            f"{path}: {key} packed width={value.shape[-1]}, "
            f"expected={expected_width}"
        )
    unpacked = np.unpackbits(value, axis=-1)
    if n_visual % 8 and np.any(unpacked[..., n_visual:]):
        raise ValueError(f"{path}: {key} has nonzero packed padding bits")
    return unpacked[..., :n_visual].astype(bool)


def _validate_budget(
    masks: Mapping[str, np.ndarray],
    *,
    n_keep: int,
    path: Path,
) -> None:
    shapes = {name: value.shape for name, value in masks.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{path}: mask shapes differ={shapes}")
    for name, mask in masks.items():
        if not np.all(mask.sum(axis=-1) == n_keep):
            raise ValueError(
                f"{path}: {name} does not keep exactly K={n_keep}"
            )


def _load_native_masks(
    record: Mapping[str, Any],
    path: Path,
) -> dict[str, np.ndarray]:
    if record.get("schema_version") != NATIVE_SCHEMA_VERSION:
        raise ValueError(f"{path}: native record schema mismatch")
    n_visual = int(record["n_visual"])
    n_keep = int(record["n_visual_kept_per_head"])
    with np.load(path, allow_pickle=False) as packed:
        if set(packed.files) != {"n_visual", "n_keep", *NATIVE_MASK_NAMES}:
            raise ValueError(
                f"{path}: native NPZ keys mismatch={sorted(packed.files)}"
            )
        if int(_npz_scalar(packed, "n_visual", path)) != n_visual:
            raise ValueError(f"{path}: native n_visual mismatch")
        if int(_npz_scalar(packed, "n_keep", path)) != n_keep:
            raise ValueError(f"{path}: native n_keep mismatch")
        masks = {
            name: _unpack_mask(
                packed,
                name,
                n_visual=n_visual,
                expected_ndim=3,
                path=path,
            )
            for name in NATIVE_MASK_NAMES
        }
    _validate_budget(masks, n_keep=n_keep, path=path)
    return masks


def _load_shared_masks(
    record: Mapping[str, Any],
    path: Path,
) -> dict[str, np.ndarray]:
    if record.get("schema_version") != SHARED_SCHEMA_VERSION:
        raise ValueError(f"{path}: shared record schema mismatch")
    if record.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"{path}: shared experiment_id mismatch")
    if record.get("policy_version") != SHARED_POLICY_VERSION:
        raise ValueError(f"{path}: shared policy_version mismatch")
    if record.get("oracle_decode") != "omitted":
        raise ValueError(f"{path}: shared Oracle decode was not omitted")
    if not record.get("all_compared_masks_shared_across_kv_heads"):
        raise ValueError(f"{path}: shared-mask invariant flag is false")
    expected_record_policy = {
        "selection_policy": "shared_layerwise_head_mean_topk",
        "mask_policy": (
            "one shared layer-wise visual mask broadcast unchanged to all "
            "KV heads"
        ),
        "score_shape": "[L,N_visual]",
        "head_reduction": "arithmetic mean before Top-K",
        "future_reference_source": (
            "raw Full-cache greedy generation attention; head-averaged "
            "before shared Top-K"
        ),
    }
    for key, expected in expected_record_policy.items():
        if record.get(key) != expected:
            raise ValueError(
                f"{path}: {key}={record.get(key)!r}, expected={expected!r}"
            )
    if record.get("future_trajectory") != FUTURE_TRAJECTORY:
        raise ValueError(f"{path}: shared Future trajectory changed")
    if record.get("mask_sha256") != _sha256_file(path):
        raise ValueError(f"{path}: shared mask SHA-256 mismatch")

    n_visual = int(record["n_visual"])
    n_keep = int(record["n_visual_kept_per_layer_and_kv_head"])
    n_kv_heads = int(record["n_kv_heads"])
    if n_kv_heads < 1:
        raise ValueError(f"{path}: n_kv_heads must be positive")
    with np.load(path, allow_pickle=False) as packed:
        expected_keys = {
            "schema_version",
            "policy_version",
            "mask_policy",
            "future_reference",
            "future_trajectory",
            "n_visual",
            "n_keep",
            "n_kv_heads",
            *SHARED_MASK_NAMES,
        }
        if set(packed.files) != expected_keys:
            raise ValueError(
                f"{path}: shared NPZ keys mismatch; "
                f"missing={sorted(expected_keys - set(packed.files))}, "
                f"extra={sorted(set(packed.files) - expected_keys)}"
            )
        expected_scalars = {
            "schema_version": SHARED_SCHEMA_VERSION,
            "policy_version": SHARED_POLICY_VERSION,
            "mask_policy": "shared_across_all_kv_heads",
            "future_reference": "raw_full_cache_generate_attention",
            "future_trajectory": FUTURE_TRAJECTORY,
            "n_visual": n_visual,
            "n_keep": n_keep,
            "n_kv_heads": n_kv_heads,
        }
        for key, expected in expected_scalars.items():
            actual = _npz_scalar(packed, key, path)
            if actual != expected:
                raise ValueError(
                    f"{path}: {key}={actual!r}, expected={expected!r}"
                )
        masks = {
            name: _unpack_mask(
                packed,
                name,
                n_visual=n_visual,
                expected_ndim=2,
                path=path,
            )
            for name in SHARED_MASK_NAMES
        }
    _validate_budget(masks, n_keep=n_keep, path=path)

    fingerprints = record.get("applied_visual_mask_sha256")
    if not isinstance(fingerprints, dict):
        raise ValueError(f"{path}: missing applied mask fingerprints")
    for score_name in (
        "all_prefill_shared",
        "question_shared",
        "qvik_shared",
    ):
        expected = _sha256_mask(masks[f"{score_name}_keep"])
        if fingerprints.get(score_name) != expected:
            raise ValueError(
                f"{path}: applied {score_name} fingerprint mismatch"
            )
    expected_future = _sha256_mask(masks["future_shared_keep"])
    if record.get("future_reference_mask_sha256") != expected_future:
        raise ValueError(f"{path}: Future reference fingerprint mismatch")
    return masks


def _short(value: Any, limit: int = 180) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _require_equal(
    native: Any,
    shared: Any,
    *,
    identity: Identity,
    label: str,
) -> None:
    if native != shared:
        raise ValueError(
            f"{identity!r}: invariant {label} differs; "
            f"native={_short(native)}, shared={_short(shared)}"
        )


def _mapping(
    record: Mapping[str, Any],
    key: str,
    *,
    identity: Identity,
) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{identity!r}: {key} must be an object")
    return value


def _finite_score(
    value: Any,
    *,
    identity: Identity,
    selector: str,
) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{identity!r}: invalid score for {selector}={value!r}"
        ) from error
    if not math.isfinite(score):
        raise ValueError(
            f"{identity!r}: non-finite score for {selector}={score}"
        )
    return score


def _prediction_text(
    value: Any,
    *,
    identity: Identity,
    selector: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{identity!r}: prediction for {selector} is not text"
        )
    return value


def _validate_recomputed_budget(
    native_record: Mapping[str, Any],
    shared_record: Mapping[str, Any],
    *,
    identity: Identity,
    expected_total_keep_ratio: float | None,
) -> None:
    prompt_len = int(native_record["prompt_len"])
    n_text = int(native_record["n_text"])
    n_visual = int(native_record["n_visual"])
    ratio = float(native_record["requested_total_keep_ratio"])
    if prompt_len < 1 or n_text < 0 or n_visual < 1:
        raise ValueError(f"{identity!r}: invalid prompt budget dimensions")
    if n_text + n_visual != prompt_len:
        raise ValueError(
            f"{identity!r}: n_text+n_visual != prompt_len"
        )
    if not 0 < ratio <= 1:
        raise ValueError(
            f"{identity!r}: requested_total_keep_ratio={ratio}"
        )
    if expected_total_keep_ratio is not None and not math.isclose(
        ratio,
        expected_total_keep_ratio,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{identity!r}: requested_total_keep_ratio={ratio}, "
            f"expected={expected_total_keep_ratio}"
        )
    if (
        native_record.get("same_processed_image_for_prefill_and_future")
        is not True
        or shared_record.get(
            "same_processed_image_for_prefill_and_future"
        )
        is not True
    ):
        raise ValueError(
            f"{identity!r}: processed-image reuse invariant is not true"
        )
    expected_keep = min(
        n_visual,
        max(0, math.ceil(ratio * prompt_len) - n_text),
    )
    if expected_keep < 1:
        raise ValueError(
            f"{identity!r}: recomputed visual keep budget is zero"
        )
    native_keep = int(native_record["n_visual_kept_per_head"])
    shared_keep = int(
        shared_record["n_visual_kept_per_layer_and_kv_head"]
    )
    if native_keep != expected_keep or shared_keep != expected_keep:
        raise ValueError(
            f"{identity!r}: stored K native={native_keep}, "
            f"shared={shared_keep}, recomputed={expected_keep}"
        )
    expected_total_ratio = (n_text + expected_keep) / prompt_len
    expected_visual_ratio = expected_keep / n_visual
    for label, record in (
        ("native", native_record),
        ("shared", shared_record),
    ):
        actual_total = float(record["actual_total_keep_ratio"])
        actual_visual = float(record["actual_visual_keep_ratio"])
        if not math.isclose(
            actual_total,
            expected_total_ratio,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"{identity!r}: {label} actual_total_keep_ratio="
                f"{actual_total}, recomputed={expected_total_ratio}"
            )
        if not math.isclose(
            actual_visual,
            expected_visual_ratio,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"{identity!r}: {label} actual_visual_keep_ratio="
                f"{actual_visual}, recomputed={expected_visual_ratio}"
            )


def _compare_vector_pair(
    native_pair: Any,
    shared_pair: Any,
    *,
    identity: Identity,
    pair_label: str,
) -> float:
    if not isinstance(native_pair, dict) or not isinstance(shared_pair, dict):
        raise ValueError(f"{identity!r}: vector pair {pair_label} is not an object")
    if set(native_pair) != set(shared_pair):
        raise ValueError(
            f"{identity!r}: vector metric keys differ for {pair_label}; "
            f"native={sorted(native_pair)}, shared={sorted(shared_pair)}"
        )
    maximum = 0.0
    for metric in native_pair:
        try:
            native_values = np.asarray(native_pair[metric], dtype=np.float64)
            shared_values = np.asarray(shared_pair[metric], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{identity!r}: invalid vector values for "
                f"{pair_label}/{metric}"
            ) from error
        if native_values.shape != shared_values.shape:
            raise ValueError(
                f"{identity!r}: vector shape differs for "
                f"{pair_label}/{metric}: "
                f"{native_values.shape} != {shared_values.shape}"
            )
        if not np.all(np.isfinite(native_values)) or not np.all(
            np.isfinite(shared_values)
        ):
            raise ValueError(
                f"{identity!r}: non-finite vector values for "
                f"{pair_label}/{metric}"
            )
        if native_values.size:
            maximum = max(
                maximum,
                float(np.max(np.abs(native_values - shared_values))),
            )
    if maximum > VECTOR_TOLERANCE:
        raise ValueError(
            f"{identity!r}: non-Future vector invariant {pair_label} "
            f"max_abs_error={maximum} exceeds {VECTOR_TOLERANCE}"
        )
    return maximum


def _validate_join(
    native: ArtifactSnapshot,
    shared: ArtifactSnapshot,
    *,
    expected_total: int,
) -> list[Identity]:
    native_ids = set(native.records)
    shared_ids = set(shared.records)
    intersection = native_ids & shared_ids
    if (
        len(native_ids) != expected_total
        or len(shared_ids) != expected_total
        or len(intersection) != expected_total
        or native_ids != shared_ids
    ):
        raise ValueError(
            "Hard identity join failed for "
            "(dataset, row_index, sample_id): "
            f"native={len(native_ids)}, shared={len(shared_ids)}, "
            f"intersection={len(intersection)}, expected={expected_total}, "
            f"native_only={sorted(native_ids - shared_ids)[:3]}, "
            f"shared_only={sorted(shared_ids - native_ids)[:3]}"
        )
    return sorted(intersection)


def _comparison_row(
    group: str,
    identities: Sequence[Identity],
    native: ArtifactSnapshot,
    shared: ArtifactSnapshot,
) -> dict[str, Any]:
    native_scores: list[float] = []
    shared_scores: list[float] = []
    raw_scores: list[float] = []
    manual_scores: list[float] = []
    native_raw_matches: list[bool] = []
    shared_raw_matches: list[bool] = []
    native_manual_matches: list[bool] = []
    shared_manual_matches: list[bool] = []
    prefill_matches: list[bool] = []

    for identity in identities:
        native_record = native.records[identity]
        shared_record = shared.records[identity]
        native_predictions = _mapping(
            native_record,
            "predictions",
            identity=identity,
        )
        shared_predictions = _mapping(
            shared_record,
            "predictions",
            identity=identity,
        )
        native_score_map = _mapping(native_record, "scores", identity=identity)
        shared_score_map = _mapping(shared_record, "scores", identity=identity)

        raw = _prediction_text(
            native_predictions["full_cache_generate"],
            identity=identity,
            selector="full_cache_generate",
        )
        manual = _prediction_text(
            native_predictions["full_cache_manual"],
            identity=identity,
            selector="full_cache_manual",
        )
        native_prediction = _prediction_text(
            native_predictions["h2o_prefill"],
            identity=identity,
            selector="h2o_prefill",
        )
        shared_prediction = _prediction_text(
            shared_predictions["all_prefill_shared"],
            identity=identity,
            selector="all_prefill_shared",
        )
        native_raw_matches.append(native_prediction == raw)
        shared_raw_matches.append(shared_prediction == raw)
        native_manual_matches.append(native_prediction == manual)
        shared_manual_matches.append(shared_prediction == manual)
        prefill_matches.append(native_prediction == shared_prediction)

        native_scores.append(
            _finite_score(
                native_score_map["h2o_prefill"],
                identity=identity,
                selector="h2o_prefill",
            )
        )
        shared_scores.append(
            _finite_score(
                shared_score_map["all_prefill_shared"],
                identity=identity,
                selector="all_prefill_shared",
            )
        )
        raw_scores.append(
            _finite_score(
                native_score_map["full_cache_generate"],
                identity=identity,
                selector="full_cache_generate",
            )
        )
        manual_scores.append(
            _finite_score(
                native_score_map["full_cache_manual"],
                identity=identity,
                selector="full_cache_manual",
            )
        )

    total = len(identities)
    if total < 1:
        raise ValueError(f"Cannot summarize empty group={group!r}")

    native_raw_count = sum(native_raw_matches)
    shared_raw_count = sum(shared_raw_matches)
    native_manual_count = sum(native_manual_matches)
    shared_manual_count = sum(shared_manual_matches)
    prefill_match_count = sum(prefill_matches)
    score_deltas = [
        shared_score - native_score
        for native_score, shared_score in zip(
            native_scores,
            shared_scores,
            strict=True,
        )
    ]

    return {
        "group": group,
        "n_samples": total,
        "native_selector": "h2o_prefill",
        "shared_selector": "all_prefill_shared",
        "raw_full_task_score_mean": math.fsum(raw_scores) / total,
        "manual_full_task_score_mean": math.fsum(manual_scores) / total,
        "native_task_score_mean": math.fsum(native_scores) / total,
        "shared_task_score_mean": math.fsum(shared_scores) / total,
        "shared_minus_native_task_score_mean": (
            math.fsum(score_deltas) / total
        ),
        "shared_task_score_wins": sum(delta > 0 for delta in score_deltas),
        "task_score_ties": sum(delta == 0 for delta in score_deltas),
        "native_task_score_wins": sum(delta < 0 for delta in score_deltas),
        "native_raw_full_exact_matches": native_raw_count,
        "native_raw_full_exact_match_rate": native_raw_count / total,
        "shared_raw_full_exact_matches": shared_raw_count,
        "shared_raw_full_exact_match_rate": shared_raw_count / total,
        "shared_minus_native_raw_full_exact_match_rate": (
            (shared_raw_count - native_raw_count) / total
        ),
        "shared_only_raw_full_matches": sum(
            shared_match and not native_match
            for native_match, shared_match in zip(
                native_raw_matches,
                shared_raw_matches,
                strict=True,
            )
        ),
        "native_only_raw_full_matches": sum(
            native_match and not shared_match
            for native_match, shared_match in zip(
                native_raw_matches,
                shared_raw_matches,
                strict=True,
            )
        ),
        "native_manual_full_exact_matches": native_manual_count,
        "native_manual_full_exact_match_rate": native_manual_count / total,
        "shared_manual_full_exact_matches": shared_manual_count,
        "shared_manual_full_exact_match_rate": shared_manual_count / total,
        "shared_minus_native_manual_full_exact_match_rate": (
            (shared_manual_count - native_manual_count) / total
        ),
        "native_shared_prefill_exact_matches": prefill_match_count,
        "native_shared_prefill_exact_match_rate": (
            prefill_match_count / total
        ),
    }


def compare_artifacts(
    native_artifact: Path,
    shared_artifact: Path,
    *,
    dataset_counts: Mapping[str, int] = DEFAULT_DATASET_COUNTS,
    expected_total_keep_ratio: float | None = 0.2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate both artifacts and return an audit plus paired CSV rows."""

    counts, expected_total = _expected_counts(dataset_counts)
    native = _load_snapshot(
        native_artifact,
        dataset_counts=counts,
        label="native",
    )
    shared = _load_snapshot(
        shared_artifact,
        dataset_counts=counts,
        label="shared",
    )
    if native.root == shared.root:
        raise ValueError("Native and shared artifacts must be different")

    manifest_audit: dict[str, dict[str, Any]] = {}
    for dataset in counts:
        native_sha = native.manifest_sha256[dataset]
        shared_sha = shared.manifest_sha256[dataset]
        if native_sha != shared_sha:
            raise ValueError(
                f"{dataset}: native/shared manifest SHA-256 differs; "
                f"{native_sha} != {shared_sha}"
            )
        manifest_audit[dataset] = {
            "native_sha256": native_sha,
            "shared_sha256": shared_sha,
            "identical": True,
        }

    identities = _validate_join(
        native,
        shared,
        expected_total=expected_total,
    )
    selector_match_counts = {
        f"{native_name}->{shared_name}": 0
        for native_name, shared_name in INVARIANT_SELECTOR_MAP.items()
    }
    selector_score_match_counts = dict.fromkeys(selector_match_counts, 0)
    vector_max_errors = {
        f"{native_name}->{shared_name}": 0.0
        for native_name, shared_name in NON_FUTURE_VECTOR_PAIR_MAP.items()
    }
    question_head_shared_count = 0
    qvik_head_shared_count = 0
    question_cross_artifact_mask_count = 0
    qvik_cross_artifact_mask_count = 0

    for identity in identities:
        native_record = native.records[identity]
        shared_record = shared.records[identity]
        for field in INVARIANT_FIELDS:
            if field not in native_record or field not in shared_record:
                raise ValueError(
                    f"{identity!r}: invariant field missing={field!r}"
                )
            _require_equal(
                native_record[field],
                shared_record[field],
                identity=identity,
                label=field,
            )
        if native_record["future_trajectory"] != FUTURE_TRAJECTORY:
            raise ValueError(
                f"{identity!r}: native Future trajectory changed from "
                f"{FUTURE_TRAJECTORY!r}"
            )
        _require_equal(
            int(native_record["n_visual_kept_per_head"]),
            int(shared_record["n_visual_kept_per_layer_and_kv_head"]),
            identity=identity,
            label=(
                "n_visual_kept_per_head->"
                "n_visual_kept_per_layer_and_kv_head"
            ),
        )
        _validate_recomputed_budget(
            native_record,
            shared_record,
            identity=identity,
            expected_total_keep_ratio=expected_total_keep_ratio,
        )

        native_predictions = _mapping(
            native_record,
            "predictions",
            identity=identity,
        )
        shared_predictions = _mapping(
            shared_record,
            "predictions",
            identity=identity,
        )
        native_scores = _mapping(native_record, "scores", identity=identity)
        shared_scores = _mapping(shared_record, "scores", identity=identity)
        if set(native_predictions) != NATIVE_SELECTORS:
            raise ValueError(
                f"{identity!r}: native prediction keys mismatch; "
                f"got={sorted(native_predictions)}"
            )
        if set(native_scores) != NATIVE_SELECTORS:
            raise ValueError(
                f"{identity!r}: native score keys mismatch; "
                f"got={sorted(native_scores)}"
            )
        if set(shared_predictions) != SHARED_SELECTORS:
            raise ValueError(
                f"{identity!r}: shared prediction keys mismatch; "
                f"got={sorted(shared_predictions)}"
            )
        if set(shared_scores) != SHARED_SELECTORS:
            raise ValueError(
                f"{identity!r}: shared score keys mismatch; "
                f"got={sorted(shared_scores)}"
            )
        for native_name, shared_name in INVARIANT_SELECTOR_MAP.items():
            label = f"{native_name}->{shared_name}"
            if native_name not in native_predictions:
                raise ValueError(
                    f"{identity!r}: missing native prediction={native_name}"
                )
            if shared_name not in shared_predictions:
                raise ValueError(
                    f"{identity!r}: missing shared prediction={shared_name}"
                )
            _require_equal(
                native_predictions[native_name],
                shared_predictions[shared_name],
                identity=identity,
                label=f"prediction/{label}",
            )
            selector_match_counts[label] += 1
            if native_name not in native_scores or shared_name not in shared_scores:
                raise ValueError(
                    f"{identity!r}: missing score for invariant={label}"
                )
            native_score = _finite_score(
                native_scores[native_name],
                identity=identity,
                selector=native_name,
            )
            shared_score = _finite_score(
                shared_scores[shared_name],
                identity=identity,
                selector=shared_name,
            )
            _require_equal(
                native_score,
                shared_score,
                identity=identity,
                label=f"score/{label}",
            )
            selector_score_match_counts[label] += 1

        native_masks = _load_native_masks(
            native_record,
            native.mask_paths[identity],
        )
        shared_masks = _load_shared_masks(
            shared_record,
            shared.mask_paths[identity],
        )
        reference_shape = native_masks["question_keep"].shape
        if len(reference_shape) != 3:
            raise ValueError(
                f"{identity!r}: native masks are not [L,H,N]"
            )
        layers, heads, n_visual = reference_shape
        if shared_masks["question_shared_keep"].shape != (layers, n_visual):
            raise ValueError(
                f"{identity!r}: native/shared mask dimensions differ"
            )
        if heads != int(shared_record["n_kv_heads"]):
            raise ValueError(
                f"{identity!r}: native heads={heads}, "
                f"shared n_kv_heads={shared_record['n_kv_heads']}"
            )

        native_question = native_masks["question_keep"]
        if not np.all(native_question == native_question[:, :1, :]):
            raise ValueError(
                f"{identity!r}: native Question mask differs across heads"
            )
        question_head_shared_count += 1
        if not np.array_equal(
            native_question[:, 0, :],
            shared_masks["question_shared_keep"],
        ):
            raise ValueError(
                f"{identity!r}: native Question and new shared Question "
                "masks differ"
            )
        question_cross_artifact_mask_count += 1

        native_qvik = native_masks["qvik_keep"]
        if not np.all(native_qvik == native_qvik[:, :1, :]):
            raise ValueError(
                f"{identity!r}: native Q-ViK mask differs across heads"
            )
        qvik_head_shared_count += 1
        if not np.array_equal(
            native_qvik[:, 0, :],
            shared_masks["qvik_shared_keep"],
        ):
            raise ValueError(
                f"{identity!r}: native Q-ViK and new shared Q-ViK masks differ"
            )
        qvik_cross_artifact_mask_count += 1

        native_pairs = _mapping(
            native_record,
            "vector_pairs",
            identity=identity,
        )
        shared_pairs = _mapping(
            shared_record,
            "vector_pairs",
            identity=identity,
        )
        for native_name, shared_name in NON_FUTURE_VECTOR_PAIR_MAP.items():
            if native_name not in native_pairs or shared_name not in shared_pairs:
                raise ValueError(
                    f"{identity!r}: missing non-Future vector pair "
                    f"{native_name}->{shared_name}"
                )
            label = f"{native_name}->{shared_name}"
            maximum = _compare_vector_pair(
                native_pairs[native_name],
                shared_pairs[shared_name],
                identity=identity,
                pair_label=label,
            )
            vector_max_errors[label] = max(
                vector_max_errors[label],
                maximum,
            )

    per_dataset_join = {
        dataset: sum(identity[0] == dataset for identity in identities)
        for dataset in counts
    }
    if per_dataset_join != counts:
        raise ValueError(
            f"Joined per-dataset counts={per_dataset_join}, expected={counts}"
        )

    groups: list[tuple[str, list[Identity]]] = [("all", identities)]
    groups.extend(
        (
            dataset,
            [identity for identity in identities if identity[0] == dataset],
        )
        for dataset in counts
    )
    csv_rows = [
        _comparison_row(group, selected, native, shared)
        for group, selected in groups
    ]

    audit: dict[str, Any] = {
        "status": "pass",
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": {
            "native": str(native.root),
            "shared": str(shared.root),
            "write_scope": str(shared.root / "summary"),
            "native_artifact_was_read_only": True,
            "shared_artifact_outside_summary_was_read_only": True,
        },
        "identity_join": {
            "key_fields": ["dataset", "row_index", "sample_id"],
            "expected": expected_total,
            "native": len(native.records),
            "shared": len(shared.records),
            "intersection": len(identities),
            "union": len(set(native.records) | set(shared.records)),
            "exact_set_equality": True,
            "per_dataset": per_dataset_join,
        },
        "manifest_sha256": manifest_audit,
        "invariants": {
            "scalar_metadata": {
                "fields": list(INVARIANT_FIELDS),
                "samples_checked": expected_total,
                "field_comparisons": expected_total * len(INVARIANT_FIELDS),
            },
            "predictions_exact": selector_match_counts,
            "scores_exact": selector_score_match_counts,
            "mapped_keep_budget_exact": {
                "samples_checked": expected_total,
                "native_field": "n_visual_kept_per_head",
                "shared_field": (
                    "n_visual_kept_per_layer_and_kv_head"
                ),
                "independently_recomputed_from": (
                    "ceil(requested_total_keep_ratio * prompt_len) - n_text"
                ),
                "ratios_recomputed": True,
                "expected_requested_total_keep_ratio": (
                    expected_total_keep_ratio
                ),
            },
            "mask_equivalence": {
                "native_question_identical_across_all_heads": (
                    question_head_shared_count
                ),
                "native_qvik_identical_across_all_heads": (
                    qvik_head_shared_count
                ),
                "native_question_equals_shared_question": (
                    question_cross_artifact_mask_count
                ),
                "native_qvik_equals_shared_qvik": (
                    qvik_cross_artifact_mask_count
                ),
                "samples_checked": expected_total,
            },
            "non_future_vector_pairs": {
                "tolerance": VECTOR_TOLERANCE,
                "max_abs_error": vector_max_errors,
                "samples_checked": expected_total,
            },
            "future_trajectory_unchanged": FUTURE_TRAJECTORY,
            "shared_oracle_decode": "omitted",
        },
        "policy_comparison": {
            "native_selector": "h2o_prefill",
            "native_policy": "per-head native H2O Top-K",
            "shared_selector": "all_prefill_shared",
            "shared_policy": "head-average [L,N] then one shared Top-K",
            "primary_response_reference": "full_cache_generate",
            "secondary_response_reference": "full_cache_manual",
            "output_csv": "summary/native_vs_shared_prefill.csv",
        },
        "excluded_cross_definition_comparisons": [
            {
                "native": "future_keep [L,H,N]",
                "shared": "future_shared_keep [L,N]",
                "status": "not_compared",
                "reason": (
                    "native applies Top-K per head; shared averages heads "
                    "before Top-K"
                ),
            },
            {
                "native": "token_metrics.*Future*",
                "shared": "token_metrics.future_agreement",
                "status": "not_compared",
                "reason": "the Future Top-K reference definition changed",
            },
            {
                "native": "vector_pairs.*_vs_future",
                "shared": "vector_pairs.*__future_shared",
                "status": "not_compared",
                "reason": (
                    "Future-derived cross-run metrics are excluded to avoid "
                    "mixing head-wise and shared definitions"
                ),
            },
            {
                "native": "future_oracle decode prediction/score",
                "shared": "no Oracle decode",
                "status": "not_compared",
                "reason": "the shared experiment intentionally omits Oracle decode",
            },
        ],
    }
    return audit, csv_rows


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("Refusing to write an empty comparison CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("Comparison CSV rows have inconsistent columns")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_reports(
    shared_artifact: Path,
    audit: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Write both reports only under the shared artifact's summary directory."""

    shared_root = shared_artifact.resolve(strict=True)
    summary = shared_root / "summary"
    if summary.exists():
        resolved_summary = summary.resolve()
        try:
            resolved_summary.relative_to(shared_root)
        except ValueError as error:
            raise ValueError(
                f"Refusing summary symlink outside shared artifact: {summary}"
            ) from error
    summary.mkdir(parents=True, exist_ok=True)

    audit_path = summary / "invariance_audit.json"
    csv_path = summary / "native_vs_shared_prefill.csv"
    audit_text = (
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    comparison_text = _csv_text(csv_rows)
    _atomic_write_text(csv_path, comparison_text)
    # Write the pass audit last so its presence implies that the CSV exists.
    _atomic_write_text(audit_path, audit_text)
    return audit_path, csv_path


def run_comparison(
    native_artifact: Path,
    shared_artifact: Path,
    *,
    dataset_counts: Mapping[str, int] = DEFAULT_DATASET_COUNTS,
    expected_total_keep_ratio: float | None = 0.2,
) -> tuple[Path, Path]:
    """Validate first, then write reports into the shared summary directory."""

    audit, csv_rows = compare_artifacts(
        native_artifact,
        shared_artifact,
        dataset_counts=dataset_counts,
        expected_total_keep_ratio=expected_total_keep_ratio,
    )
    return write_reports(shared_artifact, audit, csv_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit_path, csv_path = run_comparison(
            args.native_artifact,
            args.shared_artifact,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[compare-native-shared] ERROR: {error}", file=sys.stderr)
        return 1
    print(f"[compare-native-shared] wrote {audit_path}")
    print(f"[compare-native-shared] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
