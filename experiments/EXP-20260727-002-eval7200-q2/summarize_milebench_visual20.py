#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the MileBench Visual-KV 20% selector experiment.

This is deliberately separate from the seven standard benchmarks.  The
standard benchmarks use a *total prompt-cache* budget, whereas this MileBench
run keeps ``ceil(0.2 * N_visual)`` visual KVs while preserving every text KV.
Combining the two suites into one macro would therefore compare different
budgets and is forbidden here.

One completed JSON contains predictions and scores for every selector, and its
matching NPZ contains every selector's packed keep mask.  The summary includes
only complete, validated JSON+NPZ pairs.  Consequently an OOM exclusion is a
shared sample exclusion: Full, H2O, semantic-user-prompt, Q-ViK, and Future
Oracle are always evaluated on exactly the same retained sample set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = (
    ZAP_ROOT
    / "artifacts"
    / "rebuttal_milebench5800_q2_llava15_visual0p2"
)
MILEBENCH_PREFIX = "milebench__"
VISUAL_KEEP_RATIO = 0.2

# Mirrored from eval7200_data.py.  Keeping this CPU-only summarizer independent
# of model/tokenizer imports makes it safe to run while inference is active.
OFFICIAL_MILEBENCH_TASKS = (
    "ALFRED",
    "ActionLocalization",
    "ActionPrediction",
    "ActionSequence",
    "CLEVR-Change",
    "CharacterOrder",
    "CounterfactualInference",
    "DocVQA",
    "EgocentricNavigation",
    "GPR1200",
    "IEdit",
    "ImageNeedleInAHaystack",
    "MMCoQA",
    "MovingAttribute",
    "MovingDirection",
    "MultiModalQA",
    "OCR-VQA",
    "ObjectExistence",
    "ObjectInteraction",
    "ObjectShuffle",
    "SceneTransition",
    "SlideVQA",
    "Spot-the-Diff",
    "StateChange",
    "TQA",
    "TextNeedleInAHaystack",
    "WebQA",
    "WikiVQA",
)
LOCAL_EXTENSION_TASKS = ("nuscenes",)
LOCAL_MILEBENCH_TASKS = OFFICIAL_MILEBENCH_TASKS + LOCAL_EXTENSION_TASKS
OFFICIAL_DATASETS = tuple(
    f"{MILEBENCH_PREFIX}{task}" for task in OFFICIAL_MILEBENCH_TASKS
)
LOCAL_DATASETS = tuple(
    f"{MILEBENCH_PREFIX}{task}" for task in LOCAL_MILEBENCH_TASKS
)

SELECTORS = (
    "full_cache_generate",
    "full_cache_manual",
    "h2o_prefill",
    "question_prefill",
    "qvik",
    "future_oracle",
)
SELECTOR_LABELS = {
    "full_cache_generate": "Full",
    "full_cache_manual": "Matched Full",
    "h2o_prefill": "H2O",
    "question_prefill": "Semantic user-prompt",
    "qvik": "Q-ViK",
    "future_oracle": "Future Oracle",
}
VECTOR_PAIRS = (
    "h2o_vs_future",
    "question_vs_future",
    "qvik_vs_future",
    "qvik_vs_h2o",
    "qvik_vs_question",
)
VECTOR_METRICS = (
    "spearman",
    "cosine",
    "jaccard_10",
    "jaccard_20",
    "jaccard_50",
)
TOKEN_METRICS = (
    "h2o_future_topk_recall",
    "h2o_jaccard_vs_future",
    "h2o_visual_future_mass_retained",
    "question_future_topk_recall",
    "question_jaccard_vs_future",
    "question_visual_future_mass_retained",
    "qvik_future_topk_recall",
    "qvik_jaccard_vs_future",
    "qvik_visual_future_mass_retained",
    "qvik_minus_h2o_future_recall",
    "qvik_minus_h2o_jaccard",
    "qvik_minus_h2o_future_mass",
    "qvik_useful_additions_per_k",
    "qvik_harmful_removals_per_k",
    "qvik_head_win_fraction",
    "qvik_head_tie_fraction",
    "qvik_head_loss_fraction",
    "qvik_h2o_keep_overlap_per_k",
    "qvik_h2o_keep_jaccard",
    "qvik_h2o_swap_fraction",
    "question_h2o_keep_overlap_per_k",
    "question_h2o_keep_jaccard",
    "question_h2o_swap_fraction",
    "qvik_question_keep_overlap_per_k",
    "qvik_question_keep_jaccard",
    "qvik_question_swap_fraction",
)
MASK_KEYS = (
    "h2o_keep",
    "question_keep",
    "qvik_keep",
    "future_keep",
)
MASK_METRICS = (
    "h2o_future_topk_recall",
    "h2o_jaccard_vs_future",
    "question_future_topk_recall",
    "question_jaccard_vs_future",
    "qvik_future_topk_recall",
    "qvik_jaccard_vs_future",
    "qvik_h2o_keep_overlap_per_k",
    "qvik_h2o_keep_jaccard",
    "qvik_h2o_swap_fraction",
    "question_h2o_keep_overlap_per_k",
    "question_h2o_keep_jaccard",
    "question_h2o_swap_fraction",
    "qvik_question_keep_overlap_per_k",
    "qvik_question_keep_jaccard",
    "qvik_question_swap_fraction",
)


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    sample_id: str
    row_index: int
    stem: str


@dataclass
class Record:
    dataset: str
    sample_id: str
    row_index: int
    scores: np.ndarray
    predictions: tuple[str, ...]
    vector: np.ndarray
    token: np.ndarray
    masks: np.ndarray
    n_visual: int
    n_keep: int
    prompt_len: int
    n_text: int
    actual_total_keep_ratio: float
    answer_steps: int
    hit_max_new_tokens: bool


@dataclass
class GroupStats:
    name: str
    group_type: str
    task_names: tuple[str, ...]
    keys: tuple[str, ...]
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    bootstrap: np.ndarray
    n_samples: int
    complete_suite: bool

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    def value(self, key: str) -> tuple[float, float, float]:
        index = self.keys.index(key)
        return (
            float(self.mean[index]),
            float(self.low[index]),
            float(self.high[index]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize complete MileBench JSON+mask pairs at Visual-KV 20%."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        help=(
            "Destination for summary files. Defaults to "
            "<output-root>/summary_milebench_visual20."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(LOCAL_MILEBENCH_TASKS),
        help="Original task names or milebench__-prefixed dataset names.",
    )
    parser.add_argument("--expected-per-task", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--layer-bootstrap-replicates",
        type=int,
        default=2_000,
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Permit missing manifests, in-progress samples, dangling pairs, or "
            "non-OOM failures for a read-only progress summary. OOM failures "
            "are always permitted as shared sample exclusions."
        ),
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write tables and report without rendering PNG/PDF.",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.summary_dir = (
        args.summary_dir.resolve()
        if args.summary_dir is not None
        else (args.output_root / "summary_milebench_visual20").resolve()
    )
    args.datasets = normalize_datasets(args.tasks)
    if args.expected_per_task <= 0:
        parser.error("--expected-per-task must be positive")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if args.layer_bootstrap_replicates <= 0:
        parser.error("--layer-bootstrap-replicates must be positive")
    forbidden = {
        (args.output_root / name).resolve()
        for name in ("samples", "masks", "manifests", "failures")
    }
    if args.summary_dir in forbidden:
        parser.error("--summary-dir may not overwrite an inference directory")
    return args


def normalize_datasets(values: Sequence[str]) -> tuple[str, ...]:
    datasets: list[str] = []
    for value in values:
        dataset = (
            value
            if value.startswith(MILEBENCH_PREFIX)
            else f"{MILEBENCH_PREFIX}{value}"
        )
        if dataset not in LOCAL_DATASETS:
            raise ValueError(
                f"Unknown MileBench task {value!r}; expected one of "
                f"{LOCAL_MILEBENCH_TASKS}"
            )
        if dataset not in datasets:
            datasets.append(dataset)
    if not datasets:
        raise ValueError("At least one MileBench task is required")
    return tuple(datasets)


def safe_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return sanitized[:160] or "sample"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def is_oom_reason(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        marker in lowered
        for marker in (
            "outofmemory",
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
    )


def finite_array(
    value: Any,
    *,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name}: expected one dimension, got {array.shape}")
    if expected_length is not None and len(array) != expected_length:
        raise ValueError(
            f"{name}: expected {expected_length} layers, got {len(array)}"
        )
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name}: empty or non-finite values")
    return array


def unpack_and_validate_masks(
    path: Path,
    *,
    n_visual: int,
    n_keep: int,
    n_layers: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as packed:
        required = {"n_visual", "n_keep", *MASK_KEYS}
        missing = required - set(packed.files)
        if missing:
            raise ValueError(f"{path}: missing packed arrays {sorted(missing)}")
        if int(packed["n_visual"]) != n_visual:
            raise ValueError(f"{path}: packed n_visual disagrees with JSON")
        if int(packed["n_keep"]) != n_keep:
            raise ValueError(f"{path}: packed n_keep disagrees with JSON")
        masks = {
            key: np.unpackbits(packed[key], axis=-1)[..., :n_visual].astype(
                bool
            )
            for key in MASK_KEYS
        }
    shapes = {key: value.shape for key, value in masks.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{path}: selector mask shapes differ: {shapes}")
    shape = next(iter(shapes.values()))
    if len(shape) != 3 or shape[0] != n_layers or shape[-1] != n_visual:
        raise ValueError(
            f"{path}: expected [layers,heads,{n_visual}], got {shape}"
        )
    for key, mask in masks.items():
        counts = mask.sum(axis=-1)
        if not np.all(counts == n_keep):
            raise ValueError(
                f"{path}: {key} does not retain exactly {n_keep} tokens/head"
            )

    layer_values: dict[str, np.ndarray] = {}
    future = masks["future_keep"]
    for prefix in ("h2o", "question", "qvik"):
        selected = masks[f"{prefix}_keep"]
        intersection = np.logical_and(selected, future).sum(axis=-1)
        union = np.logical_or(selected, future).sum(axis=-1)
        layer_values[f"{prefix}_future_topk_recall"] = (
            intersection / n_keep
        ).mean(axis=1)
        layer_values[f"{prefix}_jaccard_vs_future"] = (
            intersection / np.maximum(union, 1)
        ).mean(axis=1)

    for label, first, second in (
        ("qvik_h2o", "qvik_keep", "h2o_keep"),
        ("question_h2o", "question_keep", "h2o_keep"),
        ("qvik_question", "qvik_keep", "question_keep"),
    ):
        intersection = np.logical_and(masks[first], masks[second]).sum(axis=-1)
        union = np.logical_or(masks[first], masks[second]).sum(axis=-1)
        overlap = (intersection / n_keep).mean(axis=1)
        layer_values[f"{label}_keep_overlap_per_k"] = overlap
        layer_values[f"{label}_keep_jaccard"] = (
            intersection / np.maximum(union, 1)
        ).mean(axis=1)
        layer_values[f"{label}_swap_fraction"] = 1.0 - overlap
    return np.stack([layer_values[key] for key in MASK_METRICS])


def validate_record(
    *,
    record_path: Path,
    mask_path: Path,
    manifest: ManifestRow,
    output_root: Path,
) -> Record:
    payload = json.loads(record_path.read_text())
    if payload.get("dataset") != manifest.dataset:
        raise ValueError(
            f"{record_path}: dataset={payload.get('dataset')!r}, expected "
            f"{manifest.dataset!r}"
        )
    if str(payload.get("sample_id")) != manifest.sample_id:
        raise ValueError(f"{record_path}: sample_id disagrees with manifest")
    if int(payload.get("row_index", -1)) != manifest.row_index:
        raise ValueError(f"{record_path}: row_index disagrees with manifest")
    if not str(payload["dataset"]).startswith(MILEBENCH_PREFIX):
        raise ValueError(f"{record_path}: standard-eval records are forbidden")
    if payload.get("budget_mode") != "visual":
        raise ValueError(
            f"{record_path}: budget_mode must be 'visual', got "
            f"{payload.get('budget_mode')!r}"
        )
    requested_visual = payload.get(
        "requested_visual_keep_ratio",
        payload.get("requested_keep_ratio"),
    )
    if not math.isclose(
        float(requested_visual),
        VISUAL_KEEP_RATIO,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{record_path}: requested visual keep ratio is "
            f"{requested_visual}, not {VISUAL_KEEP_RATIO}"
        )
    if payload.get("requested_total_keep_ratio") is not None:
        raise ValueError(
            f"{record_path}: total-cache budget record cannot enter visual20"
        )

    n_visual = int(payload["n_visual"])
    n_keep = int(payload["n_visual_kept_per_head"])
    expected_keep = int(math.ceil(VISUAL_KEEP_RATIO * n_visual))
    if n_keep != expected_keep:
        raise ValueError(
            f"{record_path}: n_keep={n_keep}, expected ceil(.2*"
            f"{n_visual})={expected_keep}"
        )
    if not math.isclose(
        float(payload["actual_visual_keep_ratio"]),
        n_keep / n_visual,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{record_path}: incorrect actual visual keep ratio")
    expected_relative_mask = mask_path.relative_to(output_root).as_posix()
    if payload.get("mask_file") != expected_relative_mask:
        raise ValueError(
            f"{record_path}: mask_file={payload.get('mask_file')!r}, "
            f"expected {expected_relative_mask!r}"
        )

    predictions_payload = payload.get("predictions", {})
    scores_payload = payload.get("scores", {})
    missing_predictions = set(SELECTORS) - set(predictions_payload)
    missing_scores = set(SELECTORS) - set(scores_payload)
    if missing_predictions or missing_scores:
        raise ValueError(
            f"{record_path}: incomplete all-method result; predictions "
            f"missing={sorted(missing_predictions)}, scores "
            f"missing={sorted(missing_scores)}"
        )
    scores = np.asarray(
        [float(scores_payload[key]) for key in SELECTORS],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise ValueError(f"{record_path}: non-finite downstream score")
    predictions = tuple(str(predictions_payload[key]) for key in SELECTORS)

    vector_payload = payload.get("vector_pairs", {})
    token_payload = payload.get("token_metrics", {})
    missing_pairs = set(VECTOR_PAIRS) - set(vector_payload)
    missing_tokens = set(TOKEN_METRICS) - set(token_payload)
    if missing_pairs or missing_tokens:
        raise ValueError(
            f"{record_path}: missing vector pairs={sorted(missing_pairs)} or "
            f"token metrics={sorted(missing_tokens)}"
        )
    first = finite_array(
        vector_payload[VECTOR_PAIRS[0]][VECTOR_METRICS[0]],
        name=f"{record_path}: first vector metric",
    )
    n_layers = len(first)
    vector = np.empty(
        (len(VECTOR_PAIRS), len(VECTOR_METRICS), n_layers),
        dtype=np.float64,
    )
    for pair_index, pair in enumerate(VECTOR_PAIRS):
        missing_metrics = set(VECTOR_METRICS) - set(vector_payload[pair])
        if missing_metrics:
            raise ValueError(
                f"{record_path}: {pair} missing {sorted(missing_metrics)}"
            )
        for metric_index, metric in enumerate(VECTOR_METRICS):
            vector[pair_index, metric_index] = finite_array(
                vector_payload[pair][metric],
                name=f"{record_path}:{pair}/{metric}",
                expected_length=n_layers,
            )
    token = np.stack(
        [
            finite_array(
                token_payload[key],
                name=f"{record_path}:token/{key}",
                expected_length=n_layers,
            )
            for key in TOKEN_METRICS
        ]
    )
    masks = unpack_and_validate_masks(
        mask_path,
        n_visual=n_visual,
        n_keep=n_keep,
        n_layers=n_layers,
    )

    # Ensure the metrics saved in JSON describe the actual deployed masks.
    token_index = {key: index for index, key in enumerate(TOKEN_METRICS)}
    for mask_index, key in enumerate(MASK_METRICS):
        if key not in token_index:
            continue
        if not np.allclose(
            masks[mask_index],
            token[token_index[key]],
            rtol=1e-6,
            atol=2e-6,
        ):
            raise ValueError(
                f"{record_path}: packed-mask metric {key} disagrees with JSON"
            )

    prompt_len = int(payload["prompt_len"])
    n_text = int(payload["n_text"])
    if prompt_len != n_text + n_visual:
        raise ValueError(f"{record_path}: prompt_len != n_text + n_visual")
    return Record(
        dataset=manifest.dataset,
        sample_id=manifest.sample_id,
        row_index=manifest.row_index,
        scores=scores,
        predictions=predictions,
        vector=vector,
        token=token,
        masks=masks,
        n_visual=n_visual,
        n_keep=n_keep,
        prompt_len=prompt_len,
        n_text=n_text,
        actual_total_keep_ratio=float(payload["actual_total_keep_ratio"]),
        answer_steps=int(payload["answer_steps"]),
        hit_max_new_tokens=bool(payload["hit_max_new_tokens"]),
    )


def load_complete_records(
    args: argparse.Namespace,
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    records_by_task: dict[str, list[Record]] = {}
    exclusions: list[dict[str, Any]] = []
    task_audit: dict[str, dict[str, Any]] = {}
    fatal: list[str] = []

    for dataset in args.datasets:
        manifest_path = args.output_root / "manifests" / f"{dataset}.json"
        if not manifest_path.exists():
            row = {
                "dataset": dataset,
                "sample_id": None,
                "row_index": None,
                "kind": "manifest_missing",
                "reason": "manifest missing; task has not started",
            }
            exclusions.append(row)
            task_audit[dataset] = {
                "manifest_rows": 0,
                "complete_pairs": 0,
                "shared_oom_exclusions": 0,
                "other_exclusions": 1,
            }
            if not args.allow_partial:
                fatal.append(f"{dataset}: manifest missing")
            continue
        manifest_payload = json.loads(manifest_path.read_text())
        manifest_rows = manifest_payload.get("samples", [])
        if len(manifest_rows) != args.expected_per_task:
            message = (
                f"{dataset}: manifest has {len(manifest_rows)} rows, expected "
                f"{args.expected_per_task}"
            )
            if not args.allow_partial:
                fatal.append(message)

        seen: set[tuple[int, str]] = set()
        complete: list[Record] = []
        oom_count = 0
        other_count = 0
        expected_stems: set[str] = set()
        for raw in manifest_rows:
            sample_id = str(raw["sample_id"])
            row_index = int(raw["row_index"])
            identity = (row_index, sample_id)
            if identity in seen:
                fatal.append(f"{dataset}: duplicate manifest identity {identity}")
                continue
            seen.add(identity)
            stem = f"{row_index:06d}_{safe_name(sample_id)}"
            expected_stems.add(stem)
            manifest = ManifestRow(
                dataset=dataset,
                sample_id=sample_id,
                row_index=row_index,
                stem=stem,
            )
            result_path = (
                args.output_root / "samples" / dataset / f"{stem}.json"
            )
            mask_path = args.output_root / "masks" / dataset / f"{stem}.npz"
            failure_path = (
                args.output_root / "failures" / dataset / f"{stem}.txt"
            )
            result_exists = result_path.exists()
            mask_exists = mask_path.exists()
            if result_exists and mask_exists:
                try:
                    complete.append(
                        validate_record(
                            record_path=result_path,
                            mask_path=mask_path,
                            manifest=manifest,
                            output_root=args.output_root,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    reason = f"{type(error).__name__}: {error}"
                    exclusions.append(
                        {
                            "dataset": dataset,
                            "sample_id": sample_id,
                            "row_index": row_index,
                            "kind": "invalid_complete_pair",
                            "reason": reason,
                        }
                    )
                    other_count += 1
                    if not args.allow_partial:
                        fatal.append(reason)
                continue

            reason = (
                failure_path.read_text().strip()
                if failure_path.exists()
                else "inference output not completed"
            )
            if not result_exists and not mask_exists and is_oom_reason(reason):
                kind = "shared_oom"
                oom_count += 1
            elif result_exists != mask_exists:
                kind = "dangling_json_or_mask"
                other_count += 1
                if not args.allow_partial:
                    fatal.append(
                        f"{dataset}/{stem}: JSON/mask pair is incomplete"
                    )
            else:
                kind = "other_failure_or_in_progress"
                other_count += 1
                if not args.allow_partial:
                    fatal.append(f"{dataset}/{stem}: {reason}")
            exclusions.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "row_index": row_index,
                    "kind": kind,
                    "reason": reason,
                }
            )

        for kind, directory_name, suffix in (
            ("json", "samples", ".json"),
            ("mask", "masks", ".npz"),
        ):
            directory = args.output_root / directory_name / dataset
            actual_stems = (
                {path.stem for path in directory.glob(f"*{suffix}")}
                if directory.exists()
                else set()
            )
            orphaned = sorted(actual_stems - expected_stems)
            if orphaned:
                message = (
                    f"{dataset}: {len(orphaned)} orphan {kind} files outside "
                    "the manifest"
                )
                if not args.allow_partial:
                    fatal.append(message)
                exclusions.append(
                    {
                        "dataset": dataset,
                        "sample_id": None,
                        "row_index": None,
                        "kind": f"orphan_{kind}",
                        "reason": message,
                        "stems": orphaned,
                    }
                )
                other_count += len(orphaned)

        complete.sort(key=lambda item: (item.row_index, item.sample_id))
        if complete:
            records_by_task[dataset] = complete
        elif not args.allow_partial:
            fatal.append(f"{dataset}: no complete JSON+mask records")
        task_audit[dataset] = {
            "manifest_rows": len(manifest_rows),
            "complete_pairs": len(complete),
            "shared_oom_exclusions": oom_count,
            "other_exclusions": other_count,
        }

    if fatal:
        preview = "\n".join(f"  - {line}" for line in fatal[:25])
        remainder = len(fatal) - min(len(fatal), 25)
        if remainder:
            preview += f"\n  - ... and {remainder} more"
        raise RuntimeError(
            "MileBench visual20 summary validation failed:\n" + preview
        )
    if not records_by_task:
        raise RuntimeError("No complete MileBench JSON+mask pairs were found")

    audit = {
        "schema_version": 1,
        "input_root": str(args.output_root),
        "budget_scope": "visual_kv",
        "requested_visual_keep_ratio": VISUAL_KEEP_RATIO,
        "standard_total20_records_included": False,
        "sample_inclusion_rule": (
            "validated MileBench JSON plus matching packed all-method mask NPZ"
        ),
        "shared_exclusion_rule": (
            "a missing/failed sample is removed once from every selector"
        ),
        "allow_partial": bool(args.allow_partial),
        "requested_tasks": list(args.datasets),
        "tasks_with_complete_records": list(records_by_task),
        "task_audit": task_audit,
        "complete_pairs": sum(map(len, records_by_task.values())),
        "shared_oom_exclusions": sum(
            row["kind"] == "shared_oom" for row in exclusions
        ),
        "other_exclusions": sum(
            row["kind"] != "shared_oom" for row in exclusions
        ),
        "exclusions": exclusions,
    }
    return records_by_task, audit


def scalar_features(record: Record) -> dict[str, float]:
    values: dict[str, float] = {}
    selector_index = {name: index for index, name in enumerate(SELECTORS)}
    for selector, index in selector_index.items():
        values[f"downstream/{selector}"] = float(record.scores[index])
        values[f"preservation/{selector}"] = float(
            record.predictions[index]
            == record.predictions[selector_index["full_cache_manual"]]
        )
    values["downstream/qvik_minus_h2o_prefill"] = float(
        record.scores[selector_index["qvik"]]
        - record.scores[selector_index["h2o_prefill"]]
    )
    for label, first, second in (
        ("qvik_minus_h2o", "qvik", "h2o_prefill"),
        (
            "future_oracle_minus_h2o",
            "future_oracle",
            "h2o_prefill",
        ),
        (
            "question_minus_h2o",
            "question_prefill",
            "h2o_prefill",
        ),
    ):
        reference = record.predictions[selector_index["full_cache_manual"]]
        values[f"preservation_delta/{label}"] = float(
            record.predictions[selector_index[first]] == reference
        ) - float(record.predictions[selector_index[second]] == reference)

    pair_index = {name: index for index, name in enumerate(VECTOR_PAIRS)}
    metric_index = {
        name: index for index, name in enumerate(VECTOR_METRICS)
    }
    for pair in VECTOR_PAIRS:
        for metric in VECTOR_METRICS:
            values[f"vector/{pair}/{metric}"] = float(
                record.vector[pair_index[pair], metric_index[metric]].mean()
            )
    for metric in VECTOR_METRICS:
        values[f"vector/qvik_minus_h2o_vs_future/{metric}"] = float(
            record.vector[
                pair_index["qvik_vs_future"],
                metric_index[metric],
            ].mean()
            - record.vector[
                pair_index["h2o_vs_future"],
                metric_index[metric],
            ].mean()
        )
    for index, metric in enumerate(TOKEN_METRICS):
        values[f"token/{metric}"] = float(record.token[index].mean())
    for index, metric in enumerate(MASK_METRICS):
        values[f"mask/{metric}"] = float(record.masks[index].mean())
    return values


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode()).digest()
    return int((seed + int.from_bytes(digest[:4], "little")) % (2**32))


def bootstrap_matrix(
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact nonparametric sample bootstrap for every value column."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"Expected nonempty [samples,metrics], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values contain NaN/Inf")
    n_samples, n_metrics = values.shape
    mean = values.mean(axis=0)
    if n_samples == 1:
        bootstrap = np.repeat(mean[None, :], repeats, axis=0)
    else:
        rng = np.random.default_rng(seed)
        bootstrap = np.empty((repeats, n_metrics), dtype=np.float32)
        probabilities = np.full(n_samples, 1.0 / n_samples)
        # Bound the integer weight matrix to roughly 32 MiB.
        chunk_size = min(512, max(1, 4_000_000 // n_samples))
        for start in range(0, repeats, chunk_size):
            count = min(chunk_size, repeats - start)
            weights = rng.multinomial(
                n_samples,
                probabilities,
                size=count,
            )
            bootstrap[start : start + count] = (
                weights @ values / n_samples
            ).astype(np.float32)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return mean, low, high, bootstrap


def make_group(
    *,
    name: str,
    group_type: str,
    task_names: Sequence[str],
    task_groups: dict[str, GroupStats],
    complete_suite: bool,
) -> GroupStats:
    selected = [task_groups[name] for name in task_names]
    keys = selected[0].keys
    if any(group.keys != keys for group in selected):
        raise ValueError("Task bootstrap columns differ")
    mean = sum((group.mean for group in selected), np.zeros_like(selected[0].mean))
    mean /= len(selected)
    bootstrap = sum(
        (
            group.bootstrap.astype(np.float64)
            for group in selected
        ),
        np.zeros_like(selected[0].bootstrap, dtype=np.float64),
    )
    bootstrap /= len(selected)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return GroupStats(
        name=name,
        group_type=group_type,
        task_names=tuple(task_names),
        keys=keys,
        mean=mean,
        low=low,
        high=high,
        bootstrap=bootstrap.astype(np.float32),
        n_samples=sum(group.n_samples for group in selected),
        complete_suite=complete_suite,
    )


def build_scalar_groups(
    records_by_task: dict[str, list[Record]],
    args: argparse.Namespace,
) -> dict[str, GroupStats]:
    first_record = next(iter(next(iter(records_by_task.values()))))
    keys = tuple(sorted(scalar_features(first_record)))
    task_groups: dict[str, GroupStats] = {}
    for dataset, records in records_by_task.items():
        rows = []
        for record in records:
            features = scalar_features(record)
            if tuple(sorted(features)) != keys:
                raise ValueError(f"{dataset}/{record.sample_id}: feature drift")
            rows.append([features[key] for key in keys])
        mean, low, high, bootstrap = bootstrap_matrix(
            np.asarray(rows),
            repeats=args.bootstrap_replicates,
            seed=stable_seed(args.seed, f"scalar/{dataset}"),
        )
        task_groups[dataset] = GroupStats(
            name=dataset,
            group_type="task",
            task_names=(dataset,),
            keys=keys,
            mean=mean,
            low=low,
            high=high,
            bootstrap=bootstrap,
            n_samples=len(records),
            complete_suite=True,
        )

    groups = dict(task_groups)
    present = tuple(
        dataset for dataset in LOCAL_DATASETS if dataset in task_groups
    )
    groups["milebench_present_task_macro"] = make_group(
        name="milebench_present_task_macro",
        group_type="partial_macro",
        task_names=present,
        task_groups=task_groups,
        complete_suite=set(present) == set(args.datasets),
    )
    if all(dataset in task_groups for dataset in OFFICIAL_DATASETS):
        groups["milebench_official28"] = make_group(
            name="milebench_official28",
            group_type="official_macro",
            task_names=OFFICIAL_DATASETS,
            task_groups=task_groups,
            complete_suite=True,
        )
    if all(dataset in task_groups for dataset in LOCAL_DATASETS):
        groups["milebench_local29"] = make_group(
            name="milebench_local29",
            group_type="local_macro",
            task_names=LOCAL_DATASETS,
            task_groups=task_groups,
            complete_suite=True,
        )
    return groups


def group_rows(groups: dict[str, GroupStats]) -> Iterable[GroupStats]:
    for dataset in LOCAL_DATASETS:
        if dataset in groups:
            yield groups[dataset]
    for name in (
        "milebench_official28",
        "milebench_local29",
        "milebench_present_task_macro",
    ):
        if name in groups:
            if name == "milebench_present_task_macro" and (
                "milebench_local29" in groups
            ):
                continue
            yield groups[name]


def common_fields(group: GroupStats) -> dict[str, Any]:
    return {
        "group": group.name,
        "group_type": group.group_type,
        "n_tasks": group.n_tasks,
        "n_samples": group.n_samples,
        "complete_suite": group.complete_suite,
        "budget": "Visual-KV 20%",
    }


def make_scalar_tables(
    groups: dict[str, GroupStats],
) -> dict[str, list[dict[str, Any]]]:
    downstream: list[dict[str, Any]] = []
    vector: list[dict[str, Any]] = []
    token: list[dict[str, Any]] = []
    mask: list[dict[str, Any]] = []
    preservation: list[dict[str, Any]] = []
    preservation_paired: list[dict[str, Any]] = []
    for group in group_rows(groups):
        fields = common_fields(group)
        for selector in (*SELECTORS, "qvik_minus_h2o_prefill"):
            mean, low, high = group.value(f"downstream/{selector}")
            downstream.append(
                {
                    **fields,
                    "selector": selector,
                    "mean": 100.0 * mean,
                    "ci95_low": 100.0 * low,
                    "ci95_high": 100.0 * high,
                    "scale": "0-100",
                }
            )
        for selector in SELECTORS:
            mean, low, high = group.value(f"preservation/{selector}")
            raw_matches = None
            if group.group_type == "task":
                raw_matches = int(round(mean * group.n_samples))
            preservation.append(
                {
                    **fields,
                    "selector": selector,
                    "reference": "full_cache_manual",
                    "exact_matches": raw_matches,
                    "match_rate": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "aggregation": (
                        "sample mean"
                        if group.group_type == "task"
                        else "unweighted task macro"
                    ),
                }
            )
        for comparison in (
            "qvik_minus_h2o",
            "future_oracle_minus_h2o",
            "question_minus_h2o",
        ):
            mean, low, high = group.value(
                f"preservation_delta/{comparison}"
            )
            preservation_paired.append(
                {
                    **fields,
                    "comparison": comparison,
                    "mean_delta": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        for pair in (*VECTOR_PAIRS, "qvik_minus_h2o_vs_future"):
            for metric in VECTOR_METRICS:
                mean, low, high = group.value(f"vector/{pair}/{metric}")
                vector.append(
                    {
                        **fields,
                        "pair": pair,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
        for metric in TOKEN_METRICS:
            mean, low, high = group.value(f"token/{metric}")
            token.append(
                {
                    **fields,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        for metric in MASK_METRICS:
            mean, low, high = group.value(f"mask/{metric}")
            mask.append(
                {
                    **fields,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "source": "recomputed from packed keep masks",
                }
            )
    return {
        "downstream": downstream,
        "vector": vector,
        "token": token,
        "mask": mask,
        "preservation": preservation,
        "preservation_paired": preservation_paired,
    }


def layer_features(record: Record) -> tuple[tuple[str, ...], np.ndarray]:
    pair_index = {name: index for index, name in enumerate(VECTOR_PAIRS)}
    metric_index = {
        name: index for index, name in enumerate(VECTOR_METRICS)
    }
    mask_index = {name: index for index, name in enumerate(MASK_METRICS)}
    keys: list[str] = []
    values: list[np.ndarray] = []
    for method, pair in (
        ("h2o", "h2o_vs_future"),
        ("question", "question_vs_future"),
        ("qvik", "qvik_vs_future"),
    ):
        keys.append(f"spearman/{method}")
        values.append(
            record.vector[
                pair_index[pair],
                metric_index["spearman"],
            ]
        )
    for metric in ("future_topk_recall", "jaccard_vs_future"):
        for method in ("h2o", "question", "qvik"):
            key = f"{method}_{metric}"
            keys.append(f"{metric}/{method}")
            values.append(record.masks[mask_index[key]])
    return tuple(keys), np.stack(values)


def build_layerwise(
    records_by_task: dict[str, list[Record]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    first = next(iter(next(iter(records_by_task.values()))))
    keys, first_values = layer_features(first)
    n_layers = first_values.shape[1]
    repeats = min(
        args.bootstrap_replicates,
        args.layer_bootstrap_replicates,
    )
    task_means: dict[str, np.ndarray] = {}
    task_bootstraps: dict[str, np.ndarray] = {}
    for dataset, records in records_by_task.items():
        rows = []
        for record in records:
            record_keys, values = layer_features(record)
            if record_keys != keys or values.shape != first_values.shape:
                raise ValueError(
                    f"{dataset}/{record.sample_id}: layer feature drift"
                )
            rows.append(values.reshape(-1))
        mean, _low, _high, bootstrap = bootstrap_matrix(
            np.asarray(rows),
            repeats=repeats,
            seed=stable_seed(args.seed, f"layer/{dataset}"),
        )
        task_means[dataset] = mean
        task_bootstraps[dataset] = bootstrap

    if all(dataset in task_means for dataset in LOCAL_DATASETS):
        group_name = "milebench_local29"
        selected = LOCAL_DATASETS
    elif all(dataset in task_means for dataset in OFFICIAL_DATASETS):
        group_name = "milebench_official28"
        selected = OFFICIAL_DATASETS
    else:
        group_name = "milebench_present_task_macro"
        selected = tuple(
            dataset for dataset in LOCAL_DATASETS if dataset in task_means
        )
    mean = sum(
        (task_means[dataset] for dataset in selected),
        np.zeros_like(next(iter(task_means.values()))),
    ) / len(selected)
    bootstrap = sum(
        (
            task_bootstraps[dataset].astype(np.float64)
            for dataset in selected
        ),
        np.zeros_like(next(iter(task_bootstraps.values())), dtype=np.float64),
    ) / len(selected)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    mean = mean.reshape(len(keys), n_layers)
    low = low.reshape(len(keys), n_layers)
    high = high.reshape(len(keys), n_layers)
    rows: list[dict[str, Any]] = []
    for key_index, key in enumerate(keys):
        metric, method = key.split("/", 1)
        for layer in range(n_layers):
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "method": method,
                    "layer": layer,
                    "mean": float(mean[key_index, layer]),
                    "ci95_low": float(low[key_index, layer]),
                    "ci95_high": float(high[key_index, layer]),
                    "n_tasks": len(selected),
                    "n_samples": sum(
                        len(records_by_task[dataset])
                        for dataset in selected
                    ),
                    "bootstrap_unit": "sample within task",
                    "bootstrap_replicates": repeats,
                    "budget": "Visual-KV 20%",
                }
            )
    return rows, group_name


def write_layerwise_figure(
    summary_dir: Path,
    rows: list[dict[str, Any]],
    group_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (row["metric"], row["method"], int(row["layer"])): row
        for row in rows
    }
    layers = sorted({int(row["layer"]) for row in rows})
    methods = (
        ("h2o", "H2O all-prefill", "#d55e00"),
        ("question", "Semantic user-prompt", "#009e73"),
        ("qvik", "Q-ViK", "#0072b2"),
    )
    panels = (
        ("spearman", "Spearman vs. Future"),
        ("future_topk_recall", "Future Top-K recall"),
        ("jaccard_vs_future", "Jaccard vs. Future"),
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 3.9),
        sharex=True,
        constrained_layout=True,
    )
    for axis, (metric, ylabel) in zip(axes, panels, strict=True):
        for method, label, color in methods:
            selected = [lookup[(metric, method, layer)] for layer in layers]
            means = np.asarray([row["mean"] for row in selected])
            low = np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected])
            axis.plot(
                layers,
                means,
                label=label,
                color=color,
                linewidth=2,
            )
            axis.fill_between(layers, low, high, color=color, alpha=0.14)
        axis.set_xlabel("Eviction layer")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.set_xlim(min(layers), max(layers))
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"MileBench Visual-KV 20%: Future agreement ({group_name})",
        fontsize=12,
    )
    figure.savefig(
        summary_dir / "layerwise_visual20_future_agreement.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        summary_dir / "layerwise_visual20_future_agreement.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def audit_budget(records_by_task: dict[str, list[Record]]) -> dict[str, Any]:
    records = [
        record for values in records_by_task.values() for record in values
    ]
    visual = np.asarray([record.n_visual for record in records])
    kept = np.asarray([record.n_keep for record in records])
    prompts = np.asarray([record.prompt_len for record in records])
    text = np.asarray([record.n_text for record in records])
    total_ratios = np.asarray(
        [record.actual_total_keep_ratio for record in records]
    )
    answers = np.asarray([record.answer_steps for record in records])
    return {
        "n_samples": len(records),
        "n_tasks": len(records_by_task),
        "budget_scope": "visual_kv",
        "requested_visual_keep_ratio": VISUAL_KEEP_RATIO,
        "all_keep_counts_equal_ceil_ratio_times_visual": bool(
            np.all(kept == np.ceil(VISUAL_KEEP_RATIO * visual))
        ),
        "mean_visual_tokens": float(visual.mean()),
        "mean_visual_tokens_kept": float(kept.mean()),
        "mean_actual_visual_keep_ratio": float((kept / visual).mean()),
        "min_actual_visual_keep_ratio": float((kept / visual).min()),
        "max_actual_visual_keep_ratio": float((kept / visual).max()),
        "mean_prompt_tokens": float(prompts.mean()),
        "mean_text_tokens": float(text.mean()),
        "mean_incidental_total_cache_ratio": float(total_ratios.mean()),
        "min_incidental_total_cache_ratio": float(total_ratios.min()),
        "max_incidental_total_cache_ratio": float(total_ratios.max()),
        "mean_full_answer_steps": float(answers.mean()),
        "hit_max_new_tokens": int(
            sum(record.hit_max_new_tokens for record in records)
        ),
        "standard_total20_records_included": False,
        "warning": (
            "The incidental total-cache ratio is descriptive only; this "
            "experiment's controlled budget is Visual-KV 20%."
        ),
    }


def task_label(group: str) -> str:
    return (
        group[len(MILEBENCH_PREFIX) :]
        if group.startswith(MILEBENCH_PREFIX)
        else group
    )


def ci_text(group: GroupStats, key: str, scale: float = 1.0) -> str:
    mean, low, high = group.value(key)
    return (
        f"{scale * mean:.4f} "
        f"[{scale * low:.4f}, {scale * high:.4f}]"
    )


def report_macro_groups(
    groups: dict[str, GroupStats],
) -> list[GroupStats]:
    names = [
        name
        for name in ("milebench_official28", "milebench_local29")
        if name in groups
    ]
    if not names:
        names = ["milebench_present_task_macro"]
    return [groups[name] for name in names]


def write_report(
    *,
    args: argparse.Namespace,
    records_by_task: dict[str, list[Record]],
    groups: dict[str, GroupStats],
    audit: dict[str, Any],
    budget: dict[str, Any],
    figure_group: str,
) -> None:
    complete = sum(map(len, records_by_task.values()))
    macros = report_macro_groups(groups)
    lines = [
        "# MileBench selector comparison — Visual-KV 20%",
        "",
        f"Complete shared sample set: {complete} samples across "
        f"{len(records_by_task)} tasks. Shared OOM exclusions: "
        f"{audit['shared_oom_exclusions']}.",
        "",
        "This report contains **only MileBench Visual-KV 20%** records. It "
        "does not read or aggregate the standard-eval total-cache 20% run.",
        "",
        "A sample enters the report only when both its all-method JSON and "
        "packed all-method mask NPZ are complete and validated. Thus every "
        "selector uses exactly the same sample set; an OOM removes the sample "
        "from all selectors together.",
        "",
        f"Confidence intervals use {args.bootstrap_replicates:,} sample "
        "bootstrap replicates within each task. Suite macros are unweighted "
        "task means, with samples resampled inside every task.",
        "",
        "## Downstream macro",
        "",
        "| Suite | Tasks | Samples | Full | Matched Full | H2O | "
        "Semantic user-prompt | Q-ViK | Future Oracle | Q-ViK − H2O |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in macros:
        cells = [
            ci_text(group, f"downstream/{selector}", scale=100.0)
            for selector in SELECTORS
        ]
        cells.append(
            ci_text(
                group,
                "downstream/qvik_minus_h2o_prefill",
                scale=100.0,
            )
        )
        lines.append(
            f"| {group.name} | {group.n_tasks} | {group.n_samples} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-task downstream score",
            "",
            "| Task | n | Full | Matched Full | H2O | Semantic user-prompt | "
            "Q-ViK | Future Oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in LOCAL_DATASETS:
        if dataset not in groups:
            continue
        group = groups[dataset]
        cells = [
            f"{100.0 * group.value(f'downstream/{selector}')[0]:.2f}"
            for selector in SELECTORS
        ]
        lines.append(
            f"| {task_label(dataset)} | {group.n_samples} | "
            + " | ".join(cells)
            + " |"
        )

    primary = macros[-1]
    lines.extend(
        [
            "",
            f"## Vector agreement ({primary.name})",
            "",
            "| Signal | Spearman vs. Future | Cosine vs. Future | "
            "Vector Jaccard@20% |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, pair in (
        ("H2O all-prefill", "h2o_vs_future"),
        ("Semantic user-prompt", "question_vs_future"),
        ("Q-ViK", "qvik_vs_future"),
        ("Q-ViK − H2O paired delta", "qvik_minus_h2o_vs_future"),
    ):
        lines.append(
            f"| {label} | "
            f"{ci_text(primary, f'vector/{pair}/spearman')} | "
            f"{ci_text(primary, f'vector/{pair}/cosine')} | "
            f"{ci_text(primary, f'vector/{pair}/jaccard_20')} |"
        )

    lines.extend(
        [
            "",
            f"## Deployed-token agreement ({primary.name})",
            "",
            "These rows are recomputed from the packed masks, not inferred "
            "from score-vector similarity.",
            "",
            "| Selector | Future Top-K recall | Jaccard vs. Future | "
            "Future mass retained |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, prefix in (
        ("H2O all-prefill", "h2o"),
        ("Semantic user-prompt", "question"),
        ("Q-ViK", "qvik"),
    ):
        lines.append(
            f"| {label} | "
            f"{ci_text(primary, f'mask/{prefix}_future_topk_recall')} | "
            f"{ci_text(primary, f'mask/{prefix}_jaccard_vs_future')} | "
            f"{ci_text(primary, f'token/{prefix}_visual_future_mass_retained')} |"
        )

    lines.extend(
        [
            "",
            f"## Full-response preservation ({primary.name})",
            "",
            "| Selector | Exact match with Matched Full |",
            "|---|---:|",
        ]
    )
    for selector in SELECTORS:
        lines.append(
            f"| {SELECTOR_LABELS[selector]} | "
            f"{ci_text(primary, f'preservation/{selector}')} |"
        )
    lines.extend(
        [
            "",
            "## Budget audit",
            "",
            f"- Controlled budget: **Visual-KV 20%**, "
            f"`K_v = ceil(0.2 N_v)`.",
            f"- Mean visual/kept visual tokens: "
            f"{budget['mean_visual_tokens']:.1f} / "
            f"{budget['mean_visual_tokens_kept']:.1f}.",
            f"- Mean actual visual keep ratio: "
            f"{budget['mean_actual_visual_keep_ratio']:.6f}.",
            f"- Incidental total-cache ratio: "
            f"{budget['mean_incidental_total_cache_ratio']:.4f} "
            f"(range {budget['min_incidental_total_cache_ratio']:.4f}–"
            f"{budget['max_incidental_total_cache_ratio']:.4f}); this is not "
            "the controlled budget.",
            f"- Layerwise figure aggregation: `{figure_group}`.",
            "",
            "Detailed per-task/sample-bootstrap tables, packed-mask audits, "
            "shared exclusions, and provenance are saved beside this report.",
        ]
    )
    atomic_text(args.summary_dir / "RESULTS.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    records_by_task, audit = load_complete_records(args)
    groups = build_scalar_groups(records_by_task, args)
    tables = make_scalar_tables(groups)
    layer_rows, figure_group = build_layerwise(records_by_task, args)
    budget = audit_budget(records_by_task)

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    downstream_tasks = [
        row for row in tables["downstream"] if row["group_type"] == "task"
    ]
    downstream_macros = [
        row for row in tables["downstream"] if row["group_type"] != "task"
    ]
    included_rows = [
        {
            "dataset": record.dataset,
            "task": task_label(record.dataset),
            "row_index": record.row_index,
            "sample_id": record.sample_id,
            "n_visual": record.n_visual,
            "n_visual_kept": record.n_keep,
            "actual_visual_keep_ratio": record.n_keep / record.n_visual,
            "prompt_len": record.prompt_len,
            "n_text": record.n_text,
            "answer_steps": record.answer_steps,
            "budget": "Visual-KV 20%",
            "all_selectors_present": True,
            "json_and_mask_validated": True,
        }
        for dataset in LOCAL_DATASETS
        for record in records_by_task.get(dataset, ())
    ]
    write_csv(
        args.summary_dir / "included_shared_samples.csv",
        included_rows,
    )
    write_csv(
        args.summary_dir / "downstream_per_task.csv",
        downstream_tasks,
    )
    write_csv(
        args.summary_dir / "downstream_macros.csv",
        downstream_macros,
    )
    write_csv(
        args.summary_dir / "downstream_paired_deltas.csv",
        [
            row
            for row in tables["downstream"]
            if row["selector"] == "qvik_minus_h2o_prefill"
        ],
    )
    write_csv(
        args.summary_dir / "vector_agreement.csv",
        tables["vector"],
    )
    write_csv(
        args.summary_dir / "token_keep_metrics.csv",
        tables["token"],
    )
    write_csv(
        args.summary_dir / "deployed_mask_agreement.csv",
        tables["mask"],
    )
    write_csv(
        args.summary_dir / "response_preservation.csv",
        tables["preservation"],
    )
    write_csv(
        args.summary_dir / "response_preservation_paired.csv",
        tables["preservation_paired"],
    )
    write_csv(
        args.summary_dir / "layerwise_visual20_metrics.csv",
        layer_rows,
    )
    atomic_json(args.summary_dir / "exclusion_audit.json", audit)
    atomic_json(args.summary_dir / "budget_audit.json", budget)
    atomic_json(
        args.summary_dir / "provenance.json",
        {
            "schema_version": 1,
            "input_root": str(args.output_root),
            "summary_dir": str(args.summary_dir),
            "budget_name": "Visual-KV 20%",
            "budget_scope": "visual_kv",
            "requested_visual_keep_ratio": VISUAL_KEEP_RATIO,
            "standard_budget_scope": "total_prompt_cache",
            "standard_records_included": False,
            "cross_budget_macro_reported": False,
            "selectors": list(SELECTORS),
            "bootstrap_unit": "sample within task",
            "bootstrap_replicates": args.bootstrap_replicates,
            "layer_bootstrap_replicates": min(
                args.bootstrap_replicates,
                args.layer_bootstrap_replicates,
            ),
            "bootstrap_seed": args.seed,
            "official_suite_tasks": list(OFFICIAL_DATASETS),
            "local_extension_tasks": list(LOCAL_DATASETS),
            "official_macro_present": "milebench_official28" in groups,
            "local_macro_present": "milebench_local29" in groups,
            "shared_oom_exclusions": audit["shared_oom_exclusions"],
            "sample_inclusion_rule": audit["sample_inclusion_rule"],
        },
    )
    if not args.skip_figure:
        write_layerwise_figure(
            args.summary_dir,
            layer_rows,
            figure_group,
        )
    write_report(
        args=args,
        records_by_task=records_by_task,
        groups=groups,
        audit=audit,
        budget=budget,
        figure_group=figure_group,
    )
    print(
        f"[done] summarized {audit['complete_pairs']} complete shared "
        f"JSON+mask pairs; shared OOM exclusions="
        f"{audit['shared_oom_exclusions']}"
    )
    print(f"[done] wrote {args.summary_dir / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
