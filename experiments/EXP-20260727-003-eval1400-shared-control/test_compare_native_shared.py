# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the completed-artifact comparator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import compare_native_shared as comparator


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _base_record(index: int) -> dict[str, Any]:
    return {
        "dataset": "gqa",
        "sample_id": f"sample-{index}",
        "row_index": index,
        "context": f"question {index}\nAnswer briefly.",
        "semantic_question": f"question {index}",
        "references": [f"answer-{index}"],
        "max_new_tokens": 8,
        "answer_steps": 2,
        "hit_max_new_tokens": False,
        "processed_image_sha256": f"{index + 1:064x}",
        "prompt_len": 6,
        "n_text": 2,
        "n_visual": 4,
        "requested_total_keep_ratio": 0.6,
        "actual_total_keep_ratio": 4 / 6,
        "actual_visual_keep_ratio": 0.5,
        "question_token_count": 2,
        "student_prompt_tail_token_count": 2,
        "qvik_checkpoint": "synthetic-checkpoint",
        "same_processed_image_for_prefill_and_future": True,
        "full_cache_manual_matches_generate": True,
        "future_trajectory": comparator.FUTURE_TRAJECTORY,
    }


def _mask_values() -> dict[str, np.ndarray]:
    question = np.asarray(
        [
            [True, True, False, False],
            [False, True, True, False],
        ]
    )
    qvik = np.asarray(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    h2o = np.asarray(
        [
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, True, True, False],
            ],
            [
                [True, False, False, True],
                [False, True, False, True],
                [False, False, True, True],
            ],
        ]
    )
    future = np.asarray(
        [
            [
                [True, False, False, True],
                [False, True, True, False],
                [True, False, True, False],
            ],
            [
                [True, True, False, False],
                [False, True, False, True],
                [True, False, False, True],
            ],
        ]
    )
    return {
        "question": question,
        "qvik": qvik,
        "h2o": h2o,
        "future": future,
    }


def _write_native_mask(path: Path, *, disagree_question: bool = False) -> None:
    values = _mask_values()
    question = np.repeat(values["question"][:, None, :], 3, axis=1)
    if disagree_question:
        question[0, 1] = np.asarray([False, False, True, True])
    qvik = np.repeat(values["qvik"][:, None, :], 3, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_visual=np.int32(4),
        n_keep=np.int32(2),
        h2o_keep=np.packbits(values["h2o"], axis=-1),
        question_keep=np.packbits(question, axis=-1),
        qvik_keep=np.packbits(qvik, axis=-1),
        future_keep=np.packbits(values["future"], axis=-1),
    )


def _write_shared_mask(path: Path) -> dict[str, np.ndarray]:
    values = _mask_values()
    masks = {
        "all_prefill_shared_keep": np.asarray(
            [
                [False, True, False, True],
                [True, True, False, False],
            ]
        ),
        "question_shared_keep": values["question"],
        "qvik_shared_keep": values["qvik"],
        # Intentionally not derived from the native head-wise Future mask.
        "future_shared_keep": np.asarray(
            [
                [False, False, True, True],
                [True, False, True, False],
            ]
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.int32(comparator.SHARED_SCHEMA_VERSION),
        policy_version=np.str_(comparator.SHARED_POLICY_VERSION),
        mask_policy=np.str_("shared_across_all_kv_heads"),
        future_reference=np.str_("raw_full_cache_generate_attention"),
        future_trajectory=np.str_(comparator.FUTURE_TRAJECTORY),
        n_visual=np.int32(4),
        n_keep=np.int32(2),
        n_kv_heads=np.int32(3),
        **{
            name: np.packbits(mask, axis=-1)
            for name, mask in masks.items()
        },
    )
    return masks


def _vector_pair(value: float) -> dict[str, list[float]]:
    return {
        "cosine": [value, value + 0.1],
        "spearman": [value + 0.2, value + 0.3],
    }


def _build_artifacts(
    tmp_path: Path,
    *,
    sample_count: int = 2,
    disagree_question: bool = False,
) -> tuple[Path, Path]:
    native = tmp_path / "native"
    shared = tmp_path / "shared"
    manifest = {
        "schema_version": 1,
        "sample_count": sample_count,
        "datasets": {"gqa": sample_count},
        "samples": [
            {
                key: _base_record(index)[key]
                for key in (
                    "dataset",
                    "sample_id",
                    "row_index",
                    "context",
                    "references",
                    "max_new_tokens",
                )
            }
            for index in range(sample_count)
        ],
    }
    for index, sample in enumerate(manifest["samples"]):
        sample["question_span"] = _base_record(index)["semantic_question"]
    _write_json(native / "manifests/gqa.json", manifest)
    _write_json(shared / "manifests/gqa.json", manifest)

    for index in range(sample_count):
        stem = f"{index:06d}_sample-{index}"
        native_mask_path = native / f"masks/gqa/{stem}.npz"
        shared_mask_path = shared / f"masks/gqa/{stem}.npz"
        _write_native_mask(
            native_mask_path,
            disagree_question=disagree_question and index == 0,
        )
        shared_masks = _write_shared_mask(shared_mask_path)

        base = _base_record(index)
        raw_answer = f"answer-{index}"
        native_prefill = raw_answer if index == 0 else "native-different"
        shared_prefill = raw_answer
        invariant_predictions = {
            "full_cache_generate": raw_answer,
            "full_cache_manual": raw_answer,
            "question_prefill": f"question-prediction-{index}",
            "qvik": f"qvik-prediction-{index}",
        }
        invariant_scores = {
            "full_cache_generate": 1.0,
            "full_cache_manual": 1.0,
            "question_prefill": 0.5,
            "qvik": 0.75,
        }
        native_record = {
            **base,
            "schema_version": 2,
            "n_visual_kept_per_head": 2,
            "mask_file": f"masks/gqa/{stem}.npz",
            "predictions": {
                **invariant_predictions,
                "h2o_prefill": native_prefill,
                "future_oracle": "unused-native-oracle",
            },
            "scores": {
                **invariant_scores,
                "h2o_prefill": 0.25 if index else 1.0,
                "future_oracle": 0.0,
            },
            "vector_pairs": {
                "qvik_vs_h2o": _vector_pair(0.1),
                "qvik_vs_question": _vector_pair(0.2),
                "h2o_vs_future": _vector_pair(0.3),
                "question_vs_future": _vector_pair(0.4),
                "qvik_vs_future": _vector_pair(0.5),
            },
            "token_metrics": {
                "h2o_future_topk_recall": [[0.0]],
            },
        }
        shared_record = {
            **base,
            "schema_version": comparator.SHARED_SCHEMA_VERSION,
            "experiment_id": comparator.EXPERIMENT_ID,
            "policy_version": comparator.SHARED_POLICY_VERSION,
            "oracle_decode": "omitted",
            "all_compared_masks_shared_across_kv_heads": True,
            "selection_policy": "shared_layerwise_head_mean_topk",
            "mask_policy": (
                "one shared layer-wise visual mask broadcast unchanged to "
                "all KV heads"
            ),
            "score_shape": "[L,N_visual]",
            "head_reduction": "arithmetic mean before Top-K",
            "future_reference_source": (
                "raw Full-cache greedy generation attention; head-averaged "
                "before shared Top-K"
            ),
            "n_visual_kept_per_layer_and_kv_head": 2,
            "n_kv_heads": 3,
            "mask_file": f"masks/gqa/{stem}.npz",
            "mask_sha256": comparator._sha256_file(shared_mask_path),
            "applied_visual_mask_sha256": {
                name: comparator._sha256_mask(
                    shared_masks[f"{name}_keep"]
                )
                for name in (
                    "all_prefill_shared",
                    "question_shared",
                    "qvik_shared",
                )
            },
            "future_reference_mask_sha256": comparator._sha256_mask(
                shared_masks["future_shared_keep"]
            ),
            "predictions": {
                "full_cache_generate": raw_answer,
                "full_cache_manual": raw_answer,
                "all_prefill_shared": shared_prefill,
                "question_shared": invariant_predictions["question_prefill"],
                "qvik_shared": invariant_predictions["qvik"],
            },
            "scores": {
                "full_cache_generate": 1.0,
                "full_cache_manual": 1.0,
                "all_prefill_shared": 1.0,
                "question_shared": invariant_scores["question_prefill"],
                "qvik_shared": invariant_scores["qvik"],
            },
            "vector_pairs": {
                "qvik_shared__all_prefill_shared": _vector_pair(0.1),
                "qvik_shared__question_shared": _vector_pair(0.2),
                # Deliberate mismatches: Future-derived cross-run metrics
                # must remain excluded.
                "all_prefill_shared__future_shared": _vector_pair(0.9),
                "question_shared__future_shared": _vector_pair(0.8),
                "qvik_shared__future_shared": _vector_pair(0.7),
            },
            "token_metrics": {
                "future_agreement": {"deliberately": "different"},
            },
        }
        _write_json(native / f"samples/gqa/{stem}.json", native_record)
        _write_json(shared / f"samples/gqa/{stem}.json", shared_record)
    return native, shared


def test_completed_artifacts_write_strict_audit_and_paired_csv(
    tmp_path: Path,
) -> None:
    native, shared = _build_artifacts(tmp_path)

    audit_path, csv_path = comparator.run_comparison(
        native,
        shared,
        dataset_counts={"gqa": 2},
        expected_total_keep_ratio=0.6,
    )

    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "pass"
    assert audit["identity_join"]["intersection"] == 2
    assert audit["identity_join"]["exact_set_equality"] is True
    assert audit["manifest_sha256"]["gqa"]["identical"] is True
    assert (
        audit["invariants"]["mask_equivalence"][
            "native_question_equals_shared_question"
        ]
        == 2
    )
    assert set(
        audit["invariants"]["non_future_vector_pairs"]["max_abs_error"]
    ) == {
        "qvik_vs_h2o->qvik_shared__all_prefill_shared",
        "qvik_vs_question->qvik_shared__question_shared",
    }
    excluded = audit["excluded_cross_definition_comparisons"]
    assert all(row["status"] == "not_compared" for row in excluded)
    assert any(row["native"] == "future_keep [L,H,N]" for row in excluded)

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["group"] for row in rows] == ["all", "gqa"]
    overall = rows[0]
    assert int(overall["n_samples"]) == 2
    assert float(overall["native_raw_full_exact_match_rate"]) == 0.5
    assert float(overall["shared_raw_full_exact_match_rate"]) == 1.0
    assert int(overall["shared_only_raw_full_matches"]) == 1
    assert float(overall["native_shared_prefill_exact_match_rate"]) == 0.5
    assert not (native / "summary").exists()


@pytest.mark.parametrize("failure", ("identity", "manifest_sha"))
def test_hard_join_and_manifest_sha_fail_before_writing(
    tmp_path: Path,
    failure: str,
) -> None:
    native, shared = _build_artifacts(tmp_path, sample_count=1)
    if failure == "identity":
        result_path = shared / "samples/gqa/000000_sample-0.json"
        record = json.loads(result_path.read_text())
        record["sample_id"] = "different-id"
        _write_json(result_path, record)
    else:
        manifest_path = shared / "manifests/gqa.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["synthetic_difference"] = True
        _write_json(manifest_path, manifest)

    with pytest.raises(ValueError):
        comparator.run_comparison(
            native,
            shared,
            dataset_counts={"gqa": 1},
            expected_total_keep_ratio=0.6,
        )
    assert not (shared / "summary").exists()


def test_native_question_head_disagreement_is_not_collapsed(
    tmp_path: Path,
) -> None:
    native, shared = _build_artifacts(
        tmp_path,
        sample_count=1,
        disagree_question=True,
    )

    with pytest.raises(ValueError, match="Question mask differs across heads"):
        comparator.compare_artifacts(
            native,
            shared,
            dataset_counts={"gqa": 1},
            expected_total_keep_ratio=0.6,
        )
