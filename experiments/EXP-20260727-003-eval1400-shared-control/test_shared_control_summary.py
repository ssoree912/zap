# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU integration test for strict summary and common exclusions."""

from __future__ import annotations

import csv
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import run_eval1400_shared_control as runner


@pytest.mark.parametrize(
    ("prompt_len", "expected_visual_keep"),
    (
        (576, 116),
        (600, 96),
        (610, 88),
        (650, 56),
    ),
)
def test_exact_total_budget_is_ceil_twenty_percent(
    prompt_len: int,
    expected_visual_keep: int,
) -> None:
    n_visual = 576
    n_text = prompt_len - n_visual
    n_keep = runner.q2.exact_total_budget(
        n_visual=n_visual,
        prompt_len=prompt_len,
        total_keep_ratio=0.2,
    )
    assert n_keep == expected_visual_keep
    assert n_text + n_keep == math.ceil(0.2 * prompt_len)


def test_summary_uses_complete_pairs_and_declared_common_exclusions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    dataset = "gqa"
    rows = [
        {
            "dataset": dataset,
            "sample_id": f"sample-{index}",
            "row_index": index,
        }
        for index in range(200)
    ]
    runner.atomic_json(
        output / "manifests" / f"{dataset}.json",
        {
            "schema_version": 1,
            "sample_count": 200,
            "datasets": {dataset: 200},
            "samples": rows,
        },
    )

    shared_scores = {
        "all_prefill_shared": torch.tensor(
            [[9.0, 7.0, 2.0, 1.0], [1.0, 3.0, 8.0, 5.0]]
        ),
        "question_shared": torch.tensor(
            [[1.0, 8.0, 7.0, 2.0], [2.0, 7.0, 6.0, 1.0]]
        ),
        "qvik_shared": torch.tensor(
            [[8.0, 2.0, 6.0, 1.0], [5.0, 1.0, 7.0, 3.0]]
        ),
        "future_shared": torch.tensor(
            [[2.0, 9.0, 6.0, 1.0], [1.0, 8.0, 7.0, 2.0]]
        ),
    }
    token_metrics, masks = runner.control.shared_control_metrics(
        shared_scores,
        k=2,
    )
    vector_pairs = {
        runner.control.pair_key(first, second): (
            runner.q2.agreement.compute_pair_metrics(
                shared_scores[first],
                shared_scores[second],
            )
        )
        for first, second in runner.control.PAIRWISE_SCORE_PAIRS
    }
    stem = f"{0:06d}_{runner.safe_name(rows[0]['sample_id'])}"
    result_path = output / "samples" / dataset / f"{stem}.json"
    mask_path = output / "masks" / dataset / f"{stem}.npz"
    runner.save_packed_shared_masks(
        mask_path,
        masks,
        n_visual=4,
        n_keep=2,
        n_kv_heads=3,
    )
    predictions = {
        "full_cache_generate": "raw answer",
        "full_cache_manual": "manual answer",
        "all_prefill_shared": "raw answer",
        "question_shared": "other answer",
        "qvik_shared": "raw answer",
    }
    record = {
        "schema_version": runner.SCHEMA_VERSION,
        "experiment_id": runner.EXPERIMENT_ID,
        "policy_version": runner.POLICY_VERSION,
        "dataset": dataset,
        "sample_id": rows[0]["sample_id"],
        "row_index": 0,
        "future_trajectory": runner.FUTURE_TRAJECTORY,
        "all_compared_masks_shared_across_kv_heads": True,
        "n_visual": 4,
        "n_visual_kept_per_layer_and_kv_head": 2,
        "n_kv_heads": 3,
        "actual_visual_keep_ratio": 0.5,
        "hit_max_new_tokens": False,
        "full_cache_manual_matches_generate": False,
        "references": ["raw answer"],
        "token_metrics": token_metrics,
        "vector_pairs": vector_pairs,
        "predictions": predictions,
        "scores": {
            selector: float(prediction == "raw answer")
            for selector, prediction in predictions.items()
        },
        "applied_visual_mask_sha256": {
            score_name: runner.sha256_mask(
                masks[f"{score_name}_keep"]
            )
            for score_name in runner.SCORE_TO_SELECTOR
        },
        "future_reference_mask_sha256": runner.sha256_mask(
            masks["future_shared_keep"]
        ),
        "mask_sha256": runner.sha256_file(mask_path),
    }
    runner.atomic_json(result_path, record)

    for row in rows[1:]:
        failure_path = (
            output
            / "failures"
            / dataset
            / (
                f"{int(row['row_index']):06d}_"
                f"{runner.safe_name(row['sample_id'])}.txt"
            )
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text("RuntimeError: synthetic sample failure\n")

    args = Namespace(
        output_dir=output,
        datasets=[dataset],
        n_samples=200,
        sample_seed=42,
        sample_start=0,
        sample_end=None,
        total_keep_ratio=0.2,
        bootstrap_replicates=20,
        summary_seed=20260727,
    )
    assert runner.summarize(args) == 0

    report = (output / "summary" / "report.md").read_text()
    assert "Response preservation against raw Full Cache generate" in report
    assert "Future shared Oracle decode" not in report
    assert report.index(
        "Response preservation against raw Full Cache generate"
    ) < report.index(
        "Response preservation against matched manual Full Cache"
    )
    assert (output / "summary" / "response_preservation_paired.csv").exists()
    future_paired_path = (
        output / "summary" / "future_agreement_paired.csv"
    )
    vector_paired_path = (
        output / "summary" / "vector_agreement_paired.csv"
    )
    assert future_paired_path.exists()
    assert vector_paired_path.exists()
    with future_paired_path.open(newline="") as handle:
        future_paired = list(csv.DictReader(handle))
    future_row = next(
        row
        for row in future_paired
        if row["group"] == "all"
        and row["comparison"]
        == "qvik_shared_minus_all_prefill_shared"
        and row["metric"] == "topk_recall"
    )
    expected_future_delta = (
        sum(
            token_metrics["future_agreement"]["qvik_shared"][
                "topk_recall"
            ]
        )
        / 2
        - sum(
            token_metrics["future_agreement"][
                "all_prefill_shared"
            ]["topk_recall"]
        )
        / 2
    )
    assert float(future_row["mean_delta"]) == pytest.approx(
        expected_future_delta
    )
    with vector_paired_path.open(newline="") as handle:
        vector_paired = list(csv.DictReader(handle))
    vector_comparison = (
        "qvik_shared_vs_future_shared_minus_"
        "question_shared_vs_future_shared"
    )
    vector_row = next(
        row
        for row in vector_paired
        if row["group"] == "all"
        and row["comparison"] == vector_comparison
        and row["metric"] == "spearman"
    )
    qvik_future = vector_pairs[
        runner.control.pair_key("qvik_shared", "future_shared")
    ]["spearman"]
    question_future = vector_pairs[
        runner.control.pair_key(
            "question_shared",
            "future_shared",
        )
    ]["spearman"]
    expected_vector_delta = (
        sum(qvik_future) / len(qvik_future)
        - sum(question_future) / len(question_future)
    )
    assert float(vector_row["mean_delta"]) == pytest.approx(
        expected_vector_delta
    )
    assert "Paired deltas in Future agreement" in report
    assert "Paired deltas in score-vector agreement" in report
    provenance = json.loads(
        (output / "summary" / "provenance.json").read_text()
    )
    assert provenance["n_complete_samples"] == 1
    assert provenance["n_declared_common_exclusions"] == 199
    assert provenance["declared_common_exclusion_counts"]["gqa"] == 199
    assert provenance["paired_delta_outputs"] == {
        "future_agreement": "future_agreement_paired.csv",
        "vector_agreement": "vector_agreement_paired.csv",
    }
