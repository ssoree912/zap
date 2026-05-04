#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score selected MileBench outputs and write dataset/category CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

LOOKM_ROOT = Path("/workspace/look-m")
if str(LOOKM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOKM_ROOT))

from evaluate import Eval  # noqa: E402

DATA_ROOT = Path("/workspace/zap/data/MileBench")

CATEGORIES: dict[str, list[str]] = {
    "T-2": ["MovingAttribute", "ObjectExistence", "ObjectInteraction", "ObjectShuffle"],
    "T-3": ["EgocentricNavigation", "MovingDirection"],
    "T-4": ["CharacterOrder", "CounterfactualInference", "SceneTransition", "StateChange"],
    "S-4": ["ALFRED", "MMCoQA"],
}

DATASETS: list[str] = [dataset for datasets in CATEGORIES.values() for dataset in datasets]


def keep_tag_to_ratio(tag: str) -> float:
    return int(tag.replace("keep", "")) / 100.0


def normalize_sample_ids(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for pred in predictions:
        item = dict(pred)
        try:
            item["sample_id"] = int(item["sample_id"])
        except (TypeError, ValueError):
            pass
        normalized.append(item)
    return normalized


def score_dataset(pred_path: Path, dataset: str) -> tuple[dict[str, Any], str, float]:
    core_path = DATA_ROOT / dataset / f"{dataset}.json"
    core = json.loads(core_path.read_text())
    predictions = normalize_sample_ids(json.loads(pred_path.read_text()))
    question_type = core["meta_data"]["question_type"]
    scorer = Eval()

    if "NeedleInAHaystack" in dataset or dataset == "MMCoQA":
        eval_result, eval_list = scorer.evaluate_needle(
            predictions,
            core,
            needle="NeedleInAHaystack" in dataset,
        )
        metric = "Accuracy"
        score = float(eval_result[metric])
    elif question_type == "open-ended":
        eval_result, eval_list = scorer.evaluate_rouge(predictions, core)
        metric = "Rouge-L f"
        score = float(eval_result[metric])
    elif question_type == "multi-choice":
        predictions_with_extracted, eval_result, eval_list = scorer.evaluate_multichoice(
            predictions,
            core,
        )
        (pred_path.parent / "pred_with_extracted.json").write_text(
            json.dumps(predictions_with_extracted, ensure_ascii=False, indent=2)
        )
        metric = "Accuracy"
        score = float(eval_result[metric])
    else:
        raise ValueError(f"Unsupported question_type={question_type} for {dataset}")

    (pred_path.parent / "eval.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2))
    (pred_path.parent / "eval_score.json").write_text(json.dumps(eval_list, ensure_ascii=False, indent=2))
    return eval_result, metric, score


def category_for_dataset(dataset: str) -> str:
    for category, datasets in CATEGORIES.items():
        if dataset in datasets:
            return category
    raise KeyError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--keep-tags", nargs="+", default=["keep050", "keep010", "keep005"])
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for keep_tag in args.keep_tags:
        keep_ratio = keep_tag_to_ratio(keep_tag)
        for dataset in DATASETS:
            pred_path = output_root / keep_tag / dataset / "pred.json"
            if not pred_path.exists():
                missing.append({"keep_ratio": keep_ratio, "dataset": dataset, "path": str(pred_path)})
                continue
            eval_result, metric, score = score_dataset(pred_path, dataset)
            rows.append(
                {
                    "keep_ratio": keep_ratio,
                    "category": category_for_dataset(dataset),
                    "dataset": dataset,
                    "metric": metric,
                    "score": round(score * 100.0, 2),
                    "n": int(eval_result.get("image_quantity_level-Result", {}).get("Few", [0, 0])[1])
                    + int(eval_result.get("image_quantity_level-Result", {}).get("Medium", [0, 0])[1])
                    + int(eval_result.get("image_quantity_level-Result", {}).get("Many", [0, 0])[1]),
                }
            )

    dataset_csv = summary_dir / "milebench_dataset_scores.csv"
    with dataset_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["keep_ratio", "category", "dataset", "metric", "score", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)

    category_rows: list[dict[str, Any]] = []
    for keep_tag in args.keep_tags:
        keep_ratio = keep_tag_to_ratio(keep_tag)
        keep_rows = [row for row in rows if row["keep_ratio"] == keep_ratio]
        for category, datasets in CATEGORIES.items():
            vals = [row["score"] for row in keep_rows if row["dataset"] in datasets]
            category_rows.append(
                {
                    "keep_ratio": keep_ratio,
                    "category": category,
                    "avg_score": round(sum(vals) / len(vals), 2) if vals else "",
                    "n_datasets": len(vals),
                }
            )
        vals = [row["score"] for row in keep_rows]
        category_rows.append(
            {
                "keep_ratio": keep_ratio,
                "category": "Overall",
                "avg_score": round(sum(vals) / len(vals), 2) if vals else "",
                "n_datasets": len(vals),
            }
        )

    category_csv = summary_dir / "milebench_category_averages.csv"
    with category_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keep_ratio", "category", "avg_score", "n_datasets"])
        writer.writeheader()
        writer.writerows(category_rows)

    (summary_dir / "missing.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2))
    print(f"[save] {dataset_csv}")
    print(f"[save] {category_csv}")
    print(f"[save] {summary_dir / 'missing.json'} missing={len(missing)}")


if __name__ == "__main__":
    main()
