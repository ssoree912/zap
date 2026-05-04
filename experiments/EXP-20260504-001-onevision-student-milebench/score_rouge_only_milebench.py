#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score ROUGE-only MileBench outputs and write compact CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from score_and_summarize_milebench import keep_tag_to_ratio, score_dataset

ROUGE_DATASETS = ["ALFRED", "CLEVR-Change", "IEdit", "Spot-the-Diff"]


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
        for dataset in ROUGE_DATASETS:
            pred_path = output_root / keep_tag / dataset / "pred.json"
            if not pred_path.exists():
                missing.append({"keep_ratio": str(keep_ratio), "dataset": dataset, "path": str(pred_path)})
                continue
            eval_result, metric, score = score_dataset(pred_path, dataset)
            rows.append(
                {
                    "keep_ratio": keep_ratio,
                    "dataset": dataset,
                    "metric": metric,
                    "score": round(score * 100.0, 2),
                    "n": sum(
                        int(v[1])
                        for v in eval_result.get("image_quantity_level-Result", {}).values()
                    ),
                }
            )

    dataset_csv = summary_dir / "milebench_rouge_only_dataset_scores.csv"
    with dataset_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keep_ratio", "dataset", "metric", "score", "n"])
        writer.writeheader()
        writer.writerows(rows)

    average_rows: list[dict[str, Any]] = []
    for keep_tag in args.keep_tags:
        keep_ratio = keep_tag_to_ratio(keep_tag)
        vals = [row["score"] for row in rows if row["keep_ratio"] == keep_ratio]
        average_rows.append(
            {
                "keep_ratio": keep_ratio,
                "dataset": "ROUGE average",
                "metric": "Rouge-L f",
                "score": round(sum(vals) / len(vals), 2) if vals else "",
                "n": len(vals),
            }
        )

    average_csv = summary_dir / "milebench_rouge_only_average.csv"
    with average_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keep_ratio", "dataset", "metric", "score", "n"])
        writer.writeheader()
        writer.writerows(average_rows)

    missing_path = summary_dir / "milebench_rouge_only_missing.json"
    missing_path.write_text(json.dumps(missing, ensure_ascii=False, indent=2))
    print(f"[save] {dataset_csv}")
    print(f"[save] {average_csv}")
    print(f"[save] {missing_path} missing={len(missing)}")


if __name__ == "__main__":
    main()
