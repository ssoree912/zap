#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
CASES = (
    ("scratch", 450),
    ("warm", 450),
    ("scratch", 900),
    ("warm", 900),
)
EPOCHS = (1, 3, 5, 15)


def find_result(case_tag: str) -> Path:
    root = EXP_DIR / "outputs" / "chartqa" / case_tag
    matches = sorted(root.rglob("*_results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one result for {case_tag}, found {matches}")
    return matches[0]


def main() -> int:
    rows: list[dict] = []
    for mode, total_samples in CASES:
        train_root = EXP_DIR / "checkpoints" / f"{mode}_n{total_samples}_e15"
        warm_report = train_root / "warm_start_report.json"
        warm_fraction = None
        if warm_report.is_file():
            warm_fraction = json.loads(warm_report.read_text())["transferred_fraction"]
        for epoch in EPOCHS:
            case_tag = f"{mode}_n{total_samples}_epoch{epoch:03d}"
            result_path = find_result(case_tag)
            result = json.loads(result_path.read_text())["results"]["chartqa_local"]
            metrics_path = train_root / f"epoch_{epoch:03d}" / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "initialization": mode,
                    "teacher_samples": total_samples,
                    "epoch": epoch,
                    "train_gpu_hours": metrics["elapsed"] / 3600,
                    "val_loss": metrics["val_loss"],
                    "chartqa_overall": result["relaxed_overall,none"],
                    "chartqa_human": result["relaxed_human_split,none"],
                    "chartqa_augmented": result["relaxed_augmented_split,none"],
                    "warm_transferred_fraction": warm_fraction,
                    "result_path": str(result_path),
                }
            )

    output_csv = EXP_DIR / "results.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = {
        "initialization": "scratch",
        "teacher_samples": 1800,
        "epoch": 15,
        "train_gpu_hours": 33481.9 / 3600,
        "chartqa_overall": 0.7756,
        "chartqa_human": 0.6424,
        "chartqa_augmented": 0.9088,
    }
    summary = {"rows": rows, "existing_full_data_baseline": baseline}
    (EXP_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {output_csv}")
    print(f"[save] {EXP_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
