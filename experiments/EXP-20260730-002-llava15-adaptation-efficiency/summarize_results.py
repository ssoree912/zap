#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
TEACHER_ROOT = Path("/workspace/nips/data/train/teacher")
CONDITIONS = (
    ("vicuna_generated_n900_e15", 900, 15, 0.105, None),
    ("vicuna_generated_n300_e15", 300, 15, 0.035, None),
    ("vicuna_generated_n1800_e3", 1800, 3, 0.210, None),
    (
        "vicuna_reference_n1800_e15",
        1800,
        15,
        None,
        "llava15_rff3_reference_vicuna_n600perds_seed0",
    ),
)


def teacher_hours(root_name: str) -> float:
    root = TEACHER_ROOT / root_name
    elapsed = 0.0
    for dataset in ("textvqa", "scienceqa", "gqa"):
        summary = json.loads((root / dataset / "_summary.json").read_text())
        elapsed += float(summary["elapsed_seconds"])
    return elapsed / 3600


def metric(tag: str, task: str) -> float:
    result_paths = sorted((EXP_DIR / "outputs" / tag / task).rglob("*_results.json"))
    if len(result_paths) != 1:
        raise FileNotFoundError(f"Expected one {tag}/{task} result, found {result_paths}")
    results = json.loads(result_paths[0].read_text())["results"]
    task_result = results[f"{task}_local"]
    keys = {
        "chartqa": "relaxed_overall,none",
        "docvqa": "anls,none",
        "textvqa": "exact_match,none",
    }
    return 100.0 * float(task_result[keys[task]])


def main() -> int:
    rows: list[dict[str, object]] = [
        {
            "setting": "base_existing",
            "samples": 1800,
            "epochs": 15,
            "response_source": "generated",
            "teacher_gpu_hours": 0.2101,
            "training_gpu_hours": 2.9420,
            "total_gpu_hours": 3.1521,
            "chartqa": 17.84,
            "docvqa": 26.86,
            "textvqa": 45.59,
            "average": (17.84 + 26.86 + 45.59) / 3,
        }
    ]

    for tag, samples, epochs, attributed_teacher_hours, teacher_root in CONDITIONS:
        training_summary = json.loads(
            (EXP_DIR / "checkpoints" / tag / "training_summary.json").read_text()
        )
        teacher_gpu_hours = (
            float(attributed_teacher_hours)
            if attributed_teacher_hours is not None
            else teacher_hours(str(teacher_root))
        )
        training_gpu_hours = float(training_summary["elapsed_seconds"]) / 3600
        scores = {task: metric(tag, task) for task in ("chartqa", "docvqa", "textvqa")}
        rows.append(
            {
                "setting": tag,
                "samples": samples,
                "epochs": epochs,
                "response_source": "reference" if "reference" in tag else "generated",
                "teacher_gpu_hours": teacher_gpu_hours,
                "training_gpu_hours": training_gpu_hours,
                "total_gpu_hours": teacher_gpu_hours + training_gpu_hours,
                **scores,
                "average": sum(scores.values()) / len(scores),
            }
        )

    with (EXP_DIR / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EXP_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"[save] {EXP_DIR / 'results.csv'}")
    print(f"[save] {EXP_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
