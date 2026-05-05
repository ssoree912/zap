# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate joint-eviction lmms-eval results into a single CSV.

Mirrors the format of
EXP-20260502-024/.../keep_ratio_dataset_performance.csv so the numbers can be
compared row-for-row against the visual-only original-student baseline.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

PRIMARY_METRIC: dict[str, tuple[str, float]] = {
    "textvqa_val": ("exact_match,none", 100.0),
    "gqa": ("exact_match,none", 100.0),
    "docvqa_val": ("anls,none", 100.0),
    "chartqa": ("relaxed_overall,none", 100.0),
    "scienceqa": ("exact_match,none", 100.0),
    "coco2017_cap_val": ("coco_CIDEr,none", 100.0),
    "nocaps_val": ("nocaps_CIDEr,none", 100.0),
    "textcaps_val": ("textcaps_CIDEr,none", 100.0),
}

TASKS_ORDER = [
    "textvqa_val",
    "gqa",
    "docvqa_val",
    "chartqa",
    "mme",
    "scienceqa",
    "coco2017_cap_val",
    "nocaps_val",
    "textcaps_val",
]


def keep_ratio_from_dirname(name: str) -> float | None:
    if not name.startswith("keep_"):
        return None
    tag = name.removeprefix("keep_")
    if not tag.isdigit() or len(tag) < 2:
        return None
    return float(f"0.{tag[1:]}") if tag.startswith("0") else float(f"0.{tag}")


def collect_run(run_root: Path) -> list[tuple[float, str, float]]:
    rows: list[tuple[float, str, float]] = []
    for keep_dir in sorted(run_root.iterdir()):
        if not keep_dir.is_dir():
            continue
        keep = keep_ratio_from_dirname(keep_dir.name)
        if keep is None:
            continue
        nested = keep_dir / "ckpts__llava-v1.5-7b"
        if not nested.is_dir():
            continue
        for jf in sorted(nested.glob("*_results.json")):
            with open(jf) as f:
                doc = json.load(f)
            for task, metrics in doc["results"].items():
                if task == "mme":
                    cog = metrics.get("mme_cognition_score,none", 0.0)
                    perc = metrics.get("mme_percetion_score,none", 0.0)
                    rows.append((keep, task, float(cog) + float(perc)))
                    continue
                metric_key, scale = PRIMARY_METRIC.get(task, (None, 1.0))
                if metric_key is None or metric_key not in metrics:
                    continue
                rows.append((keep, task, float(metrics[metric_key]) * scale))
    return rows


def write_csv(run_root: Path, rows: list[tuple[float, str, float]]) -> Path:
    out = run_root / "keep_ratio_dataset_performance.csv"
    rows_sorted = sorted(
        rows,
        key=lambda r: (-r[0], TASKS_ORDER.index(r[1]) if r[1] in TASKS_ORDER else 999),
    )
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keep_ratio", "dataset", "performance"])
        for keep, task, val in rows_sorted:
            w.writerow([keep, task, round(val, 1)])
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: aggregate_results.py <run_root>", file=sys.stderr)
        return 1
    run_root = Path(argv[1])
    if not run_root.is_dir():
        print(f"not a directory: {run_root}", file=sys.stderr)
        return 2
    rows = collect_run(run_root)
    if not rows:
        print(f"no results found under {run_root}", file=sys.stderr)
        return 3
    out = write_csv(run_root, rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
