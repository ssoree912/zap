#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize OneVision MMVet/detail student eval results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _keep_from_dir(path: Path) -> float:
    match = re.search(r"keep_?(\d+)$", path.name)
    if not match:
        raise ValueError(f"Cannot parse keep ratio from {path}")
    digits = match.group(1)
    return float(f"0.{digits}") if not digits.startswith("0.") else float(digits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []
    input_root = Path(args.input_root)
    for keep_dir in sorted(input_root.glob("keep_*")):
        keep_ratio = _keep_from_dir(keep_dir)
        for result_path in sorted(keep_dir.glob("*/result.json")):
            with result_path.open() as f:
                result = json.load(f)
            rows.append(
                {
                    "keep_ratio": keep_ratio,
                    "dataset": result["dataset"],
                    "rouge_l": round(float(result["rouge_l_f_mean"]) * 100.0, 2),
                    "ppl": round(float(result["ppl"]), 2),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keep_ratio", "dataset", "rouge_l", "ppl"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {output_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
