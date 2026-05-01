# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Summarize OneVision ChartQA seed-sweep outputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def read_metric(csv_path: Path) -> dict[str, str | float]:
    with csv_path.open(newline="") as f:
        row = next(csv.DictReader(f))
    return {
        "overall": float(row["Overall"]),
        "test_human": float(row["test_human"]),
        "test_augmented": float(row["test_augmented"]),
    }


def find_metric_file(seed_dir: Path) -> Path | None:
    files = list(seed_dir.glob("ov7b_baseline_local/**/ov7b_baseline_local_ChartQA_TEST_acc.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def collect(root: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        match = re.fullmatch(r"seed_(\d+)", seed_dir.name)
        if not match or not seed_dir.is_dir():
            continue
        seed = int(match.group(1))
        metric_file = find_metric_file(seed_dir)
        row: dict[str, str | int | float] = {
            "seed": seed,
            "overall": "",
            "test_human": "",
            "test_augmented": "",
            "csv": str(metric_file) if metric_file else "",
            "work_dir": str(seed_dir),
            "log": str(root / "logs" / f"seed_{seed}.log"),
        }
        if metric_file is not None:
            row.update(read_metric(metric_file))
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, str | int | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "overall", "test_human", "test_augmented", "csv", "work_dir", "log"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    return [row for row in rows if isinstance(row["overall"], float)]


def print_summary(rows: list[dict[str, str | int | float]], target: float | None) -> None:
    done = numeric_rows(rows)
    if not done:
        print("No completed seed metrics found yet.")
        return
    best = max(done, key=lambda row: float(row["overall"]))
    print(
        "Best: seed={seed} overall={overall:.2f} human={test_human:.2f} augmented={test_augmented:.2f}".format(
            **best
        )
    )
    if target is not None:
        print(f"Target: {target:.2f}; met={float(best['overall']) >= target}")
    print("Top completed seeds:")
    for row in sorted(done, key=lambda r: float(r["overall"]), reverse=True)[:10]:
        print(
            "  seed={seed:<6} overall={overall:.2f} human={test_human:.2f} augmented={test_augmented:.2f}".format(
                **row
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--check-target", action="store_true")
    args = parser.parse_args()

    rows = collect(args.root)
    write_summary(rows, args.output)
    print_summary(rows, args.target)

    if args.check_target:
        done = numeric_rows(rows)
        if done and max(float(row["overall"]) for row in done) >= float(args.target):
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
