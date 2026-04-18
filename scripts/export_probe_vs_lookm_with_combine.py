#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
    "dataset",
    "metric",
    "probe_mlp_r020",
    "lookm_r020",
    "winner",
    "probe_n_samples",
    "combine_metric",
    "combine_probe_mlp_r020",
    "combine_winner",
    "combine_n_samples",
    "iterative_4_metric",
    "iterative_4_probe_mlp_r020",
    "iterative_4_winner",
    "iterative_4_n_samples",
    "oracle_metric",
    "oracle_r020",
    "oracle_winner",
    "oracle_n_samples",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in RESULT_FIELDS})


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_primary_metric(payload: dict[str, Any]) -> tuple[str, float | None]:
    for key in ("Accuracy", "ROUGE-L", "Rouge-L f"):
        if key in payload:
            metric_name = "ROUGE-L" if key == "Rouge-L f" else key
            return metric_name, float(payload[key])
    for key, value in payload.items():
        if isinstance(value, dict):
            continue
        try:
            return key, float(value)
        except Exception:
            continue
    return "", None


def compare_values(left: float | None, right: float | None, left_label: str) -> str:
    if left is None or right is None:
        return ""
    if abs(left - right) < 1e-12:
        return "tie"
    return left_label if left > right else "look_m"


def parse_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    return float(text)


def variant_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("iterative_"):
        try:
            return (2, int(name.split("_", 1)[1]), name)
        except Exception:
            return (2, 0, name)
    if name == "probe_mlp":
        return (1, 0, name)
    return (0, 0, name)


def auto_variant(base_rows: list[dict[str, str]], combine_root: Path, keep_dir: str) -> str:
    candidates: set[str] | None = None
    for row in base_rows:
        dataset_dir = combine_root / row["dataset"]
        available = {
            variant_dir.name
            for variant_dir in dataset_dir.iterdir()
            if variant_dir.is_dir() and (variant_dir / keep_dir / "eval.json").is_file() and (variant_dir / keep_dir / "metrics.json").is_file()
        }
        candidates = available if candidates is None else candidates.intersection(available)
    common = sorted(candidates or set(), key=variant_sort_key, reverse=True)
    if not common:
        raise FileNotFoundError(f"No common combine variant found under {combine_root} for {keep_dir}")
    return common[0]


def previous_oracle_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    _, rows = read_csv(path)
    return {row["dataset"]: row for row in rows}


def load_variant_result(
    combine_root: Path,
    dataset: str,
    variant: str,
    keep_dir: str,
) -> tuple[str, float | None, Any]:
    eval_path = combine_root / dataset / variant / keep_dir / "eval.json"
    metrics_path = combine_root / dataset / variant / keep_dir / "metrics.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing eval.json for {dataset}: {eval_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics.json for {dataset}: {metrics_path}")

    eval_payload = load_json(eval_path)
    metrics_payload = load_json(metrics_path)
    metric_name, metric_value = extract_primary_metric(eval_payload)
    return metric_name, metric_value, metrics_payload.get("n_samples", "")


def build_rows(
    base_rows: list[dict[str, str]],
    combine_root: Path,
    keep_dir: str,
    combine_variant: str,
    iterative_variant: str,
    previous_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        dataset = base_row["dataset"]
        combine_metric, combine_value, combine_n_samples = load_variant_result(
            combine_root=combine_root,
            dataset=dataset,
            variant=combine_variant,
            keep_dir=keep_dir,
        )
        iterative_metric, iterative_value, iterative_n_samples = load_variant_result(
            combine_root=combine_root,
            dataset=dataset,
            variant=iterative_variant,
            keep_dir=keep_dir,
        )
        lookm_value = parse_float(base_row.get("lookm_r020"))
        prev_row = previous_rows.get(dataset, {})

        row = {field: "" for field in RESULT_FIELDS}
        row["dataset"] = dataset
        row["metric"] = base_row.get("metric", "")
        row["probe_mlp_r020"] = base_row.get("probe_mlp_r020", "")
        row["lookm_r020"] = base_row.get("lookm_r020", "")
        row["winner"] = base_row.get("winner", compare_values(parse_float(base_row.get("probe_mlp_r020")), lookm_value, "probe"))
        row["probe_n_samples"] = base_row.get("probe_n_samples", "")
        row["combine_metric"] = combine_metric
        row["combine_probe_mlp_r020"] = combine_value
        row["combine_winner"] = compare_values(combine_value, lookm_value, "combine_probe")
        row["combine_n_samples"] = combine_n_samples
        row["iterative_4_metric"] = iterative_metric
        row["iterative_4_probe_mlp_r020"] = iterative_value
        row["iterative_4_winner"] = compare_values(iterative_value, lookm_value, "iterative_4")
        row["iterative_4_n_samples"] = iterative_n_samples
        row["oracle_metric"] = prev_row.get("oracle_metric", "")
        row["oracle_r020"] = prev_row.get("oracle_r020", "")
        row["oracle_winner"] = prev_row.get("oracle_winner", "")
        row["oracle_n_samples"] = prev_row.get("oracle_n_samples", "")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_csv",
        type=Path,
        default=Path("/workspace/zap/artifacts/probe_global/probe_vs_lookm_r020_truncated.csv"),
    )
    parser.add_argument(
        "--combine_root",
        type=Path,
        default=Path("/workspace/zap/artifacts/combine_prob"),
    )
    parser.add_argument(
        "--combine_variant",
        type=str,
        default="probe_mlp",
        help="Combine result subdir to use for the existing combine columns.",
    )
    parser.add_argument(
        "--iterative_variant",
        type=str,
        default="auto",
        help="Iterative result subdir to use for iterative_4 columns, e.g. iterative_4 or auto.",
    )
    parser.add_argument(
        "--keep_dir",
        type=str,
        default="keep_0p20",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("/workspace/zap/artifacts/results/probe_vs_lookm_r020_truncated_with_combine.csv"),
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path("/workspace/zap/artifacts/results/results.csv"),
    )
    parser.add_argument(
        "--results_export_script",
        type=Path,
        default=Path("/workspace/zap/scripts/export_milebench_results_csv.py"),
    )
    args = parser.parse_args()

    _, base_rows = read_csv(args.base_csv.resolve())
    previous_rows = previous_oracle_rows(args.output_csv.resolve())
    combine_root = args.combine_root.resolve()
    keep_dir = args.keep_dir
    combine_variant = args.combine_variant
    iterative_variant = args.iterative_variant
    if iterative_variant == "auto":
        iterative_variant = auto_variant(base_rows, combine_root, keep_dir)

    rows = build_rows(
        base_rows=base_rows,
        combine_root=combine_root,
        keep_dir=keep_dir,
        combine_variant=combine_variant,
        iterative_variant=iterative_variant,
        previous_rows=previous_rows,
    )
    write_csv(args.output_csv.resolve(), rows)

    subprocess.run(
        [
            sys.executable,
            str(args.results_export_script.resolve()),
            "--input_csv",
            str(args.output_csv.resolve()),
            "--output_csv",
            str(args.results_csv.resolve()),
        ],
        check=True,
    )

    print(f"combine_variant={combine_variant}")
    print(f"iterative_variant={iterative_variant}")
    print(args.output_csv.resolve())
    print(args.results_csv.resolve())


if __name__ == "__main__":
    main()
