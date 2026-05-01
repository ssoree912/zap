#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


GROUPS: list[tuple[str, list[str]]] = [
    ("T-1", ["actionlocalization", "actionprediction", "actionsequence"]),
    ("T-2", ["objectexistence", "objectinteraction", "movingattribute", "objectshuffle"]),
    ("T-3", ["egocentricnavigation", "movingdirection"]),
    ("T-4", ["counterfactualinference", "statechange", "characterorder", "scenetransition"]),
    ("S-1", ["webqa", "tqa", "multimodalqa", "wikivqa"]),
    ("S-2", ["slidevqa", "ocr_vqa", "docvqa"]),
    ("S-3", ["clevr_change", "spot_the_diff", "iedit"]),
    ("S-4", ["mmcoqa", "alfred"]),
    ("S-5", ["nuscenes"]),
    ("N-1", ["textneedleinahaystack"]),
    ("N-2", ["imageneedleinahaystack"]),
    ("I-1", ["gpr1200"]),
]

OUTPUT_FIELDS = [
    "dataset",
    "group_code",
    "metric",
    "probe_mlp_r020",
    "lookm_r020",
    "combine_probe_mlp_r020",
    "iterative_4_probe_mlp_r020",
    "datasets_included",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in OUTPUT_FIELDS})


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    return float(text)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def group_lookup() -> dict[str, str]:
    dataset_to_group: dict[str, str] = {}
    for group_code, datasets in GROUPS:
        for dataset in datasets:
            dataset_to_group[dataset] = group_code
    return dataset_to_group


def resolve_metric(rows: list[dict[str, str]]) -> str:
    metrics = sorted({(row.get("metric") or "").strip() for row in rows if (row.get("metric") or "").strip()})
    if not metrics:
        return ""
    if len(metrics) == 1:
        return metrics[0]
    return "mixed_mean"


def build_dataset_rows(rows: list[dict[str, str]], dataset_to_group: dict[str, str]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        dataset = row["dataset"]
        output_rows.append(
            {
                "dataset": dataset,
                "group_code": dataset_to_group[dataset],
                "metric": row.get("metric", ""),
                "probe_mlp_r020": row.get("probe_mlp_r020", ""),
                "lookm_r020": row.get("lookm_r020", ""),
                "combine_probe_mlp_r020": row.get("combine_probe_mlp_r020", ""),
                "iterative_4_probe_mlp_r020": row.get("iterative_4_probe_mlp_r020", ""),
                "datasets_included": "",
            }
        )
    return output_rows


def build_group_rows(rows_by_dataset: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for group_code, datasets in GROUPS:
        group_rows = [rows_by_dataset[dataset] for dataset in datasets]
        output_rows.append(
            {
                "dataset": f"{group_code}_avg",
                "group_code": group_code,
                "metric": resolve_metric(group_rows),
                "probe_mlp_r020": format_float(mean([parse_float(row.get("probe_mlp_r020")) for row in group_rows])),
                "lookm_r020": format_float(mean([parse_float(row.get("lookm_r020")) for row in group_rows])),
                "combine_probe_mlp_r020": format_float(mean([parse_float(row.get("combine_probe_mlp_r020")) for row in group_rows])),
                "iterative_4_probe_mlp_r020": format_float(mean([parse_float(row.get("iterative_4_probe_mlp_r020")) for row in group_rows])),
                "datasets_included": "|".join(datasets),
            }
        )
    return output_rows


def validate_source(rows: list[dict[str, str]], dataset_to_group: dict[str, str]) -> None:
    source_datasets = [row["dataset"] for row in rows]
    missing = [dataset for dataset in source_datasets if dataset not in dataset_to_group]
    if missing:
        raise ValueError(f"Unmapped datasets in source CSV: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("/workspace/zap/artifacts/results/probe_vs_lookm_r020_truncated_with_combine.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("/workspace/zap/artifacts/results/results.csv"),
    )
    args = parser.parse_args()

    rows = read_csv(args.input_csv.resolve())
    dataset_to_group = group_lookup()
    validate_source(rows, dataset_to_group)

    rows_by_dataset = {row["dataset"]: row for row in rows}
    output_rows = build_dataset_rows(rows, dataset_to_group)
    output_rows.extend(build_group_rows(rows_by_dataset))

    write_csv(args.output_csv.resolve(), output_rows)
    print(args.output_csv.resolve())


if __name__ == "__main__":
    main()
