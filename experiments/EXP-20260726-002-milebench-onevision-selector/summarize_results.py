#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate sample-level agreement, bootstrap CIs, and MileBench ROUGE-L."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

ZAP_ROOT = Path("/workspace/nips/zap")
LOOKM_ROOT = Path("/workspace/nips/LOOK-M")
if str(LOOKM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOKM_ROOT))

from evaluate import Eval  # noqa: E402

DATA_ROOT = ZAP_ROOT / "data" / "MileBench"
DATASETS = ("ALFRED", "CLEVR-Change", "IEdit", "Spot-the-Diff")
SELECTORS = ("full", "prefill", "smoothed_prefill", "qvik", "future_oracle")
PAIRS = (
    "prefill_vs_future",
    "smoothed_prefill_vs_future",
    "qvik_vs_future",
    "qvik_vs_prefill",
)
METRICS = ("spearman", "cosine", "jaccard_10", "jaccard_20", "jaccard_50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ZAP_ROOT
        / "artifacts"
        / "rebuttal_milebench_onevision_selector_total0p2",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def bootstrap_mean(
    values: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(repeats, dtype=np.float64)
    chunk = 1000
    for begin in range(0, repeats, chunk):
        count = min(chunk, repeats - begin)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[begin : begin + count] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def load_records(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    records_by_dataset = {}
    for dataset in DATASETS:
        paths = sorted((output_root / dataset / "records").glob("*.json"))
        records = [json.loads(path.read_text()) for path in paths]
        if len(records) != 200:
            raise RuntimeError(f"{dataset}: expected 200 records, found {len(records)}")
        records_by_dataset[dataset] = records
    return records_by_dataset


def sample_metric(record: dict[str, Any], pair: str, metric: str) -> float:
    return float(np.mean(record["agreement"][pair][metric]))


def score_predictions(
    dataset: str,
    records: list[dict[str, Any]],
    selector: str,
) -> float:
    core = json.loads((DATA_ROOT / dataset / f"{dataset}.json").read_text())
    predictions = [
        {
            "sample_id": int(record["sample_id"]),
            "pred_response": record["predictions"][selector],
            "gt_response": record["gt_response"],
        }
        for record in records
    ]
    result, _ = Eval().evaluate_rouge(copy.deepcopy(predictions), core)
    return float(result["Rouge-L f"]) * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_ci(row: dict[str, Any]) -> str:
    return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"


def main() -> int:
    args = parse_args()
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    records_by_dataset = load_records(args.output_root)
    all_records = [
        record
        for dataset in DATASETS
        for record in records_by_dataset[dataset]
    ]
    rng = np.random.default_rng(args.seed)

    agreement_rows = []
    for group in ("all", *DATASETS):
        records = (
            all_records if group == "all" else records_by_dataset[group]
        )
        for pair in PAIRS:
            for metric in METRICS:
                values = np.array(
                    [sample_metric(record, pair, metric) for record in records]
                )
                mean, low, high = bootstrap_mean(
                    values, repeats=args.bootstrap_repeats, rng=rng
                )
                agreement_rows.append(
                    {
                        "group": group,
                        "pair": pair,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_samples": len(records),
                    }
                )
    write_csv(summary_dir / "agreement.csv", agreement_rows)

    delta_rows = []
    for group in ("all", *DATASETS):
        records = (
            all_records if group == "all" else records_by_dataset[group]
        )
        for metric in METRICS:
            values = np.array(
                [
                    sample_metric(record, "qvik_vs_future", metric)
                    - sample_metric(record, "prefill_vs_future", metric)
                    for record in records
                ]
            )
            mean, low, high = bootstrap_mean(
                values, repeats=args.bootstrap_repeats, rng=rng
            )
            delta_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "qvik_minus_prefill": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_samples": len(records),
                }
            )
    write_csv(summary_dir / "paired_deltas.csv", delta_rows)

    layer_rows = []
    n_layers = len(all_records[0]["agreement"]["qvik_vs_future"]["spearman"])
    for layer_index in range(n_layers):
        for pair in PAIRS:
            for metric in METRICS:
                values = [
                    record["agreement"][pair][metric][layer_index]
                    for record in all_records
                ]
                layer_rows.append(
                    {
                        "layer": layer_index,
                        "pair": pair,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "n_samples": len(values),
                    }
                )
    write_csv(summary_dir / "layer_agreement.csv", layer_rows)

    answer_length_rows = []
    for bin_name, low, high in (
        ("1-8", 1, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33-127", 33, 127),
        ("128", 128, 10_000),
    ):
        records = [
            record
            for record in all_records
            if low <= int(record["answer_tokens"]) <= high
        ]
        for metric in ("spearman", "cosine", "jaccard_20"):
            prefill_values = np.array(
                [
                    sample_metric(record, "prefill_vs_future", metric)
                    for record in records
                ]
            )
            qvik_values = np.array(
                [
                    sample_metric(record, "qvik_vs_future", metric)
                    for record in records
                ]
            )
            delta = qvik_values - prefill_values
            mean, ci_low, ci_high = bootstrap_mean(
                delta, repeats=args.bootstrap_repeats, rng=rng
            )
            answer_length_rows.append(
                {
                    "answer_length_bin": bin_name,
                    "metric": metric,
                    "n_samples": len(records),
                    "mean_answer_tokens": float(
                        np.mean([record["answer_tokens"] for record in records])
                    ),
                    "prefill_vs_future": float(prefill_values.mean()),
                    "qvik_vs_future": float(qvik_values.mean()),
                    "qvik_minus_prefill": mean,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                }
            )
    write_csv(summary_dir / "agreement_by_answer_length.csv", answer_length_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    layer_lookup = {
        (row["layer"], row["pair"], row["metric"]): row["mean"]
        for row in layer_rows
    }
    plot_specs = (
        ("prefill_vs_future", "Prefill vs. Future", "#4c78a8", "-"),
        ("smoothed_prefill_vs_future", "Smoothed Prefill vs. Future", "#72b7b2", "--"),
        ("qvik_vs_future", "Q-ViK vs. Future", "#f58518", "-"),
    )
    for axis, metric, title in (
        (axes[0], "spearman", "Spearman"),
        (axes[1], "jaccard_20", "Jaccard@20%"),
    ):
        for pair, label, color, linestyle in plot_specs:
            values = [
                layer_lookup[(layer, pair, metric)] for layer in range(n_layers)
            ]
            axis.plot(
                range(n_layers),
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MileBench visual-token agreement (n=800)")
    fig.savefig(summary_dir / "layer_agreement.png", dpi=200)
    plt.close(fig)

    downstream_rows = []
    for selector in SELECTORS:
        scores = []
        row: dict[str, Any] = {"selector": selector}
        for dataset in DATASETS:
            score = score_predictions(
                dataset, records_by_dataset[dataset], selector
            )
            row[dataset] = score
            scores.append(score)
            selector_dir = args.output_root / dataset / selector
            selector_dir.mkdir(exist_ok=True)
            (selector_dir / "eval.json").write_text(
                json.dumps({"Rouge-L f": score / 100.0}, indent=2)
            )
        row["macro_mean"] = float(np.mean(scores))
        downstream_rows.append(row)
    write_csv(summary_dir / "downstream_rouge_l.csv", downstream_rows)

    budget_values = np.array(
        [record["actual_total_keep_ratio"] for record in all_records]
    )
    budget_summary = {
        "n_samples": len(all_records),
        "requested_total_keep_ratio": 0.2,
        "mean_actual_total_keep_ratio": float(budget_values.mean()),
        "min_actual_total_keep_ratio": float(budget_values.min()),
        "max_actual_total_keep_ratio": float(budget_values.max()),
        "mean_prompt_len": float(
            np.mean([record["prompt_len"] for record in all_records])
        ),
        "mean_visual_tokens": float(
            np.mean([record["n_visual"] for record in all_records])
        ),
        "mean_visual_tokens_kept": float(
            np.mean([record["n_visual_kept"] for record in all_records])
        ),
        "mean_full_answer_tokens": float(
            np.mean([record["answer_tokens"] for record in all_records])
        ),
    }
    (summary_dir / "budget_summary.json").write_text(
        json.dumps(budget_summary, indent=2)
    )

    lookup = {
        (row["group"], row["pair"], row["metric"]): row
        for row in agreement_rows
    }
    delta_lookup = {
        (row["group"], row["metric"]): row for row in delta_rows
    }
    lines = [
        "# MileBench LLaVA-OneVision selector comparison",
        "",
        f"Samples: {len(all_records)} (200 per task). Total-token keep ratio: 20%.",
        "Future is extracted by replaying the normal-SDPA Full Cache greedy answer.",
        "",
        "## Agreement with Future",
        "",
        "| Compared signal | Spearman | Cosine | Jaccard@10% | Jaccard@20% | Jaccard@50% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, pair in (
        ("Prefill attention", "prefill_vs_future"),
        ("1D-smoothed prefill", "smoothed_prefill_vs_future"),
        ("Q-ViK prediction", "qvik_vs_future"),
    ):
        cells = [
            fmt_ci(lookup[("all", pair, metric)]) for metric in METRICS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Direct Q-ViK versus Prefill",
            "",
            "| Spearman | Cosine |",
            "|---:|---:|",
            "| "
            + " | ".join(
                fmt_ci(lookup[("all", "qvik_vs_prefill", metric)])
                for metric in ("spearman", "cosine")
            )
            + " |",
            "",
            "## Paired Q-ViK minus Prefill agreement with Future",
            "",
            "| Metric | Delta (95% sample-bootstrap CI) |",
            "|---|---:|",
        ]
    )
    for metric in METRICS:
        row = delta_lookup[("all", metric)]
        lines.append(
            f"| {metric} | {row['qvik_minus_prefill']:+.4f} "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    length_lookup = {
        (row["answer_length_bin"], row["metric"]): row
        for row in answer_length_rows
    }
    lines.extend(
        [
            "",
            "## Agreement by Full Cache answer length",
            "",
            "| Answer tokens | n | Prefill–Future Spearman | Q-ViK–Future Spearman | Delta |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for bin_name in ("1-8", "9-16", "17-32", "33-127", "128"):
        row = length_lookup[(bin_name, "spearman")]
        lines.append(
            f"| {bin_name} | {row['n_samples']} | "
            f"{row['prefill_vs_future']:.4f} | {row['qvik_vs_future']:.4f} | "
            f"{row['qvik_minus_prefill']:+.4f} "
            f"[{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Downstream MileBench ROUGE-L",
            "",
            "| Selector | "
            + " | ".join((*DATASETS, "Macro mean"))
            + " |",
            "|---|" + "---:|" * (len(DATASETS) + 1),
        ]
    )
    for row in downstream_rows:
        lines.append(
            f"| {row['selector']} | "
            + " | ".join(
                f"{row[column]:.2f}" for column in (*DATASETS, "macro_mean")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Budget audit",
            "",
            f"- Actual total keep ratio: {budget_summary['mean_actual_total_keep_ratio']:.4f} "
            f"(min {budget_summary['min_actual_total_keep_ratio']:.4f}, "
            f"max {budget_summary['max_actual_total_keep_ratio']:.4f})",
            f"- Mean prompt/visual/kept visual tokens: "
            f"{budget_summary['mean_prompt_len']:.1f} / "
            f"{budget_summary['mean_visual_tokens']:.1f} / "
            f"{budget_summary['mean_visual_tokens_kept']:.1f}",
            f"- Mean Full Cache answer length: "
            f"{budget_summary['mean_full_answer_tokens']:.1f} tokens",
            "",
            "Smoothed Prefill uses a 1D three-token average because OneVision AnyRes "
            "produces a variable-length visual sequence without one fixed 2D grid.",
        ]
    )
    (summary_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {summary_dir / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
