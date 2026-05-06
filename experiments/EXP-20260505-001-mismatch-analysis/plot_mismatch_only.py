#!/usr/bin/env python3
"""Visualize all 3 mismatch metrics × keep_ratio across multiple datasets.

Outputs (saved to --out-dir):
  - mismatch_combined.png        3 panels (cos_sim, spearman, jaccard@K) ×
                                 line per method averaged over all datasets.
                                 The headline figure for the paper.
  - mismatch_per_dataset_lines.png   3 panels × line per (method, dataset).
                                     Detailed view; use color=method, style=dataset.
  - mismatch_dataset_summary.png  bar grid 2 rows × 4 cols, one panel per dataset
                                  showing all metrics side-by-side.
  - mismatch_layerwise.png        per-layer cos_sim/Spearman/Jaccard@0.2 (combined).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABEL = {
    "h2o_vs_teacher": "Prefill saliency (H2O)",
    "student_vs_teacher": "Q-ViK student",
}
METHOD_COLOR = {"h2o_vs_teacher": "#d62728", "student_vs_teacher": "#2ca02c"}


# stable per-dataset palette (line styles cycle through these)
DATASET_LS = [
    "-", "--", ":", "-.",
    (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mismatch-json", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--show", choices=["sim", "1-sim"], default="sim")
    return p.parse_args()


def per_sample(mm_json, method, metric):
    out = {}
    for s in mm_json["samples"]:
        v = s["mean"].get(method, {}).get(metric)
        if v is None: continue
        out.setdefault(s["dataset"], []).append(float(v))
    return out


def per_layer_mean(mm_json, method, metric, n_layers):
    bucket = [[] for _ in range(n_layers)]
    for s in mm_json["samples"]:
        per = s["per_layer"].get(method, {})
        for li_str, mdict in per.items():
            li = int(li_str)
            if metric in mdict and li < n_layers:
                bucket[li].append(float(mdict[metric]))
    return np.array([np.mean(b) if b else np.nan for b in bucket])


def transform(v, mode):
    return 1.0 - v if mode == "1-sim" else v


def plot_combined(mm_json, out_path: Path, mode: str):
    """Headline figure: 3 metrics × 1 line per method (mean over all datasets)."""
    keep_ratios = mm_json["config"]["keep_ratios"]
    methods = list(METHOD_LABEL.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for ax, kind, title in zip(
        axes,
        ["cos_sim", "spearman", "jaccard"],
        ["Cosine sim (probability-normalized)", "Spearman ρ", "Jaccard@K"],
    ):
        for method in methods:
            if kind == "jaccard":
                ys, errs = [], []
                for kr in keep_ratios:
                    per_ds = per_sample(mm_json, method, f"jaccard@{kr:g}")
                    pooled = [v for vs in per_ds.values() for v in vs]
                    ys.append(np.mean(pooled)); errs.append(np.std(pooled))
                y = transform(np.array(ys), mode)
                ax.errorbar(keep_ratios, y, yerr=errs, marker="o", lw=2, capsize=3,
                            color=METHOD_COLOR[method], label=METHOD_LABEL[method])
            else:
                per_ds = per_sample(mm_json, method, kind)
                pooled = [v for vs in per_ds.values() for v in vs]
                val = float(np.mean(pooled)); err = float(np.std(pooled))
                if mode == "1-sim": val = 1.0 - val
                y = np.full(len(keep_ratios), val)
                ax.plot(keep_ratios, y, marker="o", lw=2,
                        color=METHOD_COLOR[method], label=METHOD_LABEL[method])
                ax.fill_between(keep_ratios, y - err, y + err,
                                color=METHOD_COLOR[method], alpha=0.12)
        ax.set_xticks(keep_ratios)
        ax.set_xlabel("Keep ratio")
        ax.set_ylabel(("1 − " if mode == "1-sim" else "") + title)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("Mismatch vs future-decode oracle  (mean over all datasets, ±std)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_per_dataset_lines(mm_json, out_path: Path, mode: str):
    """3 metrics × line per (method, dataset). Color=method, style=dataset."""
    keep_ratios = mm_json["config"]["keep_ratios"]
    datasets = sorted({s["dataset"] for s in mm_json["samples"]})
    ds_styles = {ds: DATASET_LS[i % len(DATASET_LS)] for i, ds in enumerate(datasets)}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
    for ax, kind, title in zip(
        axes,
        ["cos_sim", "spearman", "jaccard"],
        ["Cosine sim (probability-normalized)", "Spearman ρ", "Jaccard@K"],
    ):
        for method in METHOD_LABEL:
            for ds in datasets:
                if kind == "jaccard":
                    ys = []
                    for kr in keep_ratios:
                        per_ds = per_sample(mm_json, method, f"jaccard@{kr:g}")
                        ys.append(np.mean(per_ds.get(ds, [np.nan])))
                    y = transform(np.array(ys), mode)
                else:
                    per_ds = per_sample(mm_json, method, kind)
                    val = float(np.mean(per_ds.get(ds, [np.nan])))
                    if mode == "1-sim": val = 1.0 - val
                    y = np.full(len(keep_ratios), val)
                ax.plot(keep_ratios, y, marker="o", markersize=4, lw=1.4,
                        color=METHOD_COLOR[method], linestyle=ds_styles[ds],
                        alpha=0.9,
                        label=f"{ds} · {'H2O' if 'h2o' in method else 'Stud'}")
        ax.set_xticks(keep_ratios)
        ax.set_xlabel("Keep ratio")
        ax.set_ylabel(("1 − " if mode == "1-sim" else "") + title)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    # one shared legend on the rightmost panel
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_dataset_summary(mm_json, out_path: Path, mode: str):
    """Bar grid 2×4: per dataset, all metrics side-by-side."""
    keep_ratios = mm_json["config"]["keep_ratios"]
    datasets = sorted({s["dataset"] for s in mm_json["samples"]})
    methods = list(METHOD_LABEL.keys())
    metrics = ["cos_sim", "spearman"] + [f"jaccard@{kr:g}" for kr in keep_ratios]
    n_metrics = len(metrics)

    n_ds = len(datasets)
    n_cols = 4
    n_rows = (n_ds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
                             sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    width = 0.36
    x = np.arange(n_metrics)

    for ax, ds in zip(axes_flat, datasets):
        for i, m in enumerate(methods):
            means, stds = [], []
            for met in metrics:
                vals = per_sample(mm_json, m, met).get(ds, [])
                if not vals:
                    means.append(np.nan); stds.append(0); continue
                v = np.array(vals)
                means.append(transform(np.array([v.mean()]), mode)[0])
                stds.append(v.std())
            ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
                   color=METHOD_COLOR[m], label=METHOD_LABEL[m], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("jaccard@", "J@") for m in metrics],
                           rotation=30, fontsize=8)
        n = sum(1 for s in mm_json['samples'] if s['dataset'] == ds)
        ax.set_title(f"{ds}  (n={n})", fontsize=10)
        ax.set_ylim(-0.05, 1.05); ax.axhline(0, color="black", lw=0.4)
        ax.grid(alpha=0.3, axis="y")
    for ax in axes_flat[len(datasets):]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8, loc="upper left")
    for r in range(n_rows):
        axes[r][0].set_ylabel(("1 − " if mode == "1-sim" else "") + "metric value")
    fig.suptitle("Per-dataset mismatch metrics", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_layerwise(mm_json, out_path: Path, mode: str):
    n_layers = max(int(s["n_layers"]) for s in mm_json["samples"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
    for ax, metric, title in zip(
        axes, ["cos_sim", "spearman", "jaccard@0.2"],
        ["Cosine sim", "Spearman ρ", "Jaccard@0.2"],
    ):
        for method, label in METHOD_LABEL.items():
            arr = per_layer_mean(mm_json, method, metric, n_layers)
            y = transform(arr, mode) if metric != "spearman" else arr
            ax.plot(np.arange(n_layers), y, marker="o", markersize=3.5,
                    color=METHOD_COLOR[method], label=label, lw=1.5)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(("1 − " if mode == "1-sim" and metric != "spearman" else "") + title)
        ax.set_title(f"Per-layer  {title}")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Per-layer mismatch  (combined over all 7 datasets)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mm_json = json.loads(args.mismatch_json.read_text())

    plot_combined(mm_json, args.out_dir / "mismatch_combined.png", args.show)
    plot_per_dataset_lines(mm_json, args.out_dir / "mismatch_per_dataset_lines.png", args.show)
    plot_dataset_summary(mm_json, args.out_dir / "mismatch_dataset_summary.png", args.show)
    plot_layerwise(mm_json, args.out_dir / "mismatch_layerwise.png", args.show)


if __name__ == "__main__":
    main()
