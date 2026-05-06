#!/usr/bin/env python3
"""Clean two-figure layout for the paper.

Outputs (in --out-dir):
  - mismatch_jaccard_keep_ratio.png   line plot of Jaccard@K vs keep_ratio
                                      (decreasing x: 0.8 → 0.2). One line per method.
  - mismatch_global_table.png         small 2×2 table for global metrics
                                      (cos_sim, Spearman) per method.

If `--eval-json` is provided, also writes:
  - accuracy_vs_keep_ratio.png         same x-axis, accuracy lines per method.
  - mismatch_vs_accuracy_scatter.png   x = mismatch (1 - Jaccard), y = accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD = {
    "h2o": ("Prefill saliency", "#d62728", "o"),
    "student": ("Q-ViK student", "#2ca02c", "s"),
    "full_cache": ("Full cache (no eviction)", "#7f7f7f", "x"),
}
MISMATCH_KEY = {"h2o": "h2o_vs_teacher", "student": "student_vs_teacher"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mismatch-json", required=True, type=Path)
    p.add_argument("--eval-json", type=Path, default=None,
                   help="Optional: from run_eval_with_press.py")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--filter-datasets", nargs="+", default=None,
                   help="If given, only average over these dataset names.")
    return p.parse_args()


def per_sample(mm, method, metric, datasets=None):
    out = []
    for s in mm["samples"]:
        if datasets is not None and s["dataset"] not in datasets:
            continue
        v = s["mean"].get(method, {}).get(metric)
        if v is not None:
            out.append(float(v))
    return np.array(out)


def plot_jaccard_keep_ratio(mm, out_path: Path, datasets=None):
    keep_ratios = sorted(mm["config"]["keep_ratios"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for short in ("h2o", "student"):
        label, color, marker = METHOD[short]
        means = []
        for kr in keep_ratios:
            vals = per_sample(mm, MISMATCH_KEY[short], f"jaccard@{kr:g}", datasets)
            means.append(vals.mean())
        ax.plot(keep_ratios, means, marker=marker, lw=4.0,
                color=color, label=label, markersize=15)
    ax.set_xticks(keep_ratios)
    # Extra right padding to fit annotation labels at smallest keep ratio
    ax.set_xlim(max(keep_ratios) + 0.05, min(keep_ratios) - 0.10)
    ax.set_xlabel("Keep ratio", fontsize=30)
    ax.set_ylabel("Jaccard@K ", fontsize=30)
    ax.set_title("Jaccard@K↑", fontsize=30, pad=20)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=20, loc="lower left")
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    # Save both PDF and PNG
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.with_suffix('.pdf')}", flush=True)
    print(f"[saved] {out_path.with_suffix('.png')}", flush=True)


def plot_global_table(mm, out_path: Path):
    """2 cols × 2 rows table: rows=metric, cols=method."""
    rows = []
    for short in ("h2o", "student"):
        cs = per_sample(mm, MISMATCH_KEY[short], "cos_sim")
        sp = per_sample(mm, MISMATCH_KEY[short], "spearman")
        rows.append((short, cs.mean(), sp.mean()))
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.axis("off")
    cell_text = [
        [f"{rows[0][1]:.3f}", f"{rows[1][1]:.3f}"],
        [f"{rows[0][2]:.3f}", f"{rows[1][2]:.3f}"],
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=["Cosine sim (prob-norm)  ↑", "Spearman ρ  ↑"],
        colLabels=[METHOD[rows[0][0]][0], METHOD[rows[1][0]][0]],
        cellLoc="center", loc="center",
        colWidths=[0.32, 0.32],
    )
    table.auto_set_font_size(False); table.set_fontsize(20)
    table.scale(1.1, 2.6)
    # color headers by method
    for j, (short, _, _) in enumerate(rows):
        table[0, j].set_facecolor(METHOD[short][1])
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Global mismatch (constant across keep ratio)",
                 fontsize=24, pad=16)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.with_suffix('.pdf')}", flush=True)
    print(f"[saved] {out_path.with_suffix('.png')}", flush=True)


def plot_accuracy_keep_ratio(eval_json: dict, out_path: Path, datasets: list[str]):
    """Lines per method. Single panel when only one dataset, else per-dataset + combined."""
    keep_ratios = sorted(eval_json["config"]["keep_ratios"], reverse=True)
    summary = {(s["dataset"], s["method"], s["keep_ratio"]): s
               for s in eval_json["summary"]}

    single = len(datasets) == 1
    panels = list(datasets) if single else list(datasets) + ["combined"]
    n_panels = len(panels)

    if single:
        fig, axes = plt.subplots(1, 1, figsize=(9.0, 9.0))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(9.0 * n_panels, 9.0), sharey=False)

    for ax, ds in zip(axes, panels):
        for short in ("h2o", "student"):
            label, color, marker = METHOD[short]
            ys = []
            for kr in keep_ratios:
                if ds == "combined":
                    parts = [summary[(d, short, kr)]
                             for d in datasets if (d, short, kr) in summary]
                    ys.append(np.mean([p["mean"] for p in parts]) if parts else np.nan)
                else:
                    p = summary.get((ds, short, kr))
                    ys.append(p["mean"] if p else np.nan)
            ax.plot(keep_ratios, ys, marker=marker, lw=4.0,
                    color=color, label=label, markersize=15)

        # full-cache horizontal line
        if ds == "combined":
            fc_vals = [summary[(d, "full_cache", 1.0)]["mean"]
                       for d in datasets if (d, "full_cache", 1.0) in summary]
            fc = float(np.mean(fc_vals)) if fc_vals else None
        else:
            fc_entry = summary.get((ds, "full_cache", 1.0))
            fc = fc_entry["mean"] if fc_entry else None
        if fc is not None:
            ax.axhline(fc, color=METHOD["full_cache"][1], ls="--", lw=2.5,
                       label=f"Full cache = {fc:.3f}")

        ax.set_xticks(keep_ratios)
        ax.set_xlim(max(keep_ratios) + 0.05, min(keep_ratios) - 0.10)
        ax.set_xlabel("Keep ratio", fontsize=30)
        ax.set_ylabel("Accuracy", fontsize=30)
        ax.set_title("Accuracy↑", fontsize=30, pad=20)
        ax.grid(alpha=0.5)
        ax.legend(fontsize=15, loc="lower left")
        ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.with_suffix('.pdf')}", flush=True)
    print(f"[saved] {out_path.with_suffix('.png')}", flush=True)


def plot_mismatch_vs_accuracy(mm, eval_json: dict, out_path: Path,
                              datasets: list[str]):
    """Line plot: x = 1-Jaccard@kr, y = accuracy. One line per method, marker per kr."""
    keep_ratios = sorted(eval_json["config"]["keep_ratios"], reverse=True)
    summary = {(s["dataset"], s["method"], s["keep_ratio"]): s
               for s in eval_json["summary"]}

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    all_x, all_y = [], []
    for short in ("h2o", "student"):
        label, color, marker = METHOD[short]
        xs, ys = [], []
        for kr in keep_ratios:
            jacc_metric = f"jaccard@{kr:g}"
            mm_per_ds = {}
            for s in mm["samples"]:
                v = s["mean"].get(MISMATCH_KEY[short], {}).get(jacc_metric)
                if v is not None and s["dataset"] in datasets:
                    mm_per_ds.setdefault(s["dataset"], []).append(float(v))
            for ds in datasets:
                acc = summary.get((ds, short, kr))
                if acc is None or ds not in mm_per_ds:
                    continue
                jacc = float(np.mean(mm_per_ds[ds]))
                mismatch = 1.0 - jacc
                xs.append(mismatch); ys.append(acc["mean"])
                all_x.append(mismatch); all_y.append(acc["mean"])
        if len(xs) >= 2:
            xs_a, ys_a = np.array(xs), np.array(ys)
            order = np.argsort(xs_a)
            ax.plot(xs_a[order], ys_a[order], color=color, lw=4.0,
                    marker=marker, markersize=15, label=label)

    title = "Accuracy↑"
    ax.set_title(title, fontsize=30, pad=20)
    ax.set_xlabel("Mismatch↓ (1 − Jaccard@K)", fontsize=30)
    ax.set_ylabel("Accuracy", fontsize=30)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.with_suffix('.pdf')}", flush=True)
    print(f"[saved] {out_path.with_suffix('.png')}", flush=True)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mm = json.loads(args.mismatch_json.read_text())

    plot_jaccard_keep_ratio(mm, args.out_dir / "mismatch_jaccard_keep_ratio",
                            datasets=args.filter_datasets)
    plot_global_table(mm, args.out_dir / "mismatch_global_table.png")

    if args.eval_json:
        eval_json = json.loads(args.eval_json.read_text())
        eval_datasets = eval_json["config"]["datasets"]
        # match eval datasets to those present in both
        mm_datasets = {s["dataset"] for s in mm["samples"]}
        common = [d for d in eval_datasets if d in mm_datasets]
        plot_accuracy_keep_ratio(eval_json,
                                 args.out_dir / "accuracy_vs_keep_ratio.png", common)
        plot_mismatch_vs_accuracy(mm, eval_json,
                                  args.out_dir / "mismatch_vs_accuracy.png",
                                  common)


if __name__ == "__main__":
    main()
