#!/usr/bin/env python3
"""Plot final DocVQA random eviction-scope figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "random_image_only": "Image-token only",
    "random_all_token": "All tokens",
}
METHOD_COLORS = {
    "random_image_only": "#2f80ed",
    "random_all_token": "#d94841",
}
RATIOS_DESC = [i / 10 for i in range(9, 0, -1)]


def load_summary(path: Path) -> tuple[dict[tuple[str, str, str], float], float]:
    values: dict[tuple[str, str, str], float] = {}
    full_ppl = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            ratio = row["keep_ratio"]
            metric = row["metric"]
            value = float(row["value"])
            values[(method, ratio, metric)] = value
            if method == "full" and metric == "ppl_gt":
                full_ppl = value
    if full_ppl is None:
        raise RuntimeError(f"missing full-cache PPL in {path}")
    return values, full_ppl


def format_ratio(ratio: float) -> str:
    return f"{ratio:.1f}"


def style_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Keep ratio")
    ax.set_ylabel(ylabel)
    ax.set_xticks(RATIOS_DESC)
    ax.set_xticklabels([format_ratio(x) for x in RATIOS_DESC])
    ax.set_xlim(0.95, 0.05)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_ppl_plot(values: dict[tuple[str, str, str], float], full_ppl: float, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method in METHOD_LABELS:
        ys = [values[(method, format_ratio(r), "ppl_gt")] for r in RATIOS_DESC]
        ax.plot(
            RATIOS_DESC,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.axhline(
        full_ppl,
        color="#333333",
        linestyle=(0, (5, 3)),
        linewidth=1.6,
        label=f"Full cache ({full_ppl:.3f})",
    )
    ax.set_yscale("log")
    style_axis(ax, "PPL vs GT (log scale, lower is better)")
    ax.set_title("DocVQA random eviction: PPL", pad=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outputs / "docvqa_random_scope_ppl.png", dpi=240)
    fig.savefig(outputs / "docvqa_random_scope_ppl.pdf")
    plt.close(fig)


def save_rouge_plot(values: dict[tuple[str, str, str], float], outputs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method in METHOD_LABELS:
        ys = [values[(method, format_ratio(r), "rouge_vs_full")] for r in RATIOS_DESC]
        ax.plot(
            RATIOS_DESC,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.axhline(
        1.0,
        color="#333333",
        linestyle=(0, (5, 3)),
        linewidth=1.6,
        label="Full cache (1.000)",
    )
    ax.set_ylim(0.0, 1.05)
    style_axis(ax, "ROUGE-L vs full cache (higher is better)")
    ax.set_title("DocVQA random eviction: ROUGE-L", pad=10)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(outputs / "docvqa_random_scope_rouge.png", dpi=240)
    fig.savefig(outputs / "docvqa_random_scope_rouge.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    outputs = exp_dir / "outputs"
    values, full_ppl = load_summary(outputs / "summary.csv")
    save_ppl_plot(values, full_ppl, outputs)
    save_rouge_plot(values, outputs)
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_ppl.png'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_ppl.pdf'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_rouge.png'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_rouge.pdf'}")


if __name__ == "__main__":
    main()
