#!/usr/bin/env python3
"""Plot DocVQA random eviction-scope figures without full-cache baseline lines."""

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


def format_ratio(ratio: float) -> str:
    return f"{ratio:.1f}"


def load_summary(path: Path) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            values[(row["method"], row["keep_ratio"], row["metric"])] = float(row["value"])
    return values


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


def save_ppl_plot(values: dict[tuple[str, str, str], float], outputs: Path) -> None:
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

    ax.set_yscale("log")
    style_axis(ax, "PPL vs GT (log scale, lower is better)")
    ax.set_title("DocVQA random eviction: PPL", pad=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outputs / "docvqa_random_scope_ppl_nofull.png", dpi=240)
    fig.savefig(outputs / "docvqa_random_scope_ppl_nofull.pdf")
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

    ax.set_ylim(0.0, 1.02)
    style_axis(ax, "ROUGE-L vs full cache (higher is better)")
    ax.set_title("DocVQA random eviction: ROUGE-L", pad=10)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(outputs / "docvqa_random_scope_rouge_nofull.png", dpi=240)
    fig.savefig(outputs / "docvqa_random_scope_rouge_nofull.pdf")
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
    values = load_summary(outputs / "summary.csv")
    save_ppl_plot(values, outputs)
    save_rouge_plot(values, outputs)
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_ppl_nofull.png'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_ppl_nofull.pdf'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_rouge_nofull.png'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_rouge_nofull.pdf'}")


if __name__ == "__main__":
    main()
