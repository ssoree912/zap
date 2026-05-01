#!/usr/bin/env python3
"""Plot DocVQA random eviction-scope figures with original GT metrics.

PPL is read from the existing teacher-forced GT runs. ROUGE-L is read from
random-method generations scored against the DocVQA ground-truth answers.
Full-cache metrics are recorded in the CSV for reference but are not plotted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("random_image_only", "random_all_token")
METHOD_LABELS = {
    "random_image_only": "Image-token only",
    "random_all_token": "All tokens",
}
METHOD_COLORS = {
    "random_image_only": "#2f80ed",
    "random_all_token": "#d94841",
}
RATIOS_ASC = [f"{i / 10:.1f}" for i in range(1, 10)]
RATIOS_DESC = [i / 10 for i in range(9, 0, -1)]


def tag(ratio: str) -> str:
    return ratio.replace(".", "p")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def add_metric(
    rows: list[dict[str, object]],
    *,
    method: str,
    keep_ratio: str,
    metric: str,
    value: float,
    n_samples: int,
    n_failures: int,
    path: Path,
) -> None:
    rows.append({
        "method": method,
        "keep_ratio": keep_ratio,
        "metric": metric,
        "value": value,
        "n_samples": n_samples,
        "n_failures": n_failures,
        "path": str(path),
    })


def collect_rows(outputs: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    full_ppl_path = outputs / "full" / "ppl_gt" / "result.json"
    full_rouge_path = outputs / "full" / "rouge_gt" / "result.json"
    full_ppl = load_json(full_ppl_path)
    full_rouge = load_json(full_rouge_path)
    if full_ppl:
        add_metric(
            rows,
            method="full",
            keep_ratio="",
            metric="ppl_gt",
            value=full_ppl["ppl"],
            n_samples=full_ppl["n_samples"],
            n_failures=full_ppl["n_failures"],
            path=full_ppl_path,
        )
    if full_rouge:
        add_metric(
            rows,
            method="full",
            keep_ratio="",
            metric="rouge_gt",
            value=full_rouge["rouge_l_f_mean"],
            n_samples=full_rouge["n_samples"],
            n_failures=full_rouge["n_failures"],
            path=full_rouge_path,
        )

    for method in METHODS:
        for ratio in RATIOS_ASC:
            base = outputs / method / f"keep_{tag(ratio)}"
            ppl_path = base / "ppl_gt" / "result.json"
            rouge_path = base / "rouge_gt" / "result.json"
            ppl = load_json(ppl_path)
            rouge = load_json(rouge_path)
            if ppl:
                add_metric(
                    rows,
                    method=method,
                    keep_ratio=ratio,
                    metric="ppl_gt",
                    value=ppl["ppl"],
                    n_samples=ppl["n_samples"],
                    n_failures=ppl["n_failures"],
                    path=ppl_path,
                )
            if rouge:
                add_metric(
                    rows,
                    method=method,
                    keep_ratio=ratio,
                    metric="rouge_gt",
                    value=rouge["rouge_l_f_mean"],
                    n_samples=rouge["n_samples"],
                    n_failures=rouge["n_failures"],
                    path=rouge_path,
                )
    return rows


def write_csv(rows: list[dict[str, object]], outputs: Path) -> Path:
    path = outputs / "summary_original_gt.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "keep_ratio", "metric", "value", "n_samples", "n_failures", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def values_by_key(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], float]:
    return {
        (str(row["method"]), str(row["keep_ratio"]), str(row["metric"])): float(row["value"])
        for row in rows
    }


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


def save_ppl_plot(values: dict[tuple[str, str, str], float], outputs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method in METHODS:
        ys = [values.get((method, format_ratio(r), "ppl_gt")) for r in RATIOS_DESC]
        if not all(y is not None for y in ys):
            continue
        ax.plot(
            RATIOS_DESC,
            [float(y) for y in ys],
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
    fig.savefig(outputs / "docvqa_random_scope_ppl_gt_nofull.png", dpi=240)
    fig.savefig(outputs / "docvqa_random_scope_ppl_gt_nofull.pdf")
    plt.close(fig)


def save_rouge_plot(values: dict[tuple[str, str, str], float], outputs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    max_y = 0.0
    for method in METHODS:
        ys = [values.get((method, format_ratio(r), "rouge_gt")) for r in RATIOS_DESC]
        if not all(y is not None for y in ys):
            continue
        y_float = [float(y) for y in ys]
        max_y = max(max_y, max(y_float))
        ax.plot(
            RATIOS_DESC,
            y_float,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.set_ylim(0.0, min(1.0, max(0.42, max_y + 0.05)))
    style_axis(ax, "ROUGE-L vs GT (higher is better)")
    ax.set_title("DocVQA random eviction: ROUGE-L", pad=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outputs / "docvqa_random_scope_rouge_gt_nofull.png", dpi=240)
    fig.savefig(outputs / "docvqa_random_scope_rouge_gt_nofull.pdf")
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
    outputs.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(outputs)
    csv_path = write_csv(rows, outputs)
    values = values_by_key(rows)
    save_ppl_plot(values, outputs)
    save_rouge_plot(values, outputs)

    print(f"[summary] wrote {csv_path}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_ppl_gt_nofull.png'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_ppl_gt_nofull.pdf'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_rouge_gt_nofull.png'}")
    print(f"[plot] wrote {outputs / 'docvqa_random_scope_rouge_gt_nofull.pdf'}")


if __name__ == "__main__":
    main()
