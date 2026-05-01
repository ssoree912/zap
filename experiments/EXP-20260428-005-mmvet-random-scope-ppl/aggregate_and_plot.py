#!/usr/bin/env python3
"""Aggregate and plot mm-vet random eviction PPL sweep."""

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


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ratio_tag(ratio: str) -> str:
    return ratio.replace(".", "p")


def collect_rows(exp_dir: Path) -> list[dict[str, object]]:
    outputs = exp_dir / "outputs"
    rows: list[dict[str, object]] = []

    full = load_json(outputs / "full" / "ppl" / "result.json")
    if full is not None:
        rows.append({
            "method": "full",
            "keep_ratio": "",
            "ppl": full["ppl"],
            "n_samples": full["n_samples"],
            "n_failures": full["n_failures"],
            "path": str(outputs / "full" / "ppl" / "result.json"),
        })

    for method in METHODS:
        for ratio in RATIOS_ASC:
            path = outputs / method / f"keep_{ratio_tag(ratio)}" / "ppl" / "result.json"
            payload = load_json(path)
            if payload is None:
                continue
            rows.append({
                "method": method,
                "keep_ratio": ratio,
                "ppl": payload["ppl"],
                "n_samples": payload["n_samples"],
                "n_failures": payload["n_failures"],
                "path": str(path),
            })
    return rows


def write_summary(exp_dir: Path, rows: list[dict[str, object]]) -> Path:
    outputs = exp_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    csv_path = outputs / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "keep_ratio", "ppl", "n_samples", "n_failures", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def row_map(rows: list[dict[str, object]]) -> dict[tuple[str, str], float]:
    return {(str(r["method"]), str(r["keep_ratio"])): float(r["ppl"]) for r in rows}


def style_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Keep ratio")
    ax.set_ylabel(ylabel)
    ax.set_xticks(RATIOS_DESC)
    ax.set_xticklabels([f"{x:.1f}" for x in RATIOS_DESC])
    ax.set_xlim(0.95, 0.05)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_if_ready(exp_dir: Path, rows: list[dict[str, object]]) -> None:
    values = row_map(rows)
    full_ppl = values.get(("full", ""))
    if full_ppl is None:
        return
    if not all((method, f"{ratio:.1f}") in values for method in METHODS for ratio in RATIOS_DESC):
        return

    import matplotlib.pyplot as plt

    outputs = exp_dir / "outputs"

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in METHODS:
        ys = [values[(method, f"{ratio:.1f}")] for ratio in RATIOS_DESC]
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
    ax.set_title("mm-vet random eviction: PPL", pad=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outputs / "mmvet_random_scope_ppl.png", dpi=240)
    fig.savefig(outputs / "mmvet_random_scope_ppl.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in METHODS:
        ys = [values[(method, f"{ratio:.1f}")] for ratio in RATIOS_DESC]
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
    image_vals = [values[("random_image_only", f"{ratio:.1f}")] for ratio in RATIOS_DESC]
    ymin = min([full_ppl, *image_vals]) * 0.985
    ymax = max([full_ppl, *image_vals]) * 1.025
    if ymax - ymin < 0.1:
        ymax = ymin + 0.1
    ax.set_ylim(ymin, ymax)
    style_axis(ax, "PPL vs GT (zoomed, lower is better)")
    ax.set_title("mm-vet random eviction: PPL zoom", pad=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outputs / "mmvet_random_scope_ppl_zoom.png", dpi=240)
    fig.savefig(outputs / "mmvet_random_scope_ppl_zoom.pdf")
    plt.close(fig)


def write_result(exp_dir: Path, rows: list[dict[str, object]]) -> Path:
    values = row_map(rows)
    full_ppl = values.get(("full", ""))
    lines = [
        "## Experiment Result",
        "",
        "**ID**: EXP-20260428-005-mmvet-random-scope-ppl",
        f"**Status**: {'done' if len(rows) == 19 else 'running / partial'}",
        "",
        "### Summary",
        "",
        f"- Full-cache PPL: `{full_ppl:.6f}`" if isinstance(full_ppl, float) else "- Full-cache PPL: pending",
        "",
        "| keep | random_image_only PPL ↓ | Δ vs full | random_all_token PPL ↓ | Δ vs full |",
        "|------|--------------------------|-----------|------------------------|-----------|",
    ]
    for ratio in RATIOS_ASC:
        img = values.get(("random_image_only", ratio))
        all_tok = values.get(("random_all_token", ratio))
        if isinstance(full_ppl, float) and img is not None and all_tok is not None:
            lines.append(
                f"| {ratio} | {img:.4f} | {img - full_ppl:+.4f} | "
                f"{all_tok:.4f} | {all_tok - full_ppl:+.4f} |"
            )
        else:
            lines.append(f"| {ratio} | {img if img is not None else ''} |  | {all_tok if all_tok is not None else ''} |  |")
    lines.extend([
        "",
        "### Artifacts",
        "",
        f"- CSV: `{exp_dir / 'outputs' / 'summary.csv'}`",
        f"- PPL plot: `{exp_dir / 'outputs' / 'mmvet_random_scope_ppl.png'}`",
        f"- PPL zoom plot: `{exp_dir / 'outputs' / 'mmvet_random_scope_ppl_zoom.png'}`",
        f"- Logs: `{exp_dir / 'logs'}` and per-run `run.log` under each output directory",
        "",
    ])
    path = exp_dir / "RESULT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="/workspace/zap/experiments/EXP-20260428-005-mmvet-random-scope-ppl",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    rows = collect_rows(exp_dir)
    csv_path = write_summary(exp_dir, rows)
    result_path = write_result(exp_dir, rows)
    plot_if_ready(exp_dir, rows)
    print(f"[aggregate] rows={len(rows)} csv={csv_path}")
    print(f"[aggregate] result={result_path}")


if __name__ == "__main__":
    main()
