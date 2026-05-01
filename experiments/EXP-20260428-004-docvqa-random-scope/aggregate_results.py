#!/usr/bin/env python3
"""Aggregate DocVQA random scope PPL/ROUGE sweep into CSV, markdown, and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("random_image_only", "random_all_token")
RATIOS = [f"{i / 10:.1f}" for i in range(1, 10)]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def tag(ratio: str) -> str:
    return ratio.replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    outputs = exp_dir / "outputs"
    rows: list[dict[str, object]] = []

    full_ppl = load_json(outputs / "full" / "ppl_gt" / "result.json")
    full_rouge = load_json(outputs / "full" / "rouge_gt" / "result.json")
    if full_ppl:
        rows.append({
            "method": "full",
            "keep_ratio": "",
            "metric": "ppl_gt",
            "value": full_ppl["ppl"],
            "n_samples": full_ppl["n_samples"],
            "n_failures": full_ppl["n_failures"],
            "path": str(outputs / "full" / "ppl_gt" / "result.json"),
        })
    if full_rouge:
        rows.append({
            "method": "full",
            "keep_ratio": "",
            "metric": "rouge_gt",
            "value": full_rouge["rouge_l_f_mean"],
            "n_samples": full_rouge["n_samples"],
            "n_failures": full_rouge["n_failures"],
            "path": str(outputs / "full" / "rouge_gt" / "result.json"),
        })

    for method in METHODS:
        for ratio in RATIOS:
            base = outputs / method / f"keep_{tag(ratio)}"
            ppl = load_json(base / "ppl_gt" / "result.json")
            rouge = load_json(base / "rouge_vs_full" / "result.json")
            if ppl:
                rows.append({
                    "method": method,
                    "keep_ratio": ratio,
                    "metric": "ppl_gt",
                    "value": ppl["ppl"],
                    "n_samples": ppl["n_samples"],
                    "n_failures": ppl["n_failures"],
                    "path": str(base / "ppl_gt" / "result.json"),
                })
            if rouge:
                rows.append({
                    "method": method,
                    "keep_ratio": ratio,
                    "metric": "rouge_vs_full",
                    "value": rouge["rouge_l_f_mean"],
                    "n_samples": rouge["n_samples"],
                    "n_failures": rouge["n_failures"],
                    "path": str(base / "rouge_vs_full" / "result.json"),
                })

    outputs.mkdir(parents=True, exist_ok=True)
    csv_path = outputs / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "keep_ratio", "metric", "value", "n_samples", "n_failures", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = exp_dir / "RESULT.md"
    md_lines = [
        "## Experiment Result",
        "",
        "**ID**: EXP-20260428-004-docvqa-random-scope",
        "**Completed**: pending if some rows are missing",
        "",
        "### 결과 요약",
        "",
        "| keep | random_image_only PPL ↓ | random_all_token PPL ↓ | random_image_only ROUGE-L ↑ | random_all_token ROUGE-L ↑ |",
        "|------|--------------------------|------------------------|------------------------------|----------------------------|",
    ]
    by_key = {(r["method"], r["keep_ratio"], r["metric"]): r["value"] for r in rows}
    for ratio in RATIOS:
        img_ppl = by_key.get(("random_image_only", ratio, "ppl_gt"), "")
        all_ppl = by_key.get(("random_all_token", ratio, "ppl_gt"), "")
        img_rg = by_key.get(("random_image_only", ratio, "rouge_vs_full"), "")
        all_rg = by_key.get(("random_all_token", ratio, "rouge_vs_full"), "")
        md_lines.append(
            f"| {ratio} | {float(img_ppl):.4f} | {float(all_ppl):.4f} | "
            f"{float(img_rg):.4f} | {float(all_rg):.4f} |"
            if all(v != "" for v in (img_ppl, all_ppl, img_rg, all_rg))
            else f"| {ratio} | {img_ppl} | {all_ppl} | {img_rg} | {all_rg} |"
        )

    md_lines.extend([
        "",
        "### 체크포인트 / 아티팩트 위치",
        f"- CSV: `{csv_path}`",
        f"- Outputs: `{outputs}`",
        f"- Logs: `{exp_dir / 'logs'}`",
        "",
        "### Caveat",
        "- `random_image_only`는 text token을 항상 보존하므로 낮은 keep ratio에서 effective prompt keep ratio가 nominal ratio보다 커질 수 있다.",
        "- PPL은 GT answer 기준이고, ROUGE-L은 full-cache LLaVA generation 기준이다.",
        "",
    ])
    md_path.write_text("\n".join(md_lines))
    print(f"[aggregate] wrote {csv_path}")
    print(f"[aggregate] wrote {md_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        xs = [float(r) for r in RATIOS]
        for method, label in [("random_image_only", "image-only random"), ("random_all_token", "all-token random")]:
            ppl_vals = [by_key.get((method, r, "ppl_gt")) for r in RATIOS]
            rouge_vals = [by_key.get((method, r, "rouge_vs_full")) for r in RATIOS]
            if all(v is not None for v in ppl_vals):
                axes[0].plot(xs, [float(v) for v in ppl_vals], marker="o", label=label)
            if all(v is not None for v in rouge_vals):
                axes[1].plot(xs, [float(v) for v in rouge_vals], marker="o", label=label)
        axes[0].set_xlabel("Keep ratio")
        axes[0].set_ylabel("PPL vs GT (lower is better)")
        axes[0].set_title("DocVQA PPL")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Keep ratio")
        axes[1].set_ylabel("ROUGE-L vs full-cache (higher is better)")
        axes[1].set_title("DocVQA ROUGE-L")
        axes[1].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        plot_path = outputs / "docvqa_random_scope_sweep.png"
        fig.savefig(plot_path, dpi=180)
        print(f"[plot] wrote {plot_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] skipped: {exc!r}")


if __name__ == "__main__":
    main()

