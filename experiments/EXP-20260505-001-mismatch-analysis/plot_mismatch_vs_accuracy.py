#!/usr/bin/env python3
"""Plot future-utility mismatch vs downstream eviction accuracy.

Inputs
------
1) Mismatch JSON  produced by `compute_mismatch.py`
2) Eval manifest  JSON mapping (method, keep_ratio) -> eval result file + metric
   path. Schema:

   {
     "metric_path": ["results", "docvqa_val", "anls,none"],   # default for all
     "full_cache":  {"path": "...", "metric_path": [...]},     # optional baseline
     "methods": {
       "h2o":     {"0.2": "/.../h2o_keep02.json", "0.4": "...", "0.6": "...", "0.8": "..."},
       "student": {"0.2": "/.../stu_keep02.json", "0.4": "...", "0.6": "...", "0.8": "..."}
     }
   }

The script averages mismatch metrics over samples and pairs each
(method, keep_ratio) point with the eval accuracy. Figures saved:

  - mismatch_vs_keep_ratio.png   (one line per method)
  - accuracy_vs_keep_ratio.png   (one line per method, dashed for full cache)
  - mismatch_vs_accuracy.png     (scatter; lower mismatch → higher acc?)

Usage
-----
  python plot_mismatch_vs_accuracy.py \
      --mismatch-json mismatch_textvqa_gqa.json \
      --eval-manifest eval_manifest.json \
      --metric jaccard@0.2 \
      --out-dir figs/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "h2o_vs_teacher": "Prefill saliency (H2O)",
    "student_vs_teacher": "Q-ViK student",
}
METHOD_COLORS = {
    "h2o": "#d62728",        # red
    "student": "#2ca02c",    # green
    "full_cache": "#7f7f7f", # gray
}
METHOD_MARKERS = {
    "h2o": "o",
    "student": "s",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _resolve_metric(blob: Any, path: list[str]) -> float:
    cur = blob
    for key in path:
        cur = cur[key]
    return float(cur)


def collect_mismatch(mismatch_json: dict, method_key: str, metric: str) -> dict[str, list[float]]:
    """Return {dataset: [per-sample metric]} for the requested method/metric."""
    out: dict[str, list[float]] = {}
    for s in mismatch_json["samples"]:
        ds = s["dataset"]
        m = s["mean"].get(method_key, {}).get(metric)
        if m is None:
            continue
        out.setdefault(ds, []).append(float(m))
    return out


def collect_accuracy(manifest: dict, method: str, kr: str) -> float | None:
    methods = manifest.get("methods", {})
    if method not in methods or kr not in methods[method]:
        return None
    eval_path = Path(methods[method][kr])
    blob = _read_json(eval_path)
    metric_path = manifest.get("metric_path") or []
    return _resolve_metric(blob, metric_path)


def collect_full_cache(manifest: dict) -> float | None:
    fc = manifest.get("full_cache")
    if not fc:
        return None
    blob = _read_json(Path(fc["path"]))
    return _resolve_metric(blob, fc.get("metric_path") or manifest.get("metric_path") or [])


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_mismatch_vs_keep_ratio(
    rows: list[dict],
    out_path: Path,
    metric_label: str,
) -> None:
    """rows: each {method, keep_ratio (float), mismatch_mean, mismatch_std}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        pts = sorted([r for r in rows if r["method"] == m], key=lambda x: x["keep_ratio"])
        x = [r["keep_ratio"] for r in pts]
        y = [r["mismatch_mean"] for r in pts]
        e = [r["mismatch_std"] for r in pts]
        ax.errorbar(x, y, yerr=e, marker=METHOD_MARKERS.get(m, "o"),
                    color=METHOD_COLORS.get(m, "black"), label=m, capsize=3, lw=1.5)
    ax.set_xlabel("Keep ratio")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Mismatch ({metric_label}) vs keep ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_accuracy_vs_keep_ratio(
    rows: list[dict],
    full_cache_acc: float | None,
    out_path: Path,
    acc_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = sorted({r["method"] for r in rows if r["accuracy"] is not None})
    for m in methods:
        pts = sorted([r for r in rows if r["method"] == m and r["accuracy"] is not None],
                     key=lambda x: x["keep_ratio"])
        if not pts:
            continue
        x = [r["keep_ratio"] for r in pts]
        y = [r["accuracy"] for r in pts]
        ax.plot(x, y, marker=METHOD_MARKERS.get(m, "o"),
                color=METHOD_COLORS.get(m, "black"), label=m, lw=1.5)
    if full_cache_acc is not None:
        ax.axhline(full_cache_acc, color=METHOD_COLORS["full_cache"], ls="--",
                   lw=1.0, label=f"Full cache = {full_cache_acc:.3f}")
    ax.set_xlabel("Keep ratio")
    ax.set_ylabel(acc_label)
    ax.set_title(f"{acc_label} vs keep ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_mismatch_vs_accuracy(
    rows: list[dict],
    full_cache_acc: float | None,
    out_path: Path,
    metric_label: str,
    acc_label: str,
) -> None:
    """Scatter: x = mismatch (1 - similarity), y = accuracy. Marker = method, size = keep_ratio."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = sorted({r["method"] for r in rows if r["accuracy"] is not None})
    # one Spearman line over all points (qualitative)
    all_x, all_y = [], []
    for m in methods:
        pts = [r for r in rows if r["method"] == m and r["accuracy"] is not None]
        x = np.array([r["mismatch_mean"] for r in pts])
        y = np.array([r["accuracy"] for r in pts])
        sizes = np.array([60 + 200 * r["keep_ratio"] for r in pts])
        ax.scatter(x, y, s=sizes, marker=METHOD_MARKERS.get(m, "o"),
                   color=METHOD_COLORS.get(m, "black"), label=m, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
        for r in pts:
            ax.annotate(f"k={r['keep_ratio']:g}",
                        (r["mismatch_mean"], r["accuracy"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        all_x.extend(x.tolist()); all_y.extend(y.tolist())

    if full_cache_acc is not None:
        ax.axhline(full_cache_acc, color=METHOD_COLORS["full_cache"], ls="--",
                   lw=1.0, label=f"Full cache = {full_cache_acc:.3f}")

    # qualitative correlation
    if len(all_x) >= 3:
        from scipy.stats import pearsonr, spearmanr
        pr, _ = pearsonr(all_x, all_y)
        sp, _ = spearmanr(all_x, all_y)
        ax.set_title(f"Mismatch vs accuracy   (Pearson={pr:.2f}, Spearman={sp:.2f})")
    else:
        ax.set_title("Mismatch vs accuracy")

    ax.set_xlabel(f"Mismatch ({metric_label})")
    ax.set_ylabel(acc_label)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from short method name (in eval manifest) to mismatch JSON key.
EVAL_TO_MISMATCH = {
    "h2o": "h2o_vs_teacher",
    "student": "student_vs_teacher",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mismatch-json", required=True, type=Path)
    p.add_argument("--eval-manifest", required=True, type=Path)
    p.add_argument("--metric", default="jaccard@0.2",
                   help="Mismatch metric to use (cos_sim, spearman, jaccard@0.2, jaccard@0.4, ...).")
    p.add_argument("--mismatch-as", choices=["sim", "1-sim"], default="1-sim",
                   help="Plot raw similarity, or 1-similarity (= mismatch).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--acc-label", default="Accuracy",
                   help="Y-axis label for accuracy (e.g., 'DocVQA ANLS').")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mm = _read_json(args.mismatch_json)
    manifest = _read_json(args.eval_manifest)

    # Build (method, keep_ratio) rows
    rows: list[dict] = []
    methods_in_manifest = list(manifest.get("methods", {}).keys())
    for method in methods_in_manifest:
        mismatch_key = EVAL_TO_MISMATCH.get(method)
        if mismatch_key is None:
            print(f"[warn] unknown method '{method}' — skip", flush=True)
            continue
        per_sample = collect_mismatch(mm, mismatch_key, args.metric)
        # Average over all samples (across datasets)
        all_vals = [v for vs in per_sample.values() for v in vs]
        if not all_vals:
            print(f"[warn] no mismatch values for method={method} metric={args.metric}", flush=True)
            continue
        m_mean = float(np.mean(all_vals))
        m_std = float(np.std(all_vals))

        # Mismatch transform: 1 - sim if requested
        if args.mismatch_as == "1-sim":
            disp_mean, disp_std = 1.0 - m_mean, m_std
        else:
            disp_mean, disp_std = m_mean, m_std

        for kr_str in sorted(manifest["methods"][method].keys(), key=float):
            kr = float(kr_str)
            acc = collect_accuracy(manifest, method, kr_str)
            rows.append({
                "method": method,
                "keep_ratio": kr,
                "mismatch_mean": disp_mean,   # method-level (does not vary with keep_ratio for sim/spearman)
                "mismatch_std": disp_std,
                "accuracy": acc,
            })

    # If user picked jaccard@K, recompute mismatch *per keep_ratio* using the
    # matching jaccard metric (since K varies with keep_ratio).
    if args.metric.startswith("jaccard@"):
        # Build per-(method, keep_ratio) recomputed mismatch
        for r in rows:
            metric_for_kr = f"jaccard@{r['keep_ratio']:g}"
            mismatch_key = EVAL_TO_MISMATCH.get(r["method"])
            if mismatch_key is None:
                continue
            per_sample = collect_mismatch(mm, mismatch_key, metric_for_kr)
            vals = [v for vs in per_sample.values() for v in vs]
            if not vals:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals))
            r["mismatch_mean"] = (1.0 - m) if args.mismatch_as == "1-sim" else m
            r["mismatch_std"] = s

    full_cache_acc = collect_full_cache(manifest)
    metric_disp = f"1 - {args.metric}" if args.mismatch_as == "1-sim" else args.metric

    plot_mismatch_vs_keep_ratio(
        rows, args.out_dir / "mismatch_vs_keep_ratio.png", metric_disp)
    plot_accuracy_vs_keep_ratio(
        rows, full_cache_acc, args.out_dir / "accuracy_vs_keep_ratio.png", args.acc_label)
    plot_mismatch_vs_accuracy(
        rows, full_cache_acc, args.out_dir / "mismatch_vs_accuracy.png",
        metric_disp, args.acc_label)

    # Also dump tabular data for paper
    table_path = args.out_dir / "table.json"
    with table_path.open("w") as f:
        json.dump({
            "config": {
                "mismatch_json": str(args.mismatch_json),
                "eval_manifest": str(args.eval_manifest),
                "metric": args.metric,
                "mismatch_as": args.mismatch_as,
            },
            "full_cache_accuracy": full_cache_acc,
            "rows": rows,
        }, f, indent=2)
    print(f"[saved] {table_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
