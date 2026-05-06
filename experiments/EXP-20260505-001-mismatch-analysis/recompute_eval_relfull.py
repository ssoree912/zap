#!/usr/bin/env python3
"""Recompute eval scores using full-cache prediction as the reference.

Reads an eval JSON from run_eval_with_press.py (which already contains full_cache
predictions at keep_ratio=1.0) and rescores every (method, keep_ratio) row
with ROUGE-L against the full-cache pred for the same sample.

This removes the length-bias of comparing against short GT captions and
instead measures "how well does the press preserve the original model
output" — the quantity KV eviction methods actually target.

Output JSON has the same schema as the input (config / summary / rows)
so plot_clean.py can be reused as-is.

Usage:
    python recompute_eval_relfull.py \
        --input eval_coco_100.json \
        --output eval_coco_100_relfull.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


_PUNC_RE = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    return _PUNC_RE.sub("", s.lower()).strip()


def _lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev = tmp
    return dp[m]


def rouge_l(pred: str, ref: str) -> float:
    p_toks = _normalize(pred).split()
    r_toks = _normalize(ref).split()
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_length(p_toks, r_toks)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_toks)
    rec = lcs / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    blob = json.loads(args.input.read_text())
    rows = blob["rows"]

    # Build (dataset, sample_id) → full_cache pred
    fc_pred: dict[tuple[str, str], str] = {}
    for r in rows:
        if r["method"] == "full_cache":
            fc_pred[(r["dataset"], str(r["sample_id"]))] = r.get("pred") or ""

    new_rows: list[dict] = []
    skipped_no_fc = 0
    for r in rows:
        key = (r["dataset"], str(r["sample_id"]))
        ref = fc_pred.get(key, "")
        new_r = dict(r)
        if r["method"] == "full_cache":
            # full vs full = 1.0 by definition (kept for completeness; aggregator excludes it via filter below)
            new_r["score"] = 1.0
            new_r["ref"] = "(self)"
        else:
            if not ref:
                new_r["score"] = None
                skipped_no_fc += 1
            else:
                new_r["score"] = float(rouge_l(r.get("pred") or "", ref))
            new_r["ref"] = ref
        new_rows.append(new_r)

    # Aggregate per (dataset, method, keep_ratio); keep full_cache row at 1.0 for plotter compatibility
    agg: dict[tuple, list[float]] = defaultdict(list)
    for r in new_rows:
        if r["score"] is None:
            continue
        agg[(r["dataset"], r["method"], r["keep_ratio"])].append(r["score"])
    summary = []
    for (ds, m, kr), vals in agg.items():
        summary.append({
            "dataset": ds, "method": m, "keep_ratio": kr,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        })
    summary.sort(key=lambda x: (x["dataset"], x["method"], x["keep_ratio"]))

    new_cfg = dict(blob.get("config", {}))
    new_cfg["score_reference"] = "full_cache_pred"
    new_cfg["score_metric"] = "rouge_l"
    payload = {"config": new_cfg, "summary": summary, "rows": new_rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {args.output}  rows={len(new_rows)}  skipped_no_fc={skipped_no_fc}")
    print()
    print("=== summary (ROUGE-L vs full-cache pred) ===")
    for s in summary:
        print(f"  {s['dataset']:20s}  {s['method']:10s}  k={s['keep_ratio']:.2f}  "
              f"score={s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
