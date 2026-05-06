#!/usr/bin/env python3
"""Merge eval JSONs produced by run_eval_with_press.py with disjoint sample shards.

Usage:
    python merge_eval_shards.py \
        --inputs eval_coco_100_p1.json eval_coco_100_p2.json eval_coco_100_p3.json \
        --output eval_coco_100.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    blobs = [json.loads(f.read_text()) for f in args.inputs]
    cfg = blobs[0]["config"].copy()

    rows: list[dict] = []
    for b in blobs:
        rows.extend(b["rows"])

    # De-dup on (dataset, sample_id, method, keep_ratio); keep last
    dedup: dict[tuple, dict] = {}
    for r in rows:
        key = (r["dataset"], r["sample_id"], r["method"], r["keep_ratio"])
        dedup[key] = r
    rows = list(dedup.values())

    agg: dict[tuple, list[float]] = {}
    for r in rows:
        if r["score"] is None:
            continue
        k = (r["dataset"], r["method"], r["keep_ratio"])
        agg.setdefault(k, []).append(r["score"])
    summary = []
    for (ds, m, kr), vals in agg.items():
        summary.append({
            "dataset": ds, "method": m, "keep_ratio": kr,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        })
    summary.sort(key=lambda x: (x["dataset"], x["method"], x["keep_ratio"]))

    payload = {"config": cfg, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {args.output}  rows={len(rows)}  configs={len(summary)}")
    print()
    print("=== summary ===")
    for s in summary:
        print(f"  {s['dataset']:20s}  {s['method']:10s}  k={s['keep_ratio']:.2f}  "
              f"score={s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
