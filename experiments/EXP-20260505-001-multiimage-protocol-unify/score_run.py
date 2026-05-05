#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coerce sample_id to int + invoke look-m evaluate.py for every keep/dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

LOOKM_PY = "/workspace/look-m/.conda/look-m/bin/python"
LOOKM_EVAL = "/workspace/look-m/evaluate.py"
DATA_ROOT = "/workspace/zap/data/MileBench"


def coerce_sample_ids(pred_path: Path) -> int:
    preds = json.loads(pred_path.read_text())
    n_changed = 0
    for d in preds:
        sid = d.get("sample_id")
        try:
            new_sid = int(sid)
            if new_sid != sid:
                n_changed += 1
            d["sample_id"] = new_sid
        except Exception:
            pass
    pred_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    return n_changed


def parse_score(stdout: str) -> dict:
    # Look-M evaluate.py prints e.g.:
    # keep050:  ALFRED:  {'Rouge-L f': 0.6075..., ...}
    m = re.search(r"\{[^}]*'Rouge-L f'\s*:\s*([0-9.]+)", stdout)
    if m:
        return {"metric": "Rouge-L f", "score": float(m.group(1))}
    m = re.search(r"'Accuracy'\s*:\s*([0-9.]+)", stdout)
    if m:
        return {"metric": "Accuracy", "score": float(m.group(1))}
    return {"metric": None, "score": None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_root", type=Path)
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args()

    if not args.run_root.is_dir():
        raise SystemExit(f"not a dir: {args.run_root}")

    rows = []
    for keep_dir in sorted(args.run_root.iterdir()):
        if not keep_dir.is_dir() or not keep_dir.name.startswith("keep"):
            continue
        keep_tag = keep_dir.name
        for ds_dir in sorted(keep_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            if args.datasets and ds not in args.datasets:
                continue
            pred = ds_dir / "pred.json"
            if not pred.exists():
                continue
            n_coerced = coerce_sample_ids(pred)
            print(f"[{keep_tag}/{ds}] sample_id coerced: {n_coerced}")
            res = subprocess.run(
                [LOOKM_PY, LOOKM_EVAL,
                 "--data-dir", DATA_ROOT,
                 "--dataset", ds,
                 "--result-dir", str(keep_dir)],
                capture_output=True, text=True,
            )
            tail = res.stdout.strip().splitlines()
            score_line = tail[-1] if tail else ""
            parsed = parse_score(res.stdout)
            score_pct = (parsed["score"] or 0.0) * 100
            print(f"  → {score_line}")
            rows.append({
                "keep_tag": keep_tag,
                "dataset": ds,
                "metric": parsed["metric"],
                "score_pct": round(score_pct, 2),
            })

    out_csv = args.run_root / "scores_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["keep_tag", "dataset", "metric", "score_pct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
