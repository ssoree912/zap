#!/usr/bin/env python3
"""Compute prefill vs future attention mismatch metrics per sample.

Reads .pt files from attn_dump/{dataset}/ (output of collect_attn_dump.py).
For each sample computes (averaged over layers 0-31):
  - Spearman rank correlation between prefill score and future score
  - Top-K overlap at keep_ratio r (default 0.2, 0.5)

Writes candidates.csv sorted by ascending Spearman (most mismatch first).

Usage:
  python compute_mismatch.py --dataset DocVQA
  python compute_mismatch.py --dataset mmvet --keep-ratios 0.2 0.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from scipy.stats import spearmanr
from tqdm import tqdm

DUMP_ROOT = "/workspace/zap/artifacts/EXP-20260426-001-figure/attn_dump"
OUT_ROOT  = "/workspace/zap/artifacts/EXP-20260426-001-figure"


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    """Fraction of top-k indices shared between two score vectors."""
    if k <= 0:
        return 0.0
    top_a = set(torch.topk(a, k=min(k, len(a))).indices.tolist())
    top_b = set(torch.topk(b, k=min(k, len(b))).indices.tolist())
    return len(top_a & top_b) / max(k, 1)


def mismatch_token_counts(a: torch.Tensor, b: torch.Tensor, k: int) -> tuple[int, int]:
    """
    Returns (n_hidden_gems, n_false_salient):
      hidden_gems:    low prefill, high future  (top-k future but NOT top-k prefill)
      false_salient:  high prefill, low future  (top-k prefill but NOT top-k future)
    """
    if k <= 0:
        return 0, 0
    top_a = set(torch.topk(a, k=min(k, len(a))).indices.tolist())
    top_b = set(torch.topk(b, k=min(k, len(b))).indices.tolist())
    hidden_gems   = len(top_b - top_a)   # in future top-k but not prefill top-k
    false_salient = len(top_a - top_b)   # in prefill top-k but not future top-k
    return hidden_gems, false_salient


def process_file(pt_path: Path, keep_ratios: list[float]) -> dict | None:
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[WARN] cannot load {pt_path}: {e}")
        return None

    prefill = data["prefill"].float()  # [L, N_I]
    future  = data["future"].float()   # [L, N_I]
    L, N_I  = prefill.shape

    # Layer-averaged scores [N_I]
    pref_avg = prefill.mean(dim=0)
    fut_avg  = future.mean(dim=0)

    # Spearman (layer-averaged)
    rho_avg, _ = spearmanr(pref_avg.numpy(), fut_avg.numpy())

    # Per-layer Spearman then averaged
    rhos = []
    for l in range(L):
        r, _ = spearmanr(prefill[l].numpy(), future[l].numpy())
        rhos.append(float(r))
    rho_perlayer_mean = sum(rhos) / len(rhos) if rhos else float("nan")

    row: dict = {
        "sample_id":          data.get("sample_id", pt_path.stem),
        "image_path":         data.get("image_path", ""),
        "question":           data.get("question", "")[:120],
        "answer":             data.get("answer", "")[:80],
        "N_I":                N_I,
        "L":                  L,
        "spearman_avg_layer": round(float(rho_avg), 4),
        "spearman_perlayer_mean": round(rho_perlayer_mean, 4),
    }

    for r in keep_ratios:
        k = max(1, int(N_I * r))
        overlap = topk_overlap(pref_avg, fut_avg, k)
        gems, false_sal = mismatch_token_counts(pref_avg, fut_avg, k)
        tag = f"r{int(r*100):02d}"
        row[f"overlap_{tag}"]       = round(overlap, 4)
        row[f"hidden_gems_{tag}"]   = gems
        row[f"false_sal_{tag}"]     = false_sal

    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--keep-ratios", type=float, nargs="+", default=[0.2, 0.5])
    p.add_argument("--dump-root", default=DUMP_ROOT)
    p.add_argument("--out-root", default=OUT_ROOT)
    args = p.parse_args()

    dump_dir = Path(args.dump_root) / args.dataset
    pt_files = sorted(dump_dir.glob("*.pt"))
    if not pt_files:
        print(f"[ERROR] No .pt files in {dump_dir}")
        return

    rows = []
    for pt in tqdm(pt_files, desc=f"mismatch {args.dataset}"):
        row = process_file(pt, args.keep_ratios)
        if row:
            rows.append(row)

    # sort by ascending spearman (most mismatch first)
    rows.sort(key=lambda x: x["spearman_avg_layer"])

    out_path = Path(args.out_root) / f"candidates_{args.dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # print summary
    if rows:
        spears = [r["spearman_avg_layer"] for r in rows]
        print(f"\n[{args.dataset}]  n={len(rows)}")
        print(f"  Spearman (avg layer):  mean={sum(spears)/len(spears):.3f}  "
              f"min={min(spears):.3f}  max={max(spears):.3f}")
        for r_val in args.keep_ratios:
            tag = f"r{int(r_val*100):02d}"
            overlaps = [row[f"overlap_{tag}"] for row in rows]
            print(f"  Overlap@{int(r_val*100)}%:           mean={sum(overlaps)/len(overlaps):.3f}  "
                  f"min={min(overlaps):.3f}  max={max(overlaps):.3f}")
        print(f"\nTop-10 mismatch samples (lowest Spearman):")
        for row in rows[:10]:
            r20_key = f"overlap_r{int(args.keep_ratios[0]*100):02d}"
            print(f"  {row['sample_id']:20s}  rho={row['spearman_avg_layer']:.3f}  "
                  f"overlap@{int(args.keep_ratios[0]*100)}%={row.get(r20_key, '?'):.3f}  "
                  f"Q: {row['question'][:60]}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
