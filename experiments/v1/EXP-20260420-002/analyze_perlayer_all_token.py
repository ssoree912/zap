#!/usr/bin/env python3
"""Compare all-token per-layer Hybrid variants against flat α=0.5 hybrid + Future-all.

Columns:
  H2O-all, Future-all, Hybrid-all(flat), Hybrid-all(L24-31), Hybrid-all(L0-23)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

ALL_ROOT  = Path("/workspace/zap/artifacts/EXP-20260420-002/all_token_eval")
PL_ROOT   = Path("/workspace/zap/artifacts/EXP-20260420-002/perlayer_all_token_eval")
OUT       = Path("/workspace/zap/experiments/EXP-20260420-002")
KEEP_TAG  = "k0p2"

DATASETS = [
    ("spot_the_diff",           "Spot-the-Diff"),
    ("clevr_change",            "CLEVR-Change"),
    ("webqa",                   "WebQA"),
    ("alfred",                  "ALFRED"),
    ("iedit",                   "IEdit"),
    ("mmcoqa",                  "MMCoQA"),
    ("actionlocalization",      "ActionLocalization"),
    ("actionprediction",        "ActionPrediction"),
    ("actionsequence",          "ActionSequence"),
    ("characterorder",          "CharacterOrder"),
    ("counterfactualinference", "CounterfactualInference"),
    ("docvqa",                  "DocVQA"),
    ("egocentricnavigation",    "EgocentricNavigation"),
    ("gpr1200",                 "GPR1200"),
    ("imageneedleinahaystack",  "ImageNeedleInAHaystack"),
    ("movingattribute",         "MovingAttribute"),
    ("movingdirection",         "MovingDirection"),
    ("multimodalqa",            "MultiModalQA"),
    ("nuscenes",                "nuscenes"),
    ("objectexistence",         "ObjectExistence"),
    ("objectinteraction",       "ObjectInteraction"),
    ("objectshuffle",           "ObjectShuffle"),
    ("ocr_vqa",                 "OCR-VQA"),
    ("scenetransition",         "SceneTransition"),
    ("slidevqa",                "SlideVQA"),
    ("statechange",             "StateChange"),
    ("textneedleinahaystack",   "TextNeedleInAHaystack"),
    ("tqa",                     "TQA"),
    ("wikivqa",                 "WikiVQA"),
]


def load_metric(path: Path) -> Optional[float]:
    f = path / "metrics.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    le = d.get("look_eval") or {}
    for key in ("ROUGE-L", "Accuracy"):
        v = le.get(key)
        if v is not None: return float(v)
    em = d.get("exact_match_accuracy")
    return float(em) if em is not None else None


def fmt(x): return f"{x:.4f}" if x is not None else "  —   "
def fmt_d(x): return f"{x:+.4f}" if x is not None else "  —   "


rows = []
n_all = 0
sums = {"h2o": 0, "fut": 0, "hyb_flat": 0, "hyb_late": 0, "hyb_early": 0}
for tag, pretty in DATASETS:
    m = {
        "h2o":       load_metric(ALL_ROOT / tag / f"h2o_all_{KEEP_TAG}"),
        "fut":       load_metric(ALL_ROOT / tag / f"future_all_{KEEP_TAG}"),
        "hyb_flat":  load_metric(ALL_ROOT / tag / f"hybrid_h2o_future_all_a050_{KEEP_TAG}"),
        "hyb_late":  load_metric(PL_ROOT  / tag / f"hybrid_all_L24-31_a050_{KEEP_TAG}"),
        "hyb_early": load_metric(PL_ROOT  / tag / f"hybrid_all_L0-23_a050_{KEEP_TAG}"),
    }
    rows.append((tag, pretty, m))
    if all(v is not None for v in m.values()):
        n_all += 1
        for k, v in m.items(): sums[k] += v


lines = [
    "# EXP-20260420-002 — All-token per-layer Hybrid (α=0.5, keep_ratio=0.2)",
    "",
    "Comparing flat vs layer-gated Hybrid(H2O + Future-all):",
    "- **Hyb-flat**: α=0.5 at every one of 32 layers.",
    "- **Hyb-late**: α=0.5 at layers 24–31 only (H2O-only at 0–23).",
    "- **Hyb-early**: α=0.5 at layers 0–23 only (H2O-only at 24–31).",
    "",
    "| Dataset                |  H2O   | Future | Hyb-flat | Hyb-late | Hyb-early | Δ(late − flat) | Δ(early − flat) |",
    "|------------------------|--------|--------|----------|----------|-----------|----------------|-----------------|",
]

w_late = w_early = 0
best_counts = {"h2o": 0, "fut": 0, "hyb_flat": 0, "hyb_late": 0, "hyb_early": 0}

for tag, pretty, m in rows:
    d_late  = (m["hyb_late"]  - m["hyb_flat"]) if (m["hyb_late"]  is not None and m["hyb_flat"] is not None) else None
    d_early = (m["hyb_early"] - m["hyb_flat"]) if (m["hyb_early"] is not None and m["hyb_flat"] is not None) else None
    if d_late is not None and d_late > 1e-6:  w_late += 1
    if d_early is not None and d_early > 1e-6: w_early += 1
    # best-method per dataset
    valid = {k: v for k, v in m.items() if v is not None}
    if valid:
        best = max(valid.items(), key=lambda kv: kv[1])[0]
        best_counts[best] = best_counts.get(best, 0) + 1
    lines.append(
        f"| {pretty:<22} | {fmt(m['h2o'])} | {fmt(m['fut'])} | {fmt(m['hyb_flat'])} | "
        f"{fmt(m['hyb_late'])} | {fmt(m['hyb_early'])} | {fmt_d(d_late):>13} | {fmt_d(d_early):>14} |"
    )

if n_all > 0:
    lines.append("|" + "-" * 24 + "|" + "|".join(["-" * 8] * 5) + "|----------------|-----------------|")
    a = {k: sums[k] / n_all for k in sums}
    lines.append(
        f"| **Average ({n_all:d} ds)**       | {a['h2o']:.4f} | {a['fut']:.4f} | {a['hyb_flat']:.4f} | "
        f"{a['hyb_late']:.4f} | {a['hyb_early']:.4f} | {a['hyb_late']-a['hyb_flat']:+.4f}     | "
        f"{a['hyb_early']-a['hyb_flat']:+.4f}      |"
    )

lines += [
    "",
    "## Best-per-dataset counts",
    "",
    f"- H2O wins:       {best_counts.get('h2o', 0)} / {n_all}",
    f"- Future wins:    {best_counts.get('fut', 0)} / {n_all}",
    f"- Hyb-flat wins:  {best_counts.get('hyb_flat', 0)} / {n_all}",
    f"- Hyb-late wins:  {best_counts.get('hyb_late', 0)} / {n_all}",
    f"- Hyb-early wins: {best_counts.get('hyb_early', 0)} / {n_all}",
    "",
    f"## Per-layer vs flat deltas",
    "",
    f"- Hyb-late > Hyb-flat: {w_late}/{n_all}",
    f"- Hyb-early > Hyb-flat: {w_early}/{n_all}",
]
out_path = OUT / "results_perlayer_all_token.md"
out_path.write_text("\n".join(lines) + "\n")
print(f"wrote: {out_path}")
print()
print("\n".join(lines))
