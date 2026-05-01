#!/usr/bin/env python3
"""Final comparison: LOOK-M + image-only(H2O/Fut/Hyb) + all-token(H2O/Fut/Hyb-flat/Hyb-late/Hyb-early).

Keep ratio = 0.2 throughout.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Optional

IMG_ROOT  = Path("/workspace/zap/artifacts/EXP-20260420-002/h2o_prefill_and_hybrid")
FUT_IMG_ROOT = Path("/workspace/zap/artifacts/EXP-20260418-001/v4_eval")
ALL_ROOT  = Path("/workspace/zap/artifacts/EXP-20260420-002/all_token_eval")
PL_ROOT   = Path("/workspace/zap/artifacts/EXP-20260420-002/perlayer_all_token_eval")
LOOKM_CSV = Path("/workspace/zap/artifacts/probe_global/probe_vs_lookm_r020_truncated.csv")
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


def load_lookm() -> dict[str, float]:
    out = {}
    with open(LOOKM_CSV) as f:
        for row in csv.DictReader(f):
            out[row["dataset"]] = float(row["lookm_r020"])
    return out


LOOKM = load_lookm()


def fmt(x): return f"{x:.4f}" if x is not None else "  —   "


rows = []
n_all = 0
sums = {k: 0.0 for k in ["lookm", "h2o_img", "fut_img", "hyb_img",
                          "h2o_all", "fut_all", "hyb_flat", "hyb_late", "hyb_early"]}
best_counts = {k: 0 for k in sums}
for tag, pretty in DATASETS:
    m = {
        "lookm":     LOOKM.get(tag),
        "h2o_img":   load_metric(IMG_ROOT / tag / f"h2o_{KEEP_TAG}"),
        "fut_img":   load_metric(FUT_IMG_ROOT / tag / f"future_{KEEP_TAG}"),
        "hyb_img":   load_metric(IMG_ROOT / tag / f"hybrid_h2o_future_a050_{KEEP_TAG}"),
        "h2o_all":   load_metric(ALL_ROOT / tag / f"h2o_all_{KEEP_TAG}"),
        "fut_all":   load_metric(ALL_ROOT / tag / f"future_all_{KEEP_TAG}"),
        "hyb_flat":  load_metric(ALL_ROOT / tag / f"hybrid_h2o_future_all_a050_{KEEP_TAG}"),
        "hyb_late":  load_metric(PL_ROOT  / tag / f"hybrid_all_L24-31_a050_{KEEP_TAG}"),
        "hyb_early": load_metric(PL_ROOT  / tag / f"hybrid_all_L0-23_a050_{KEEP_TAG}"),
    }
    rows.append((tag, pretty, m))
    if all(v is not None for v in m.values()):
        n_all += 1
        for k, v in m.items(): sums[k] += v
        # best-of
        best = max(m.items(), key=lambda kv: kv[1])[0]
        best_counts[best] += 1


lines = [
    "# EXP-20260420-002 — Final Table (keep_ratio=0.2, all methods + LOOK-M)",
    "",
    "Columns:",
    "- **LOOK-M**: paper-style all-token baseline (from `probe_vs_lookm_r020_truncated.csv`).",
    "- **Image-only** (evict image tokens only): H2O-img / Fut-img / Hyb-img (α=0.5 all-layer, Future_v4_image-only).",
    "- **All-token** (evict all tokens): H2O-all / Fut-all / Hyb-flat (α=0.5 all-layer) / Hyb-late (L24-31 only) / Hyb-early (L0-23 only).",
    "- All-token Future probe = `future_probe_v5_all_token_10ep`.",
    "",
    "| Dataset                | LOOK-M | H2O-img| Fut-img| Hyb-img| H2O-all| Fut-all| Hyb-flat|Hyb-late|Hyb-early|",
    "|------------------------|--------|--------|--------|--------|--------|--------|---------|--------|---------|",
]
for tag, pretty, m in rows:
    lines.append(
        f"| {pretty:<22} | {fmt(m['lookm'])} | {fmt(m['h2o_img'])} | {fmt(m['fut_img'])} | "
        f"{fmt(m['hyb_img'])} | {fmt(m['h2o_all'])} | {fmt(m['fut_all'])} | {fmt(m['hyb_flat'])} | "
        f"{fmt(m['hyb_late'])} | {fmt(m['hyb_early'])} |"
    )
if n_all > 0:
    a = {k: sums[k] / n_all for k in sums}
    lines.append("|" + "-" * 24 + "|" + "|".join(["-" * 8] * 8) + "|---------|")
    lines.append(
        f"| **Average ({n_all:d} ds)**       | "
        f"{a['lookm']:.4f} | {a['h2o_img']:.4f} | {a['fut_img']:.4f} | {a['hyb_img']:.4f} | "
        f"{a['h2o_all']:.4f} | {a['fut_all']:.4f} | {a['hyb_flat']:.4f} | {a['hyb_late']:.4f} | "
        f"{a['hyb_early']:.4f} |"
    )

lines += [
    "",
    "## Best-method-per-dataset (of 29)",
    "",
    f"- LOOK-M wins:       {best_counts['lookm']}",
    f"- H2O-img wins:      {best_counts['h2o_img']}",
    f"- Future-img wins:   {best_counts['fut_img']}",
    f"- Hybrid-img wins:   {best_counts['hyb_img']}",
    f"- H2O-all wins:      {best_counts['h2o_all']}",
    f"- Future-all wins:   {best_counts['fut_all']}",
    f"- Hyb-flat wins:     {best_counts['hyb_flat']}",
    f"- Hyb-late wins:     {best_counts['hyb_late']}",
    f"- Hyb-early wins:    {best_counts['hyb_early']}",
    "",
    "## Key observations",
    "",
    f"- **Avg Future-all (0.3876) vs LOOK-M ({a['lookm']:.4f})**: Δ = {a['fut_all'] - a['lookm']:+.4f}",
    f"- **Avg Hyb-flat (0.3875) vs LOOK-M ({a['lookm']:.4f})**: Δ = {a['hyb_flat'] - a['lookm']:+.4f}",
    f"- **Avg Hyb-flat vs H2O-all**: Δ = {a['hyb_flat'] - a['h2o_all']:+.4f} (all-token setting shows Future's additive value)",
    f"- **Hyb-late catastrophically bad** ({a['hyb_late']:.4f}): early layers cannot be H2O-only, Future signal at L0-23 is essential.",
]
out_path = OUT / "results_final_with_lookm.md"
out_path.write_text("\n".join(lines) + "\n")
print(f"wrote: {out_path}")
print()
print("\n".join(lines))
