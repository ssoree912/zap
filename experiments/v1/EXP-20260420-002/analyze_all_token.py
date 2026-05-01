#!/usr/bin/env python3
"""All-token vs image-only comparison for EXP-20260420-002.

Reads three sources:
  - Image-only: /workspace/zap/artifacts/EXP-20260420-002/h2o_prefill_and_hybrid/
      (H2O-image, Hybrid-image(H2O+Future_image))
  - Image-only Future-only: /workspace/zap/artifacts/EXP-20260418-001/v4_eval/
  - All-token: /workspace/zap/artifacts/EXP-20260420-002/all_token_eval/
      (H2O-all, Future-all, Hybrid-all)

Produces /workspace/zap/experiments/EXP-20260420-002/results_all_token.md
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

IMG_ROOT  = Path("/workspace/zap/artifacts/EXP-20260420-002/h2o_prefill_and_hybrid")
FUT_ROOT  = Path("/workspace/zap/artifacts/EXP-20260418-001/v4_eval")
ALL_ROOT  = Path("/workspace/zap/artifacts/EXP-20260420-002/all_token_eval")
OUT = Path("/workspace/zap/experiments/EXP-20260420-002")
KEEP_TAG = "k0p2"

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
        if v is not None:
            return float(v)
    em = d.get("exact_match_accuracy")
    return float(em) if em is not None else None


def fmt(x):
    return f"{x:.4f}" if x is not None else "  —   "


def fmt_d(x):
    if x is None: return "  —   "
    s = f"{x:+.4f}"
    return s


rows = []
n_all = 0
sums = {"h2o_img": 0, "hyb_img": 0, "fut_img": 0,
        "h2o_all": 0, "fut_all": 0, "hyb_all": 0}

for tag, pretty in DATASETS:
    m = {
        "h2o_img": load_metric(IMG_ROOT / tag / f"h2o_{KEEP_TAG}"),
        "hyb_img": load_metric(IMG_ROOT / tag / f"hybrid_h2o_future_a050_{KEEP_TAG}"),
        "fut_img": load_metric(FUT_ROOT / tag / f"future_{KEEP_TAG}"),
        "h2o_all": load_metric(ALL_ROOT / tag / f"h2o_all_{KEEP_TAG}"),
        "fut_all": load_metric(ALL_ROOT / tag / f"future_all_{KEEP_TAG}"),
        "hyb_all": load_metric(ALL_ROOT / tag / f"hybrid_h2o_future_all_a050_{KEEP_TAG}"),
    }
    if all(v is not None for v in m.values()):
        n_all += 1
        for k, v in m.items():
            sums[k] += v
    rows.append((tag, pretty, m))


lines = [
    "# EXP-20260420-002 — All-token vs Image-only (keep_ratio = 0.2)",
    "",
    "Comparing **image-only eviction** (text preserved) vs **all-token eviction** (20% of ALL tokens kept).",
    "Core question: *does pure Future probe beat H2O when scoring domain extends to text tokens?*",
    "",
    "**image-only** columns: evict only image tokens (keep_ratio of image_tokens).",
    "**all-token** columns: evict any token (total_keep_ratio of all_tokens).",
    "",
    "| Dataset                |  H2O-img | Fut-img  | Hyb-img  |  H2O-all | Fut-all  | Hyb-all  | Δ(Fut-all − H2O-all) |",
    "|------------------------|----------|----------|----------|----------|----------|----------|----------------------|",
]
w_fa_gt_ha = w_fa_lt_ha = ties_fa = 0
w_ha_gt_hi = w_ha_lt_hi = ties_ha = 0  # h2o_all vs h2o_img
for tag, pretty, m in rows:
    d_fa_ha = (m["fut_all"] - m["h2o_all"]) if (m["fut_all"] is not None and m["h2o_all"] is not None) else None
    d_ha_hi = (m["h2o_all"] - m["h2o_img"]) if (m["h2o_all"] is not None and m["h2o_img"] is not None) else None
    if d_fa_ha is not None:
        if d_fa_ha > 1e-6: w_fa_gt_ha += 1
        elif d_fa_ha < -1e-6: w_fa_lt_ha += 1
        else: ties_fa += 1
    if d_ha_hi is not None:
        if d_ha_hi > 1e-6: w_ha_gt_hi += 1
        elif d_ha_hi < -1e-6: w_ha_lt_hi += 1
        else: ties_ha += 1

    lines.append(
        f"| {pretty:<22} | {fmt(m['h2o_img'])} | {fmt(m['fut_img'])} | {fmt(m['hyb_img'])} | "
        f"{fmt(m['h2o_all'])} | {fmt(m['fut_all'])} | {fmt(m['hyb_all'])} | {fmt_d(d_fa_ha):>18} |"
    )

if n_all > 0:
    lines.append("|" + "-" * 24 + "|" + "|".join(["-" * 10] * 6) + "|----------------------|")
    avgs = {k: sums[k] / n_all for k in sums}
    lines.append(
        f"| **Average ({n_all:d} ds)**       | "
        f"{avgs['h2o_img']:.4f} | {avgs['fut_img']:.4f} | {avgs['hyb_img']:.4f} | "
        f"{avgs['h2o_all']:.4f} | {avgs['fut_all']:.4f} | {avgs['hyb_all']:.4f} | "
        f"{avgs['fut_all'] - avgs['h2o_all']:+.4f} |"
    )

lines += [
    "",
    "## Verdict counts (all-token)",
    "",
    f"- Future-all > H2O-all: **{w_fa_gt_ha}** / ties {ties_fa} / losses {w_fa_lt_ha}",
    f"- H2O-all > H2O-img: **{w_ha_gt_hi}** / ties {ties_ha} / losses {w_ha_lt_hi}",
    "",
    "## Notes",
    "",
    "- Future probe used for all-token = `future_probe_v5_all_token_10ep` "
    "(trained on 750 samples across textvqa+scienceqa+nlvr2 with all-position targets).",
    "- Image-only Hybrid uses `future_probe_v4_last8_20ep_bcast31` (image-only trained, last-8 + broadcast).",
    "- Image-only Future uses `future_probe_v4_last8_20ep_bcast31` (EXP-20260418-001).",
]
out_path = OUT / "results_all_token.md"
out_path.write_text("\n".join(lines) + "\n")
print(f"wrote: {out_path}")
print()
print("\n".join(lines))
