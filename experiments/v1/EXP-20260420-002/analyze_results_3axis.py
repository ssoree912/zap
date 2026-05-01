#!/usr/bin/env python3
"""3-axis main table for EXP-20260420-002.

Axes:
  H2O-prefill         (observable prefill saliency)
  Future              (unobservable-at-prefill decode utility)   — reuses EXP-20260418-001
  Hybrid(H2O+Future)  (α=0.5, all-layer blend)

Datasets: whichever have metrics.json available (supports partial sweep).

Produces: /workspace/zap/experiments/EXP-20260420-002/results_3axis.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

H2O_ROOT = Path("/workspace/zap/artifacts/EXP-20260420-002/h2o_prefill_and_hybrid")
FUTURE_ROOT = Path("/workspace/zap/artifacts/EXP-20260418-001/v4_eval")
OUT = Path("/workspace/zap/experiments/EXP-20260420-002")
KEEP_TAG = "k0p2"

# (tag, pretty)
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


def load_metric(path: Path) -> tuple[Optional[float], Optional[str]]:
    """Return (score, metric_name) — tries ROUGE-L then Accuracy."""
    f = path / "metrics.json"
    if not f.exists():
        return None, None
    try:
        d = json.loads(f.read_text())
    except Exception:
        return None, None
    le = d.get("look_eval") or {}
    for key in ("ROUGE-L", "Accuracy"):
        val = le.get(key)
        if val is not None:
            return float(val), key
    # fallback
    em = d.get("exact_match_accuracy")
    return (float(em) if em is not None else None), ("EM" if em is not None else None)


def h2o_path(tag: str) -> Path:
    return H2O_ROOT / tag / f"h2o_{KEEP_TAG}"


def hybrid_path(tag: str) -> Path:
    return H2O_ROOT / tag / f"hybrid_h2o_future_a050_{KEEP_TAG}"


def future_path(tag: str) -> Path:
    return FUTURE_ROOT / tag / f"future_{KEEP_TAG}"


def fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "  —   "


def main():
    rows: list[dict] = []
    for tag, pretty in DATASETS:
        h2o, m1 = load_metric(h2o_path(tag))
        fut, m2 = load_metric(future_path(tag))
        hyb, m3 = load_metric(hybrid_path(tag))
        metric = next((m for m in (m1, m2, m3) if m), "?")
        rows.append({
            "tag": tag, "pretty": pretty, "metric": metric,
            "h2o": h2o, "future": fut, "hybrid": hyb,
        })

    # Build table
    lines = [
        "# EXP-20260420-002 — 3-axis Main Table (keep_ratio = 0.2)",
        "",
        "**H2O** = prefill accumulated attention (observable at prefill).",
        "**Future** = MLP probe, decode→image attention prediction (unobservable at prefill).",
        "**Hybrid** = α · softmax(H2O) + (1−α) · softmax(Future),  α=0.5 at every layer.",
        "",
        "Metric automatically picked per dataset: `ROUGE-L` (generative) or `Accuracy` (choice).",
        "",
        "| Dataset                | metric   |   H2O   | Future  | Hybrid  | Δ(Hyb−H2O) | Δ(Hyb−Fut) |",
        "|------------------------|----------|---------|---------|---------|------------|------------|",
    ]
    n_completed = 0
    wins_hyb_vs_h2o = ties_hyb_vs_h2o = losses_hyb_vs_h2o = 0
    wins_hyb_vs_fut = ties_hyb_vs_fut = losses_hyb_vs_fut = 0
    wins_fut_vs_h2o = ties_fut_vs_h2o = losses_fut_vs_h2o = 0

    sum_h2o = sum_fut = sum_hyb = 0.0
    n_all_three = 0

    for r in rows:
        tag = r["pretty"]
        h2o, fut, hyb = r["h2o"], r["future"], r["hybrid"]
        d_hh = hyb - h2o if (hyb is not None and h2o is not None) else None
        d_hf = hyb - fut if (hyb is not None and fut is not None) else None
        if hyb is not None and h2o is not None:
            if d_hh > 1e-6: wins_hyb_vs_h2o += 1
            elif d_hh < -1e-6: losses_hyb_vs_h2o += 1
            else: ties_hyb_vs_h2o += 1
        if hyb is not None and fut is not None:
            if d_hf > 1e-6: wins_hyb_vs_fut += 1
            elif d_hf < -1e-6: losses_hyb_vs_fut += 1
            else: ties_hyb_vs_fut += 1
        if fut is not None and h2o is not None:
            d_fh = fut - h2o
            if d_fh > 1e-6: wins_fut_vs_h2o += 1
            elif d_fh < -1e-6: losses_fut_vs_h2o += 1
            else: ties_fut_vs_h2o += 1
        if h2o is not None and fut is not None and hyb is not None:
            n_all_three += 1
            sum_h2o += h2o; sum_fut += fut; sum_hyb += hyb
        if h2o is not None or fut is not None or hyb is not None:
            n_completed += 1
        lines.append(
            f"| {tag:<22} | {r['metric']:<8} | {fmt(h2o)} | {fmt(fut)} | {fmt(hyb)} | "
            f"{('+' if d_hh and d_hh > 0 else '') + f'{d_hh:.4f}' if d_hh is not None else '  —  '} | "
            f"{('+' if d_hf and d_hf > 0 else '') + f'{d_hf:.4f}' if d_hf is not None else '  —  '} |"
        )

    # Averages over the subset where all 3 are present
    if n_all_three:
        avg_h = sum_h2o / n_all_three
        avg_f = sum_fut / n_all_three
        avg_y = sum_hyb / n_all_three
        lines.append("|" + "-" * 24 + "|" + "-" * 10 + "|---------|---------|---------|------------|------------|")
        lines.append(
            f"| **Average ({n_all_three:d} ds)**{'':<6}  |    —     | {avg_h:.4f} | {avg_f:.4f} | {avg_y:.4f} | "
            f"{avg_y - avg_h:+.4f} | {avg_y - avg_f:+.4f} |"
        )

    lines += [
        "",
        "## Verdict counts",
        "",
        f"- Hybrid > H2O: **{wins_hyb_vs_h2o}** / ties {ties_hyb_vs_h2o} / losses {losses_hyb_vs_h2o}",
        f"- Hybrid > Future: **{wins_hyb_vs_fut}** / ties {ties_hyb_vs_fut} / losses {losses_hyb_vs_fut}",
        f"- Future > H2O: **{wins_fut_vs_h2o}** / ties {ties_fut_vs_h2o} / losses {losses_fut_vs_h2o}",
        "",
        f"- Completed datasets (at least one method): **{n_completed}** / {len(DATASETS)}",
        f"- Datasets with all three methods scored: **{n_all_three}** / {len(DATASETS)}",
        "",
        "## Notes",
        "",
        "- Hybrid α=0.5 is blended at every one of the 32 layers (no per-layer gating).",
        "- Future probe weights are trained on layers 24-31; layers 0-23 use broadcast-31 "
        "(broadcast noise acknowledged, not dominant per EXP-20260420-001).",
        "- Missing cells (`—`) indicate the sweep is still in progress or the run failed.",
    ]

    out_path = OUT / "results_3axis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote: {out_path}")
    print()
    print("\n".join(lines[:15 + n_completed]))


if __name__ == "__main__":
    main()
