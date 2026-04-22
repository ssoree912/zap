#!/usr/bin/env python3
"""Aggregate EXP-20260420-001 per-layer Hybrid sweep results and compare vs global α sweep.

Produces:
  - results_table.md     : headline comparison table (6 ds × 2 k × 9 methods)
  - per_layer_vs_global.md: delta matrix (per-layer best α vs global best α per cell)
  - summary.md            : H1/H2/H3 success-criteria verdict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

ART = Path("/workspace/zap/artifacts")
V4_EVAL = ART / "EXP-20260418-001" / "v4_eval"
V4_ALPHA = ART / "EXP-20260418-001" / "v4_alpha_sweep"
V5_PERL = ART / "EXP-20260420-001" / "v5_perlayer_sweep"
OUT = Path("/workspace/zap/experiments/EXP-20260420-001")

DATASETS = ["spot_the_diff", "clevr_change", "webqa", "alfred", "iedit", "mmcoqa"]
KEEPS = [0.2, 0.1]
GLOBAL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0.0=future_k*, 0.5=hybrid_k*, 1.0=pv_k*
PERL_ALPHAS = [0.0, 0.25, 0.5, 0.75]
METRIC_KEY = {"webqa": "Accuracy"}          # others default to ROUGE-L


def ktag(k: float) -> str:
    return "k" + str(k).replace(".", "p")


def atag(a: float) -> str:
    return f"a{int(round(a * 100)):03d}"


def read_metric(path: Path, ds: str) -> Optional[float]:
    f = path / "metrics.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text())
    except Exception:
        return None
    key = METRIC_KEY.get(ds, "ROUGE-L")
    val = d.get("look_eval", {}).get(key)
    if val is None:
        return None
    return float(val)


def global_path(ds: str, alpha: float, k: float) -> Path:
    kt = ktag(k)
    if alpha == 1.0:
        return V4_EVAL / ds / f"pv_{kt}"
    if alpha == 0.0:
        return V4_EVAL / ds / f"future_{kt}"
    if alpha == 0.5:
        return V4_EVAL / ds / f"hybrid_{kt}"
    # 0.25 or 0.75 → alpha_sweep
    return V4_ALPHA / ds / f"hybrid_{atag(alpha)}_{kt}"


def perl_path(ds: str, alpha: float, k: float) -> Path:
    return V5_PERL / ds / f"hybrid_{atag(alpha)}_L24-31_{ktag(k)}"


def fmt(x: Optional[float], pad: int = 7) -> str:
    return f"{x:.4f}".rjust(pad) if x is not None else "   —   "


# ── Collect ───────────────────────────────────────────────────────────────
global_scores: Dict[tuple, Optional[float]] = {}
perl_scores: Dict[tuple, Optional[float]] = {}

for ds in DATASETS:
    for k in KEEPS:
        for a in GLOBAL_ALPHAS:
            global_scores[(ds, k, a)] = read_metric(global_path(ds, a, k), ds)
        for a in PERL_ALPHAS:
            perl_scores[(ds, k, a)] = read_metric(perl_path(ds, a, k), ds)


# ── results_table.md ─────────────────────────────────────────────────────
def build_results_table() -> str:
    lines = [
        "# EXP-20260420-001 Per-Layer Hybrid Sweep — Results Table",
        "",
        f"Metric: ROUGE-L (Accuracy for webqa). Keep ratios: {KEEPS}.",
        "",
        "`global` = EXP-20260418-001 Phase 3/4 (Future blended at every layer 0-31).",
        "`perL`  = EXP-20260420-001 Phase 5 (Future blended ONLY at layers 24-31; PV-only elsewhere).",
        "",
    ]
    for k in KEEPS:
        lines.append(f"## keep_ratio = {k}")
        lines.append("")
        header = "| method              | " + " | ".join(f"{ds:^14}" for ds in DATASETS) + " |  mean  |"
        sep = "|---------------------|" + "|".join(["-" * 16] * len(DATASETS)) + "|--------|"
        lines += [header, sep]
        # global rows
        for a in GLOBAL_ALPHAS:
            lbl_map = {0.0: "global Future (α=0)", 1.0: "global PV (α=1)", 0.5: "global Hybrid α=.5"}
            lbl = lbl_map.get(a, f"global Hybrid α={a}")
            vals = [global_scores[(ds, k, a)] for ds in DATASETS]
            mean = sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None))
            row = f"| {lbl:<19} | " + " | ".join(fmt(v, 14) for v in vals) + f" | {mean:.4f} |"
            lines.append(row)
        # spacer
        lines.append("|" + " " * 21 + "|" + "|".join([" " * 16] * len(DATASETS)) + "|        |")
        # per-layer rows
        for a in PERL_ALPHAS:
            lbl = f"perL Hybrid α={a}"
            vals = [perl_scores[(ds, k, a)] for ds in DATASETS]
            valid = [v for v in vals if v is not None]
            mean = sum(valid) / len(valid) if valid else float("nan")
            row = f"| {lbl:<19} | " + " | ".join(fmt(v, 14) for v in vals) + f" | {mean:.4f} |"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


# ── per_layer_vs_global.md ───────────────────────────────────────────────
def build_delta_table() -> str:
    """For each cell (ds, k): best_perL_alpha, best_global_alpha, delta."""
    lines = [
        "# Per-layer vs Global: Best-α Comparison",
        "",
        "For each (dataset, keep_ratio) cell we pick:",
        "  - `perL_best` = max over α ∈ {0.0, 0.25, 0.5, 0.75} of per-layer Hybrid score.",
        "  - `glob_best` = max over α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} of global baseline.",
        "  - `Δ = perL_best - glob_best`.  Positive = per-layer wins.",
        "",
        "| dataset       |  k   | perL α* | perL score | glob α* | glob score |   Δ    |",
        "|---------------|------|---------|------------|---------|------------|--------|",
    ]
    wins = ties = losses = 0
    sig_wins = 0  # Δ ≥ 0.005
    for ds in DATASETS:
        for k in KEEPS:
            g_pairs = [(a, global_scores[(ds, k, a)]) for a in GLOBAL_ALPHAS if global_scores[(ds, k, a)] is not None]
            p_pairs = [(a, perl_scores[(ds, k, a)]) for a in PERL_ALPHAS if perl_scores[(ds, k, a)] is not None]
            if not g_pairs or not p_pairs:
                lines.append(f"| {ds:<13} | {k:<4} | — | — | — | — | — |")
                continue
            pa, ps = max(p_pairs, key=lambda x: x[1])
            ga, gs = max(g_pairs, key=lambda x: x[1])
            d = ps - gs
            verdict = "+" if d > 0 else ("=" if abs(d) < 1e-6 else "-")
            if d >= 0.005:
                sig_wins += 1
            if d > 0:
                wins += 1
            elif abs(d) < 1e-6:
                ties += 1
            else:
                losses += 1
            lines.append(
                f"| {ds:<13} | {k:<4} | {pa:<7} | {ps:.4f}    | {ga:<7} | {gs:.4f}    | {d:+.4f} {verdict} |"
            )
    total = len(DATASETS) * len(KEEPS)
    lines += [
        "",
        f"**Score**: per-layer wins/ties/losses = {wins}/{ties}/{losses} of {total}.",
        f"**Significant wins (Δ ≥ 0.005)**: {sig_wins}/{total}.",
    ]
    return "\n".join(lines)


# ── summary.md ───────────────────────────────────────────────────────────
def build_summary() -> str:
    """Check H1/H2/H3 from PLAN.md §2."""
    lines = ["# Summary — PLAN Success Criteria Verdict", ""]

    # H1: per-layer Hybrid best avg ≥ global Hybrid best avg (over datasets, at each k).
    h1_lines = ["## H1 — per-layer Hybrid ≥ global Hybrid (average over 6 datasets)", ""]
    h1_pass_count = 0
    for k in KEEPS:
        p_avg = []
        g_avg = []
        for ds in DATASETS:
            p_best = max((v for v in (perl_scores[(ds, k, a)] for a in PERL_ALPHAS) if v is not None), default=None)
            g_best = max((v for v in (global_scores[(ds, k, a)] for a in GLOBAL_ALPHAS) if v is not None), default=None)
            if p_best is not None:
                p_avg.append(p_best)
            if g_best is not None:
                g_avg.append(g_best)
        pm = sum(p_avg) / len(p_avg) if p_avg else float("nan")
        gm = sum(g_avg) / len(g_avg) if g_avg else float("nan")
        delta = pm - gm
        verdict = "PASS" if delta >= 0 else "FAIL"
        if delta >= 0:
            h1_pass_count += 1
        h1_lines.append(f"- k={k}: perL_avg={pm:.4f}  global_avg={gm:.4f}  Δ={delta:+.4f}  → **{verdict}**")
    lines += h1_lines + [""]

    # H2: per-layer best α is more skewed toward Future (lower α) than global best α.
    lines += ["## H2 — per-layer α* shifted toward Future (lower α) vs global", ""]
    lines.append("| dataset       |  k   | perL α* | glob α* | diff (perL − glob) |")
    lines.append("|---------------|------|---------|---------|---------------------|")
    h2_toward_future = 0
    h2_eval = 0
    for ds in DATASETS:
        for k in KEEPS:
            g_pairs = [(a, global_scores[(ds, k, a)]) for a in GLOBAL_ALPHAS if global_scores[(ds, k, a)] is not None]
            p_pairs = [(a, perl_scores[(ds, k, a)]) for a in PERL_ALPHAS if perl_scores[(ds, k, a)] is not None]
            if not g_pairs or not p_pairs:
                continue
            pa = max(p_pairs, key=lambda x: x[1])[0]
            ga = max(g_pairs, key=lambda x: x[1])[0]
            dif = pa - ga
            h2_eval += 1
            if dif < 0:
                h2_toward_future += 1
            lines.append(f"| {ds:<13} | {k:<4} | {pa:<7} | {ga:<7} | {dif:+.2f} |")
    lines.append("")
    lines.append(f"**Toward-Future count**: {h2_toward_future}/{h2_eval} cells have perL_α* < global_α*.")
    lines.append("")

    # H3: Future-dominant datasets (clevr_change, alfred) — per-layer Hybrid ≥ Future-only (α=0.0).
    lines += ["## H3 — on Future-dominant datasets, per-layer Hybrid ≥ Future-only (α=0.0 global)", ""]
    h3_ok = 0
    h3_eval = 0
    for ds in ["clevr_change", "alfred"]:
        for k in KEEPS:
            p_best = max(
                ((a, v) for a, v in ((a, perl_scores[(ds, k, a)]) for a in PERL_ALPHAS) if v is not None),
                default=None,
                key=lambda x: x[1] if x else -1,
            )
            fut = global_scores[(ds, k, 0.0)]
            if p_best is None or fut is None:
                continue
            pa, ps = p_best
            h3_eval += 1
            ok = ps >= fut
            if ok:
                h3_ok += 1
            verdict = "PASS" if ok else "FAIL"
            lines.append(f"- {ds} k={k}: perL(α={pa})={ps:.4f}  vs Future-only={fut:.4f}  Δ={ps - fut:+.4f}  → **{verdict}**")
    lines.append("")
    lines.append(f"**H3 score**: {h3_ok}/{h3_eval} ≥ (broadcast-noise removal benefit).")
    lines.append("")

    # Overall
    lines += ["## Overall verdict", ""]
    lines.append(f"- H1 (average gain): {h1_pass_count}/{len(KEEPS)} keep ratios pass.")
    lines.append(f"- H2 (shift toward Future): {h2_toward_future}/{h2_eval} cells.")
    lines.append(f"- H3 (Future-dominant datasets): {h3_ok}/{h3_eval} cells.")
    return "\n".join(lines)


# ── write ──────────────────────────────────────────────────────────────────
def main():
    (OUT / "results_table.md").write_text(build_results_table() + "\n")
    (OUT / "per_layer_vs_global.md").write_text(build_delta_table() + "\n")
    (OUT / "summary.md").write_text(build_summary() + "\n")
    print("wrote:")
    print(f"  {OUT / 'results_table.md'}")
    print(f"  {OUT / 'per_layer_vs_global.md'}")
    print(f"  {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
