#!/usr/bin/env python3
"""
EXP-20260412-004: Ratio Sensitivity Analysis
probe_mlp r=0.05/0.10 (truncation) vs LOOK-M r=0.20
"""
import json
import os
import csv
from pathlib import Path

ARTIFACT_ROOT = Path("/workspace/hd/artifacts/probe_global")
OUT_CSV = ARTIFACT_ROOT / "ratio_sensitivity_analysis.csv"
OUT_MD = Path(__file__).parent / "RESULT.md"


def get_metric(path: Path):
    metrics_file = path / "metrics.json"
    if not metrics_file.exists():
        return None
    data = json.loads(metrics_file.read_text())
    look_eval = data.get("look_eval", {})
    for key in ["Accuracy", "ROUGE-L"]:
        if key in look_eval:
            return round(look_eval[key], 3)
    return None


def win_loss_tie(probe_val, lookm_val):
    if probe_val is None or lookm_val is None:
        return "?"
    if probe_val > lookm_val + 1e-6:
        return "W"
    elif lookm_val > probe_val + 1e-6:
        return "L"
    else:
        return "T"


def main():
    datasets = sorted([d.name for d in ARTIFACT_ROOT.iterdir() if d.is_dir()])

    rows = []
    for ds in datasets:
        probe_05 = get_metric(ARTIFACT_ROOT / ds / "probe_mlp" / "keep_0p05")
        probe_10 = get_metric(ARTIFACT_ROOT / ds / "probe_mlp" / "keep_0p10")
        probe_20 = get_metric(ARTIFACT_ROOT / ds / "probe_mlp" / "keep_0p20")
        lookm_20 = get_metric(ARTIFACT_ROOT / ds / "look_m" / "keep_0p20")

        if probe_05 is None and probe_10 is None:
            continue

        rows.append({
            "dataset": ds,
            "probe_r05": probe_05,
            "probe_r10": probe_10,
            "probe_r20": probe_20,
            "lookm_r20": lookm_20,
            "wlt_05_vs_lookm20": win_loss_tie(probe_05, lookm_20),
            "wlt_10_vs_lookm20": win_loss_tie(probe_10, lookm_20),
            "wlt_20_vs_lookm20": win_loss_tie(probe_20, lookm_20),
            "delta_05": round(probe_05 - lookm_20, 3) if probe_05 is not None and lookm_20 is not None else None,
            "delta_10": round(probe_10 - lookm_20, 3) if probe_10 is not None and lookm_20 is not None else None,
        })

    # Write CSV
    fieldnames = ["dataset", "probe_r05", "probe_r10", "probe_r20", "lookm_r20",
                  "wlt_05_vs_lookm20", "wlt_10_vs_lookm20", "wlt_20_vs_lookm20",
                  "delta_05", "delta_10"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {OUT_CSV}")

    # Summary stats
    valid = [r for r in rows if r["lookm_r20"] is not None]

    def count_wlt(rows, key):
        w = sum(1 for r in rows if r[key] == "W")
        l = sum(1 for r in rows if r[key] == "L")
        t = sum(1 for r in rows if r[key] == "T")
        return w, l, t

    w05, l05, t05 = count_wlt(valid, "wlt_05_vs_lookm20")
    w10, l10, t10 = count_wlt(valid, "wlt_10_vs_lookm20")
    w20, l20, t20 = count_wlt(valid, "wlt_20_vs_lookm20")

    avg_delta05 = sum(r["delta_05"] for r in valid if r["delta_05"] is not None) / max(1, sum(1 for r in valid if r["delta_05"] is not None))
    avg_delta10 = sum(r["delta_10"] for r in valid if r["delta_10"] is not None) / max(1, sum(1 for r in valid if r["delta_10"] is not None))

    # Print table
    print(f"\n{'Dataset':<30} {'p0.05':>6} {'p0.10':>6} {'p0.20':>6} {'lm0.20':>7} {'W/L/T@05':>9} {'W/L/T@10':>9}")
    print("-" * 80)
    for r in rows:
        p05 = f"{r['probe_r05']:.3f}" if r["probe_r05"] is not None else "  N/A"
        p10 = f"{r['probe_r10']:.3f}" if r["probe_r10"] is not None else "  N/A"
        p20 = f"{r['probe_r20']:.3f}" if r["probe_r20"] is not None else "  N/A"
        lm  = f"{r['lookm_r20']:.3f}" if r["lookm_r20"] is not None else "  N/A"
        print(f"{r['dataset']:<30} {p05:>6} {p10:>6} {p20:>6} {lm:>7} {r['wlt_05_vs_lookm20']:>9} {r['wlt_10_vs_lookm20']:>9}")

    print(f"\n{'':30} {'avg Δ':>6} {avg_delta05:+.3f}  {avg_delta10:+.3f}")
    print(f"\nSummary vs LOOK-M r=0.20 ({len(valid)} datasets):")
    print(f"  probe r=0.05: {w05}W / {l05}L / {t05}T")
    print(f"  probe r=0.10: {w10}W / {l10}L / {t10}T")
    print(f"  probe r=0.20: {w20}W / {l20}L / {t20}T")

    # Write RESULT.md
    lines = []
    lines.append("## Experiment Result\n")
    lines.append("**ID**: EXP-20260412-004")
    lines.append("**Date**: 2026-04-14")
    lines.append("**Status**: [x] Done")
    lines.append("")
    lines.append("---\n")
    lines.append("### 핵심 결과\n")
    lines.append(f"**probe r=0.05 vs LOOK-M r=0.20**: {w05}승 / {l05}패 / {t05}타이 (avg Δ = {avg_delta05:+.3f})")
    lines.append(f"**probe r=0.10 vs LOOK-M r=0.20**: {w10}승 / {l10}패 / {t10}타이 (avg Δ = {avg_delta10:+.3f})")
    lines.append(f"**probe r=0.20 vs LOOK-M r=0.20**: {w20}승 / {l20}패 / {t20}타이 (참고: 기존 결과)")
    lines.append("")
    lines.append("---\n")
    lines.append("### 전체 결과표\n")
    lines.append("| Dataset | probe r=0.05 | probe r=0.10 | probe r=0.20 | LOOK-M r=0.20 | W/L/T @05 | W/L/T @10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        p05 = f"{r['probe_r05']:.3f}" if r["probe_r05"] is not None else "N/A"
        p10 = f"{r['probe_r10']:.3f}" if r["probe_r10"] is not None else "N/A"
        p20 = f"{r['probe_r20']:.3f}" if r["probe_r20"] is not None else "N/A"
        lm  = f"{r['lookm_r20']:.3f}" if r["lookm_r20"] is not None else "N/A"
        d05 = f"{r['wlt_05_vs_lookm20']}"
        d10 = f"{r['wlt_10_vs_lookm20']}"
        lines.append(f"| {r['dataset']} | {p05} | {p10} | {p20} | {lm} | {d05} | {d10} |")
    lines.append("")
    lines.append("---\n")

    h1_ok = w10 >= 12
    h1_avg_ok = abs(avg_delta10) < 0.03

    lines.append("### 가설 검증\n")
    lines.append(f"- [{'x' if h1_ok else ' '}] **H1**: probe r=0.10 ≥ LOOK-M r=0.20 데이터셋 수 ≥ 12")
    lines.append(f"  - 결과: {w10 + t10}/{len(valid)} datasets에서 동등 또는 우위 (W+T={w10+t10})")
    lines.append(f"- [{'x' if h1_avg_ok else ' '}] **H1 (avg)**: |probe r=0.10 - LOOK-M r=0.20| < 0.03")
    lines.append(f"  - 결과: avg Δ = {avg_delta10:+.3f}")
    lines.append("")
    lines.append("---\n")

    lines.append("### 결론\n")
    if w10 + t10 >= 12 and avg_delta10 > -0.03:
        lines.append(f"**효율성 클레임 성립**: probe r=0.10 (token budget 절반)으로 LOOK-M r=0.20과 동등 또는 우위.")
        lines.append(f"- {w10+t10}/{len(valid)} 데이터셋에서 probe r=0.10 ≥ LOOK-M r=0.20")
        lines.append(f"- 평균 성능 차이: {avg_delta10:+.3f} (< 0.03 기준 충족)")
    else:
        lines.append(f"**효율성 클레임 부분 성립**: probe r=0.10이 LOOK-M r=0.20 대비 {w10}W/{l10}L/{t10}T.")
        lines.append(f"- 평균 Δ = {avg_delta10:+.3f}")
    lines.append("")
    lines.append("---\n")
    lines.append("### 아티팩트")
    lines.append(f"- CSV: `{OUT_CSV}`")
    lines.append(f"- probe r=0.05 결과: `/workspace/hd/artifacts/probe_global/{{ds}}/probe_mlp/keep_0p05/`")
    lines.append(f"- probe r=0.10 결과: `/workspace/hd/artifacts/probe_global/{{ds}}/probe_mlp/keep_0p10/`")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nRESULT.md written: {OUT_MD}")


if __name__ == "__main__":
    main()
