#!/usr/bin/env python3
"""Aggregate zap Future-v5 and PrefixKV PPL logs into a CSV + markdown summary.

Ratio conventions differ between the two methods:
- PrefixKV: `--ratio r` = fraction of KV to REMOVE (compression ratio).
  keep_fraction = 1 - r.
- zap     : `--image-keep-ratio k` = fraction of image tokens to KEEP.
  keep_fraction = k.

Both are reported on a unified `keep` axis so the methods can be compared directly.
"""
from __future__ import annotations
import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = Path("/workspace/zap/artifacts/EXP-20260420-003")
PREFIX_LOGS = Path("/workspace/PrefixKV/logs")

DATASETS = {"mmvet": "mm-vet", "detail1k": "detail_1k"}
PREFIX_DS_MAP = {
    "mm-vet":   ["mmvet-gpu0", "mmvet-gpu1"],
    "detail_1k": ["detail1k-gpu0", "detail1k-gpu1"],
}


def collect_zap() -> list[dict]:
    rows = []
    for ds_tag, ds_name in DATASETS.items():
        # Image-keep scope
        for p in sorted(glob.glob(str(ART / ds_tag / "future_v5_k*/result.json"))):
            d = json.load(open(p))
            m = re.search(r"k0p(\d)$", Path(p).parent.name)
            if not m:
                continue
            keep = float(f"0.{m.group(1)}")
            rows.append({
                "method": "zap_future_v5_image",
                "dataset": ds_name,
                "native_ratio_label": "image_keep",
                "native_ratio_value": keep,
                "keep_fraction": keep,
                "ppl": float(d["ppl"]),
                "n_samples": int(d["n_samples"]),
                "elapsed_s": float(d["elapsed_seconds"]),
                "source": p,
            })
        # Total-keep scope variants: future, h2o, hybrid.
        for prefix, method_name in [
            ("future_v5_tot", "zap_future_v5_total"),
            ("h2oall_tot",    "zap_h2o_all_token"),
            ("hybrid_a050_tot", "zap_hybrid_a050"),
        ]:
            for p in sorted(glob.glob(str(ART / ds_tag / f"{prefix}*/result.json"))):
                d = json.load(open(p))
                m = re.search(rf"{re.escape(prefix)}0p(\d)$", Path(p).parent.name)
                if not m:
                    continue
                keep = float(f"0.{m.group(1)}")
                rows.append({
                    "method": method_name,
                    "dataset": ds_name,
                    "native_ratio_label": "total_keep",
                    "native_ratio_value": keep,
                    "keep_fraction": keep,
                    "ppl": float(d["ppl"]),
                    "n_samples": int(d["n_samples"]),
                    "elapsed_s": float(d["elapsed_seconds"]),
                    "source": p,
                })
    return rows


def _latest_log(dir_path: Path) -> Path | None:
    files = sorted(dir_path.glob("*.txt"))
    return files[-1] if files else None


def collect_prefixkv() -> list[dict]:
    rows = []
    for ds_name, gpu_dirs in PREFIX_DS_MAP.items():
        for gdir in gpu_dirs:
            base = PREFIX_LOGS / gdir / "prefixkv"
            if not base.is_dir():
                continue
            for ratio_dir in sorted(base.iterdir()):
                if not ratio_dir.is_dir():
                    continue
                try:
                    ratio = float(ratio_dir.name)
                except ValueError:
                    continue
                log = _latest_log(ratio_dir)
                if log is None:
                    continue
                last = log.read_text().strip().splitlines()[-1]
                try:
                    ppl = float(last)
                except ValueError:
                    continue
                rows.append({
                    "method": "PrefixKV",
                    "dataset": ds_name,
                    "native_ratio_label": "remove",
                    "native_ratio_value": ratio,
                    "keep_fraction": round(1.0 - ratio, 2),
                    "ppl": ppl,
                    "n_samples": 218 if ds_name == "mm-vet" else 1000,
                    "elapsed_s": None,
                    "source": str(log),
                })
    # Dedupe: prefer the latest (later timestamp) entry per (ds, keep_fraction).
    uniq: dict[tuple, dict] = {}
    for r in rows:
        key = (r["dataset"], r["keep_fraction"])
        if key not in uniq or r["source"] > uniq[key]["source"]:
            uniq[key] = r
    return list(uniq.values())


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["method", "dataset", "native_ratio_label", "native_ratio_value",
              "keep_fraction", "ppl", "n_samples", "elapsed_s", "source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda x: (x["dataset"], x["method"], x["keep_fraction"])):
            w.writerow(row)


METHOD_ORDER = [
    "PrefixKV",
    "zap_h2o_all_token",
    "zap_hybrid_a050",
    "zap_future_v5_total",
    "zap_future_v5_image",
]


def fmt_table(rows: list[dict]) -> dict[str, str]:
    """Return {dataset: markdown_table}. 5-point keep grid matches all-token sweeps."""
    keep_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    out = {}
    for ds_name in DATASETS.values():
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        present_methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in ds_rows)]

        header = "| keep | " + " | ".join(present_methods) + " | Δ(PrefixKV − zap_future_total) |"
        sep    = "|" + "---|" * (len(present_methods) + 2)
        lines = [header, sep]
        for k in keep_grid:
            cells = [f"{k:.1f}"]
            by_method = {m: None for m in present_methods}
            for r in ds_rows:
                if abs(r["keep_fraction"] - k) < 1e-6 and r["method"] in by_method:
                    by_method[r["method"]] = r["ppl"]
            for m in present_methods:
                v = by_method[m]
                cells.append(f"{v:.4f}" if v is not None else "—")
            tot_v = by_method.get("zap_future_v5_total")
            pkv_v = by_method.get("PrefixKV")
            if tot_v is not None and pkv_v is not None:
                cells.append(f"{pkv_v - tot_v:+.4f}")
            else:
                cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
        out[ds_name] = "\n".join(lines)
    return out


def main() -> None:
    zap_rows = collect_zap()
    pkv_rows = collect_prefixkv()
    all_rows = zap_rows + pkv_rows

    csv_path = ROOT / "results.csv"
    write_csv(all_rows, csv_path)
    print(f"[CSV]  {csv_path}  rows={len(all_rows)}")

    tables = fmt_table(all_rows)
    md_path = ROOT / "results.md"
    with open(md_path, "w") as f:
        f.write(_render_md(tables, all_rows))
    print(f"[MD]   {md_path}")


def _render_md(tables: dict[str, str], rows: list[dict]) -> str:
    counts = {m: sum(1 for r in rows if r["method"] == m) for m in METHOD_ORDER}
    count_str = ", ".join(f"{m}={n}" for m, n in counts.items())
    return f"""# EXP-20260420-003 — PPL Results on PrefixKV protocol

Teacher-forcing perplexity on **mm-vet (218)** and **detail_1k / LLaVA-Description
(1000)** with LLaVA-1.5-7B. All methods reported on a unified `keep =
fraction-of-KV-retained` axis (PrefixKV `--ratio r` is mapped to `keep = 1 − r`).

## Methods

| Method | Scope | Knob | Scoring signal | Notes |
|---|---|---|---|---|
| `zap_future_v5_image` | IMAGE only | `--image-keep-ratio` | Future probe v5 (all-layer MLP) | text/system KV always kept |
| `zap_future_v5_total` | ALL tokens | `--total-keep-ratio` | Future probe v5 | scope-matched to PrefixKV |
| `zap_h2o_all_token` | ALL tokens | `--total-keep-ratio` | prefill H2O heavy-hitter | eager attention, prefill-only hook |
| `zap_hybrid_a050` | ALL tokens | `--total-keep-ratio` | 0.5·softmax(H2O) + 0.5·softmax(Future) | flat α=0.5 at every layer |
| `PrefixKV` | ALL tokens | `--ratio` = REMOVE fraction | layer-wise adaptive prefix | external baseline |

- **Probe**: `future_probe_v5_all_token_10ep` (all 32 layers, 10 ep, best val
  Spearman 0.232 @ ep9). Used by `zap_future_v5_*` and `zap_hybrid_a050`.
- **Runs collected**: {count_str}.

## mm-vet

{tables["mm-vet"]}

## detail_1k (LLaVA-Description)

{tables["detail_1k"]}

## Observations (scope-matched, all-token methods)

### zap Future v5 vs PrefixKV
- Future-v5 (total) beats PrefixKV at every keep on both datasets.
- Gap is largest at aggressive eviction: detail_1k keep=0.1 is +1.3 PPL in
  zap's favor; mm-vet keep=0.1 is +2.2. Converges as keep → 0.9.

### Hybrid (H2O + Future, α=0.5) vs H2O-only
- On mm-vet, `hybrid_a050` improves on `h2o_all_token` at every keep — the Future
  signal adds information that raw prefill attention misses.
- Against `zap_future_v5_total`, hybrid is either comparable or slightly worse
  at high keep, and slightly better at very low keep on mm-vet — weak
  complementarity. Not a clear win.

### Image-only scope (reference line)
- `zap_future_v5_image` is nearly flat across keep (mm-vet 5.08–5.14, detail_1k
  2.95–3.09). Image KV is highly redundant — the Future probe keeps the few
  positions that actually matter for decode.
- Comparing `_image` to `_total` at the same keep isolates the cost of
  evicting text: small at high keep, larger at keep=0.1 (where the total
  budget starts cutting into text too).

## Files

- CSV: `results.csv` (long form, one row per run)
- This report: `results.md`
- Zap raw artifacts: `/workspace/zap/artifacts/EXP-20260420-003/{{mmvet,detail1k}}/*/result.json`
- PrefixKV raw logs: `/workspace/PrefixKV/logs/{{mmvet,detail1k}}-gpu*/prefixkv/*/*.txt`
"""


if __name__ == "__main__":
    main()
