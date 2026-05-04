#!/usr/bin/env python3
"""Re-compute FLOPS at hypothetical decode lengths T using the same prompt
distribution as the 100-sample MileBench bench (no re-running of the model).

Reads `per_sample_runs.json` from a previous run, applies the analytical
FLOPS formulas for full_cache vs student_keep_{050,020} at user-specified
decode lengths, and prints a side-by-side table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from bench_original_llava15_flops import (  # noqa: E402
    LlamaFlops,
    VitFlops,
    compute_n_image_keep,
    make_vit_flops_default,
)


TFLOPS = 1e12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-json", required=True, help="per_sample_runs.json from a previous bench run")
    parser.add_argument("--t-values", type=int, nargs="+", default=[8, 32, 128, 512, 1024])
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    args = parser.parse_args()

    runs = json.loads(Path(args.runs_json).read_text())
    print(f"loaded {len(runs)} samples from {args.runs_json}")

    llama = LlamaFlops(num_layers=32, hidden_size=4096, intermediate_size=11008, vocab_size=32000)
    vit = make_vit_flops_default()
    vit_per_image = vit.total()

    headers = ["T_decode", "method", "prompt_mm_mean", "prefill_TF", "decode_TF", "vision_TF", "total_TF",
               "decode_pct_of_full", "total_pct_of_full"]
    rows: list[list] = []

    for T in args.t_values:
        full_total_per_sample = []
        full_decode_per_sample = []
        method_totals: dict[float, dict[str, list[float]]] = {}
        for run in runs:
            L_p = int(run["prompt_mm_len"])
            n_text = int(run["n_text"])
            n_img = int(run["n_img"])
            prefill = llama.prefill(L_p)
            full_decode = llama.decode_total(L_p, T)
            full_total = prefill + full_decode + vit_per_image
            full_decode_per_sample.append(full_decode)
            full_total_per_sample.append(full_total)
            for r in args.ratios:
                if math.isclose(r, 1.0):
                    keep = n_img
                else:
                    keep = compute_n_image_keep(n_img, n_text, r)
                retained = n_text + keep
                decode = llama.decode_total(L_p, T, retained_prompt_len=retained)
                total = prefill + decode + vit_per_image
                method_totals.setdefault(r, {"decode": [], "total": [], "decode_pct": [], "total_pct": []})
                method_totals[r]["decode"].append(decode)
                method_totals[r]["total"].append(total)
                method_totals[r]["decode_pct"].append(100.0 * decode / max(1, full_decode))
                method_totals[r]["total_pct"].append(100.0 * total / max(1, full_total))

        prompt_mean = mean(int(r["prompt_mm_len"]) for r in runs)
        prefill_tf = mean(llama.prefill(int(r["prompt_mm_len"])) for r in runs) / TFLOPS
        for r in args.ratios:
            mt = method_totals[r]
            method_name = "full_cache_100" if math.isclose(r, 1.0) else f"student_keep_{int(round(r * 100)):03d}"
            rows.append([
                T,
                method_name,
                f"{prompt_mean:.1f}",
                f"{prefill_tf:.4f}",
                f"{mean(mt['decode']) / TFLOPS:.4f}",
                f"{vit_per_image / TFLOPS:.4f}",
                f"{mean(mt['total']) / TFLOPS:.4f}",
                f"{mean(mt['decode_pct']):.2f}%" if not math.isclose(r, 1.0) else "100.00%",
                f"{mean(mt['total_pct']):.2f}%" if not math.isclose(r, 1.0) else "100.00%",
            ])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    fmt = " | ".join("{:<" + str(w) + "}" for w in col_widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in col_widths))
    last_T = None
    for row in rows:
        if last_T is not None and row[0] != last_T:
            print()
        print(fmt.format(*row))
        last_T = row[0]


if __name__ == "__main__":
    main()
