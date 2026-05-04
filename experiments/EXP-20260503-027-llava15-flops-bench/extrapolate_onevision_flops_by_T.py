#!/usr/bin/env python3
"""Re-compute FLOPS at hypothetical decode lengths T using the OneVision
100-sample manifest (no re-run of the model)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from bench_original_onevision_flops import (  # noqa: E402
    Qwen2Flops,
    SiglipFlops,
    compute_n_image_keep,
    make_qwen2_flops,
    make_siglip_flops_default,
)

TFLOPS = 1e12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-json", required=True)
    parser.add_argument("--t-values", type=int, nargs="+", default=[5, 32, 256, 1024, 2048, 4096])
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    args = parser.parse_args()

    runs = json.loads(Path(args.runs_json).read_text())
    print(f"loaded {len(runs)} samples from {args.runs_json}")

    qwen = Qwen2Flops(
        num_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=152064,
    )
    siglip = make_siglip_flops_default().total_per_crop()

    headers = ["T", "method", "L_p_mean", "ret_mean", "prefill_TF", "decode_TF",
               "vision_TF", "total_TF", "decode%", "total%"]
    fmt = "{:>5} | {:<11} | {:>8} | {:>8} | {:>10} | {:>9} | {:>9} | {:>9} | {:>7} | {:>7}"
    print(fmt.format(*headers))
    print("-" * 110)

    last_T = None
    for T in args.t_values:
        full_decode_per = []
        full_total_per = []
        for run in runs:
            L_p = int(run["prompt_mm_len"])
            full_decode_per.append(qwen.decode_total(L_p, T))
            full_total_per.append(qwen.prefill(L_p) + full_decode_per[-1] + siglip * int(run.get("n_crops", 1)))

        for r in args.ratios:
            decode_list = []
            total_list = []
            retained_list = []
            decode_pct_list = []
            total_pct_list = []
            for run, fdec, ftot in zip(runs, full_decode_per, full_total_per):
                L_p = int(run["prompt_mm_len"])
                n_text = int(run["n_text"])
                n_img = int(run["n_img"])
                if math.isclose(r, 1.0):
                    keep = n_img
                else:
                    keep = compute_n_image_keep(n_img, n_text, r)
                retained = n_text + keep
                decode = qwen.decode_total(L_p, T, retained_prompt_len=retained)
                total = qwen.prefill(L_p) + decode + siglip * int(run.get("n_crops", 1))
                decode_list.append(decode)
                total_list.append(total)
                retained_list.append(retained)
                decode_pct_list.append(100.0 * decode / max(1, fdec))
                total_pct_list.append(100.0 * total / max(1, ftot))
            method = "full_cache" if math.isclose(r, 1.0) else f"keep_{int(round(r * 100)):03d}"
            L_p_mean = mean(int(run["prompt_mm_len"]) for run in runs)
            prefill_tf_mean = mean(qwen.prefill(int(run["prompt_mm_len"])) for run in runs) / TFLOPS
            if last_T is not None and last_T != T:
                print()
            print(fmt.format(
                T, method,
                f"{L_p_mean:.1f}",
                f"{mean(retained_list):.1f}",
                f"{prefill_tf_mean:.4f}",
                f"{mean(decode_list) / TFLOPS:.4f}",
                f"{siglip / TFLOPS:.4f}",
                f"{mean(total_list) / TFLOPS:.4f}",
                f"{mean(decode_pct_list):.2f}%" if not math.isclose(r, 1.0) else "100.00%",
                f"{mean(total_pct_list):.2f}%" if not math.isclose(r, 1.0) else "100.00%",
            ))
            last_T = T


if __name__ == "__main__":
    main()
