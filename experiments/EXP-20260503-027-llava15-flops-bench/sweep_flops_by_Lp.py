#!/usr/bin/env python3
"""Sweep analytical FLOPS reduction over prompt length L_p, fixed n_img=576.

For each (L_p, T_decode, ratio) triple, computes:
  - decode_flops_pct_of_full
  - total_flops_pct_of_full   (vision_tower + prefill + decode)

LLaMA-7B + CLIP ViT-L/14 (LLaVA-1.5-7B):
  N=32, D=4096, I=11008, V=32000, ViT(N=24,D=1024,I=4096,P=576+1)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from bench_original_llava15_flops import LlamaFlops, compute_n_image_keep, make_vit_flops_default

TFLOPS = 1e12

llama = LlamaFlops(num_layers=32, hidden_size=4096, intermediate_size=11008, vocab_size=32000)
vit_flops = make_vit_flops_default().total()


def report(L_p_values: list[int], T_values: list[int], ratios: list[float], n_img: int = 576) -> None:
    headers = ["L_p", "n_text", "T", "method", "decode_TF", "total_TF", "decode_pct", "total_pct"]
    print(f"n_image_tokens = {n_img} (fixed for LLaVA-1.5 single image)\n")
    print(("{:>5} | {:>6} | {:>5} | {:<16} | {:>10} | {:>9} | {:>10} | {:>10}").format(*headers))
    print("-" * 90)
    last_L = None
    for L_p in L_p_values:
        n_text = max(0, L_p - n_img)
        prefill = llama.prefill(L_p)
        for T in T_values:
            full_decode = llama.decode_total(L_p, T)
            full_total = prefill + full_decode + vit_flops
            for r in ratios:
                if math.isclose(r, 1.0):
                    keep = n_img
                else:
                    keep = compute_n_image_keep(n_img, n_text, r)
                retained = n_text + keep
                decode = llama.decode_total(L_p, T, retained_prompt_len=retained)
                total = prefill + decode + vit_flops
                method = "full_cache" if math.isclose(r, 1.0) else f"keep_{int(round(r * 100)):03d}"
                decode_pct = 100.0 * decode / max(1, full_decode)
                total_pct = 100.0 * total / max(1, full_total)
                if last_L is not None and last_L != L_p:
                    print()
                print(("{:>5} | {:>6} | {:>5} | {:<16} | {:>10.4f} | {:>9.4f} | {:>9.2f}% | {:>9.2f}%").format(
                    L_p, n_text, T, method, decode / TFLOPS, total / TFLOPS, decode_pct, total_pct
                ))
                last_L = L_p


if __name__ == "__main__":
    L_p_values = [800, 1500, 2500, 4000, 8000, 16000, 32000]
    T_values = [32, 256, 1024]
    ratios = [1.0, 0.5, 0.2]
    report(L_p_values, T_values, ratios)
