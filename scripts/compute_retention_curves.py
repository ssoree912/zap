#!/usr/bin/env python3
"""Compute "Future-mass retention" curves for H2O vs Probe at each keep ratio.

For one sample at each layer we already have three per-image-token vectors:
  * H2O    = attn_prefill.sum(dim=Q) at image keys (prefill self-attention)
  * Future = attn_decode[:, :, image_idx].mean(Q).mean(H)  (oracle)
  * Probe  = softmax(MLP_l(hidden_image_l))

Retention metric (for keep ratio r, for a method m that picks top-k_r tokens):
  R_m(r) = sum_{i in top_k_r(m)} Future_i  /  sum_i Future_i

This measures how much of the decode's actual attention mass each method
preserves when evicting (1-r) fraction of image tokens.

Averaged over N samples → robust comparison.

Usage:
  python scripts/compute_retention_curves.py \\
      --dataset wikivqa --n-samples 20 \\
      --probe-path /workspace/zap/ckpts/future_probe_allL_limit100 \\
      --out-dir /workspace/zap/artifacts/EXP-20260422-001/viz/retention_wikivqa
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse loaders / forward helpers
from viz_h2o_vs_future_vs_probe import (
    load_sample,
    run_forward,
    compute_scores,
    N_IMAGE,
)

from kvpress.presses.kvzap_press import KVzapModel
from kvzap.llava_extractor import configure_llava_processor


def retention_at_ratio(scores_method: np.ndarray, scores_future: np.ndarray, ratio: float) -> float:
    """Top-k (by method) future-mass / total future-mass."""
    n = len(scores_method)
    k = max(1, int(round(n * ratio)))
    top = np.argpartition(-scores_method, k - 1)[:k]
    num = float(scores_future[top].sum())
    den = float(scores_future.sum()) + 1e-12
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["scienceqa", "mm-vet", "wikivqa"], default="wikivqa")
    p.add_argument("--sample-indices", type=int, nargs="+", default=None,
                   help="Explicit sample indices. If omitted, uses range(n_samples).")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--probe-path", type=str,
                   default="/workspace/zap/ckpts/future_probe_allL_limit100")
    p.add_argument("--model-path", type=str,
                   default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch-dtype", default="float16")
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Which layers to compute. Default: all 32.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = args.sample_indices or list(range(args.n_samples))
    layers = args.layers or list(range(32))

    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.torch_dtype),
        attn_implementation="eager",
    ).to(torch.device(args.device))
    configure_llava_processor(processor, model.config)
    model.eval()

    print(f"Loading probe: {args.probe_path}")
    probe = KVzapModel.from_pretrained(args.probe_path).to(torch.device(args.device))
    probe.eval()

    # accumulators: layer → ratio → list of retention values (one per sample)
    accum = {l: {r: {"h2o": [], "probe": [], "random": []} for r in args.ratios}
             for l in layers}
    failed = []

    t0 = time.time()
    for k_idx, s_idx in enumerate(indices):
        print(f"[{k_idx+1}/{len(indices)}] sample {args.dataset}[{s_idx}]")
        try:
            sample = load_sample(args.dataset, s_idx)
            forward = run_forward(model, processor, sample)
            scores = compute_scores(forward, probe, layers)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}"); torch.cuda.empty_cache(); failed.append(s_idx); continue
        except Exception as e:
            print(f"  FAIL: {e}"); failed.append(s_idx); continue

        for l in layers:
            s = scores[l]
            h2o = s["h2o"]; fu = s["future"]; pr = s["probe"]
            rng = np.random.default_rng(42 + l + s_idx)
            rand_scores = rng.random(len(fu)).astype(np.float32)
            for r in args.ratios:
                accum[l][r]["h2o"].append(retention_at_ratio(h2o, fu, r))
                accum[l][r]["probe"].append(retention_at_ratio(pr, fu, r))
                accum[l][r]["random"].append(retention_at_ratio(rand_scores, fu, r))

        # free memory
        del forward, scores
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    n_ok = len(indices) - len(failed)
    print(f"Done {n_ok}/{len(indices)} samples in {elapsed:.1f}s (failed: {failed})")

    # Build mean retention matrix
    summary = {"layers": layers, "ratios": args.ratios, "n_samples_ok": n_ok, "failed": failed,
               "mean": {}, "std": {}}
    for l in layers:
        summary["mean"][str(l)] = {"h2o": [], "probe": [], "random": []}
        summary["std"][str(l)] = {"h2o": [], "probe": [], "random": []}
        for r in args.ratios:
            for key in ("h2o", "probe", "random"):
                vals = accum[l][r][key]
                if vals:
                    summary["mean"][str(l)][key].append(float(np.mean(vals)))
                    summary["std"][str(l)][key].append(float(np.std(vals)))
                else:
                    summary["mean"][str(l)][key].append(float("nan"))
                    summary["std"][str(l)][key].append(float("nan"))

    (out_dir / "retention_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote retention_metrics.json")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: per-layer retention curves for selected layers
    selected = [4, 12, 20, 28, 31] if set([4,12,20,28,31]).issubset(layers) else layers[::max(1,len(layers)//5)][:5]
    fig, axes = plt.subplots(1, len(selected), figsize=(3.2*len(selected), 3.6), squeeze=False)
    for ax, l in zip(axes[0], selected):
        mean_h = np.array(summary["mean"][str(l)]["h2o"])
        mean_p = np.array(summary["mean"][str(l)]["probe"])
        mean_r = np.array(summary["mean"][str(l)]["random"])
        std_h = np.array(summary["std"][str(l)]["h2o"])
        std_p = np.array(summary["std"][str(l)]["probe"])
        rs = np.array(args.ratios)
        ax.plot(rs, mean_h, "o-", label="H2O", color="crimson")
        ax.fill_between(rs, mean_h - std_h, mean_h + std_h, color="crimson", alpha=0.15)
        ax.plot(rs, mean_p, "s-", label="Probe (ours)", color="teal")
        ax.fill_between(rs, mean_p - std_p, mean_p + std_p, color="teal", alpha=0.15)
        ax.plot(rs, mean_r, "--", label="Random", color="gray", alpha=0.7)
        ax.plot(rs, rs, ":", label="y=x (ideal uniform)", color="black", alpha=0.4)
        ax.set_xlabel("keep ratio r")
        ax.set_title(f"layer {l}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Future attn-mass retained")
    axes[0][-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Future mass retention vs keep ratio  (n={n_ok} samples, {args.dataset})")
    fig.tight_layout()
    fig.savefig(out_dir / "retention_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote retention_curves.png")

    # Figure 2: retention at r=0.2 across all layers (bar)
    if 0.2 in args.ratios:
        r_idx = args.ratios.index(0.2)
        ret_h = [summary["mean"][str(l)]["h2o"][r_idx] for l in layers]
        ret_p = [summary["mean"][str(l)]["probe"][r_idx] for l in layers]
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(layers))
        w = 0.4
        ax.bar(x - w/2, ret_h, w, label="H2O", color="crimson")
        ax.bar(x + w/2, ret_p, w, label="Probe (ours)", color="teal")
        ax.axhline(0.2, color="gray", linestyle="--", alpha=0.6, label="uniform baseline (r=0.2)")
        ax.set_xticks(x); ax.set_xticklabels([str(l) for l in layers], fontsize=7)
        ax.set_xlabel("layer index"); ax.set_ylabel("Future mass retained @ r=0.2")
        ax.set_title(f"Per-layer retention at keep ratio 0.2  (n={n_ok} samples, {args.dataset})")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "retention_at_r0p2.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("wrote retention_at_r0p2.png")

    # Figure 3: global mean (across all layers) — single clean plot
    global_mean_h = np.array([np.mean([summary["mean"][str(l)]["h2o"][i] for l in layers]) for i in range(len(args.ratios))])
    global_mean_p = np.array([np.mean([summary["mean"][str(l)]["probe"][i] for l in layers]) for i in range(len(args.ratios))])
    global_mean_r = np.array([np.mean([summary["mean"][str(l)]["random"][i] for l in layers]) for i in range(len(args.ratios))])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    rs = np.array(args.ratios)
    ax.plot(rs, global_mean_h, "o-", label="H2O (prefill Σattn)", color="crimson", linewidth=2)
    ax.plot(rs, global_mean_p, "s-", label="Probe (ours)", color="teal", linewidth=2)
    ax.plot(rs, global_mean_r, "--", label="Random", color="gray", alpha=0.7)
    ax.plot(rs, rs, ":", label="y=x (uniform)", color="black", alpha=0.4)
    ax.set_xlabel("keep ratio r")
    ax.set_ylabel("Future attn-mass retained (mean over layers)")
    ax.set_title(f"Global retention curve ({args.dataset}, n={n_ok})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "retention_global.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote retention_global.png")

    print("\nSummary at r=0.2 (mean over layers):")
    if 0.2 in args.ratios:
        r_idx = args.ratios.index(0.2)
        h_mean = np.mean([summary["mean"][str(l)]["h2o"][r_idx] for l in layers])
        p_mean = np.mean([summary["mean"][str(l)]["probe"][r_idx] for l in layers])
        print(f"  H2O   retention = {h_mean:.4f}")
        print(f"  Probe retention = {p_mean:.4f}  (Δ = {p_mean - h_mean:+.4f})")


if __name__ == "__main__":
    main()
