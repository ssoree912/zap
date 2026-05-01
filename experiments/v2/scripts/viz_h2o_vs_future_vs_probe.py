#!/usr/bin/env python3
"""Visualize the gap between H2O prefill-attention scoring vs actual decode
usage (Future oracle), and show our probe approximates the latter.

For one sample this runs a single LLaVA forward pass (eager attention) and
extracts three per-layer scores at the 576 image-patch positions:

  1. H2O        = attn_prefill.sum(dim=Q) at image keys, mean over heads
                  (canonical H2O heavy-hitter; matches H2OImageOnlyPress).
  2. Future     = attn_decode[:, :, image_idx].mean(Q).mean(H)  (oracle).
  3. Probe      = softmax(MLP_l(hidden_image_l))  (our student).

Output:
  {out_dir}/layer{L}_heatmap.png   — 2×2 panel: original | H2O | Future | Probe
  {out_dir}/corr_scatter.png       — H2O vs Future, Probe vs Future (per layer)
  {out_dir}/metrics.json           — Spearman(H2O, Future) / Spearman(Probe, Future)

Usage:
  python scripts/viz_h2o_vs_future_vs_probe.py \\
      --dataset scienceqa --sample-index 0 \\
      --probe-path /workspace/zap/ckpts/future_probe_allL_limit100 \\
      --layers 24 25 26 27 28 29 30 31 \\
      --out-dir /workspace/zap/artifacts/EXP-20260422-001/viz/sqa_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.kvzap_press import KVzapModel
from kvzap.llava_extractor import (
    _move_batch_to_device,
    _resolve_prompt_image_mask,
    configure_llava_processor,
    enable_merge_trace,
    disable_merge_trace,
    register_analysis_hooks,
    remove_analysis_hooks,
)

PATCH = 24  # LLaVA-1.5 CLIP-ViT-L/14 at 336 → 24×24 = 576 patches
N_IMAGE = PATCH * PATCH


DATASET_TEMPLATES = {
    "textvqa": "USER: <image>\nQuestion: {question}\nASSISTANT:",
    "scienceqa": "USER: <image>\n{prompt_body}\nASSISTANT:",
    "nlvr2": (
        "USER: <image>\n<image>\nStatement: {sentence}\n"
        "Is this statement True or False?\nOptions:\nA. True\nB. False\nASSISTANT:"
    ),
}


def load_sample(dataset: str, sample_index: int) -> dict:
    if dataset == "scienceqa":
        import json as _json
        data = _json.load(open("/workspace/zap/data/scienceqa/problems.json"))
        splits = _json.load(open("/workspace/zap/data/scienceqa/pid_splits.json"))
        ids = [sid for sid in splits["train"] if data.get(sid, {}).get("image") == "image.png"]
        sid = ids[sample_index]
        v = data[sid]
        choices = v.get("choices", [])
        choice_lines = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        body = f"{v['question']}\nOptions:\n{choice_lines}"
        split_name = sid.split('_')[0]  # 'train' or 'test' or 'val'
        image_path = f"/workspace/zap/data/scienceqa/images/{split_name}/{sid}/image.png"
        return {
            "sample_id": sid,
            "question": v["question"],
            "prompt_body": body,
            "image_paths": [image_path],
            "prompt_template": DATASET_TEMPLATES["scienceqa"],
        }
    if dataset == "mm-vet":
        import json as _json
        raw = _json.load(open("/workspace/data/mm-vet/mm-vet.json"))
        sid = list(raw.keys())[sample_index]
        v = raw[sid]
        return {
            "sample_id": sid,
            "question": v["question"],
            "image_paths": [f"/workspace/data/mm-vet/images/{v['imagename']}"],
            "prompt_template": "USER: <image>\n{question}\nASSISTANT:",
        }
    if dataset == "wikivqa":
        import json as _json, re as _re
        raw = _json.load(open("/workspace/zap/data/MileBench/WikiVQA/WikiVQA.json"))
        s = raw["data"][sample_index]
        ti = s["task_instance"]
        image_root = "/workspace/zap/data/MileBench/WikiVQA/images"
        image_paths = [f"{image_root}/{p}" for p in ti["images_path"]]
        n_img = len(image_paths)
        # Keep ONLY the question + answer body; drop the long wiki context to fit 4090 memory.
        # This keeps the KV cache scope focused on image tokens and question.
        ctx = ti["context"]
        if "Question:" in ctx:
            question_part = "Question:" + ctx.split("Question:", 1)[1]
        else:
            question_part = ctx[-400:]
        choice_lines = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ti.get("choice_list", [])))
        question_tail = f"\nOptions:\n{choice_lines}\n" if choice_lines else "\n"
        # Prepend n_img <image> markers
        body = "".join(["<image>\n"] * n_img) + question_part + question_tail
        return {
            "sample_id": str(s.get("old_sample_id", s.get("sample_id"))),
            "question": question_part,
            "prompt_body": body,
            "image_paths": image_paths,
            "n_images": n_img,
            "prompt_template": "USER: {prompt_body}ASSISTANT:",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_prompt_text(sample: dict) -> str:
    tpl = sample["prompt_template"]
    if "{prompt_body}" in tpl:
        return tpl.format(prompt_body=sample["prompt_body"])
    return tpl.format(question=sample["question"])


@torch.no_grad()
def run_forward(model, processor, sample):
    images = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
    prompt_text = build_prompt_text(sample)
    prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, dtype)

    # First pass: figure out prompt_len_mm and image positions
    trace_ctx = enable_merge_trace(model)
    try:
        prompt_out = model(**prompt_inputs, use_cache=False,
                           output_attentions=False, output_hidden_states=False, return_dict=True)
    finally:
        disable_merge_trace(trace_ctx)
    prompt_len_mm = int(prompt_out.logits.shape[1])
    is_image = _resolve_prompt_image_mask(model, prompt_inputs["input_ids"], prompt_len_mm, trace_ctx)
    image_idx = is_image.nonzero(as_tuple=False).flatten().long()

    # Generate to get decode tokens
    generated = model.generate(**prompt_inputs, do_sample=False,
                               max_new_tokens=64, use_cache=True)
    full_ids_text = generated[0]

    # Second pass: capture full attention + hidden states over prompt+decode
    analysis_inputs = {
        "input_ids": full_ids_text.unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(full_ids_text).unsqueeze(0),
        "pixel_values": prompt_inputs["pixel_values"],
    }
    hook_ctx = register_analysis_hooks(model, capture_vproj=False)
    try:
        full_out = model(**analysis_inputs, use_cache=False,
                         output_attentions=True, output_hidden_states=True, return_dict=True)
    finally:
        remove_analysis_hooks(hook_ctx)

    full_len = int(full_out.hidden_states[0].shape[1])
    attentions = full_out.attentions      # tuple[L] of [1, H, full_len, full_len]
    hidden_states = full_out.hidden_states  # tuple[L+1] of [1, full_len, D]
    n_layers = len(attentions)

    return {
        "attentions": attentions,
        "hidden_states": hidden_states,
        "prompt_len_mm": prompt_len_mm,
        "full_len": full_len,
        "image_idx": image_idx,
        "n_layers": n_layers,
        "images": images,
    }


def compute_scores(forward, probe: KVzapModel, layers: list[int]):
    """Return dict: layer → {'h2o': [N_image], 'future': [N_image], 'probe': [N_image]}."""
    pm = forward["prompt_len_mm"]
    fl = forward["full_len"]
    img_idx = forward["image_idx"]
    scores = {}
    # Hidden offset (some models prepend embedding)
    hidden_offset = 1 if len(forward["hidden_states"]) == forward["n_layers"] + 1 else 0
    for l in layers:
        attn = forward["attentions"][l][0].float()  # [H, full_len, full_len]
        # H2O: attn_prefill.sum(Q) at image keys, mean over heads
        h2o = attn[:, :pm, :][:, :, img_idx].sum(dim=1).mean(dim=0)  # [N_image]
        # Future: attn_decode.mean(Q).mean(H) at image keys
        future = attn[:, pm:fl, :][:, :, img_idx].mean(dim=0).mean(dim=0)  # [N_image]
        # Probe: softmax(MLP(hidden))
        hid = forward["hidden_states"][l + hidden_offset][0, img_idx].float()  # [N_image, D]
        hid = hid.to(next(probe.parameters()).device)
        with torch.no_grad():
            logits = probe.layers[l](hid).squeeze(-1).cpu()
            pr = torch.softmax(logits, dim=0)
        scores[l] = {
            "h2o": h2o.cpu().numpy(),
            "future": future.cpu().numpy(),
            "probe": pr.detach().numpy(),
        }
    return scores


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    return x / s if s > 0 else x


def render_heatmaps(scores: dict, images, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_img = len(images)
    img_arrs = [np.array(im.resize((336, 336))) for im in images]
    methods = [("H2O (prefill Σattn)", "h2o"),
               ("Future oracle (decode→img)", "future"),
               ("Our probe", "probe")]

    for l, s in scores.items():
        # Layout: rows = n_img (one per image), cols = 4 (original + 3 methods)
        fig, axes = plt.subplots(n_img, 4, figsize=(16, 4 * n_img), squeeze=False)
        for img_i in range(n_img):
            axes[img_i, 0].imshow(img_arrs[img_i])
            axes[img_i, 0].set_title(f"image {img_i+1}")
            axes[img_i, 0].axis("off")
            for col, (name, key) in enumerate(methods, start=1):
                arr_full = s[key]  # [n_img * N_IMAGE]
                arr = arr_full[img_i * N_IMAGE:(img_i + 1) * N_IMAGE]
                grid = normalize(arr).reshape(PATCH, PATCH)
                axes[img_i, col].imshow(img_arrs[img_i], alpha=0.35)
                axes[img_i, col].imshow(grid, cmap="hot", alpha=0.75,
                                        extent=(0, img_arrs[img_i].shape[1], img_arrs[img_i].shape[0], 0))
                axes[img_i, col].set_title(name if img_i == 0 else "")
                axes[img_i, col].axis("off")
        fig.suptitle(f"layer {l}")
        fig.tight_layout()
        path = out_dir / f"layer{l:02d}_heatmap.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path.name}")


def render_scatter(scores: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(scores.keys())
    fig, axes = plt.subplots(2, len(layers), figsize=(2.8*len(layers), 5.5), squeeze=False)
    corr = {}
    for i, l in enumerate(layers):
        s = scores[l]
        fu = s["future"]
        rho_h, _ = spearmanr(s["h2o"], fu)
        rho_p, _ = spearmanr(s["probe"], fu)
        corr[l] = {"spearman_h2o_vs_future": float(rho_h), "spearman_probe_vs_future": float(rho_p)}
        axes[0, i].scatter(s["h2o"], fu, s=4, alpha=0.6, color="crimson")
        axes[0, i].set_title(f"L{l}  H2O vs Fut\nρ={rho_h:.3f}")
        axes[1, i].scatter(s["probe"], fu, s=4, alpha=0.6, color="teal")
        axes[1, i].set_title(f"L{l}  Probe vs Fut\nρ={rho_p:.3f}")
        for row in (0, 1):
            axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
    fig.tight_layout()
    path = out_dir / "corr_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return corr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["scienceqa", "mm-vet", "wikivqa"], default="scienceqa")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--probe-path", type=str,
                   default="/workspace/zap/ckpts/future_probe_allL_limit100")
    p.add_argument("--model-path", type=str,
                   default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    p.add_argument("--layers", type=int, nargs="+", default=[24, 27, 30, 31])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch-dtype", default="float16")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Loading sample: {args.dataset}[{args.sample_index}]")
    sample = load_sample(args.dataset, args.sample_index)
    print(f"  image(s): {sample['image_paths']}")
    print(f"  question: {sample['question'][:120]}")

    print("Running forward pass...")
    forward = run_forward(model, processor, sample)
    n_image_actual = len(forward["image_idx"])
    print(f"  prompt_len_mm={forward['prompt_len_mm']} full_len={forward['full_len']} "
          f"n_image={n_image_actual} n_layers={forward['n_layers']}")
    n_images_expected = sample.get("n_images", 1)
    assert n_image_actual == N_IMAGE * n_images_expected, (
        f"Expected {N_IMAGE * n_images_expected} image tokens for {n_images_expected} images, "
        f"got {n_image_actual}"
    )

    print("Computing scores...")
    scores = compute_scores(forward, probe, args.layers)

    print("Rendering heatmaps...")
    render_heatmaps(scores, forward["images"], out_dir)
    print("Rendering scatter...")
    corr = render_scatter(scores, out_dir)

    metrics = {
        "sample_id": sample.get("sample_id"),
        "dataset": args.dataset,
        "sample_index": args.sample_index,
        "layers": args.layers,
        "spearman_per_layer": corr,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to: {out_dir}")
    print("Spearman summary:")
    for l, c in corr.items():
        print(f"  L{l}: H2O↔Future={c['spearman_h2o_vs_future']:+.3f}  "
              f"Probe↔Future={c['spearman_probe_vs_future']:+.3f}")


if __name__ == "__main__":
    main()
