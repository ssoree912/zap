#!/usr/bin/env python3
"""Per-sample heatmap + top-K box visualization for coco caps teacher .pt files.

For each sample, renders 4 panels:
    Original | Prefill saliency (H2O) | Teacher (oracle) | Ours (Q-ViK student)

Scores are layer-averaged, reshaped to a 24×24 patch grid (LLaVA-1.5),
upsampled (nearest) to image space, and shown as a heatmap overlay on the
original image. Top-K patches (K = ceil(keep_ratio · 576)) are drawn as boxes.

Usage:
    python visualize_coco_samples.py \
        --teacher-dir /mnt/srv/home/dlpc.3842/zap/artifacts/eval_teacher_llava15_7b \
        --dataset coco2017_cap_val \
        --student-ckpt /mnt/srv/home/dlpc.3842/zap/artifacts/student_llava15_original_future_1800_lr1e4_15ep \
        --n-samples 5 \
        --keep-ratio 0.2 \
        --out-dir figs_coco_viz
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# LLaVA path
VFLOWOPT_LLAVA_ROOT = Path(os.environ.get(
    "VFLOWOPT_LLAVA_ROOT",
    "/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision",
))
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402

PATCHES_PER_SIDE = 24


# ──────────────────────────────────────────────────────────────────────────────
# Score extraction (mirrors compute_mismatch.py)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_prefill(model, tokenizer, image_processor, rec, device):
    image = Image.open(io.BytesIO(rec["image_bytes"])).convert("RGB")
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        rec["prompt_text"], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)

    out = model(
        input_ids=input_ids, images=image_tensor, image_sizes=[image_size],
        modalities=["image"], use_cache=False,
        output_attentions=True, output_hidden_states=True, return_dict=True,
    )
    return out.attentions, out.hidden_states, image


def h2o_layerwise(attentions, image_indices):
    out = {}
    img_idx = image_indices.long()
    for l, attn in enumerate(attentions):
        a = attn[0].float()
        col_sum = a.sum(dim=1).mean(dim=0)
        out[l] = col_sum.index_select(dim=-1, index=img_idx.to(col_sum.device)).cpu()
    return out


def lookm_layerwise(attentions, image_indices, keep_ratio):
    """LOOK-M: per-(layer, head) top-K. Score per patch = fraction of heads keeping it."""
    out = {}
    img_idx = image_indices.long()
    n_img = int(img_idx.numel())
    K = max(1, int(np.ceil(keep_ratio * n_img)))
    for l, attn in enumerate(attentions):
        a = attn[0].float()
        head_score = a.sum(dim=1).index_select(
            dim=-1, index=img_idx.to(a.device)).cpu()  # [H, N_I]
        H = head_score.size(0)
        keep_count = torch.zeros(n_img)
        for h in range(H):
            top_idx = head_score[h].argsort(descending=True)[:K]
            keep_count[top_idx] += 1
        out[l] = keep_count / float(H)
    return out


def vflowopt_layerwise(attentions, image_indices):
    """VFlowOpt: last query's attention to image keys, mean over heads."""
    out = {}
    img_idx = image_indices.long()
    for l, attn in enumerate(attentions):
        a = attn[0].float()
        last_q = a[:, -1, :].mean(dim=0)
        out[l] = last_q.index_select(dim=-1, index=img_idx.to(last_q.device)).cpu()
    return out


def prefixkv_layerwise(attentions, image_indices):
    """PrefixKV: same as H2O for image positions (prefix/recent protect doesn't apply)."""
    return h2o_layerwise(attentions, image_indices)


def student_layerwise(student, hidden_states, image_indices, question_indices, device):
    out = {}
    img_idx_dev = image_indices.long().to(device)
    q_idx_dev = question_indices.long().to(device)
    for l in student.layer_indices:
        H_l = hidden_states[l + 1].to(device=device, dtype=torch.float16)
        layer_mod = student.layers[str(l)].to(device=device, dtype=torch.float16).eval()
        with torch.no_grad():
            score = layer_mod(H_l, img_idx_dev, q_idx_dev,
                              student.grid_h, student.grid_w)
        out[l] = score.squeeze(0).float().cpu()
    return out


def layer_mean(score_dict: dict[int, torch.Tensor]) -> np.ndarray:
    vecs = [v.numpy() for _, v in sorted(score_dict.items())]
    return np.mean(np.stack(vecs, axis=0), axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def to_probability(v: np.ndarray) -> np.ndarray:
    """Convert raw scores to a sum-1 probability distribution.

    For attention-based scores (already non-negative), this is a sum-normalize.
    For student logits (possibly negative), this acts as a softmax. Either way
    the resulting distribution has the same scale as Teacher (sum=1).
    Mirrors the row-norm / softmax convention used in the deleted
    EXP-20260423-001/viz_future_vs_h2o_timewise.py (commit 4ba88bd).
    """
    v = v.astype(np.float64)
    v = v - v.max()
    p = np.exp(v)
    s = p.sum()
    return (p / s).astype(np.float32) if s > 1e-12 else np.full_like(p, 1.0 / p.size, dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-8)


def draw_panel(ax, image: Image.Image, score_24x24: np.ndarray | None,
               keep_ratio: float, label: str, cmap: str = "hot"):
    W, H = image.size
    ax.imshow(image)
    if score_24x24 is not None:
        norm = normalize(score_24x24)
        ax.imshow(norm, cmap=cmap, alpha=0.55,
                  extent=(0, W, H, 0), interpolation="bilinear", vmin=0, vmax=1)

        flat = score_24x24.ravel()
        K = int(np.ceil(keep_ratio * flat.size))
        top_idx = np.argpartition(-flat, K - 1)[:K]
        px_w = W / PATCHES_PER_SIDE
        px_h = H / PATCHES_PER_SIDE
        for idx in top_idx:
            r, c = divmod(int(idx), PATCHES_PER_SIDE)
            rect = mpatches.Rectangle(
                (c * px_w, r * px_h), px_w, px_h,
                linewidth=2.0, edgecolor="lime", facecolor="none", alpha=0.95,
            )
            ax.add_patch(rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel(label, fontsize=19, labelpad=7)


def render_sample(image: Image.Image, h2o_v: np.ndarray, lookm_v: np.ndarray,
                  vflowopt_v: np.ndarray, prefixkv_v: np.ndarray,
                  teacher_v: np.ndarray, student_v: np.ndarray,
                  keep_ratio: float, out_path: Path, caption: str = ""):
    grid = lambda v: v.reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)

    # Sum=1 probability scale (rank-preserving for top-K).
    h2o_p = to_probability(h2o_v)
    lookm_p = to_probability(lookm_v)
    vflowopt_p = to_probability(vflowopt_v)
    prefixkv_p = to_probability(prefixkv_v)
    teacher_p = to_probability(teacher_v)
    student_p = to_probability(student_v)

    fig, axes = plt.subplots(1, 7, figsize=(28, 4.8))
    axes[0].imshow(image)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_xlabel("(a) Original", fontsize=19, labelpad=7)

    draw_panel(axes[1], image, grid(h2o_p), keep_ratio, "(b) Prefill saliency")
    draw_panel(axes[2], image, grid(lookm_p), keep_ratio, "(c) LOOK-M")
    draw_panel(axes[3], image, grid(vflowopt_p), keep_ratio, "(d) VFlowOpt")
    draw_panel(axes[4], image, grid(prefixkv_p), keep_ratio, "(e) PrefixKV")
    draw_panel(axes[5], image, grid(teacher_p), keep_ratio, "(f) Future teacher")
    draw_panel(axes[6], image, grid(student_p), keep_ratio, "(g) Ours")

    if caption:
        fig.suptitle(caption, fontsize=9, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=120, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-dir", required=True, type=Path)
    p.add_argument("--dataset", default="coco2017_cap_val")
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--sample-ids", nargs="+", default=None,
                   help="If given, only visualize these sample_ids (overrides --n-samples).")
    p.add_argument("--keep-ratio", type=float, default=0.2)
    p.add_argument("--model-path",
                   default="/mnt/srv/home/dlpc.3842/zap/ckpts/llava-v1.5-7b")
    p.add_argument("--model-name", default="llava-v1.5-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[load] {args.model_path}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path, model_base=None, model_name=args.model_name,
        device_map=args.device_map, attn_implementation="eager", multimodal=True,
    )
    model.eval()

    student = VisualUtilityStudent.from_pretrained(args.student_ckpt).to(device).eval()
    print(f"[student] variant={student.variant}", flush=True)

    ds_dir = args.teacher_dir / args.dataset
    if args.sample_ids:
        pts = []
        for sid in args.sample_ids:
            cand = ds_dir / f"{sid}.pt"
            if cand.exists():
                pts.append(cand)
            else:
                print(f"[skip] sample_id {sid}: not found at {cand}", flush=True)
    else:
        pts = sorted(ds_dir.glob("*.pt"))[: args.n_samples]
    print(f"[viz] {args.dataset}  n={len(pts)}  keep_ratio={args.keep_ratio}", flush=True)

    for i, tpath in enumerate(pts, 1):
        rec = torch.load(tpath, map_location="cpu", weights_only=False)
        try:
            attns, hids, image = run_prefill(model, tokenizer, image_processor, rec, device)
        except Exception as exc:
            print(f"[skip] {tpath.stem}: {exc}", flush=True)
            continue

        image_indices = rec["image_token_indices"].long()
        question_indices = rec["question_token_indices"].long()
        teacher_norm = rec["teacher_norm"].float().numpy()  # [L, N_I]

        h2o_per_layer = h2o_layerwise(attns, image_indices)
        stu_per_layer = student_layerwise(student, hids, image_indices, question_indices, device)
        lookm_per_layer = lookm_layerwise(attns, image_indices, args.keep_ratio)
        vflowopt_per_layer = vflowopt_layerwise(attns, image_indices)
        prefixkv_per_layer = prefixkv_layerwise(attns, image_indices)
        del attns, hids
        torch.cuda.empty_cache()

        h2o_mean = layer_mean(h2o_per_layer)
        stu_mean = layer_mean(stu_per_layer)
        lookm_mean = layer_mean(lookm_per_layer)
        vflowopt_mean = layer_mean(vflowopt_per_layer)
        prefixkv_mean = layer_mean(prefixkv_per_layer)
        teacher_mean = teacher_norm.mean(axis=0)

        n_img = teacher_norm.shape[1]
        if n_img != PATCHES_PER_SIDE * PATCHES_PER_SIDE:
            print(f"[skip] {tpath.stem}: n_img={n_img} != 576", flush=True)
            continue

        sid = str(rec["sample_id"])
        question = str(rec.get("question", ""))[:80]
        decoded = str(rec.get("decoded", ""))[:80]
        gt = str(rec.get("answer", ""))[:80]
        caption = (
            f"{args.dataset}  #{sid}  |  Q: {question}\n"
            f"Pred: {decoded}   GT: {gt}"
        )

        out_path = args.out_dir / f"{i:02d}_{Path(sid).stem}_k{args.keep_ratio:g}.png"
        render_sample(image, h2o_mean, lookm_mean, vflowopt_mean, prefixkv_mean,
                      teacher_mean, stu_mean,
                      args.keep_ratio, out_path, caption)
        print(f"[saved] {out_path}", flush=True)

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
