#!/usr/bin/env python3
"""4-panel Figure 1: prefill saliency vs future utility mismatch.

Panels:
  (a) original image + question + GT answer
  (b) prefill attention heatmap  (question→image, all layers averaged)
  (c) future teacher heatmap     (decode→image, all layers averaged)
  (d) student predicted score    (VisualUtilityStudentPress forward, all layers)

Usage:
  python make_figure1.py \
      --pt   /workspace/zap/artifacts/EXP-20260426-001-figure/attn_dump/DocVQA/42.pt \
      --out  /workspace/zap/artifacts/EXP-20260426-001-figure/figure1_DocVQA_42.png

  # With student (panel d):
  python make_figure1.py \
      --pt   ...42.pt \
      --ckpt /workspace/zap/ckpts/student_v2_A_gqa_lr1e4 \
      --out  figure1_DocVQA_42.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/zap")

GRID_H, GRID_W = 24, 24
CMAP = "turbo"
ALPHA = 0.45
TOPK_FRAC = 0.15          # fraction of tokens to highlight with red outline
DPI = 150


# ── score helpers ──────────────────────────────────────────────────────────────

def avg_layers(scores: torch.Tensor) -> np.ndarray:
    """[L, N_I] float → [N_I] float32 numpy, mean over L."""
    return scores.float().mean(dim=0).numpy()


def sum_normalize_layers(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each layer over image tokens to match teacher-label semantics."""
    scores = scores.float().clamp_min(0)
    denom = scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    return scores / denom


def to_grid(vec: np.ndarray, h: int = GRID_H, w: int = GRID_W,
            pct: tuple[float, float] = (2, 98), rank_norm: bool = False) -> np.ndarray:
    """[N_I] → [H, W], normalised to [0, 1].

    rank_norm=True: convert to rank-based uniform [0,1] (best for skewed logit distributions).
    rank_norm=False: percentile-clip then min-max normalise.
    """
    g = vec[:h * w].copy()
    if rank_norm:
        order = np.argsort(g)
        g_out = np.empty_like(g, dtype=np.float32)
        g_out[order] = np.linspace(0.0, 1.0, len(g), dtype=np.float32)
        g = g_out
    else:
        lo, hi = np.percentile(g, pct)
        g = np.clip(g, lo, hi)
        g = g - g.min()
        if g.max() > 0:
            g = g / g.max()
    return g.reshape(h, w).astype(np.float32)


def topk_mask_grid(vec: np.ndarray, frac: float, h: int, w: int) -> np.ndarray:
    """Return boolean [H, W] True at top-frac positions."""
    k = max(1, int(len(vec) * frac))
    idx = np.argsort(vec)[-k:]
    mask = np.zeros(h * w, dtype=bool)
    mask[idx] = True
    return mask.reshape(h, w)


# ── student forward (optional panel d) ────────────────────────────────────────

def run_student(
    ckpt: str,
    data: dict,
    device: torch.device,
    layer_filter: set[int] | None = None,
) -> np.ndarray | None:
    """Run VisualUtilityStudent and return mean per-layer softmax probabilities [N_I]."""
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        from kvpress.presses.visual_utility_student import VisualUtilityStudent

        student = VisualUtilityStudent.from_pretrained(ckpt).to(device).eval()

        model_path = "/workspace/zap/ckpts/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=str(device)
        )
        model.eval()

        image = Image.open(data["image_path"]).convert("RGB")
        from foresight.llava_extractor import infer_llava_image_positions_no_forward
        prompt = f"USER: <image>\n{data['question']}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
            prompt_inputs=inputs, model_config=model.config, num_images=1,
        )
        img_idx = image_positions.to(device)
        last_img = int(image_positions.max().item())
        q_idx = (
            torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long, device=device)
            if last_img + 1 < prompt_len_mm
            else torch.empty(0, dtype=torch.long, device=device)
        )

        # Run prefill forward to get hidden states
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states  # tuple of L+1 tensors [1, N, D]
        layer_indices = [
            li for li in student.layer_indices
            if layer_filter is None or li in layer_filter
        ]
        if not layer_indices:
            raise ValueError("No student layers remain after applying --layers filter")
        all_scores = []

        with torch.no_grad():
            for li in layer_indices:
                H_l = hidden_states[li + 1].to(device=device, dtype=torch.float16)
                layer_mod = student.layers[str(li)].to(device=device, dtype=torch.float16).eval()
                score = layer_mod(H_l, img_idx, q_idx, GRID_H, GRID_W)  # [1, N_I]
                all_scores.append(score.squeeze(0).float().cpu())

        stacked = torch.stack(all_scores, dim=0)  # [n_layers, N_I]
        probs = torch.softmax(stacked.float(), dim=-1)
        return avg_layers(probs)
    except Exception as e:
        print(f"[WARN] student forward failed: {e}")
        return None


# ── panel drawing ──────────────────────────────────────────────────────────────

def draw_heatmap_panel(
    ax: plt.Axes,
    image: Image.Image,
    score_grid: np.ndarray,  # [H, W] normalised 0-1
    title: str,
    topk_frac: float = TOPK_FRAC,
    show_heatmap: bool = True,
    fill_color: str | None = None,   # e.g. "green", "yellow", "red" — filled semi-transparent boxes
    fill_alpha: float = 0.45,
    dim_bg: float = 0.4,             # darken background when fill_color is set (0=black, 1=original)
    box_scale: float = 1.0,          # scale box size relative to one patch cell
    raw_vec: np.ndarray | None = None,  # original scores for mask selection (bypasses clip artifact)
) -> None:
    W_img, H_img = image.size
    img_arr = np.array(image).astype(np.float32) * dim_bg 
    ax.imshow(img_arr.clip(0, 255).astype(np.uint8), extent=[0, W_img, H_img, 0])
    # Optionally darken background
    if fill_color is not None:
        img_arr = np.array(image).astype(np.float32) * dim_bg
        ax.imshow(img_arr.clip(0, 255).astype(np.uint8), extent=[0, W_img, H_img, 0])
    else:
        ax.imshow(image, extent=[0, W_img, H_img, 0])

    # heatmap overlay
    if show_heatmap:
        ax.imshow(
            score_grid,
            cmap=CMAP,
            alpha=ALPHA,
            extent=[0, W_img, H_img, 0],
            vmin=0.0, vmax=1.0,
            interpolation="bilinear",
        )

    # top-k box overlay — use raw_vec for mask if given (avoids clip-flatten ordering issue)
    mask_src = raw_vec[:GRID_H * GRID_W] if raw_vec is not None else score_grid.ravel()
    mask = topk_mask_grid(mask_src, topk_frac, GRID_H, GRID_W)
    cell_w = W_img / GRID_W
    cell_h = H_img / GRID_H
    bw = cell_w * box_scale
    bh = cell_h * box_scale
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            if mask[gy, gx]:
                # center the scaled box on the cell center
                cx = (gx + 0.5) * cell_w - bw / 2
                cy = (gy + 0.5) * cell_h - bh / 2
                if fill_color is not None:
                    rect = mpatches.Rectangle(
                        (cx, cy), bw, bh,
                        linewidth=0, edgecolor="none",
                        facecolor=fill_color, alpha=fill_alpha,
                    )
                else:
                    rect = mpatches.Rectangle(
                        (cx, cy), bw, bh,
                        linewidth=2, edgecolor="red", facecolor="none",
                    )
                ax.add_patch(rect)

    ax.set_title(title, fontsize=10, pad=4)
    ax.axis("off")


def apply_multi_color_overlay(
    image: Image.Image,
    layers: list[tuple[np.ndarray, tuple[float, float, float], np.ndarray | None]],  # (score_grid, rgb, raw_vec)
    topk_frac: float,
    box_scale: float = 1.5,
    fill_alpha: float = 0.55,
    dim_bg: float = 0.55,
) -> np.ndarray:
    """Pixel-level multi-color overlay with uniform alpha (no stacking artifact).

    Each top-k cell is painted with the layer's color. Overlapping cells show
    the average RGB of all overlapping colors — all at the same fixed alpha.
    """
    W_img, H_img = image.size
    cell_w = W_img / GRID_W
    cell_h = H_img / GRID_H
    bw = cell_w * box_scale
    bh = cell_h * box_scale

    bg = np.array(image).astype(np.float32)
    color_acc = np.zeros((H_img, W_img, 3), dtype=np.float32)
    count_map = np.zeros((H_img, W_img), dtype=np.int32)

    for score_grid, color_rgb, raw_vec in layers:
        mask_src = raw_vec[:GRID_H * GRID_W] if raw_vec is not None else score_grid.ravel()
        mask = topk_mask_grid(mask_src, topk_frac, GRID_H, GRID_W)
        c = np.array(color_rgb, dtype=np.float32) * 255.0
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                if mask[gy, gx]:
                    x1 = int(max(0, (gx + 0.5) * cell_w - bw / 2))
                    y1 = int(max(0, (gy + 0.5) * cell_h - bh / 2))
                    x2 = int(min(W_img, x1 + bw))
                    y2 = int(min(H_img, y1 + bh))
                    color_acc[y1:y2, x1:x2] += c
                    count_map[y1:y2, x1:x2] += 1

    has_color = count_map > 0
    avg_color = np.where(
        count_map[..., None] > 0,
        color_acc / np.maximum(count_map[..., None], 1),
        0.0,
    )

    # Darken background uniformly first
    result = bg * dim_bg
    # Blend colored cells at fixed alpha (no stacking)
    result[has_color] = (1 - fill_alpha) * bg[has_color] + fill_alpha * avg_color[has_color]
    return result.clip(0, 255).astype(np.uint8)


def draw_blend_panel(
    ax: plt.Axes,
    image: Image.Image,
    layers: list[tuple[np.ndarray, tuple[float, float, float], str, np.ndarray | None]],  # (grid, rgb, label, raw_vec)
    title: str,
    topk_frac: float,
    box_scale: float = 1.5,
    fill_alpha: float = 0.55,
    dim_bg: float = 0.55,
) -> None:
    layer_data = [(g, c, rv) for g, c, _, rv in layers]
    arr = apply_multi_color_overlay(image, layer_data, topk_frac, box_scale, fill_alpha, dim_bg)
    ax.imshow(arr)
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis("off")

    # legend patches
    handles = [mpatches.Patch(color=c, label=lbl, alpha=0.8) for _, c, lbl, _ in layers]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.7)


def draw_image_panel(
    ax: plt.Axes,
    image: Image.Image,
    question: str,
    generated_answer: str,
    gt_answer: str = "",
) -> None:
    ax.imshow(image)
    ax.axis("off")
    caption = f"Q: {question}\nModel: {generated_answer}"
    if gt_answer:
        caption += f"\nGT: {gt_answer}"
    ax.set_title("(a) Image + Q + Answer", fontsize=10, pad=4)
    ax.text(
        0.5, -0.02, caption,
        transform=ax.transAxes, fontsize=7,
        ha="center", va="top", wrap=True,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt",   required=True, help="Path to .pt dump from collect_attn_dump.py")
    ap.add_argument("--out",  required=True, help="Output PNG path")
    ap.add_argument("--ckpt", default="/workspace/zap/ckpts/student_v2_A_gqa_lr1e4",
                    help="Student ckpt dir for panel (d). Default: student_v2_A_gqa_lr1e4")
    ap.add_argument("--no-student", action="store_true",
                    help="Skip panel (d) — useful if running without GPU or ckpt")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="Show only top-k boxes on raw image, no colormap overlay")
    ap.add_argument("--fill-color", default=None,
                    help="Fill top-k boxes with semi-transparent color (e.g. green/yellow/red). Implies --no-heatmap + darkened bg.")
    ap.add_argument("--fill-alpha", type=float, default=0.45,
                    help="Opacity of filled boxes (default 0.45)")
    ap.add_argument("--dim-bg", type=float, default=0.5,
                    help="Background brightness when --fill-color set (0=black, 1=original, default 0.5)")
    ap.add_argument("--box-scale", type=float, default=1.0,
                    help="Scale factor for box size relative to one patch cell (default 1.0)")
    ap.add_argument("--blend", action="store_true",
                    help="3-panel blend mode: (a) original, (b) prefill+teacher, (c) student+teacher")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk-frac", type=float, default=TOPK_FRAC,
                    help="Fraction of tokens highlighted with red outline")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Layer subset to average (default: all). E.g. --layers 24 25 26 27 28 29 30 31")
    args = ap.parse_args()

    device = torch.device(args.device)
    data = torch.load(args.pt, map_location="cpu", weights_only=False)
    image = Image.open(data["image_path"]).convert("RGB")

    prefill = data["prefill"].float()  # [L, N_I]
    future  = data["future"].float()   # [L, N_I]

    layer_filter = set(args.layers) if args.layers else None
    if args.layers:
        l_idx = torch.tensor(args.layers)
        prefill = prefill[l_idx]
        future  = future[l_idx]

    pref_vec = avg_layers(sum_normalize_layers(prefill))
    fut_vec  = avg_layers(sum_normalize_layers(future))
    pref_grid = to_grid(pref_vec)
    fut_grid  = to_grid(fut_vec)

    show_student = args.ckpt and not args.no_student
    student_vec = None
    if show_student or args.blend:
        student_vec = run_student(args.ckpt, data, device, layer_filter=layer_filter)

    N_I = len(pref_vec)
    topk_pct = args.topk_frac * 100
    topk_pct_str = f"{topk_pct:.1f}".rstrip("0").rstrip(".")
    k_top = max(1, int(N_I * args.topk_frac))
    fut_set = set(np.argsort(fut_vec)[-k_top:].tolist())
    ov_pf = len(set(np.argsort(pref_vec)[-k_top:].tolist()) & fut_set) / k_top
    title = f"Overlap@{topk_pct_str}%(pf-fut)={ov_pf:.2f}"
    if student_vec is not None:
        ov_sf = len(set(np.argsort(student_vec)[-k_top:].tolist()) & fut_set) / k_top
        title += f"  Overlap@{topk_pct_str}%(tea-stu)={ov_sf:.2f}"

    # ── blend mode: 3-panel multi-color overlay ───────────────────────────────
    if args.blend:
        if student_vec is not None:
            stud_grid = to_grid(student_vec)
        else:
            stud_grid = None
        BLUE  = (0.20, 0.50, 1.00)
        RED   = (1.00, 0.20, 0.20)
        GREEN = (0.10, 0.85, 0.30)
        bkw = dict(topk_frac=args.topk_frac, box_scale=args.box_scale,
                   fill_alpha=args.fill_alpha, dim_bg=args.dim_bg)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        draw_image_panel(
            axes[0], image,
            data["question"][:80],
            data.get("generated_answer", data.get("answer", ""))[:60],
            data.get("gt_answer", "")[:40],
        )
        draw_blend_panel(axes[1], image,
                         [(pref_grid, BLUE, "Prefill", pref_vec),
                          (fut_grid,  RED,  "Teacher", fut_vec)],
                         "(b) Prefill vs Teacher", **bkw)
        if stud_grid is not None:
            draw_blend_panel(axes[2], image,
                             [(stud_grid, GREEN, "Student", student_vec),
                              (fut_grid,  RED,   "Teacher", fut_vec)],
                             "(c) Student vs Teacher", **bkw)
        else:
            axes[2].text(0.5, 0.5, "Student failed", ha="center", va="center")
            axes[2].axis("off")

        fig.suptitle(title, fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"[saved] {out_path}  ({title})")
        return

    # ── standard 4-panel mode ─────────────────────────────────────────────────
    n_panels = 4 if show_student else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))

    draw_image_panel(
        axes[0], image,
        data["question"][:80],
        data.get("generated_answer", data.get("answer", ""))[:60],
        data.get("gt_answer", "")[:40],
    )
    fc = args.fill_color
    show_hm = not args.no_heatmap and fc is None
    kw = dict(show_heatmap=show_hm, fill_color=fc,
              fill_alpha=args.fill_alpha, dim_bg=args.dim_bg, box_scale=args.box_scale)
    draw_heatmap_panel(axes[1], image, pref_grid,
                       "(b) Prefill saliency\n(all layers)", args.topk_frac, **kw)
    draw_heatmap_panel(axes[2], image, fut_grid,
                       "(c) Future teacher\n(decode→image, all layers)", args.topk_frac, **kw)

    if show_student:
        if student_vec is not None:
            stud_grid = to_grid(student_vec)

            draw_heatmap_panel(axes[3], image, stud_grid,
                               "(d) Student predicted\n(our method)", args.topk_frac,
                               raw_vec=student_vec, **kw)
        else:
            axes[3].text(0.5, 0.5, "Student failed", ha="center", va="center")
            axes[3].axis("off")

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"[saved] {out_path}  ({title})")


if __name__ == "__main__":
    main()
