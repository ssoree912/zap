#!/usr/bin/env python3
"""Render image token eviction visualizations from VizCapture .npz files.

Reads data produced by evaluate_image_teacher_pruning.py --save_viz_for_first_n.

Usage — single method:
  python scripts/render_eviction_visualizations.py \\
      --viz_dir /path/to/viz_data \\
      --out_dir /path/to/figures

Usage — compare two methods (e.g. probe vs oracle):
  python scripts/render_eviction_visualizations.py \\
      --viz_dir /path/to/probe_viz_data \\
      --compare_viz_dir /path/to/oracle_viz_data \\
      --out_dir /path/to/figures \\
      --label_a probe --label_b oracle

Output:
  {out_dir}/per_sample/  — one .png per sample
  {out_dir}/aggregate/   — mean score heatmaps, correlation plots
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── optional heavy imports (graceful message if missing) ───────────────────────

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")


def _require_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")


# ── constants ──────────────────────────────────────────────────────────────────

PATCHES_PER_SIDE = 24      # LLaVA-1.5: 24×24 = 576 patches per image
PATCHES_PER_IMAGE = PATCHES_PER_SIDE ** 2


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_image_blocks(image_positions: np.ndarray) -> List[np.ndarray]:
    """Split flat image_positions into per-image contiguous blocks (same logic as press)."""
    if image_positions.size == 0:
        return []
    blocks: List[np.ndarray] = []
    start = 0
    for i in range(1, len(image_positions)):
        if image_positions[i] != image_positions[i - 1] + 1:
            blocks.append(image_positions[start:i])
            start = i
    blocks.append(image_positions[start:])
    return blocks


def load_sample(npz_path: Path) -> Optional[Dict[str, Any]]:
    """Load a single sample: .npz arrays + .json metadata."""
    meta_path = npz_path.parent / npz_path.name.replace("_scores.npz", "_meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open() as f:
            meta = json.load(f)
        data = dict(np.load(npz_path, allow_pickle=False))
        data["meta"] = meta
        return data
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {npz_path.name}: {e}")
        return None


def find_npz_files(viz_dir: Path) -> List[Path]:
    """Return sorted list of *_scores.npz files in viz_dir."""
    return sorted(viz_dir.glob("*_scores.npz"))


def scores_to_image_heatmaps(
    layer_scores: np.ndarray,
    image_positions: np.ndarray,
    reduce: str = "mean",
) -> List[np.ndarray]:
    """Convert per-layer flat scores to per-image 24×24 heatmaps.

    Args:
        layer_scores: (n_layers, n_image) float32
        image_positions: (n_image,) int64 — flat absolute sequence positions
        reduce: how to aggregate layers — "mean" or "last"

    Returns:
        List of (24, 24) float32 heatmaps, one per image block.
        NaN if the block size ≠ 576.
    """
    # Aggregate layers
    if reduce == "mean":
        agg = layer_scores.mean(axis=0)          # (n_image,)
    elif reduce == "last":
        agg = layer_scores[-1] if layer_scores.shape[0] > 0 else layer_scores[0]
    else:
        agg = layer_scores.mean(axis=0)

    blocks = _find_image_blocks(image_positions)
    heatmaps: List[np.ndarray] = []
    offset = 0
    for block in blocks:
        n = len(block)
        block_scores = agg[offset: offset + n]
        if n == PATCHES_PER_IMAGE:
            hm = block_scores.reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        else:
            # pad/crop to 24×24 with NaN marker so viewer knows
            padded = np.full(PATCHES_PER_IMAGE, float("nan"), dtype=np.float32)
            padded[: min(n, PATCHES_PER_IMAGE)] = block_scores[: PATCHES_PER_IMAGE]
            hm = padded.reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        heatmaps.append(hm)
        offset += n
    return heatmaps


def keep_masks_to_image_masks(
    keep_masks: np.ndarray,
    image_positions: np.ndarray,
    reduce: str = "any",
) -> List[np.ndarray]:
    """Convert per-layer keep masks to per-image 24×24 bool maps.

    Args:
        keep_masks: (n_layers, n_image) bool
        image_positions: (n_image,)
        reduce: "any" (kept by any layer), "all" (kept by all), "mean" (fraction kept)

    Returns:
        List of (24, 24) arrays, one per image block.
    """
    if reduce == "any":
        agg = keep_masks.any(axis=0)    # (n_image,)
    elif reduce == "all":
        agg = keep_masks.all(axis=0)
    else:  # mean
        agg = keep_masks.mean(axis=0).astype(np.float32)

    blocks = _find_image_blocks(image_positions)
    masks: List[np.ndarray] = []
    offset = 0
    for block in blocks:
        n = len(block)
        bm = agg[offset: offset + n]
        if n == PATCHES_PER_IMAGE:
            m = bm.reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        else:
            padded = np.zeros(PATCHES_PER_IMAGE, dtype=agg.dtype)
            padded[: min(n, PATCHES_PER_IMAGE)] = bm[: PATCHES_PER_IMAGE]
            m = padded.reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        masks.append(m)
        offset += n
    return masks


def open_image_for_panel(path: str, size: int = 224):
    """Return a numpy RGB array resized to (size, size, 3)."""
    Image = _require_pil()
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:  # noqa: BLE001
        arr = np.full((size, size, 3), 128, dtype=np.uint8)
        return arr


def normalize_heatmap(h: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], handling NaN."""
    valid = h[~np.isnan(h)]
    if valid.size == 0 or valid.max() == valid.min():
        out = np.zeros_like(h)
    else:
        out = (h - valid.min()) / (valid.max() - valid.min() + 1e-8)
    return out


# ── per-sample figure ──────────────────────────────────────────────────────────

def render_sample_figure(
    sample_a: Dict[str, Any],
    sample_b: Optional[Dict[str, Any]],
    label_a: str,
    label_b: str,
    out_path: Path,
    img_size: int = 168,
    sample_c: Optional[Dict[str, Any]] = None,
    label_c: str = "method_c",
) -> None:
    """Render per-sample visualization figure (2-way or 3-way comparison).

    Layout per image (one row):
      [Original] [Score-A] [KeepMask-A]
        [Score-B (if b)] [KeepMask-B (if b)] [Diff A−B (if b)]
        [Score-C (if c)] [KeepMask-C (if c)] [Diff A−C (if c)]
    Plus a text panel at the bottom.
    """
    plt, gridspec = _require_matplotlib()

    meta = sample_a["meta"]
    layer_scores_a = sample_a["layer_scores"]      # (n_layers, n_image)
    keep_masks_a = sample_a["keep_masks"]          # (n_layers, n_image) bool
    image_positions = sample_a["image_positions"]  # (n_image,)

    n_images = len(_find_image_blocks(image_positions))
    has_b = sample_b is not None
    has_c = sample_c is not None
    # orig, score_a, mask_a, [score_b, mask_b, diff_ab], [score_c, mask_c, diff_ac]
    n_cols = 3 + (3 if has_b else 0) + (3 if has_c else 0)

    score_hm_a = scores_to_image_heatmaps(layer_scores_a, image_positions, reduce="mean")
    mask_hm_a = keep_masks_to_image_masks(keep_masks_a, image_positions, reduce="mean")

    if has_b:
        layer_scores_b = sample_b["layer_scores"]
        keep_masks_b = sample_b["keep_masks"]
        score_hm_b = scores_to_image_heatmaps(layer_scores_b, image_positions, reduce="mean")
        mask_hm_b = keep_masks_to_image_masks(keep_masks_b, image_positions, reduce="mean")

    if has_c:
        layer_scores_c = sample_c["layer_scores"]
        keep_masks_c = sample_c["keep_masks"]
        score_hm_c = scores_to_image_heatmaps(layer_scores_c, image_positions, reduce="mean")
        mask_hm_c = keep_masks_to_image_masks(keep_masks_c, image_positions, reduce="mean")

    image_paths = meta.get("image_paths", [])
    n_img_rows = max(n_images, 1)

    # Build col titles
    col_titles = ["Original", f"Score\n({label_a})", f"Keep\n({label_a})"]
    if has_b:
        col_titles += [f"Score\n({label_b})", f"Keep\n({label_b})", f"Diff\n(A−B)"]
    if has_c:
        col_titles += [f"Score\n({label_c})", f"Keep\n({label_c})", f"Diff\n(A−C)"]

    fig_w = n_cols * (img_size / 72) + 0.5
    fig_h = n_img_rows * (img_size / 72) + 1.8  # +1.8 for text panel + titles

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=96)
    gs = gridspec.GridSpec(
        n_img_rows + 1, n_cols,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
        height_ratios=[1.0] * n_img_rows + [0.28],
    )

    # Column title row at the very top (use first image row axes titles)
    for c, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, c])
        ax.set_title(title, fontsize=7, pad=2)

    for img_i in range(n_img_rows):
        # --- Original image ---
        ax_orig = fig.add_subplot(gs[img_i, 0])
        if img_i < len(image_paths):
            orig_arr = open_image_for_panel(image_paths[img_i], size=img_size)
            ax_orig.imshow(orig_arr)
        else:
            ax_orig.set_facecolor("#cccccc")
        ax_orig.axis("off")
        ax_orig.set_aspect("equal")

        # helper to plot one heatmap
        def _plot_hm(ax, hm: np.ndarray, cmap: str, vmin=None, vmax=None):
            norm_hm = normalize_heatmap(hm) if vmin is None else hm
            im = ax.imshow(norm_hm, cmap=cmap, interpolation="nearest",
                           vmin=vmin, vmax=vmax, aspect="equal")
            ax.axis("off")
            return im

        # --- Score A ---
        ax_score_a = fig.add_subplot(gs[img_i, 1])
        if img_i < len(score_hm_a):
            _plot_hm(ax_score_a, score_hm_a[img_i], "hot")
        else:
            ax_score_a.axis("off")

        # --- Keep mask A ---
        ax_mask_a = fig.add_subplot(gs[img_i, 2])
        if img_i < len(mask_hm_a):
            _plot_hm(ax_mask_a, mask_hm_a[img_i].astype(float), "RdYlGn", vmin=0, vmax=1)
        else:
            ax_mask_a.axis("off")

        if has_b:
            # --- Score B ---
            ax_score_b = fig.add_subplot(gs[img_i, 3])
            if img_i < len(score_hm_b):
                _plot_hm(ax_score_b, score_hm_b[img_i], "hot")
            else:
                ax_score_b.axis("off")

            # --- Keep mask B ---
            ax_mask_b = fig.add_subplot(gs[img_i, 4])
            if img_i < len(mask_hm_b):
                _plot_hm(ax_mask_b, mask_hm_b[img_i].astype(float), "RdYlGn", vmin=0, vmax=1)
            else:
                ax_mask_b.axis("off")

            # --- Score diff (A - B) ---
            ax_diff = fig.add_subplot(gs[img_i, 5])
            if img_i < len(score_hm_a) and img_i < len(score_hm_b):
                norm_a = normalize_heatmap(score_hm_a[img_i])
                norm_b = normalize_heatmap(score_hm_b[img_i])
                diff = norm_a - norm_b
                _plot_hm(ax_diff, diff, "bwr", vmin=-1, vmax=1)
            else:
                ax_diff.axis("off")

        if has_c:
            col_c = 3 + (3 if has_b else 0)
            # --- Score C ---
            ax_score_c = fig.add_subplot(gs[img_i, col_c])
            if img_i < len(score_hm_c):
                _plot_hm(ax_score_c, score_hm_c[img_i], "hot")
            else:
                ax_score_c.axis("off")

            # --- Keep mask C ---
            ax_mask_c = fig.add_subplot(gs[img_i, col_c + 1])
            if img_i < len(mask_hm_c):
                _plot_hm(ax_mask_c, mask_hm_c[img_i].astype(float), "RdYlGn", vmin=0, vmax=1)
            else:
                ax_mask_c.axis("off")

            # --- Score diff (A - C) ---
            ax_diff_c = fig.add_subplot(gs[img_i, col_c + 2])
            if img_i < len(score_hm_a) and img_i < len(score_hm_c):
                norm_a = normalize_heatmap(score_hm_a[img_i])
                norm_c = normalize_heatmap(score_hm_c[img_i])
                diff_c = norm_a - norm_c
                _plot_hm(ax_diff_c, diff_c, "bwr", vmin=-1, vmax=1)
            else:
                ax_diff_c.axis("off")

    # --- Text panel ---
    ax_text = fig.add_subplot(gs[n_img_rows, :])
    ax_text.axis("off")
    n_layers = layer_scores_a.shape[0]
    n_img_tok = layer_scores_a.shape[1]
    keep_ratio_str = f"{meta.get('total_keep_ratio', '?')}"
    question_short = str(meta.get("question", ""))[:120]
    pred_short = str(meta.get("prediction", ""))[:60]
    gold_short = str(meta.get("gold_answer", ""))[:60]
    info = (
        f"Dataset: {meta.get('dataset', '?')}  |  sample_id: {meta.get('sample_id', '?')}  "
        f"|  n_layers: {n_layers}  |  n_img_tokens: {n_img_tok}  |  keep_ratio: {keep_ratio_str}\n"
        f"Q: {question_short}\n"
        f"Pred: {pred_short}   Gold: {gold_short}"
    )
    ax_text.text(0.01, 0.85, info, transform=ax_text.transAxes,
                 fontsize=5.5, va="top", ha="left", family="monospace",
                 wrap=True)

    methods_str = label_a
    if has_b:
        methods_str += f" vs {label_b}"
    if has_c:
        methods_str += f" vs {label_c}"
    fig.suptitle(
        f"{methods_str} — {meta.get('dataset', '?')} #{meta.get('sample_id', '?')}",
        fontsize=8, y=1.0,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=96)
    plt.close(fig)


# ── aggregate figure ───────────────────────────────────────────────────────────

def render_aggregate_figure(
    all_scores_a: List[np.ndarray],
    all_scores_b: Optional[List[np.ndarray]],
    label_a: str,
    label_b: str,
    out_path: Path,
    patches_per_side: int = PATCHES_PER_SIDE,
) -> None:
    """Render dataset-level aggregate: mean score heatmap + correlation histogram.

    all_scores_a / all_scores_b: each element is (n_image_for_one_image_in_one_sample,)
    averaged over layers.  Only 576-patch images are included.
    """
    plt, gridspec = _require_matplotlib()

    def _mean_heatmap(score_list):
        valid = [s for s in score_list if s.size == patches_per_side ** 2]
        if not valid:
            return None
        stacked = np.stack([v.reshape(patches_per_side, patches_per_side) for v in valid])
        # normalize each sample's heatmap before averaging
        mins = stacked.min(axis=(1, 2), keepdims=True)
        maxs = stacked.max(axis=(1, 2), keepdims=True)
        normed = (stacked - mins) / (maxs - mins + 1e-8)
        return normed.mean(axis=0)

    mean_hm_a = _mean_heatmap(all_scores_a)

    has_b = all_scores_b is not None
    mean_hm_b = _mean_heatmap(all_scores_b) if has_b else None

    n_cols = 2 + (3 if has_b else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.2), dpi=96)
    if n_cols == 1:
        axes = [axes]

    col = 0
    # Mean score heatmap A
    if mean_hm_a is not None:
        axes[col].imshow(mean_hm_a, cmap="hot", aspect="equal", interpolation="nearest")
        axes[col].set_title(f"Mean score\n({label_a})", fontsize=8)
    else:
        axes[col].text(0.5, 0.5, "no data", ha="center", va="center")
        axes[col].set_title(f"Mean score\n({label_a})", fontsize=8)
    axes[col].axis("off")
    col += 1

    # Spatial score variance A
    if mean_hm_a is not None:
        valid_a = [s for s in all_scores_a if s.size == patches_per_side ** 2]
        stacked_a = np.stack([v.reshape(patches_per_side, patches_per_side) for v in valid_a])
        var_map = stacked_a.std(axis=0)
        axes[col].imshow(var_map, cmap="Blues", aspect="equal", interpolation="nearest")
        axes[col].set_title(f"Score Std\n({label_a})", fontsize=8)
    else:
        axes[col].axis("off")
    axes[col].axis("off")
    col += 1

    if has_b and mean_hm_b is not None:
        axes[col].imshow(mean_hm_b, cmap="hot", aspect="equal", interpolation="nearest")
        axes[col].set_title(f"Mean score\n({label_b})", fontsize=8)
        axes[col].axis("off")
        col += 1

        # Diff heatmap
        axes[col].imshow(mean_hm_a - mean_hm_b, cmap="bwr", vmin=-0.5, vmax=0.5,
                          aspect="equal", interpolation="nearest")
        axes[col].set_title(f"Mean diff\n(A−B)", fontsize=8)
        axes[col].axis("off")
        col += 1

        # Correlation histogram
        valid_a2 = [s for s in all_scores_a if s.size == patches_per_side ** 2]
        valid_b2 = [s for s in all_scores_b if s.size == patches_per_side ** 2]
        n = min(len(valid_a2), len(valid_b2))
        if n > 0:
            corrs = []
            for i in range(n):
                a_ = valid_a2[i].flatten()
                b_ = valid_b2[i].flatten()
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                b_ = (b_ - b_.mean()) / (b_.std() + 1e-8)
                corrs.append(float(np.corrcoef(a_, b_)[0, 1]))
            axes[col].hist(corrs, bins=20, color="steelblue", edgecolor="white")
            axes[col].set_xlabel(f"Pearson r ({label_a} vs {label_b})", fontsize=7)
            axes[col].set_ylabel("Samples", fontsize=7)
            axes[col].set_title(
                f"Score correlation\nmean r={np.mean(corrs):.3f}", fontsize=8
            )
            axes[col].tick_params(labelsize=6)
        col += 1

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viz_dir", required=True, type=str,
        help="Directory with *_scores.npz + *_meta.json files (method A).",
    )
    parser.add_argument(
        "--compare_viz_dir", default=None, type=str,
        help="Optional second directory (method B) for difference maps.",
    )
    parser.add_argument(
        "--compare_viz_dir_2", default=None, type=str,
        help="Optional third directory (method C) for 3-way comparison.",
    )
    parser.add_argument(
        "--out_dir", required=True, type=str,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--label_a", default="method_a", type=str,
        help="Label for the primary method.",
    )
    parser.add_argument(
        "--label_b", default="method_b", type=str,
        help="Label for the comparison method (B).",
    )
    parser.add_argument(
        "--label_c", default="method_c", type=str,
        help="Label for the third comparison method (C).",
    )
    parser.add_argument(
        "--img_size", default=168, type=int,
        help="Image panel pixel size (default 168).",
    )
    parser.add_argument(
        "--limit", default=None, type=int,
        help="Render only the first N samples (default: all).",
    )
    args = parser.parse_args()

    viz_dir = Path(args.viz_dir)
    out_dir = Path(args.out_dir)
    per_sample_dir = out_dir / "per_sample"
    aggregate_dir = out_dir / "aggregate"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    npz_files_a = find_npz_files(viz_dir)
    if not npz_files_a:
        print(f"No *_scores.npz files found in {viz_dir}")
        return

    has_b = args.compare_viz_dir is not None
    has_c = args.compare_viz_dir_2 is not None
    compare_viz_dir = Path(args.compare_viz_dir) if has_b else None
    compare_viz_dir_2 = Path(args.compare_viz_dir_2) if has_c else None

    if args.limit is not None:
        npz_files_a = npz_files_a[: args.limit]

    print(f"Rendering {len(npz_files_a)} samples …")

    all_layer_means_a: List[np.ndarray] = []
    all_layer_means_b: List[np.ndarray] = []
    all_layer_means_c: List[np.ndarray] = []

    for i, npz_a in enumerate(npz_files_a):
        sample_a = load_sample(npz_a)
        if sample_a is None:
            continue

        sample_b = None
        if has_b:
            npz_b = compare_viz_dir / npz_a.name
            if npz_b.exists():
                sample_b = load_sample(npz_b)

        sample_c = None
        if has_c:
            npz_c = compare_viz_dir_2 / npz_a.name
            if npz_c.exists():
                sample_c = load_sample(npz_c)

        meta = sample_a["meta"]
        stem = f"{i:04d}_{meta.get('sample_id', i)}"
        out_path = per_sample_dir / f"{stem}.png"

        try:
            render_sample_figure(
                sample_a=sample_a,
                sample_b=sample_b,
                label_a=args.label_a,
                label_b=args.label_b,
                out_path=out_path,
                img_size=args.img_size,
                sample_c=sample_c,
                label_c=args.label_c,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [error] sample {i}: {e}")
            continue

        # Collect per-image layer-mean scores for aggregate
        image_positions = sample_a["image_positions"]
        blocks = _find_image_blocks(image_positions)
        layer_mean_a = sample_a["layer_scores"].mean(axis=0)  # (n_image,)
        offset = 0
        for block in blocks:
            n = len(block)
            all_layer_means_a.append(layer_mean_a[offset: offset + n])
            offset += n

        if sample_b is not None:
            layer_mean_b = sample_b["layer_scores"].mean(axis=0)
            offset = 0
            for block in blocks:
                n = len(block)
                all_layer_means_b.append(layer_mean_b[offset: offset + n])
                offset += n

        if sample_c is not None:
            layer_mean_c = sample_c["layer_scores"].mean(axis=0)
            offset = 0
            for block in blocks:
                n = len(block)
                all_layer_means_c.append(layer_mean_c[offset: offset + n])
                offset += n

        if (i + 1) % 10 == 0 or i == len(npz_files_a) - 1:
            print(f"  {i + 1}/{len(npz_files_a)} samples done")

    # Aggregate figures
    print("Rendering aggregate figure …")
    render_aggregate_figure(
        all_scores_a=all_layer_means_a,
        all_scores_b=all_layer_means_b if (has_b and all_layer_means_b) else None,
        label_a=args.label_a,
        label_b=args.label_b,
        out_path=aggregate_dir / "mean_heatmaps_ab.png",
    )
    if has_c and all_layer_means_c:
        render_aggregate_figure(
            all_scores_a=all_layer_means_a,
            all_scores_b=all_layer_means_c,
            label_a=args.label_a,
            label_b=args.label_c,
            out_path=aggregate_dir / "mean_heatmaps_ac.png",
        )

    print(f"Done.  Per-sample: {per_sample_dir}  Aggregate: {aggregate_dir}")


if __name__ == "__main__":
    main()
