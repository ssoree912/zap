# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch


def compute_oracle_keep_masks(
    teacher_scores: torch.Tensor,
    n_image_keep_per_layer: List[int],
    head_reduce: str = "amax",
) -> np.ndarray:
    """Compute which image tokens oracle would keep, layer by layer.

    Replicates the top-k logic in ImageTokenTopKPress.compress() using oracle
    teacher scores and the same per-layer budget as the probe VizCapture.

    Args:
        teacher_scores: (n_layers, n_heads, n_image) float32 oracle scores.
        n_image_keep_per_layer: Per-layer keep budget from VizCapture.n_image_keep.
        head_reduce: "amax" or "mean" — must match probe press head_reduce.

    Returns:
        (n_layers, n_image) bool array; True = would be kept by oracle.
    """
    n_layers, n_heads, n_image = teacher_scores.shape
    masks: List[torch.Tensor] = []
    for layer_idx in range(n_layers):
        scores = teacher_scores[layer_idx].float()  # (n_heads, n_image)
        if head_reduce == "amax":
            agg = scores.amax(dim=0)
        else:
            agg = scores.mean(dim=0)

        n_keep = int(n_image_keep_per_layer[layer_idx])
        n_keep = min(n_keep, n_image)

        if n_keep <= 0:
            masks.append(torch.zeros(n_image, dtype=torch.bool))
        elif n_keep >= n_image:
            masks.append(torch.ones(n_image, dtype=torch.bool))
        else:
            topk_idx = torch.topk(agg, k=n_keep).indices
            mask = torch.zeros(n_image, dtype=torch.bool)
            mask[topk_idx] = True
            masks.append(mask)

    return torch.stack(masks).numpy()


def compute_overlap_ratios(
    probe_masks: np.ndarray,
    oracle_masks: np.ndarray,
) -> np.ndarray:
    """Per-layer overlap: (probe & oracle).sum() / oracle.sum().

    A ratio of 1.0 means probe selected exactly the same tokens as oracle.

    Args:
        probe_masks:  (n_layers, n_image) bool
        oracle_masks: (n_layers, n_image) bool

    Returns:
        (n_layers,) float32 — NaN for layers where oracle keeps nothing.
    """
    intersection = (probe_masks & oracle_masks).sum(axis=1).astype(np.float32)
    oracle_count = oracle_masks.sum(axis=1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(oracle_count > 0, intersection / oracle_count, np.nan)
    return ratios


def compute_spatial_entropy_per_layer(
    keep_masks: np.ndarray,
    n_image_per_image: int = 576,
    grid_side: int = 24,
) -> np.ndarray:
    """Per-layer Shannon entropy of selected token spatial distribution.

    For each image block (assumed contiguous, n_image_per_image tokens each)
    in the keep_mask, reshape to (grid_side, grid_side), treat token counts as
    a probability distribution, and compute Shannon entropy. Per-layer value is
    the mean entropy across all image blocks.

    A high entropy means selected tokens are spread across the image. A low
    entropy means they cluster in a small region.

    Args:
        keep_masks: (n_layers, n_image_total) bool.
        n_image_per_image: Tokens per image (576 for LLaVA-1.5 336px).
        grid_side: Square grid dimension (24 for 336px / patch-14).

    Returns:
        (n_layers,) float32 — mean entropy in nats per layer.
    """
    n_layers, n_image_total = keep_masks.shape
    n_blocks = max(1, n_image_total // n_image_per_image)
    # Remainder tokens (e.g. if total is not a perfect multiple) are ignored per block loop.

    entropies = np.zeros(n_layers, dtype=np.float32)
    for layer_idx in range(n_layers):
        layer_mask = keep_masks[layer_idx].astype(np.float32)
        block_entropies: List[float] = []
        for b in range(n_blocks):
            start = b * n_image_per_image
            end = start + n_image_per_image
            if end > n_image_total:
                break
            block = layer_mask[start:end]
            total = block.sum()
            if total == 0:
                block_entropies.append(0.0)
                continue
            grid = block.reshape(grid_side, grid_side) / total
            nonzero = grid[grid > 0]
            block_entropies.append(float(-np.sum(nonzero * np.log(nonzero))))

        entropies[layer_idx] = float(np.mean(block_entropies)) if block_entropies else 0.0

    return entropies


def summarize_phase2_sample(
    probe_masks: np.ndarray,
    oracle_masks: np.ndarray,
    head_reduce: str = "amax",
    n_image_per_image: int = 576,
    grid_side: int = 24,
) -> dict:
    """Compute all Phase 2 metrics for a single sample.

    Args:
        probe_masks:  (n_layers, n_image) bool — from VizCapture.
        oracle_masks: (n_layers, n_image) bool — from compute_oracle_keep_masks().

    Returns:
        Dict with keys:
          overlap_per_layer   (n_layers,) float32
          entropy_per_layer   (n_layers,) float32
          mean_overlap        scalar float
          mean_entropy        scalar float
    """
    overlap = compute_overlap_ratios(probe_masks, oracle_masks)
    entropy = compute_spatial_entropy_per_layer(
        probe_masks,
        n_image_per_image=n_image_per_image,
        grid_side=grid_side,
    )
    valid_overlap = overlap[~np.isnan(overlap)]
    return {
        "overlap_per_layer": overlap.tolist(),
        "entropy_per_layer": entropy.tolist(),
        "mean_overlap": float(valid_overlap.mean()) if valid_overlap.size > 0 else float("nan"),
        "mean_entropy": float(entropy.mean()),
    }


def aggregate_phase2_records(records: List[dict]) -> dict:
    """Aggregate per-sample Phase 2 summaries into dataset-level stats.

    Args:
        records: List of dicts from summarize_phase2_sample().

    Returns:
        Dict with mean/std for overlap and entropy across all samples.
    """
    if not records:
        return {}
    overlaps = np.array([r["mean_overlap"] for r in records], dtype=np.float32)
    entropies = np.array([r["mean_entropy"] for r in records], dtype=np.float32)
    valid_overlaps = overlaps[~np.isnan(overlaps)]
    return {
        "n_samples": len(records),
        "overlap_mean": float(valid_overlaps.mean()) if valid_overlaps.size > 0 else float("nan"),
        "overlap_std": float(valid_overlaps.std()) if valid_overlaps.size > 0 else float("nan"),
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
    }
