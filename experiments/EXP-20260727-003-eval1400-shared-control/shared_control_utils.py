# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure tensor helpers for the eval-1400 shared-mask control.

Every score entering this module is layer-wise and head-averaged, with shape
``[L, N_visual]``.  Consequently every returned Top-K mask is also
``[L, N_visual]`` and is intended to be broadcast unchanged to every KV head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


SCORE_NAMES = (
    "all_prefill_shared",
    "question_shared",
    "qvik_shared",
    "future_shared",
)
DEPLOYABLE_SCORE_NAMES = SCORE_NAMES[:3]
PAIRWISE_SCORE_PAIRS = (
    ("all_prefill_shared", "question_shared"),
    ("all_prefill_shared", "future_shared"),
    ("question_shared", "future_shared"),
    ("qvik_shared", "all_prefill_shared"),
    ("qvik_shared", "question_shared"),
    ("qvik_shared", "future_shared"),
)


def normalize_rows(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return finite nonnegative rows that sum to one."""

    if values.ndim != 2:
        raise ValueError(
            f"Expected [L,N] shared scores, got shape={tuple(values.shape)}"
        )
    if values.shape[-1] < 1:
        raise ValueError("Shared score tensor has no visual tokens")
    clean = torch.nan_to_num(
        values.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0)
    sums = clean.sum(dim=-1, keepdim=True)
    normalized = clean / sums.clamp_min(eps)
    empty = sums.squeeze(-1) <= eps
    if empty.any():
        normalized[empty] = 1.0 / clean.shape[-1]
    return normalized


def head_average_scores(
    values: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Average ``[L,H,N]`` scores over heads and normalize each layer."""

    if values.ndim != 3:
        raise ValueError(
            f"{name} must have shape [L,H,N], got {tuple(values.shape)}"
        )
    if values.shape[1] < 1:
        raise ValueError(f"{name} has no attention/KV heads")
    return normalize_rows(values.float().mean(dim=1))


def future_scores_from_attention_blocks(
    attention_blocks: Sequence[Sequence[torch.Tensor]],
    *,
    image_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and aggregate the causal query that predicts each answer token.

    The first generation block may contain a full prompt-query axis, so its
    final row is the last prompt query that predicts ``y_1``. Later blocks
    normally have one query row and predict subsequent answer tokens.

    Returns:
        ``future_shared [L,N]`` and the intermediate time-averaged
        ``future_heads [L,H,N]``.
    """

    if not attention_blocks:
        raise ValueError("Future attention trajectory is empty")
    layer_count = len(attention_blocks[0])
    if layer_count < 1:
        raise ValueError("Future attention trajectory has no layers")
    positions = image_positions.to(dtype=torch.long)
    by_layer: list[list[torch.Tensor]] = [
        [] for _ in range(layer_count)
    ]
    for step_index, step in enumerate(attention_blocks):
        if len(step) != layer_count:
            raise ValueError(
                f"Future step {step_index} has {len(step)} layers; "
                f"expected {layer_count}"
            )
        for layer, attention in enumerate(step):
            if attention.ndim != 4 or attention.shape[0] != 1:
                raise ValueError(
                    "Future attention blocks must have shape [1,H,Q,K], "
                    f"got {tuple(attention.shape)}"
                )
            selected = attention[
                0,
                :,
                -1,
                :,
            ].index_select(
                -1,
                positions.to(attention.device),
            )
            by_layer[layer].append(selected.float().cpu())
    future_heads = torch.stack(
        [
            torch.stack(layer_steps, dim=0).mean(dim=0)
            for layer_steps in by_layer
        ],
        dim=0,
    )
    return (
        head_average_scores(
            future_heads,
            name="raw_full_cache_future",
        ),
        future_heads,
    )


def topk_mask(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return a layer-wise boolean Top-K mask for shared ``[L,N]`` scores."""

    if values.ndim != 2:
        raise ValueError(f"Top-K input must be [L,N], got {tuple(values.shape)}")
    tokens = int(values.shape[-1])
    k = int(k)
    if not 0 < k <= tokens:
        raise ValueError(f"K must be in [1,{tokens}], got {k}")
    indices = torch.topk(values, k=k, dim=-1, largest=True).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def _validate_scores(
    scores: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    missing = sorted(set(SCORE_NAMES) - set(scores))
    extra = sorted(set(scores) - set(SCORE_NAMES))
    if missing or extra:
        raise ValueError(f"Shared score keys mismatch: missing={missing}, extra={extra}")
    normalized = {
        name: normalize_rows(scores[name].detach().cpu())
        for name in SCORE_NAMES
    }
    reference_shape = normalized[SCORE_NAMES[0]].shape
    mismatched = {
        name: tuple(value.shape)
        for name, value in normalized.items()
        if value.shape != reference_shape
    }
    if mismatched:
        raise ValueError(
            f"All shared scores must have shape {tuple(reference_shape)}; "
            f"mismatched={mismatched}"
        )
    return normalized


def pair_key(first: str, second: str) -> str:
    return f"{first}__{second}"


def shared_control_metrics(
    scores: Mapping[str, torch.Tensor],
    *,
    k: int,
) -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, torch.Tensor],
]:
    """Compute exact-budget agreement using only shared layer-wise masks.

    ``future_shared`` is the reference derived from the raw Full-cache answer
    trajectory.  It is not an Oracle decode or an attention trace from the
    second, pruned generation.
    """

    normalized = _validate_scores(scores)
    tokens = int(next(iter(normalized.values())).shape[-1])
    if not 0 < int(k) <= tokens:
        raise ValueError(f"K must be in [1,{tokens}], got {k}")
    masks = {
        f"{name}_keep": topk_mask(values, k)
        for name, values in normalized.items()
    }
    future_mask = masks["future_shared_keep"]
    future_mass = normalized["future_shared"]

    future_agreement: dict[str, dict[str, list[float]]] = {}
    for name in DEPLOYABLE_SCORE_NAMES:
        keep = masks[f"{name}_keep"]
        intersection = (keep & future_mask).sum(dim=-1).float()
        union = (keep | future_mask).sum(dim=-1).float()
        future_agreement[name] = {
            "topk_recall": (intersection / float(k)).tolist(),
            "jaccard": (intersection / union.clamp_min(1)).tolist(),
            "future_mass_retained": (
                future_mass * keep.float()
            ).sum(dim=-1).tolist(),
        }

    pairwise: dict[str, dict[str, list[float]]] = {}
    for first, second in PAIRWISE_SCORE_PAIRS:
        first_mask = masks[f"{first}_keep"]
        second_mask = masks[f"{second}_keep"]
        intersection = (first_mask & second_mask).sum(dim=-1).float()
        union = (first_mask | second_mask).sum(dim=-1).float()
        pairwise[pair_key(first, second)] = {
            "overlap_per_k": (intersection / float(k)).tolist(),
            "jaccard": (intersection / union.clamp_min(1)).tolist(),
        }

    return {
        "future_agreement": future_agreement,
        "pairwise_keep_agreement": pairwise,
    }, masks


def prompt_keep_masks(
    visual_keep: torch.Tensor,
    *,
    image_positions: torch.Tensor,
    prompt_len: int,
) -> dict[int, torch.Tensor]:
    """Embed shared visual masks into one 1-D prompt mask per layer.

    A one-dimensional prompt mask is consumed as ``cache[:, :, mask, :]``.
    There is therefore no head axis on which selectors can disagree.
    """

    if visual_keep.ndim != 2 or visual_keep.dtype != torch.bool:
        raise ValueError(
            "visual_keep must be a boolean [L,N_visual] shared mask"
        )
    positions = image_positions.detach().cpu().to(dtype=torch.long)
    if positions.ndim != 1 or positions.numel() != visual_keep.shape[-1]:
        raise ValueError(
            "image_positions must be a vector matching the visual mask width"
        )
    prompt_len = int(prompt_len)
    if positions.numel() and (
        int(positions.min()) < 0 or int(positions.max()) >= prompt_len
    ):
        raise IndexError("image_positions fall outside the prompt cache")
    if positions.unique().numel() != positions.numel():
        raise ValueError("image_positions must be unique")

    output: dict[int, torch.Tensor] = {}
    for layer, layer_visual in enumerate(visual_keep):
        prompt_mask = torch.ones(prompt_len, dtype=torch.bool)
        prompt_mask[positions] = layer_visual
        output[layer] = prompt_mask
    return output


def broadcast_to_kv_heads(
    visual_keep: torch.Tensor,
    *,
    n_kv_heads: int,
) -> torch.Tensor:
    """Expose the conceptual broadcast for tests and artifact auditing."""

    if visual_keep.ndim != 2 or visual_keep.dtype != torch.bool:
        raise ValueError("visual_keep must be boolean [L,N_visual]")
    n_kv_heads = int(n_kv_heads)
    if n_kv_heads < 1:
        raise ValueError("n_kv_heads must be positive")
    return visual_keep.unsqueeze(1).expand(-1, n_kv_heads, -1)


def validate_visual_masks(
    masks: Mapping[str, torch.Tensor],
    *,
    layers: int,
    n_visual: int,
    n_keep: int,
) -> None:
    """Fail if any persisted control mask is not an exact shared Top-K."""

    expected_keys = {f"{name}_keep" for name in SCORE_NAMES}
    missing = sorted(expected_keys - set(masks))
    extra = sorted(set(masks) - expected_keys)
    if missing or extra:
        raise ValueError(f"Mask keys mismatch: missing={missing}, extra={extra}")
    expected_shape = (int(layers), int(n_visual))
    for name, mask in masks.items():
        if mask.dtype != torch.bool or tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"{name} must be boolean {expected_shape}, got "
                f"dtype={mask.dtype}, shape={tuple(mask.shape)}"
            )
        counts = mask.sum(dim=-1)
        if not torch.equal(
            counts,
            torch.full_like(counts, int(n_keep)),
        ):
            raise ValueError(
                f"{name} is not exact Top-{n_keep}: counts={counts.tolist()}"
            )
