# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.foresight_press import KVzapModel


HeadReduce = Literal["amax", "mean"]


@dataclass
class VizCapture:
    """Lightweight per-layer score/mask logger for image token presses.

    Attach to an ImageTokenTopKPress via ``attach_viz_capture()`` before calling
    ``model.generate()``.  After generation, read ``.layer_scores`` and
    ``.keep_masks``.  Detach with ``detach_viz_capture()`` when done, or reset
    between samples with ``.reset()``.

    All tensors are stored on CPU (float32 / bool) to avoid GPU memory pressure.
    Visualization semantics are zero-overhead when not attached — the press checks
    ``self._viz_capture is not None`` before any recording.
    """

    layer_scores: List[torch.Tensor] = field(default_factory=list)
    """Per-layer amax-reduced image token scores: list of (n_image,) float32 CPU tensors."""
    keep_masks: List[torch.Tensor] = field(default_factory=list)
    """Per-layer keep/evict boolean masks: list of (n_image,) bool CPU tensors.
    True = kept in KV cache.  Union across all KV heads."""
    n_image_keep: List[int] = field(default_factory=list)
    """Number of image tokens kept per layer (after forced-keep deduction)."""

    def reset(self) -> None:
        """Clear all recorded data (call between samples)."""
        self.layer_scores.clear()
        self.keep_masks.clear()
        self.n_image_keep.clear()

    def record(
        self,
        *,
        agg_score: torch.Tensor,
        keep_mask: torch.Tensor,
        n_keep: int,
    ) -> None:
        """Record one layer's scores and keep decisions.

        Args:
            agg_score: (n_image,) float32 CPU tensor — amax across kv heads.
            keep_mask: (n_image,) bool CPU tensor — True if the token is kept
                (union across all kv heads).
            n_keep: total image tokens kept this layer.
        """
        self.layer_scores.append(agg_score)
        self.keep_masks.append(keep_mask)
        self.n_image_keep.append(n_keep)

    def to_arrays(self) -> Dict[str, Any]:
        """Return numpy arrays suitable for ``np.savez_compressed``.

        Returns a dict with keys:
            - ``layer_scores``: (n_layers, n_image) float32
            - ``keep_masks``:   (n_layers, n_image) bool
            - ``n_image_keep``: (n_layers,) int32
        """
        import numpy as np

        if not self.layer_scores:
            return {
                "layer_scores": np.zeros((0, 0), dtype=np.float32),
                "keep_masks": np.zeros((0, 0), dtype=bool),
                "n_image_keep": np.zeros(0, dtype=np.int32),
            }
        scores = torch.stack(self.layer_scores).numpy().astype(np.float32)
        masks = torch.stack(self.keep_masks).numpy()
        return {
            "layer_scores": scores,
            "keep_masks": masks,
            "n_image_keep": np.array(self.n_image_keep, dtype=np.int32),
        }


def _get_num_heads(module: nn.Module) -> int:
    return int(getattr(module, "num_heads", module.config.num_attention_heads))


def _get_num_kv_heads(module: nn.Module) -> int:
    return int(getattr(module, "num_key_value_heads", getattr(module.config, "num_key_value_heads", _get_num_heads(module))))


def _aggregate_scores_to_kv_heads(scores: torch.Tensor, module: nn.Module, reduce: HeadReduce = "amax") -> torch.Tensor:
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
    if scores.dim() != 3:
        raise ValueError(f"Expected [B, H, T] scores, got {tuple(scores.shape)}")

    num_score_heads = int(scores.shape[1])
    num_kv_heads = _get_num_kv_heads(module)
    if num_score_heads == num_kv_heads:
        return scores

    if num_score_heads == 1 and num_kv_heads > 1:
        return scores.expand(scores.shape[0], num_kv_heads, scores.shape[-1])

    if num_score_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot aggregate {num_score_heads} score heads into {num_kv_heads} kv heads for layer {module.layer_idx}"
        )

    n_groups = num_score_heads // num_kv_heads
    scores = scores.view(scores.shape[0], num_kv_heads, n_groups, scores.shape[-1])
    if reduce == "amax":
        return scores.amax(dim=2)
    if reduce == "mean":
        return scores.mean(dim=2)
    raise ValueError(f"Unsupported head reduction: {reduce}")


def _compute_n_image_keep(
    n_image: int,
    n_text: int,
    image_keep_ratio: Optional[float],
    total_keep_ratio: Optional[float],
) -> int:
    """Compute how many image tokens to keep.

    If ``total_keep_ratio`` is given, it is treated as the fraction of ALL tokens
    (text + image) to retain. Text tokens are always kept, so:
        n_image_keep = ceil(total_keep_ratio * (n_text + n_image)) - n_text

    If only ``image_keep_ratio`` is given (legacy), it is the fraction of image
    tokens to retain directly.
    """
    if total_keep_ratio is not None:
        total_tokens = n_text + n_image
        total_keep = int(math.ceil(total_keep_ratio * total_tokens))
        n_image_keep = total_keep - n_text
    else:
        assert image_keep_ratio is not None
        n_image_keep = int(math.ceil(n_image * image_keep_ratio))

    return min(n_image, max(0, n_image_keep))


def _iterative_topk(
    scores: torch.Tensor,
    final_k: int,
    n_rounds: int = 4,
) -> torch.Tensor:
    """Iterative top-k selection: narrow the candidate pool over n_rounds before final per-head top-k.

    Rounds 1 .. (n_rounds-1) use amax-across-heads to decide which tokens survive to the next
    pool. The final round uses per-head top-k so the returned indices are head-specific.

    The keep schedule is linear from n_free down to final_k:
        round_i keeps ceil(n_free - (n_free - final_k) * i / n_rounds) tokens.

    Falls back to one-shot top-k when n_rounds <= 1 or final_k >= n_free.

    Args:
        scores:  (num_kv_heads, n_free) float — scores for the free (non-forced) image tokens.
        final_k: Number of tokens to keep after the last round.
        n_rounds: Total pruning rounds (default 4).

    Returns:
        (num_kv_heads, final_k) long — indices into the original n_free axis of scores.
    """
    n_free = scores.shape[-1]
    k_clamped = min(final_k, n_free)

    if n_rounds <= 1 or k_clamped >= n_free:
        return torch.topk(scores, k=k_clamped, dim=-1).indices

    # Build linear schedule: [k after round 1, ..., k after round n_rounds]
    schedule = [
        max(k_clamped, int(math.ceil(n_free - (n_free - k_clamped) * (i + 1) / n_rounds)))
        for i in range(n_rounds)
    ]
    schedule[-1] = k_clamped  # guarantee exact final budget

    pool = torch.arange(n_free, device=scores.device, dtype=torch.long)

    for i, round_k in enumerate(schedule):
        pool_scores = scores[:, pool]  # (H, |pool|)
        k = min(round_k, pool.numel())
        if i == len(schedule) - 1:
            # Final round: per-head selection
            final_local = torch.topk(pool_scores, k=k, dim=-1).indices  # (H, k)
            return pool[final_local]  # (H, k) — original indices
        # Intermediate rounds: amax-head pool narrowing
        agg = pool_scores.amax(dim=0)  # (|pool|,)
        local_idx = torch.topk(agg, k=k).indices  # (k,)
        pool = pool[local_idx]

    # Unreachable, but keeps type checker happy
    return pool.unsqueeze(0).expand(scores.shape[0], -1)


def _find_image_blocks(image_positions: torch.Tensor) -> List[torch.Tensor]:
    """Split image token positions into per-image contiguous blocks.

    LLaVA-1.5 encodes each image as a contiguous run of patch tokens.
    A gap (non-consecutive positions) signals the boundary between images.
    Returns a list of 1-D tensors, one per image block.
    """
    if image_positions.numel() == 0:
        return []
    blocks: List[torch.Tensor] = []
    start = 0
    pos = image_positions
    for i in range(1, pos.numel()):
        if pos[i].item() != pos[i - 1].item() + 1:
            blocks.append(pos[start:i])
            start = i
    blocks.append(pos[start:])
    return blocks


@dataclass
class ImageTokenTopKPress(BasePress):
    """Keep all non-image tokens and retain only the top-k image tokens per layer/head.

    Compression budget can be specified in two ways (mutually exclusive):
    - ``image_keep_ratio``: fraction of image tokens to keep (legacy, image-only basis).
    - ``total_keep_ratio``: fraction of ALL tokens (text + image) to keep. Text tokens
      are always preserved; image tokens fill the remaining budget. This puts all methods
      on the same r_eff_prompt basis for fair comparison.

    Positional forced-keep (for EXP-20260412-003 ablation):
    - ``n_initial_keep``: always keep the first N image tokens (global or per-image).
    - ``n_recent_keep``: always keep the last N image tokens (global or per-image).
    - ``n_random_keep``: always keep N randomly chosen image tokens (control).
    - ``per_image_forced``: if True, apply initial/recent per image block instead of globally.

    Forced positions are deducted from the scoring budget so total r_eff_prompt is unchanged.
    Random positions are sampled once per sample and reused across all layers.
    """

    image_keep_ratio: Optional[float] = None
    total_keep_ratio: Optional[float] = None
    head_reduce: HeadReduce = "amax"
    # Iterative pruning: n_rounds > 1 enables multi-round pool narrowing before final top-k.
    # 1 = one-shot (default). 4 = 4-round schedule matching EXP-20260417-001 plan.
    n_iterative_rounds: int = 1
    # Positional forced-keep parameters
    n_initial_keep: int = 0
    n_recent_keep: int = 0
    n_random_keep: int = 0
    per_image_forced: bool = False
    # Per-sample state (set via set_image_positions / set_sample_teacher)
    current_image_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    # Forced positions tensor (1-D, indices into current_image_positions). Computed once per sample.
    _forced_pos_indices: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    # Optional visualization capture — zero cost when None.
    _viz_capture: Optional[VizCapture] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.total_keep_ratio is not None and self.image_keep_ratio is not None:
            raise ValueError("Specify either image_keep_ratio or total_keep_ratio, not both")
        if self.total_keep_ratio is None and self.image_keep_ratio is None:
            # Default: keep everything (no compression)
            self.image_keep_ratio = 1.0
        if self.image_keep_ratio is not None:
            assert 0.0 <= self.image_keep_ratio <= 1.0, "image_keep_ratio must be in [0, 1]"
        if self.total_keep_ratio is not None:
            assert 0.0 <= self.total_keep_ratio <= 1.0, "total_keep_ratio must be in [0, 1]"

    # ------------------------------------------------------------------
    # Forced-position helpers
    # ------------------------------------------------------------------

    def _compute_forced_pos_indices(self, image_positions: torch.Tensor) -> torch.Tensor:
        """Compute which *indices into image_positions* are unconditionally kept.

        Returns a 1-D long tensor of indices (into ``image_positions``).
        This is called once per sample so that random positions are consistent
        across all Transformer layers.
        """
        n_image = image_positions.numel()
        forced: set = set()

        if self.per_image_forced:
            blocks = _find_image_blocks(image_positions)
            for block in blocks:
                # indices of this block within image_positions
                block_start = (image_positions == block[0]).nonzero(as_tuple=True)[0][0].item()
                block_size = block.numel()
                n_init = min(self.n_initial_keep, block_size)
                n_rec = min(self.n_recent_keep, block_size)
                for i in range(n_init):
                    forced.add(int(block_start) + i)
                for i in range(n_rec):
                    forced.add(int(block_start) + block_size - 1 - i)
        else:
            # Global: first/last N of all image tokens
            for i in range(min(self.n_initial_keep, n_image)):
                forced.add(i)
            for i in range(min(self.n_recent_keep, n_image)):
                forced.add(n_image - 1 - i)

        # Random forced keep: sample from non-forced indices
        if self.n_random_keep > 0:
            candidates = [i for i in range(n_image) if i not in forced]
            n_rand = min(self.n_random_keep, len(candidates))
            if n_rand > 0:
                forced.update(random.sample(candidates, n_rand))

        if not forced:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(forced), dtype=torch.long)

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        self.current_image_positions = image_positions.detach().cpu().long().flatten()
        self._forced_pos_indices = self._compute_forced_pos_indices(self.current_image_positions)

    def clear_sample_context(self) -> None:
        self.current_image_positions = None
        self._forced_pos_indices = None

    # ------------------------------------------------------------------
    # Visualization capture helpers
    # ------------------------------------------------------------------

    def attach_viz_capture(self, capture: VizCapture) -> None:
        """Attach a VizCapture to record per-layer scores/masks during generate().

        Call ``capture.reset()`` between samples to avoid data accumulation.
        """
        self._viz_capture = capture

    def detach_viz_capture(self) -> None:
        """Remove the attached VizCapture (zero-overhead mode)."""
        self._viz_capture = None

    def _should_skip(self) -> bool:
        """Return True if no compression is needed."""
        if self.total_keep_ratio is not None:
            return self.total_keep_ratio >= 1.0
        return (self.image_keep_ratio or 1.0) >= 1.0

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._should_skip():
            return keys, values
        if self.current_image_positions is None:
            raise RuntimeError("Image positions must be set before entering the press context")
        if keys.shape[0] != 1:
            raise ValueError("ImageTokenTopKPress currently only supports batch size 1")

        image_positions = self.current_image_positions.to(keys.device)
        image_positions = image_positions[(image_positions >= 0) & (image_positions < keys.shape[2])]
        if image_positions.numel() == 0:
            return keys, values

        seq_len = keys.shape[2]
        n_image = image_positions.numel()
        n_text = seq_len - n_image
        n_image_keep = _compute_n_image_keep(n_image, n_text, self.image_keep_ratio, self.total_keep_ratio)

        score_tensor = self.score_image_tokens(module, hidden_states, keys, values, attentions, kwargs, image_positions)
        score_tensor = _aggregate_scores_to_kv_heads(score_tensor, module, reduce=self.head_reduce)

        if score_tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 in score tensor, got {tuple(score_tensor.shape)}")
        if score_tensor.shape[-1] != image_positions.numel():
            raise ValueError(
                f"Score/image length mismatch: scores {tuple(score_tensor.shape)} vs image positions {tuple(image_positions.shape)}"
            )
        if score_tensor.shape[1] != keys.shape[1]:
            raise ValueError(
                f"KV-head mismatch: scores have {score_tensor.shape[1]} heads but cache has {keys.shape[1]} heads"
            )

        all_positions = torch.arange(seq_len, device=keys.device, dtype=torch.long)
        non_image_mask = torch.ones(seq_len, dtype=torch.bool, device=keys.device)
        non_image_mask[image_positions] = False
        non_image_positions = all_positions[non_image_mask]

        # ── Forced-keep positions (initial / recent / random) ────────────────
        forced_idx = self._forced_pos_indices  # indices into image_positions, CPU
        has_forced = forced_idx is not None and forced_idx.numel() > 0
        if has_forced:
            forced_idx_dev = forced_idx.to(keys.device)
            forced_image_positions = image_positions[forced_idx_dev]  # (n_forced,)
            # Create a boolean mask over image_positions for forced tokens
            forced_mask = torch.zeros(n_image, dtype=torch.bool, device=keys.device)
            forced_mask[forced_idx_dev] = True
            n_forced = int(forced_idx_dev.numel())
        else:
            forced_image_positions = torch.empty(0, dtype=torch.long, device=keys.device)
            forced_mask = torch.zeros(n_image, dtype=torch.bool, device=keys.device)
            n_forced = 0

        # Remaining budget for score-based top-k (after reserving forced slots)
        remaining_budget = max(0, n_image_keep - n_forced)

        if remaining_budget > 0 and (~forced_mask).any():
            # Score only non-forced image tokens
            free_indices = (~forced_mask).nonzero(as_tuple=True)[0]  # indices into image_positions
            free_scores = score_tensor[0][:, free_indices]  # (num_kv_heads, n_free)
            k = min(remaining_budget, free_indices.numel())
            topk_within_free = _iterative_topk(free_scores, k, n_rounds=self.n_iterative_rounds)  # (num_kv_heads, k)
            scored_image_positions = image_positions[free_indices[topk_within_free]]  # (num_kv_heads, k)
        elif remaining_budget > 0:
            scored_image_positions = torch.empty((keys.shape[1], 0), dtype=torch.long, device=keys.device)
        else:
            scored_image_positions = torch.empty((keys.shape[1], 0), dtype=torch.long, device=keys.device)

        # Combine: text positions + forced image positions + score-selected positions
        # forced_image_positions is 1-D (same for all heads); expand to (num_kv_heads, n_forced)
        base_positions = non_image_positions.unsqueeze(0).expand(keys.shape[1], -1)
        forced_expanded = forced_image_positions.unsqueeze(0).expand(keys.shape[1], -1)
        keep_positions = torch.cat([base_positions, forced_expanded, scored_image_positions], dim=-1)
        keep_positions, _ = torch.sort(keep_positions, dim=-1)

        # ── VizCapture hook (zero-overhead when not attached) ────────────────
        if self._viz_capture is not None:
            agg_score = score_tensor[0].amax(dim=0).detach().cpu().float()  # (n_image,)
            keep_mask = torch.zeros(n_image, dtype=torch.bool)
            if has_forced:
                keep_mask[forced_idx] = True  # forced_idx is CPU
            if remaining_budget > 0 and (~forced_mask).any():
                # free_indices[topk_within_free]: indices into image_positions, per kv head
                scored_indices_union = free_indices[topk_within_free].cpu().unique()
                keep_mask[scored_indices_union] = True
            self._viz_capture.record(agg_score=agg_score, keep_mask=keep_mask, n_keep=n_image_keep)

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(1, keys.shape[1], keep_positions.shape[-1], module.head_dim)
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class OracleImageTeacherPress(ImageTokenTopKPress):
    current_teacher_scores: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def set_sample_teacher(self, image_positions: torch.Tensor, teacher_scores: torch.Tensor) -> None:
        self.set_image_positions(image_positions)
        self.current_teacher_scores = teacher_scores.detach().cpu().float()

    def clear_sample_context(self) -> None:
        super().clear_sample_context()
        self.current_teacher_scores = None

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.current_teacher_scores is None:
            raise RuntimeError("Teacher scores must be set before using OracleImageTeacherPress")
        layer_scores = self.current_teacher_scores[module.layer_idx]
        return layer_scores.to(keys.device, dtype=keys.dtype).unsqueeze(0)


@dataclass
class H2OImageOnlyPress(ImageTokenTopKPress):
    """H2O-style accumulated attention scoring, image-only eviction (Ablation A).

    Scores image tokens by summing attention weights over all query positions —
    identical to the H2O heavy-hitter criterion — but restricts eviction to image
    tokens only, preserving all non-image (text/system) tokens unconditionally.

    NOTE: Requires ``output_attentions=True`` during generation. This forces eager
    (non-SDPA) attention, making decode slower — identical to LOOK-M's situation.
    Use total_keep_ratio for fair comparison with LOOK-M (same r_eff_prompt).
    """

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if attentions is None:
            raise RuntimeError(
                "H2OImageOnlyPress requires attention weights. "
                "Pass output_attentions=True to model.generate()."
            )
        # attentions: (1, num_heads, q_len, kv_len) — sum over query positions
        importance = attentions[0].sum(dim=1)  # (num_heads, kv_len)
        image_scores = importance[:, image_positions]  # (num_heads, n_image)
        return image_scores.unsqueeze(0)  # (1, num_heads, n_image)


@dataclass
class H2OAllTokenPress(BasePress):
    """Canonical H2O-style all-token eviction (Zhang et al. 2023, image+text).

    Scores every KV token by summing attention weights received from all prefill
    queries (``attentions.sum(dim=Q)``).  Top-K tokens (by ``total_keep_ratio``)
    are kept; the rest evicted.  No image/text distinction.

    Requires ``output_attentions=True`` (eager attention).  In practice the
    kvpress forward_hook reads ``output[1]`` directly, so we simply require
    eager attention implementation — no need to set that flag globally.
    """

    total_keep_ratio: float = 1.0
    head_reduce: HeadReduce = "amax"

    # No-op sample context API (for compatibility with evaluate driver that
    # calls these on ImageTokenTopKPress-based modes). All-token presses do
    # not need image positions — eviction is over all KV positions.
    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        return

    def clear_sample_context(self) -> None:
        return

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total_keep_ratio >= 1.0:
            return keys, values
        if attentions is None:
            raise RuntimeError(
                "H2OAllTokenPress requires attention weights (use attn_implementation='eager')."
            )
        if keys.shape[0] != 1:
            raise ValueError("H2OAllTokenPress currently only supports batch size 1")

        seq_len = keys.shape[2]
        num_kv_heads = keys.shape[1]

        # (num_heads, kv_len) — sum over prefill queries
        h2o = attentions[0].sum(dim=1).float().unsqueeze(0)  # (1, num_heads, kv_len)
        h2o = _aggregate_scores_to_kv_heads(h2o, module, reduce=self.head_reduce)[0]  # (num_kv_heads, kv_len)

        total_keep = int(math.ceil(self.total_keep_ratio * seq_len))
        total_keep = min(seq_len, max(1, total_keep))

        topk = torch.topk(h2o, k=total_keep, dim=-1).indices  # (num_kv_heads, total_keep)
        keep_positions, _ = topk.sort(dim=-1)

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(
            1, num_kv_heads, total_keep, module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class FutureAllTokenPress(BasePress):
    """Future-supervised MLP on ALL prompt tokens (requires all-token Future probe).

    Assumes the Future probe was trained with targets at every token position
    (not just image).  Applies MLP per-layer to every hidden state and keeps the
    top ``total_keep_ratio`` fraction of tokens.  No H2O mixing — pure Future.

    Does NOT require output_attentions.
    """

    future_probe_name: str = ""
    total_keep_ratio: float = 1.0
    head_reduce: HeadReduce = "amax"
    _future_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _loaded_fu_name: Optional[str] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model) -> None:
        if not self.future_probe_name:
            raise ValueError("future_probe_name is required for FutureAllTokenPress")
        if self.future_probe_name != self._loaded_fu_name:
            self._future_probe = KVzapModel.from_pretrained(self.future_probe_name)
            self._loaded_fu_name = self.future_probe_name

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        return

    def clear_sample_context(self) -> None:
        return

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total_keep_ratio >= 1.0:
            return keys, values
        if self._future_probe is None:
            raise RuntimeError("Future probe not loaded")
        if keys.shape[0] != 1:
            raise ValueError("FutureAllTokenPress currently only supports batch size 1")

        seq_len = keys.shape[2]
        num_kv_heads = keys.shape[1]
        dev = hidden_states.device
        dtype = hidden_states.dtype

        fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
        with torch.no_grad():
            s_fu = fu_layer(hidden_states).transpose(1, 2)  # (1, 1, seq_len)

        # Align to kv heads via broadcast
        s_fu = _aggregate_scores_to_kv_heads(s_fu, module, reduce=self.head_reduce)[0]  # (num_kv_heads, seq_len)

        total_keep = int(math.ceil(self.total_keep_ratio * seq_len))
        total_keep = min(seq_len, max(1, total_keep))

        topk = torch.topk(s_fu, k=total_keep, dim=-1).indices  # (num_kv_heads, total_keep)
        keep_positions, _ = topk.sort(dim=-1)

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(
            1, num_kv_heads, total_keep, module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class HybridH2OFutureAllTokenPress(BasePress):
    """Blend H2O and Future (all-token trained probe) across every KV token.

    score(i) = alpha * softmax(s_H2O)(i) + (1-alpha) * softmax(s_Future)(i)

    Operates per-layer on all tokens (no image-only scope).  Requires all-token
    Future probe and eager attention.

    Per-layer gating:
      If ``future_blend_layers`` is non-empty, Future is blended (using ``alpha``)
      ONLY at those model layers; at all other layers the score falls back to
      H2O-only (effective_alpha=1.0).  Empty tuple = blend at every layer.
    """

    future_probe_name: str = ""
    alpha: float = 0.5
    total_keep_ratio: float = 1.0
    head_reduce: HeadReduce = "amax"
    future_blend_layers: tuple[int, ...] = field(default_factory=tuple)
    _future_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _loaded_fu_name: Optional[str] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model) -> None:
        if not self.future_probe_name:
            raise ValueError("future_probe_name is required for HybridH2OFutureAllTokenPress")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.future_probe_name != self._loaded_fu_name:
            self._future_probe = KVzapModel.from_pretrained(self.future_probe_name)
            self._loaded_fu_name = self.future_probe_name

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        return

    def clear_sample_context(self) -> None:
        return

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total_keep_ratio >= 1.0:
            return keys, values
        if attentions is None:
            raise RuntimeError("HybridH2OFutureAllTokenPress requires eager attention.")
        if self._future_probe is None:
            raise RuntimeError("Future probe not loaded")
        if keys.shape[0] != 1:
            raise ValueError("HybridH2OFutureAllTokenPress currently only supports batch size 1")

        seq_len = keys.shape[2]
        num_kv_heads = keys.shape[1]
        dev = hidden_states.device
        dtype = hidden_states.dtype

        # H2O: (1, 1, seq_len)
        h2o_per_tok = attentions[0].sum(dim=1).float().mean(dim=0)  # (seq_len,)
        s_h2o = h2o_per_tok.view(1, 1, -1)

        # Future: (1, 1, seq_len)
        fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
        with torch.no_grad():
            s_fu = fu_layer(hidden_states).transpose(1, 2)

        s_h2o_norm = torch.softmax(s_h2o.float(), dim=-1)
        s_fu_norm = torch.softmax(s_fu.float(), dim=-1)

        if self.future_blend_layers and module.layer_idx not in self.future_blend_layers:
            effective_alpha = 1.0  # H2O-only at this layer
        else:
            effective_alpha = self.alpha

        s_blend = effective_alpha * s_h2o_norm + (1.0 - effective_alpha) * s_fu_norm  # (1, 1, seq_len)

        s_blend = _aggregate_scores_to_kv_heads(s_blend, module, reduce=self.head_reduce)[0]  # (num_kv_heads, seq_len)

        total_keep = int(math.ceil(self.total_keep_ratio * seq_len))
        total_keep = min(seq_len, max(1, total_keep))

        topk = torch.topk(s_blend, k=total_keep, dim=-1).indices
        keep_positions, _ = topk.sort(dim=-1)

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(
            1, num_kv_heads, total_keep, module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class QuadrantEvictionPress(BasePress):
    """Evict tokens from a specific (future_score, h2o_score) quadrant.

    Quadrant key: "HH" | "HL" | "LH" | "LL"
      H = top-50% of the respective score distribution (all tokens)
      L = bottom-50%

    Evicts exactly K = floor(seq_len * evict_ratio) tokens from the target quadrant.
    If the quadrant contains fewer than K tokens, evicts the entire quadrant.
    Requires eager attention for H2O score computation.
    """

    future_probe_name: str = ""
    quadrant: str = "LL"       # "HH" | "HL" | "LH" | "LL"
    evict_ratio: float = 0.25
    head_reduce: HeadReduce = "amax"
    _future_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _loaded_fu_name: Optional[str] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model) -> None:
        if not self.future_probe_name:
            raise ValueError("future_probe_name is required for QuadrantEvictionPress")
        if self.quadrant not in ("HH", "HL", "LH", "LL"):
            raise ValueError(f"quadrant must be one of HH/HL/LH/LL, got {self.quadrant!r}")
        if self.future_probe_name != self._loaded_fu_name:
            self._future_probe = KVzapModel.from_pretrained(self.future_probe_name)
            self._loaded_fu_name = self.future_probe_name

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        return

    def clear_sample_context(self) -> None:
        return

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.evict_ratio <= 0.0:
            return keys, values
        if attentions is None:
            raise RuntimeError("QuadrantEvictionPress requires eager attention.")
        if self._future_probe is None:
            raise RuntimeError("Future probe not loaded")
        if keys.shape[0] != 1:
            raise ValueError("QuadrantEvictionPress currently only supports batch size 1")

        seq_len = keys.shape[2]
        num_kv_heads = keys.shape[1]
        dev = hidden_states.device
        dtype = hidden_states.dtype

        # H2O score: sum attention over all prefill queries, mean over heads → (seq_len,)
        s_h2o = attentions[0].sum(dim=1).float().mean(dim=0)  # (seq_len,)

        # Future score: probe output → (seq_len,)
        fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
        with torch.no_grad():
            s_fu = fu_layer(hidden_states).transpose(1, 2)[0, 0, :]  # (seq_len,)

        # Binarize at median: strictly-above-median = High
        h2o_med = s_h2o.median()
        fu_med = s_fu.median()
        is_high_h2o = s_h2o > h2o_med  # (seq_len,)
        is_high_fu = s_fu > fu_med      # (seq_len,)

        quad_masks = {
            "HH": is_high_fu & is_high_h2o,
            "HL": is_high_fu & ~is_high_h2o,
            "LH": ~is_high_fu & is_high_h2o,
            "LL": ~is_high_fu & ~is_high_h2o,
        }
        q_mask = quad_masks[self.quadrant]  # (seq_len,)

        K = max(1, int(math.floor(seq_len * self.evict_ratio)))

        # Combined score for ranking within/across quadrant
        combined = torch.softmax(s_h2o, dim=-1) + torch.softmax(s_fu, dim=-1)

        # Primary: score in-quadrant tokens highest so they get evicted first;
        # secondary: fill remaining K slots from out-of-quadrant (lowest combined score).
        # This ensures every layer evicts exactly K tokens → consistent cache size across layers.
        INF = float("inf")
        primary_score = combined.clone()
        primary_score[~q_mask] = -INF          # non-quadrant tokens selected only as fallback
        secondary_score = -combined.clone()    # for fallback: evict lowest-combined out-of-quadrant
        secondary_score[q_mask] = -INF         # in-quadrant already handled by primary

        # Merge: first fill K from primary (quadrant), remainder from secondary (non-quadrant)
        n_in_quad = int(q_mask.sum().item())
        n_from_quad = min(K, n_in_quad)
        n_from_rest = K - n_from_quad

        evict_idx_parts = []
        if n_from_quad > 0:
            evict_idx_parts.append(torch.topk(primary_score, k=n_from_quad).indices)
        if n_from_rest > 0:
            evict_idx_parts.append(torch.topk(secondary_score, k=n_from_rest).indices)
        evict_idx = torch.cat(evict_idx_parts) if evict_idx_parts else torch.empty(0, dtype=torch.long, device=dev)

        # Build keep mask and gather
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=dev)
        keep_mask[evict_idx] = False
        keep_positions = keep_mask.nonzero(as_tuple=False).squeeze(-1)  # (keep,)
        n_keep = keep_positions.shape[0]

        # Expand to (1, num_kv_heads, n_keep, head_dim)
        gather_idx = keep_positions.view(1, 1, n_keep, 1).expand(
            1, num_kv_heads, n_keep, module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class OracleAllTokenPress(BasePress):
    """att_only_postvision oracle score + all-token eviction (Ablation B).

    Budget: total_keep_ratio fraction of ALL tokens (text + image) are retained.
    Text tokens may be evicted if their H2O score is low enough.

    Scoring:
    - image tokens: att_only_postvision oracle teacher score
    - text tokens: H2O accumulated attention (sum over query positions)

    Both score types are min-max normalized per head per layer before ranking so
    they live on the same [0, 1] scale.

    NOTE: Requires ``output_attentions=True`` during generation (forces eager attn).
    """

    total_keep_ratio: float = 1.0
    head_reduce: HeadReduce = "amax"
    current_image_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    current_teacher_scores: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def set_sample_teacher(self, image_positions: torch.Tensor, teacher_scores: torch.Tensor) -> None:
        self.current_image_positions = image_positions.detach().cpu().long().flatten()
        self.current_teacher_scores = teacher_scores.detach().cpu().float()

    def clear_sample_context(self) -> None:
        self.current_image_positions = None
        self.current_teacher_scores = None

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total_keep_ratio >= 1.0:
            return keys, values
        if self.current_image_positions is None or self.current_teacher_scores is None:
            raise RuntimeError("Must call set_sample_teacher() before using OracleAllTokenPress")
        if attentions is None:
            raise RuntimeError(
                "OracleAllTokenPress requires attention weights. "
                "Pass output_attentions=True to model.generate()."
            )
        if keys.shape[0] != 1:
            raise ValueError("OracleAllTokenPress currently only supports batch size 1")

        seq_len = keys.shape[2]
        num_kv_heads = keys.shape[1]
        device = keys.device
        dtype = keys.dtype

        image_positions = self.current_image_positions.to(device)
        image_positions = image_positions[(image_positions >= 0) & (image_positions < seq_len)]

        total_keep = int(math.ceil(self.total_keep_ratio * seq_len))
        total_keep = min(seq_len, max(1, total_keep))

        # ── Build unified score: (num_kv_heads, seq_len) ──────────────────────
        # Text tokens: H2O accumulated attention — (num_heads, kv_len)
        h2o_scores = attentions[0].sum(dim=1).float()  # (num_heads, kv_len)
        # Aggregate to kv heads
        h2o_scores = _aggregate_scores_to_kv_heads(
            h2o_scores.unsqueeze(0), module, reduce=self.head_reduce
        )[0]  # (num_kv_heads, kv_len)

        # Image tokens: oracle teacher — shape [L, H, I] or [L, I]
        layer_teacher = self.current_teacher_scores[module.layer_idx]  # (H, I) or (I,)
        if layer_teacher.dim() == 1:
            layer_teacher = layer_teacher.unsqueeze(0)  # (1, I)
        image_teacher = _aggregate_scores_to_kv_heads(
            layer_teacher.unsqueeze(0).to(device, dtype=torch.float32), module, reduce=self.head_reduce
        )[0]  # (num_kv_heads, I)

        # ── Min-max normalize each score type per head ─────────────────────────
        def _minmax(x: torch.Tensor) -> torch.Tensor:
            mn = x.amin(dim=-1, keepdim=True)
            mx = x.amax(dim=-1, keepdim=True)
            denom = (mx - mn).clamp(min=1e-8)
            return (x - mn) / denom

        # Build unified score tensor (num_kv_heads, seq_len):
        # both H2O and teacher scores are min-max normalized before ranking.
        unified = _minmax(h2o_scores)  # (num_kv_heads, seq_len)
        if image_positions.numel() > 0:
            teacher_norm = _minmax(image_teacher)  # (num_kv_heads, n_image)
            unified[:, image_positions] = teacher_norm

        # ── Top-k selection across all positions ──────────────────────────────
        topk_indices = torch.topk(unified, k=total_keep, dim=-1).indices  # (num_kv_heads, total_keep)
        keep_positions, _ = topk_indices.sort(dim=-1)  # sorted for gather

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(
            1, num_kv_heads, total_keep, module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class ProbeImageTeacherPress(ImageTokenTopKPress):
    probe_model_name: str = ""
    loaded_probe_model_name: Optional[str] = field(default=None, init=False, repr=False)
    probe_model: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    # Cumulative wall-clock time (ms) spent inside probe forward calls during a single generate().
    # Call reset_probe_timing() before generate() and read probe_score_total_ms after.
    probe_score_total_ms: float = field(default=0.0, init=False, repr=False)

    def reset_probe_timing(self) -> None:
        self.probe_score_total_ms = 0.0

    def post_init_from_model(self, model):
        if not self.probe_model_name:
            raise ValueError("probe_model_name must be set for ProbeImageTeacherPress")
        if self.probe_model_name != self.loaded_probe_model_name:
            self.loaded_probe_model_name = self.probe_model_name
            self.probe_model = KVzapModel.from_pretrained(self.probe_model_name)

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.probe_model is None:
            raise RuntimeError("Probe model not loaded")
        probe_layer = self.probe_model.layers[module.layer_idx]
        probe_layer = probe_layer.to(hidden_states.device, dtype=hidden_states.dtype).eval()
        image_hidden_states = hidden_states[:, image_positions, :]
        _t0 = time.perf_counter()
        scores = probe_layer(image_hidden_states).transpose(1, 2)
        self.probe_score_total_ms += (time.perf_counter() - _t0) * 1000.0
        return scores


@dataclass
class HybridImageTeacherPress(ImageTokenTopKPress):
    """Combine PostVision probe and Future-supervised probe scores at inference time.

    Final score per image token:
        s_hybrid = alpha * softmax(s_postvision) + (1 - alpha) * softmax(s_future)

    Both probes use the same KVzapModel format and are loaded from HuggingFace-style
    checkpoint directories.

    Args:
        postvision_probe_name: path to existing PostVision KVzapModel checkpoint
        future_probe_name:     path to future-supervised KVzapModel checkpoint
        alpha:                 weight for PostVision score (0 = future-only, 1 = post-vision-only)
    """

    postvision_probe_name: str = ""
    future_probe_name: str = ""
    alpha: float = 0.5
    # model-layer indices where future probe was trained; pruning is skipped at all others.
    # Empty tuple means all layers prune (only safe when checkpoint covers all layers).
    selected_layer_indices: tuple[int, ...] = field(default_factory=tuple)
    # Per-layer alpha: Future signal is blended (using `alpha`) only at these layer indices.
    # At all other layers the score falls back to PV-only (equivalent to alpha=1.0).
    # Empty tuple (default) = blend Future at every layer (legacy behavior).
    future_blend_layers: tuple[int, ...] = field(default_factory=tuple)
    _postvision_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _future_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _loaded_pv_name: Optional[str] = field(default=None, init=False, repr=False)
    _loaded_fu_name: Optional[str] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model) -> None:
        if not self.postvision_probe_name:
            raise ValueError("postvision_probe_name must be set for HybridImageTeacherPress")
        if not self.future_probe_name:
            raise ValueError("future_probe_name must be set for HybridImageTeacherPress")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.postvision_probe_name != self._loaded_pv_name:
            self._postvision_probe = KVzapModel.from_pretrained(self.postvision_probe_name)
            self._loaded_pv_name = self.postvision_probe_name
        if self.future_probe_name != self._loaded_fu_name:
            self._future_probe = KVzapModel.from_pretrained(self.future_probe_name)
            self._loaded_fu_name = self.future_probe_name

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._postvision_probe is None or self._future_probe is None:
            raise RuntimeError("Probe models not loaded. Call post_init_from_model first.")

        dev = hidden_states.device
        dtype = hidden_states.dtype
        image_hidden = hidden_states[:, image_positions, :]  # [1, N_image, D]

        pv_layer = self._postvision_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
        fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()

        with torch.no_grad():
            # Each probe: [1, 1, N_image] after transpose (KVzapModel output is [B, L, N])
            s_pv = pv_layer(image_hidden).transpose(1, 2)  # [1, 1, N_image]
            s_fu = fu_layer(image_hidden).transpose(1, 2)  # [1, 1, N_image]

        # Softmax over image token dimension to align scales
        s_pv_norm = torch.softmax(s_pv.float(), dim=-1).to(dtype)
        s_fu_norm = torch.softmax(s_fu.float(), dim=-1).to(dtype)

        # Per-layer alpha gating: if `future_blend_layers` is set, only those layers blend
        # in the Future signal; all other layers fall back to PV-only (alpha=1.0).
        if self.future_blend_layers and module.layer_idx not in self.future_blend_layers:
            effective_alpha = 1.0
        else:
            effective_alpha = self.alpha

        s_hybrid = effective_alpha * s_pv_norm + (1.0 - effective_alpha) * s_fu_norm  # [1, 1, N_image]
        return s_hybrid

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selected_layer_indices and module.layer_idx not in self.selected_layer_indices:
            return keys, values  # no-op: untrained layer — preserve full KV cache
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)


@dataclass
class HybridH2OFuturePress(ImageTokenTopKPress):
    """Blend H2O-prefill accumulated attention with Future-supervised probe.

    Final score per image token:
        s_hybrid = alpha * softmax(s_h2o) + (1 - alpha) * softmax(s_future)

    where
        s_h2o(i)   = mean_h sum_q A^(h)[q, i]      (prefill accumulated attention)
        s_future   = Future MLP probe on hidden state at image position

    Requires ``output_attentions=True`` (eager attention) because H2O reads the
    real prefill attention tensor.

    Args:
        future_probe_name: path to future-supervised KVzapModel checkpoint
        alpha:             weight for H2O score (0 = future-only, 1 = H2O-only)
        future_blend_layers: if non-empty, Future is blended only at these layers;
                             all other layers fall back to H2O-only.
    """

    future_probe_name: str = ""
    alpha: float = 0.5
    future_blend_layers: tuple[int, ...] = field(default_factory=tuple)
    _future_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _loaded_fu_name: Optional[str] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model) -> None:
        if not self.future_probe_name:
            raise ValueError("future_probe_name must be set for HybridH2OFuturePress")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.future_probe_name != self._loaded_fu_name:
            self._future_probe = KVzapModel.from_pretrained(self.future_probe_name)
            self._loaded_fu_name = self.future_probe_name

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if attentions is None:
            raise RuntimeError(
                "HybridH2OFuturePress requires attention weights. "
                "Pass output_attentions=True (eager attention)."
            )
        if self._future_probe is None:
            raise RuntimeError("Future probe not loaded. Call post_init_from_model first.")

        dev = hidden_states.device
        dtype = hidden_states.dtype

        # H2O prefill score: sum over all query positions, mean over heads → (1, 1, n_image)
        importance = attentions[0].sum(dim=1).float()  # (num_heads, kv_len)
        h2o_per_token = importance.mean(dim=0)  # (kv_len,)
        s_h2o = h2o_per_token[image_positions].view(1, 1, -1)  # (1, 1, n_image)

        # Future probe score → (1, 1, n_image)
        image_hidden = hidden_states[:, image_positions, :]  # (1, N_image, D)
        fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
        with torch.no_grad():
            s_fu = fu_layer(image_hidden).transpose(1, 2)  # (1, 1, n_image)

        # Normalize to comparable scale via softmax over image tokens
        s_h2o_norm = torch.softmax(s_h2o.float(), dim=-1).to(dtype)
        s_fu_norm = torch.softmax(s_fu.float(), dim=-1).to(dtype)

        # Per-layer gating (optional): Future blended only at specified layers.
        if self.future_blend_layers and module.layer_idx not in self.future_blend_layers:
            effective_alpha = 1.0  # H2O-only at this layer
        else:
            effective_alpha = self.alpha

        s_hybrid = effective_alpha * s_h2o_norm + (1.0 - effective_alpha) * s_fu_norm  # (1, 1, n_image)
        return s_hybrid


@dataclass
class FutureSupervisedImagePress(ProbeImageTeacherPress):
    """Future-supervised MLP probe for image KV pruning.

    Semantically distinct from ProbeImageTeacherPress: the underlying KVzapModel was
    trained with future decode attention as the supervision signal rather than prefill
    post-vision attention.

    Because only a subset of layers (selected_layer_indices) are trained, pruning is
    skipped at all other layers — they return the full KV cache unchanged.  This
    prevents random-weight scores from corrupting the KV selection at untrained layers.

    Args:
        probe_model_name:       path to future-supervised KVzapModel checkpoint
        selected_layer_indices: model-layer indices that were trained and should prune.
                                Derived from collect_future_supervised_labels.py
                                ``selected_layers`` mapped to absolute indices.
                                If empty, all layers prune (use only if full-model trained).
    """

    selected_layer_indices: tuple[int, ...] = field(default_factory=tuple)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selected_layer_indices and module.layer_idx not in self.selected_layer_indices:
            return keys, values  # no-op: untrained layer — preserve full KV cache
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)


@dataclass
class PreselectedImagePress(BasePress):
    """Evicts all image tokens except a pre-specified keep set.

    Used after iterative probe pre-selection (A-option iterative pruning).
    The keep set is global — same positions are kept for every KV head and layer.

    Usage:
        press = PreselectedImagePress()
        press.set_selection(all_image_positions, keep_positions)
        with press(model):
            model.generate(...)
        press.clear_sample_context()
    """

    _all_image_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _keep_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def set_selection(
        self,
        all_image_positions: torch.Tensor,
        keep_positions: torch.Tensor,
    ) -> None:
        self._all_image_positions = all_image_positions.detach().cpu().long().flatten()
        self._keep_positions = keep_positions.detach().cpu().long().flatten()

    def clear_sample_context(self) -> None:
        self._all_image_positions = None
        self._keep_positions = None

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._all_image_positions is None or self._keep_positions is None:
            return keys, values

        seq_len = keys.shape[2]
        device = keys.device
        num_kv_heads = keys.shape[1]

        all_img = self._all_image_positions.to(device)
        keep = self._keep_positions.to(device)
        all_img = all_img[(all_img >= 0) & (all_img < seq_len)]
        keep = keep[(keep >= 0) & (keep < seq_len)]

        if all_img.numel() == 0:
            return keys, values

        # Evict image tokens that are not in the keep set
        evict_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        evict_mask[all_img] = True
        evict_mask[keep] = False  # un-evict survivors

        keep_positions_1d = torch.arange(seq_len, device=device)[~evict_mask]
        keep_pos = keep_positions_1d.unsqueeze(0).expand(num_kv_heads, -1)

        gather_idx = keep_pos.unsqueeze(0).unsqueeze(-1).expand(
            1, num_kv_heads, keep_positions_1d.shape[0], module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


@dataclass
class VisualUtilityStudentPress(ImageTokenTopKPress):
    """3-branch visual-utility student for image-only KV eviction.

    Loads a `VisualUtilityStudent` checkpoint trained against future-decode
    attention. At each layer in the scope, scores image tokens using the
    student's 3-branch (CNN + question + raw) head. Layers outside the scope
    are passed through unchanged (no compression at untrained layers).

    Per-sample, call `set_image_positions()` (parent) and `set_question_positions()`
    before `model.generate()`.
    """

    student_model_name: str = ""
    grid_h: int = 24
    grid_w: int = 24
    _student: Any = field(default=None, init=False, repr=False)
    _loaded_name: Optional[str] = field(default=None, init=False, repr=False)
    _scope_layers: Optional[set[int]] = field(default=None, init=False, repr=False)
    _question_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    student_score_total_ms: float = field(default=0.0, init=False, repr=False)

    def reset_student_timing(self) -> None:
        self.student_score_total_ms = 0.0

    def set_question_positions(self, q_positions: torch.Tensor) -> None:
        self._question_positions = q_positions.detach().cpu().long().flatten()

    def clear_sample_context(self) -> None:
        super().clear_sample_context()
        self._question_positions = None

    def post_init_from_model(self, model) -> None:
        if not self.student_model_name:
            raise ValueError("student_model_name must be set for VisualUtilityStudentPress")
        if self.student_model_name != self._loaded_name:
            from kvpress.presses.visual_utility_student import VisualUtilityStudent

            self._loaded_name = self.student_model_name
            self._student = VisualUtilityStudent.from_pretrained(self.student_model_name)
            self._scope_layers = set(self._student.layer_indices)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self._scope_layers is not None and module.layer_idx not in self._scope_layers:
            return keys, values
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._student is None:
            raise RuntimeError("Student model not loaded; call post_init_from_model(model) first")
        if self._question_positions is None:
            raise RuntimeError("Call set_question_positions() before generate()")
        device = hidden_states.device
        dtype = hidden_states.dtype
        layer = self._student.layers[str(module.layer_idx)]
        layer = layer.to(device=device, dtype=dtype).eval()
        image_idx = image_positions.to(device=device, dtype=torch.long).flatten()
        q_idx = self._question_positions.to(device=device, dtype=torch.long).flatten()
        _t0 = time.perf_counter()
        with torch.no_grad():
            scores = layer(hidden_states, image_idx, q_idx, self.grid_h, self.grid_w)
        self.student_score_total_ms += (time.perf_counter() - _t0) * 1000.0
        # Match ProbeImageTeacherPress contract: (1, n_heads_or_1, n_image)
        return scores.unsqueeze(1)


@dataclass
class VisualUtilityStudentOneVisionPress(ImageTokenTopKPress):
    """OneVision-Qwen2 variant of `VisualUtilityStudentPress`.

    Wires a `VisualUtilityStudentOneVision` checkpoint (1D conv branch, no
    fixed grid). Identical scope/forced-keep semantics as the LLaVA-1.5 press.
    """

    student_model_name: str = ""
    _student: Any = field(default=None, init=False, repr=False)
    _loaded_name: Optional[str] = field(default=None, init=False, repr=False)
    _scope_layers: Optional[set[int]] = field(default=None, init=False, repr=False)
    _question_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    student_score_total_ms: float = field(default=0.0, init=False, repr=False)

    def reset_student_timing(self) -> None:
        self.student_score_total_ms = 0.0

    def set_question_positions(self, q_positions: torch.Tensor) -> None:
        self._question_positions = q_positions.detach().cpu().long().flatten()

    def clear_sample_context(self) -> None:
        super().clear_sample_context()
        self._question_positions = None

    def post_init_from_model(self, model) -> None:
        if not self.student_model_name:
            raise ValueError("student_model_name must be set for VisualUtilityStudentOneVisionPress")
        if self.student_model_name != self._loaded_name:
            from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision

            self._loaded_name = self.student_model_name
            self._student = VisualUtilityStudentOneVision.from_pretrained(self.student_model_name)
            self._scope_layers = set(self._student.layer_indices)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self._scope_layers is not None and module.layer_idx not in self._scope_layers:
            return keys, values
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._student is None:
            raise RuntimeError("Student model not loaded; call post_init_from_model(model) first")
        if self._question_positions is None:
            raise RuntimeError("Call set_question_positions() before generate()")
        device = hidden_states.device
        dtype = hidden_states.dtype
        layer = self._student.layers[str(module.layer_idx)]
        layer = layer.to(device=device, dtype=dtype).eval()
        image_idx = image_positions.to(device=device, dtype=torch.long).flatten()
        q_idx = self._question_positions.to(device=device, dtype=torch.long).flatten()
        _t0 = time.perf_counter()
        with torch.no_grad():
            scores = layer(hidden_states, image_idx, q_idx)
        self.student_score_total_ms += (time.perf_counter() - _t0) * 1000.0
        return scores.unsqueeze(1)
