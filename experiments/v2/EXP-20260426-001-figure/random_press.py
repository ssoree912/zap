"""Random eviction press classes — figure baseline only (EXP-20260426-001-figure).

Two presses:
- RandomImageOnlyPress: random scores over image tokens; text always kept.
- RandomAllTokenPress:  random scores over ALL prefill tokens (text+image).

Both use total_keep_ratio. Per-layer independent random (each layer samples its own
keep mask). Set torch.manual_seed before running for reproducibility.

NOT for production. Strictly an ablation baseline to quantify the value of the
image-only eviction constraint vs. an unconstrained random selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.image_token_press import (
    ImageTokenTopKPress,
    HeadReduce,
    _aggregate_scores_to_kv_heads,
)


@dataclass
class RandomImageOnlyPress(ImageTokenTopKPress):
    """Random scores for image tokens. Text tokens never evicted.

    Equivalent budget to other image-only presses (image_keep_ratio or
    total_keep_ratio interpreted via _compute_n_image_keep)."""

    head_reduce: HeadReduce = "amax"

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
        n_image = image_positions.numel()
        device = keys.device
        # Match expected output shape: (1, num_heads_or_1, n_image)
        scores = torch.rand(1, 1, n_image, device=device, dtype=torch.float32)
        return scores


@dataclass
class RandomAllTokenPress(BasePress):
    """Random scores over ALL KV positions; text and image both evictable."""

    total_keep_ratio: float = 1.0
    head_reduce: HeadReduce = "amax"

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
        if keys.shape[0] != 1:
            raise ValueError("RandomAllTokenPress only supports batch size 1")

        seq_len = keys.shape[2]
        num_kv_heads = keys.shape[1]
        device = keys.device

        # Random scores per (head, position) — match H2OAllTokenPress shape
        rand_scores = torch.rand(1, 1, seq_len, device=device, dtype=torch.float32)
        rand_scores = _aggregate_scores_to_kv_heads(rand_scores, module, reduce=self.head_reduce)[0]
        # (num_kv_heads, seq_len)

        total_keep = int(math.ceil(self.total_keep_ratio * seq_len))
        total_keep = min(seq_len, max(1, total_keep))

        topk = torch.topk(rand_scores, k=total_keep, dim=-1).indices
        keep_positions, _ = topk.sort(dim=-1)

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(
            1, num_kv_heads, total_keep, module.head_dim
        )
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values
