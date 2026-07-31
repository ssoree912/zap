# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass(frozen=True, slots=True)
class EvictionError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class EvictionStats:
    prompt_tokens: int
    image_tokens: int
    image_tokens_kept: int
    text_tokens: int
    cache_tokens: int
    layers_compressed: int


def evict_image_kv(
    cache: DynamicCache,
    scores_by_layer: dict[int, torch.Tensor],
    image_positions: torch.Tensor,
    *,
    image_keep_ratio: float,
) -> EvictionStats:
    if not 0.0 < image_keep_ratio <= 1.0:
        raise EvictionError("image_keep_ratio must be in (0, 1]")
    if image_positions.ndim != 1 or image_positions.numel() == 0:
        raise EvictionError("image_positions must be a non-empty vector")
    if not cache.key_cache or len(cache.key_cache) != len(cache.value_cache):
        raise EvictionError("cache must contain matching key and value layers")

    prompt_tokens = int(cache.key_cache[0].shape[2])
    positions = image_positions.to(dtype=torch.long, device="cpu")
    if int(positions.min().item()) < 0 or int(positions.max().item()) >= prompt_tokens:
        raise EvictionError("image position is outside the prompt cache")

    image_tokens = int(positions.numel())
    image_tokens_kept = min(
        image_tokens,
        max(1, math.ceil(image_tokens * image_keep_ratio)),
    )
    image_mask = torch.zeros(prompt_tokens, dtype=torch.bool)
    image_mask[positions] = True

    for layer_index, (keys, values) in enumerate(
        zip(cache.key_cache, cache.value_cache, strict=True),
    ):
        scores = scores_by_layer.get(layer_index)
        if scores is None or scores.shape != (image_tokens,):
            raise EvictionError(f"invalid student scores for layer {layer_index}")
        if keys.shape != values.shape or int(keys.shape[2]) != prompt_tokens:
            raise EvictionError(f"invalid KV cache shape at layer {layer_index}")
        selected = torch.topk(
            scores.detach().to(device="cpu", dtype=torch.float32),
            k=image_tokens_kept,
            largest=True,
        ).indices
        keep_mask = ~image_mask
        keep_mask[positions[selected]] = True
        device_mask = keep_mask.to(keys.device)
        cache.key_cache[layer_index] = keys[:, :, device_mask, :].contiguous()
        cache.value_cache[layer_index] = values[:, :, device_mask, :].contiguous()

    text_tokens = prompt_tokens - image_tokens
    return EvictionStats(
        prompt_tokens=prompt_tokens,
        image_tokens=image_tokens,
        image_tokens_kept=image_tokens_kept,
        text_tokens=text_tokens,
        cache_tokens=text_tokens + image_tokens_kept,
        layers_compressed=len(cache.key_cache),
    )
