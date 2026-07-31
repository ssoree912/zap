# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from onevision_zap.eviction import evict_image_kv


def _cache() -> DynamicCache:
    cache = DynamicCache()
    tokens = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1)
    for layer_index in range(2):
        keys = tokens.add(layer_index * 100).expand(1, 2, -1, -1).clone()
        cache.key_cache.append(keys)
        cache.value_cache.append(keys.add(1000))
    return cache


def test_student_eviction_keeps_all_text_and_ten_percent_of_images() -> None:
    # Given: ten image KVs, ten text KVs, and layer-specific student scores.
    cache = _cache()
    original_keys = [keys.clone() for keys in cache.key_cache]
    image_positions = torch.arange(4, 14)
    scores_by_layer = {
        0: torch.arange(10, dtype=torch.float32),
        1: torch.arange(10, 0, -1, dtype=torch.float32),
    }

    # When: image-only eviction keeps ten percent of image tokens.
    stats = evict_image_kv(
        cache,
        scores_by_layer,
        image_positions,
        image_keep_ratio=0.1,
    )

    # Then: every text KV survives and exactly the top student image survives.
    assert stats.image_tokens == 10
    assert stats.image_tokens_kept == 1
    assert stats.text_tokens == 10
    assert stats.cache_tokens == 11
    assert stats.layers_compressed == 2
    expected_positions = (
        (*range(4), 13, *range(14, 20)),
        (*range(4), 4, *range(14, 20)),
    )
    for layer_index, positions in enumerate(expected_positions):
        torch.testing.assert_close(
            cache.key_cache[layer_index][0, 0],
            original_keys[layer_index][0, 0, list(positions)],
        )
