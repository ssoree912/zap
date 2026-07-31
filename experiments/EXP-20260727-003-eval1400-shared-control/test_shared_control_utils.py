# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the eval-1400 shared-mask control."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ZAP_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_ROOT))

from foresight.eval.kv_decode_utils import (  # noqa: E402
    trim_kv_cache_per_layer,
)
from shared_control_utils import (
    broadcast_to_kv_heads,
    future_scores_from_attention_blocks,
    head_average_scores,
    pair_key,
    prompt_keep_masks,
    shared_control_metrics,
    validate_visual_masks,
)


def test_head_average_scores_removes_the_head_axis_before_topk() -> None:
    heads = torch.tensor(
        [
            [
                [8.0, 0.0, 5.0, 0.0],
                [0.0, 6.0, 5.0, 0.0],
            ]
        ]
    )
    shared = head_average_scores(heads, name="all_prefill")
    assert shared.shape == (1, 4)
    torch.testing.assert_close(shared.sum(dim=-1), torch.ones(1))
    # Head zero prefers token 0 and head one prefers token 1, while their
    # controlled shared score prefers token 2 after averaging.
    assert int(shared.argmax(dim=-1)) == 2


def test_head_average_precedes_normalization_for_shared_future_mass() -> None:
    # The two heads have very different total visual mass. Normalizing each
    # head before averaging would incorrectly produce [0.5, 0.5].
    future_heads = torch.tensor([[[90.0, 10.0], [1.0, 9.0]]])
    shared = head_average_scores(
        future_heads,
        name="raw_full_cache_future",
    )
    torch.testing.assert_close(
        shared,
        torch.tensor([[91.0 / 110.0, 19.0 / 110.0]]),
    )


def test_future_trajectory_includes_only_last_prompt_row_then_decode_rows() -> None:
    prompt_block = torch.zeros(1, 2, 3, 5)
    # Non-final prompt rows are deliberately huge and must not contribute.
    prompt_block[:, :, 0:2, :] = 10_000.0
    prompt_block[0, 0, -1, 1] = 2.0
    prompt_block[0, 0, -1, 3] = 6.0
    prompt_block[0, 1, -1, 1] = 4.0
    prompt_block[0, 1, -1, 3] = 8.0

    decode_block = torch.zeros(1, 2, 1, 6)
    decode_block[0, 0, -1, 1] = 6.0
    decode_block[0, 0, -1, 3] = 2.0
    decode_block[0, 1, -1, 1] = 8.0
    decode_block[0, 1, -1, 3] = 4.0

    shared, heads = future_scores_from_attention_blocks(
        ((prompt_block,), (decode_block,)),
        image_positions=torch.tensor([1, 3]),
    )
    expected_heads = torch.tensor([[[4.0, 4.0], [6.0, 6.0]]])
    torch.testing.assert_close(heads, expected_heads)
    torch.testing.assert_close(shared, torch.tensor([[0.5, 0.5]]))


def test_exact_shared_metrics_use_shared_future_reference() -> None:
    scores = {
        "all_prefill_shared": torch.tensor([[9.0, 8.0, 0.0, 0.0]]),
        "question_shared": torch.tensor([[0.0, 8.0, 9.0, 0.0]]),
        "qvik_shared": torch.tensor([[9.0, 0.0, 8.0, 0.0]]),
        "future_shared": torch.tensor([[0.0, 7.0, 6.0, 1.0]]),
    }
    metrics, masks = shared_control_metrics(scores, k=2)

    assert masks["future_shared_keep"].tolist() == [[False, True, True, False]]
    assert metrics["future_agreement"]["all_prefill_shared"]["topk_recall"] == [
        0.5
    ]
    assert metrics["future_agreement"]["question_shared"]["topk_recall"] == [
        1.0
    ]
    assert metrics["future_agreement"]["question_shared"][
        "future_mass_retained"
    ] == pytest.approx([13.0 / 14.0])
    assert metrics["future_agreement"]["qvik_shared"]["topk_recall"] == [0.5]
    pair = metrics["pairwise_keep_agreement"][
        pair_key("all_prefill_shared", "question_shared")
    ]
    assert pair["overlap_per_k"] == [0.5]
    assert pair["jaccard"] == pytest.approx([1.0 / 3.0])


def test_prompt_mask_is_one_dimensional_and_broadcast_identically() -> None:
    visual = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
        ]
    )
    positions = torch.tensor([1, 3, 4])
    prompt = prompt_keep_masks(
        visual,
        image_positions=positions,
        prompt_len=6,
    )
    assert prompt[0].tolist() == [True, True, True, False, True, True]
    assert prompt[1].tolist() == [True, False, True, True, True, True]
    assert prompt[0].ndim == prompt[1].ndim == 1

    expanded = broadcast_to_kv_heads(visual, n_kv_heads=4)
    assert expanded.shape == (2, 4, 3)
    for head in range(4):
        assert torch.equal(expanded[:, head], visual)


def test_actual_cache_trim_uses_identical_positions_for_every_head() -> None:
    visual = torch.tensor([[True, False, True]])
    prompt_masks = prompt_keep_masks(
        visual,
        image_positions=torch.tensor([1, 3, 4]),
        prompt_len=6,
    )
    positions = torch.arange(6, dtype=torch.float32)
    keys = torch.stack(
        [positions + 100.0 * head for head in range(3)],
        dim=0,
    ).reshape(1, 3, 6, 1)
    cache = SimpleNamespace(
        key_cache=[keys.clone()],
        value_cache=[keys.clone() + 1000.0],
    )

    trimmed = trim_kv_cache_per_layer(cache, prompt_masks)
    expected_positions = torch.tensor([0.0, 1.0, 2.0, 4.0, 5.0])
    assert trimmed.key_cache[0].shape == (1, 3, 5, 1)
    for head in range(3):
        selected = trimmed.key_cache[0][0, head, :, 0]
        torch.testing.assert_close(
            selected - 100.0 * head,
            expected_positions,
        )


def test_mask_validator_enforces_exact_budget() -> None:
    _, masks = shared_control_metrics(
        {
            "all_prefill_shared": torch.rand(2, 5),
            "question_shared": torch.rand(2, 5),
            "qvik_shared": torch.rand(2, 5),
            "future_shared": torch.rand(2, 5),
        },
        k=2,
    )
    validate_visual_masks(masks, layers=2, n_visual=5, n_keep=2)
    masks["qvik_shared_keep"][0, 0] = ~masks["qvik_shared_keep"][0, 0]
    with pytest.raises(ValueError, match="not exact Top-2"):
        validate_visual_masks(masks, layers=2, n_visual=5, n_keep=2)


def test_zero_visual_budget_is_rejected() -> None:
    scores = {
        "all_prefill_shared": torch.rand(1, 3),
        "question_shared": torch.rand(1, 3),
        "qvik_shared": torch.rand(1, 3),
        "future_shared": torch.rand(1, 3),
    }
    with pytest.raises(ValueError, match="K must be"):
        shared_control_metrics(scores, k=0)
