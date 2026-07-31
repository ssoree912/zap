# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for attention-output compaction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from attention_compaction import compact_llava15_attentions


class TinySelfAttention(nn.Module):
    def forward(self, attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        output = torch.zeros(
            (*attention.shape[:-2], attention.shape[-2], 2),
            dtype=attention.dtype,
        )
        return output, attention, "tiny-cache"


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinySelfAttention()


class TinyModel(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers)
        self.layers = nn.ModuleList([TinyLayer() for _ in range(layers)])
        # A visual attention module must never be selected by the helper.
        self.vision_tower = TinyLayer()


def test_two_prefills_and_decode_preserve_eval700_reductions() -> None:
    model = TinyModel(layers=2)
    actual_positions = torch.tensor([1, 3], dtype=torch.long)

    def infer_positions(**_kwargs: object) -> torch.Tensor:
        return actual_positions

    agreement = SimpleNamespace(infer_user_question_positions=infer_positions)
    weights = torch.arange(1, 1 + 2 * 4 * 5, dtype=torch.float16).reshape(
        1,
        2,
        4,
        5,
    )

    with compact_llava15_attentions(model, agreement) as state:
        compact_positions = agreement.infer_user_question_positions()
        assert compact_positions.tolist() == [0]
        assert state.actual_question_count == 2
        assert torch.equal(state.actual_question_positions, actual_positions)

        first_attention = model.layers[0].self_attn(weights)[1]
        expected_question_mean = weights.index_select(
            -2,
            actual_positions,
        ).float().mean(dim=-2, keepdim=True)
        expected_h2o_sum = weights.sum(dim=-2, keepdim=True).float()

        assert first_attention.shape == (1, 2, 2, 5)
        assert first_attention.dtype == torch.float32
        torch.testing.assert_close(
            first_attention[..., 0:1, :],
            expected_question_mean,
        )
        torch.testing.assert_close(
            first_attention[..., 1:2, :],
            expected_h2o_sum - expected_question_mean,
        )
        # These are exactly the two reductions performed by eval-700 run_one.
        torch.testing.assert_close(
            first_attention.sum(dim=-2),
            expected_h2o_sum.squeeze(-2),
        )
        torch.testing.assert_close(
            first_attention.index_select(-2, compact_positions).mean(dim=-2),
            expected_question_mean.squeeze(-2),
        )

        second_attention = model.layers[0].self_attn(weights)[1]
        assert second_attention.shape == (1, 2, 1, 5)
        torch.testing.assert_close(second_attention, weights[..., -1:, :])
        assert (
            second_attention.untyped_storage().data_ptr()
            != weights.untyped_storage().data_ptr()
        )

        decode_attention = weights[..., -1:, :].clone()
        returned_decode = model.layers[0].self_attn(decode_attention)[1]
        assert returned_decode is decode_attention
        assert state.full_prefill_call_counts["layers.0.self_attn"] == 2

        # The other decoder layer has independent first/second-call state.
        other_first = model.layers[1].self_attn(weights)[1]
        assert other_first.shape[-2] == 2
        assert state.full_prefill_call_counts["layers.1.self_attn"] == 1

        vision_attention = model.vision_tower.self_attn(weights)[1]
        assert vision_attention is weights

    assert agreement.infer_user_question_positions is infer_positions
    assert not model.layers[0].self_attn._forward_hooks
    assert not model.layers[1].self_attn._forward_hooks
    assert not model.vision_tower.self_attn._forward_hooks


def test_exception_restores_original_callable_and_removes_hooks() -> None:
    model = TinyModel(layers=1)

    def infer_positions(**_kwargs: object) -> torch.Tensor:
        return torch.tensor([2], dtype=torch.long)

    agreement = SimpleNamespace(infer_user_question_positions=infer_positions)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with compact_llava15_attentions(model, agreement):
            agreement.infer_user_question_positions()
            assert agreement.infer_user_question_positions is not infer_positions
            assert model.layers[0].self_attn._forward_hooks
            raise RuntimeError("synthetic failure")

    assert agreement.infer_user_question_positions is infer_positions
    assert not model.layers[0].self_attn._forward_hooks


def test_unexpected_third_full_prefill_is_rejected() -> None:
    model = TinyModel(layers=1)
    agreement = SimpleNamespace(
        infer_user_question_positions=lambda **_kwargs: torch.tensor([1])
    )
    weights = torch.ones((1, 1, 3, 3), dtype=torch.float32)

    with compact_llava15_attentions(model, agreement):
        agreement.infer_user_question_positions()
        model.layers[0].self_attn(weights)
        model.layers[0].self_attn(weights)
        with pytest.raises(RuntimeError, match="third full-prefill"):
            model.layers[0].self_attn(weights)


def test_language_decoder_layer_count_is_validated_before_patch() -> None:
    model = TinyModel(layers=2)
    model.config.num_hidden_layers = 3

    def infer_positions(**_kwargs: object) -> torch.Tensor:
        return torch.tensor([1])

    agreement = SimpleNamespace(infer_user_question_positions=infer_positions)
    with pytest.raises(ValueError, match="layer-count mismatch"):
        with compact_llava15_attentions(model, agreement):
            raise AssertionError("unreachable")

    assert agreement.infer_user_question_positions is infer_positions
    assert not model.layers[0].self_attn._forward_hooks
    assert not model.layers[1].self_attn._forward_hooks
