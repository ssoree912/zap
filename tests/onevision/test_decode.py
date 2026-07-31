# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from onevision_zap.decode import PromptReplay, greedy_decode_after_prompt_replay, select_first_token_after_prompt_replay


@dataclass(frozen=True, slots=True)
class _DecoderOutput:
    last_hidden_state: torch.Tensor
    past_key_values: DynamicCache


class _LanguageModel:
    def __init__(self, next_tokens: tuple[int, ...]) -> None:
        self._next_tokens = iter(next_tokens)
        self.positions: list[int] = []

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1)

    def __call__(
        self,
        *,
        input_ids: None,
        attention_mask: None,
        position_ids: torch.Tensor,
        past_key_values: DynamicCache,
        inputs_embeds: torch.Tensor,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        cache_position: torch.Tensor,
    ) -> _DecoderOutput:
        del input_ids, attention_mask, position_ids, inputs_embeds
        del use_cache, output_attentions, output_hidden_states, return_dict
        self.positions.append(int(cache_position.item()))
        hidden_state = torch.full((1, 1, 10), -100.0)
        hidden_state[0, 0, next(self._next_tokens)] = 100.0
        return _DecoderOutput(hidden_state, past_key_values)


class _OneVisionModel:
    def __init__(self, next_tokens: tuple[int, ...]) -> None:
        self.language_model = _LanguageModel(next_tokens)

    def get_model(self) -> _LanguageModel:
        return self.language_model

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_first_answer_token_is_selected_after_evicted_cache_prompt_replay() -> None:
    # Given: an already-evicted cache and the final held-out prompt token.
    model = _OneVisionModel(next_tokens=(4, 6, 9))
    cache = DynamicCache()
    replay = PromptReplay(input_id=torch.tensor([[3]]), absolute_position=19)

    # When: the held-out prompt token is replayed before answer decoding.
    first_token, replayed_cache = select_first_token_after_prompt_replay(
        model,
        cache,
        replay,
    )
    generated = greedy_decode_after_prompt_replay(
        model,
        replayed_cache,
        first_token,
        prompt_length=20,
        eos_token_id=9,
        max_new_tokens=3,
    )

    # Then: the replay logit supplies token one and all positions remain absolute.
    assert generated.tolist() == [4, 6, 9]
    assert model.language_model.positions == [19, 20, 21]
