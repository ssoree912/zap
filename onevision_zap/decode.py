# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import DynamicCache


class DecoderOutput(Protocol):
    @property
    def last_hidden_state(self) -> torch.Tensor: ...

    @property
    def past_key_values(self) -> DynamicCache: ...


class DecoderModel(Protocol):
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor: ...

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
    ) -> DecoderOutput: ...


class OneVisionModel(Protocol):
    def get_model(self) -> DecoderModel: ...

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class DecodeError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class PromptReplay:
    input_id: torch.Tensor
    absolute_position: int


@torch.no_grad()
def forward_one_token(
    model: OneVisionModel,
    input_ids: torch.Tensor,
    past_key_values: DynamicCache,
    *,
    absolute_position: int,
) -> tuple[torch.Tensor, DynamicCache]:
    language_model = model.get_model()
    inputs_embeds = language_model.embed_tokens(input_ids)
    cache_position = torch.tensor(
        [absolute_position],
        dtype=torch.long,
        device=input_ids.device,
    )
    output = language_model(
        input_ids=None,
        attention_mask=None,
        position_ids=cache_position.unsqueeze(0),
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=cache_position,
    )
    return model.lm_head(output.last_hidden_state), output.past_key_values


@torch.no_grad()
def select_first_token_after_prompt_replay(
    model: OneVisionModel,
    past_key_values: DynamicCache,
    replay: PromptReplay,
) -> tuple[torch.Tensor, DynamicCache]:
    logits, replayed_cache = forward_one_token(
        model,
        replay.input_id,
        past_key_values,
        absolute_position=replay.absolute_position,
    )
    return logits[:, -1, :].argmax(dim=-1, keepdim=True), replayed_cache


@torch.no_grad()
def greedy_decode_after_prompt_replay(
    model: OneVisionModel,
    past_key_values: DynamicCache,
    first_token: torch.Tensor,
    *,
    prompt_length: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    if max_new_tokens < 1:
        raise DecodeError("max_new_tokens must be positive")
    generated = [int(first_token.item())]
    if generated[0] == eos_token_id:
        return torch.tensor(generated, dtype=torch.long)

    next_token = first_token
    absolute_position = prompt_length
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = forward_one_token(
            model,
            next_token,
            past_key_values,
            absolute_position=absolute_position,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        absolute_position += 1
        if token_id == eos_token_id:
            break
    return torch.tensor(generated, dtype=torch.long)
