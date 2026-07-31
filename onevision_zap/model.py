# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from qvik.llava_onevision.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from qvik.llava_onevision.conversation import conv_templates
from qvik.llava_onevision.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from qvik.llava_onevision.model.builder import load_pretrained_model
from transformers import DynamicCache

from onevision_zap.decode import PromptReplay, greedy_decode_after_prompt_replay, select_first_token_after_prompt_replay
from onevision_zap.eviction import EvictionStats, evict_image_kv
from onevision_zap.student import VisualUtilityStudent


@dataclass(frozen=True, slots=True)
class OneVisionAdapterError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class PreparedVisuals:
    pixel_values: torch.Tensor
    frame_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class OneVisionGeneration:
    generated_text: str
    first_token_id: int
    eviction: EvictionStats


class ZapOneVisionModel:
    def __init__(
        self,
        pretrained: Path,
        student_checkpoint: Path,
        *,
        image_keep_ratio: float,
        conv_template: str,
    ) -> None:
        if not 0.0 < image_keep_ratio <= 1.0:
            raise OneVisionAdapterError("image_keep_ratio must be in (0, 1]")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(pretrained),
            None,
            get_model_name_from_path(str(pretrained)),
            device_map="auto",
            attn_implementation="sdpa",
            multimodal=True,
        )
        self._model = model.eval()
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._device = next(model.parameters()).device
        self._conv_template = conv_template
        self._image_keep_ratio = image_keep_ratio
        self._student = (
            VisualUtilityStudent.from_pretrained(
                student_checkpoint,
            )
            .to(device=self._device, dtype=torch.float16)
            .eval()
        )

    def prepare_visuals(self, frames: list[Image.Image]) -> PreparedVisuals:
        if len(frames) > 1 or "image_aspect_ratio" not in self._model.config.__dict__:
            self._model.config.image_aspect_ratio = "pad"
        pixel_values = process_images(
            frames,
            self._image_processor,
            self._model.config,
        )
        if not isinstance(pixel_values, torch.Tensor):
            raise OneVisionAdapterError("visual preprocessing did not return a tensor")
        return PreparedVisuals(
            pixel_values=pixel_values.to(
                device=self._device,
                dtype=torch.float16,
            ),
            frame_sizes=tuple(frame.size for frame in frames),
        )

    @torch.no_grad()
    def generate_prepared(
        self,
        question: str,
        visuals: PreparedVisuals,
        *,
        max_new_tokens: int,
        is_video: bool,
    ) -> OneVisionGeneration:
        if max_new_tokens < 1:
            raise OneVisionAdapterError("max_new_tokens must be positive")
        conversation = conv_templates[self._conv_template].copy()
        placeholder_count = 1 if is_video else len(visuals.frame_sizes)
        image_tokens = " ".join(DEFAULT_IMAGE_TOKEN for _ in range(placeholder_count))
        conversation.append_message(
            conversation.roles[0],
            f"{image_tokens}\n{question}",
        )
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(
            conversation.get_prompt(),
            self._tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            raise OneVisionAdapterError("prompt tokenizer did not return a tensor")
        input_ids = input_ids.unsqueeze(0).to(self._device)
        if input_ids.shape[1] < 2:
            raise OneVisionAdapterError("prompt must contain a replay token")

        prefill_ids = input_ids[:, :-1].contiguous()
        replay = PromptReplay(
            input_id=input_ids[:, -1:].contiguous(),
            absolute_position=0,
        )
        attention_mask = torch.ones_like(prefill_ids, dtype=torch.bool)
        prepared = self._model.prepare_inputs_labels_for_multimodal(
            prefill_ids,
            None,
            attention_mask,
            None,
            prefill_ids.clone(),
            [visuals.pixel_values] if is_video else visuals.pixel_values,
            ["video"] if is_video else ["image"],
            None if is_video else list(visuals.frame_sizes),
        )
        expanded_mask = prepared[2]
        inputs_embeds = prepared[4]
        expanded_labels = prepared[5]
        if expanded_mask is None or inputs_embeds is None or expanded_labels is None:
            raise OneVisionAdapterError("multimodal prompt expansion failed")

        image_positions = (expanded_labels[0] == IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if image_positions.numel() == 0:
            raise OneVisionAdapterError("expanded prompt contains no image tokens")
        prompt_length = int(inputs_embeds.shape[1])
        replay = PromptReplay(replay.input_id, prompt_length)
        position_ids = torch.arange(
            prompt_length,
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0)
        prefill = self._model(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_mask,
            position_ids=position_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        hidden_states = prefill.hidden_states
        cache = prefill.past_key_values
        if hidden_states is None or not isinstance(cache, DynamicCache):
            raise OneVisionAdapterError("full prefill did not return hidden states and cache")

        image_positions_device = image_positions.to(self._device)
        question_positions = torch.arange(
            int(image_positions.max().item()) + 1,
            prompt_length,
            dtype=torch.long,
            device=self._device,
        )
        scores_by_layer = {
            layer_index: self._student.layers[str(layer_index)](
                hidden_states[layer_index + 1],
                image_positions_device,
                question_positions,
            ).squeeze(0)
            for layer_index in self._student.layer_indices
        }
        eviction = evict_image_kv(
            cache,
            scores_by_layer,
            image_positions,
            image_keep_ratio=self._image_keep_ratio,
        )
        del hidden_states, prefill, scores_by_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        first_token, cache = select_first_token_after_prompt_replay(
            self._model,
            cache,
            replay,
        )
        eos_token_id = int(self._tokenizer.eos_token_id) if self._tokenizer.eos_token_id is not None else 151645
        generated_ids = greedy_decode_after_prompt_replay(
            self._model,
            cache,
            first_token,
            prompt_length=prompt_length + 1,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        return OneVisionGeneration(
            generated_text=self._tokenizer.decode(
                generated_ids.tolist(),
                skip_special_tokens=True,
            ).strip(),
            first_token_id=int(first_token.item()),
            eviction=eviction,
        )
