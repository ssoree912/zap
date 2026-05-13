# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental lmms-eval wrapper: OneVision student with progressive prefill pruning.

This module intentionally registers a new model name,
`llava_onevision_student_progressive`, so existing `llava_onevision_student`
jobs are unaffected. The implementation keeps the current post-prefill wrapper
available and only changes the prefill path for this experimental model.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch

sys.path.insert(0, "/workspace/zap")

from .lmms_onevision_student import LlavaOnevisionStudent
from ..llava_onevision_extractor import infer_onevision_image_positions_no_forward
from .vlmeval_onevision_student import _greedy_decode_with_kv, _resolve_eos_token_id

try:
    from lmms_eval.api.registry import register_model
except ImportError as e:
    raise ImportError(
        "lmms_eval not found. Add VFlowOpt/src/lmms_eval-0.2.4 to PYTHONPATH."
    ) from e


def _cache_len_for_layer(past_kv: Any, layer_idx: int) -> int:
    if hasattr(past_kv, "key_cache"):
        return int(past_kv.key_cache[layer_idx].shape[2])
    if hasattr(past_kv, "layers"):
        return int(past_kv.layers[layer_idx].keys.shape[2])
    return int(past_kv[layer_idx][0].shape[2])


def _trim_computed_cache_layers(past_kv: Any, upto_layer: int, keep_mask: torch.Tensor) -> None:
    """Trim already-computed cache layers by the current active-sequence mask."""

    if hasattr(past_kv, "key_cache"):
        for li in range(min(upto_layer + 1, len(past_kv.key_cache))):
            if _cache_len_for_layer(past_kv, li) != int(keep_mask.numel()):
                continue
            mask = keep_mask.to(past_kv.key_cache[li].device)
            past_kv.key_cache[li] = past_kv.key_cache[li][:, :, mask, :].contiguous()
            past_kv.value_cache[li] = past_kv.value_cache[li][:, :, mask, :].contiguous()
        return

    if hasattr(past_kv, "layers"):
        for li, layer in enumerate(past_kv.layers[: upto_layer + 1]):
            if int(layer.keys.shape[2]) != int(keep_mask.numel()):
                continue
            mask = keep_mask.to(layer.keys.device)
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return

    for li in range(min(upto_layer + 1, len(past_kv))):
        k, v = past_kv[li]
        if int(k.shape[2]) != int(keep_mask.numel()):
            continue
        mask = keep_mask.to(k.device)
        past_kv[li] = (k[:, :, mask, :].contiguous(), v[:, :, mask, :].contiguous())


def _current_token_indices(
    active_orig_positions: torch.Tensor,
    image_positions: torch.Tensor,
    last_image_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current-sequence image and question indices.

    `active_orig_positions` maps each current hidden-state slot to its original
    prompt position. Image tokens are the surviving original image positions;
    question/text tokens are kept and pooled from positions after the image span.
    """

    image_positions = image_positions.to(active_orig_positions.device)
    is_image = torch.isin(active_orig_positions, image_positions)
    image_idx = is_image.nonzero(as_tuple=False).flatten()
    question_idx = (active_orig_positions > last_image_position).nonzero(as_tuple=False).flatten()
    return image_idx.to(torch.long), question_idx.to(torch.long)


@register_model("llava_onevision_student_progressive")
class LlavaOnevisionStudentProgressive(LlavaOnevisionStudent):
    """LLaVA-OneVision-HF with student-driven progressive prefill pruning."""

    def __init__(
        self,
        *args,
        progressive_schedule: str = "immediate",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if progressive_schedule not in {"immediate", "linear"}:
            raise ValueError(
                f"Unsupported progressive_schedule={progressive_schedule!r}; "
                "expected 'immediate' or 'linear'."
            )
        self.progressive_schedule = progressive_schedule
        self._keep_stats: list[dict] = []  # accumulated per-sample keep ratios

    @torch.no_grad()
    def _generate_with_student(self, inputs, visuals, max_new_tokens: int) -> str:
        if not visuals or self.keep_ratio >= 1.0:
            return super()._generate_with_student(inputs, visuals, max_new_tokens)

        try:
            image_positions, prompt_len = infer_onevision_image_positions_no_forward(
                prompt_inputs={"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                model_config=self._model.config,
                num_images=len(visuals),
            )
        except ValueError:
            return super()._generate_with_student(inputs, visuals, max_new_tokens)

        if "pixel_values_videos" in inputs and inputs.get("pixel_values_videos") is not None:
            raise NotImplementedError("progressive OneVision student currently supports image inputs only")

        try:
            merged_embeds = self._build_merged_inputs_embeds(inputs)
        except ValueError:
            # dynamic-res mismatch: image features ≠ image tokens → fallback to post-prefill student
            return super()._generate_with_student(inputs, visuals, max_new_tokens)

        hidden_states, past_kv, active_orig_positions, prefill_stats = self._progressive_prefill(
            merged_embeds=merged_embeds,
            attention_mask=inputs.get("attention_mask"),
            image_positions=image_positions,
            prompt_len=int(prompt_len),
        )

        # log per-sample keep ratios
        self._keep_stats.append(prefill_stats)
        n = len(self._keep_stats)
        avg_total = sum(s["total_keep_ratio"] for s in self._keep_stats) / n
        avg_image = sum(s["image_keep_ratio"] for s in self._keep_stats) / n
        print(
            f"[keep-ratio] sample={n}"
            f" total_keep={prefill_stats['total_keep_ratio']:.4f}"
            f" image_keep={prefill_stats['image_keep_ratio']:.4f}"
            f" | running_avg total={avg_total:.4f} image={avg_image:.4f}"
            f" | img_orig={prefill_stats['n_image_original']}"
            f" img_kept={prefill_stats['n_image_kept']}",
            file=sys.stderr,
            flush=True,
        )

        logits = self._model.language_model.lm_head(hidden_states[:, -1:, :])
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        eos_token_id = _resolve_eos_token_id(self._processor, self._model.config)

        # Use prefill logits directly; subsequent decode steps use the pruned KV.
        answer_ids = _greedy_decode_with_kv(
            self._model,
            past_kv,
            next_token,
            prompt_len=int(prompt_len),
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.empty_cache()
        return self._processor.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

    def _build_merged_inputs_embeds(self, inputs) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        inputs_embeds = self._model.get_input_embeddings()(input_ids)

        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return inputs_embeds

        image_features = self._model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=inputs.get("image_sizes"),
            vision_feature_layer=getattr(self._model.config, "vision_feature_layer", None),
            vision_feature_select_strategy=getattr(
                self._model.config, "vision_feature_select_strategy", None
            ),
        )
        # image_features: [N_img_tokens, D] or list of tensors
        if isinstance(image_features, (list, tuple)):
            image_features = torch.cat(image_features, dim=0)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

        # Replace image placeholder tokens with visual features
        image_token_index = getattr(self._model.config, "image_token_index", 151646)
        image_mask = (input_ids == image_token_index)  # [B, L]
        special_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        return inputs_embeds

    def _target_keep_for_step(self, original_image_tokens: int, final_keep: int, step_idx: int, n_steps: int) -> int:
        if self.progressive_schedule == "immediate":
            return final_keep
        progress = float(step_idx + 1) / float(max(1, n_steps))
        keep = round(original_image_tokens - (original_image_tokens - final_keep) * progress)
        return max(final_keep, min(original_image_tokens, int(keep)))

    def _progressive_prefill(
        self,
        merged_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_positions: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        from transformers.cache_utils import DynamicCache

        lm = self._model.language_model
        hidden_states = merged_embeds
        active_orig_positions = torch.arange(prompt_len, device=self._device, dtype=torch.long)
        image_positions = image_positions.to(self._device)
        last_image_position = int(image_positions.max().item())

        n_image_original = int(image_positions.numel())
        final_image_keep = max(1, int(round(n_image_original * self.keep_ratio)))
        student_layers = set(int(li) for li in self.student.layer_indices)
        ordered_student_layers = [li for li in range(len(lm.model.layers)) if li in student_layers]

        if not self._reported_keep_budget:
            print(
                f"[lmms-onevision-student-progressive] keep_ratio={self.keep_ratio} "
                f"schedule={self.progressive_schedule} image_tokens={n_image_original} "
                f"final_image_tokens_kept={final_image_keep}",
                file=sys.stderr,
                flush=True,
            )
            self._reported_keep_budget = True

        past_kv = DynamicCache()

        for layer_idx, decoder_layer in enumerate(lm.model.layers[: lm.config.num_hidden_layers]):
            position_ids = active_orig_positions.unsqueeze(0)
            cache_position = active_orig_positions
            position_embeddings = lm.model.rotary_emb(hidden_states, position_ids)

            seq_cache_position = torch.arange(
                hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
            )
            causal_mask = lm.model._update_causal_mask(
                attention_mask=None,
                input_tensor=hidden_states,
                cache_position=seq_cache_position,
                past_key_values=None,
                output_attentions=False,
            )

            layer_out = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            if layer_idx not in student_layers:
                continue

            step_idx = ordered_student_layers.index(layer_idx)
            target_keep = self._target_keep_for_step(
                n_image_original,
                final_image_keep,
                step_idx,
                len(ordered_student_layers),
            )
            current_image_idx, current_question_idx = _current_token_indices(
                active_orig_positions,
                image_positions,
                last_image_position,
            )
            if current_image_idx.numel() <= target_keep:
                continue

            scores = self.student.layers[str(layer_idx)](
                hidden_states,
                current_image_idx,
                current_question_idx,
            ).squeeze(0)
            top = torch.topk(scores, k=target_keep, largest=True).indices
            keep_image = torch.zeros(current_image_idx.numel(), dtype=torch.bool, device=self._device)
            keep_image[top] = True

            keep_mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=self._device)
            keep_mask[current_image_idx] = keep_image

            _trim_computed_cache_layers(past_kv, layer_idx, keep_mask)
            hidden_states = hidden_states[:, keep_mask, :]
            active_orig_positions = active_orig_positions[keep_mask]

        hidden_states = lm.model.norm(hidden_states)

        n_image_kept = int(torch.isin(active_orig_positions, image_positions).sum().item())
        stats = {
            "n_image_original": n_image_original,
            "n_image_kept": n_image_kept,
            "prompt_len": prompt_len,
            "total_kept": int(active_orig_positions.numel()),
            "total_keep_ratio": active_orig_positions.numel() / max(1, prompt_len),
            "image_keep_ratio": n_image_kept / max(1, n_image_original),
        }
        return hidden_states, past_kv, active_orig_positions, stats
