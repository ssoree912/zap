# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.kvzap_press import KVzapModel


HeadReduce = Literal["amax", "mean"]



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


@dataclass
class ImageTokenTopKPress(BasePress):
    """Keep all non-image tokens and retain only the top-k image tokens per layer/head."""

    image_keep_ratio: float = 1.0
    head_reduce: HeadReduce = "amax"
    current_image_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert 0.0 <= self.image_keep_ratio <= 1.0, "image_keep_ratio must be in [0, 1]"

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        self.current_image_positions = image_positions.detach().cpu().long().flatten()

    def clear_sample_context(self) -> None:
        self.current_image_positions = None

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
        if self.image_keep_ratio >= 1.0:
            return keys, values
        if self.current_image_positions is None:
            raise RuntimeError("Image positions must be set before entering the press context")
        if keys.shape[0] != 1:
            raise ValueError("ImageTokenTopKPress currently only supports batch size 1")

        image_positions = self.current_image_positions.to(keys.device)
        image_positions = image_positions[(image_positions >= 0) & (image_positions < keys.shape[2])]
        if image_positions.numel() == 0:
            return keys, values

        n_image_keep = int(math.ceil(image_positions.numel() * self.image_keep_ratio))
        n_image_keep = min(image_positions.numel(), max(n_image_keep, 0))

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

        all_positions = torch.arange(keys.shape[2], device=keys.device, dtype=torch.long)
        non_image_mask = torch.ones(keys.shape[2], dtype=torch.bool, device=keys.device)
        non_image_mask[image_positions] = False
        non_image_positions = all_positions[non_image_mask]

        if n_image_keep > 0:
            image_keep_indices = torch.topk(score_tensor[0], k=n_image_keep, dim=-1).indices
            image_keep_positions = image_positions[image_keep_indices]
        else:
            image_keep_positions = torch.empty((keys.shape[1], 0), dtype=torch.long, device=keys.device)

        base_positions = non_image_positions.unsqueeze(0).expand(keys.shape[1], -1)
        keep_positions = torch.cat([base_positions, image_keep_positions], dim=-1)
        keep_positions, _ = torch.sort(keep_positions, dim=-1)

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
class ProbeImageTeacherPress(ImageTokenTopKPress):
    probe_model_name: str = ""
    loaded_probe_model_name: Optional[str] = field(default=None, init=False, repr=False)
    probe_model: Optional[KVzapModel] = field(default=None, init=False, repr=False)

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
        scores = probe_layer(image_hidden_states).transpose(1, 2)
        return scores
