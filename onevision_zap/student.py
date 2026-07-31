# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class StudentCheckpointError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim: int, *, expansion: int, kernel_size: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.dwconv(inputs)
        hidden = self.norm(hidden)
        hidden = self.pwconv1(hidden)
        hidden = self.act(hidden)
        return residual + self.pwconv2(hidden)


class VisualUtilityStudentLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        conv_dim: int,
        proj_dim: int,
        mlp_dim: int,
        num_conv_blocks: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv_1x1_proj = nn.Conv1d(hidden_dim, conv_dim, kernel_size=1)
        self.conv_blocks = nn.Sequential(
            *(
                ConvNeXt1DBlock(
                    conv_dim,
                    expansion=4,
                    kernel_size=kernel_size,
                )
                for _ in range(num_conv_blocks)
            ),
        )
        self.W_c = nn.Linear(conv_dim, proj_dim) if conv_dim != proj_dim else nn.Identity()
        self.W_h = nn.Linear(hidden_dim, proj_dim)
        self.W_q = nn.Linear(hidden_dim, proj_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(proj_dim * 5, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_positions: torch.Tensor,
        question_positions: torch.Tensor,
    ) -> torch.Tensor:
        image_states = hidden_states.index_select(1, image_positions)
        batch_size = int(hidden_states.shape[0])
        hidden_dim = int(hidden_states.shape[-1])
        if question_positions.numel() == 0:
            question_state = hidden_states.new_zeros((batch_size, hidden_dim))
        else:
            question_state = hidden_states.index_select(
                1,
                question_positions,
            ).mean(dim=1)

        convolved = self.conv_1x1_proj(
            image_states.permute(0, 2, 1).contiguous(),
        )
        convolved = self.conv_blocks(convolved)
        context = self.W_c(convolved.permute(0, 2, 1).contiguous())
        image_projection = self.W_h(image_states)
        question_projection = (
            self.W_q(question_state)
            .unsqueeze(1)
            .expand(
                -1,
                image_positions.numel(),
                -1,
            )
        )
        fused = torch.cat(
            (
                image_projection,
                context,
                question_projection,
                image_projection * question_projection,
                context * question_projection,
            ),
            dim=-1,
        )
        return self.mlp_head(fused).squeeze(-1)


class VisualUtilityStudent(nn.Module):
    def __init__(
        self,
        layer_indices: tuple[int, ...],
        hidden_dim: int,
        conv_dim: int,
        proj_dim: int,
        mlp_dim: int,
        num_conv_blocks: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.layer_indices = layer_indices
        self.layers = nn.ModuleDict(
            {
                str(layer_index): VisualUtilityStudentLayer(
                    hidden_dim,
                    conv_dim,
                    proj_dim,
                    mlp_dim,
                    num_conv_blocks,
                    kernel_size,
                )
                for layer_index in layer_indices
            },
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Path,
    ) -> VisualUtilityStudent:
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "pytorch_model.bin"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("variant", "full") != "full":
            raise StudentCheckpointError("only the full student variant is supported")
        layer_indices = tuple(int(index) for index in config["layer_indices"])
        model = cls(
            layer_indices=layer_indices,
            hidden_dim=int(config["hidden_dim"]),
            conv_dim=int(config["conv_dim"]),
            proj_dim=int(config["proj_dim"]),
            mlp_dim=int(config["mlp_dim"]),
            num_conv_blocks=int(config["num_conv_blocks"]),
            kernel_size=int(config["kernel_size"]),
        )
        state = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        return model
