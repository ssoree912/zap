# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class KVzapConfig(PretrainedConfig):
    model_type: str = "kvzap"
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    n_modules: int


class KVzapModel(PreTrainedModel):
    config_class = KVzapConfig  # type: ignore[assignment]

    def __init__(self, config: KVzapConfig):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        if config.hidden_dim is None:
            self.layers = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)
