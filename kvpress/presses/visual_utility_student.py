# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual-Utility Student model for LLaVA-1.5 (2D ConvNeXt, grid 24x24).

Checkpoint layout (pytorch_model.bin):
    layers.{li}.conv_1x1_proj.{weight,bias}
    layers.{li}.conv_blocks.{n}.dwconv.{weight,bias}
    layers.{li}.conv_blocks.{n}.norm.{weight,bias}
    layers.{li}.conv_blocks.{n}.pwconv1.{weight,bias}
    layers.{li}.conv_blocks.{n}.pwconv2.{weight,bias}
    layers.{li}.W_h.{weight,bias}
    layers.{li}.W_q.{weight,bias}
    layers.{li}.mlp_head.{0,2}.{weight,bias}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        # channels-last LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        return x + residual


class VisualUtilityStudentLayer(nn.Module):
    """Per-layer scorer f_θ^{(l)} for LLaVA-1.5.

    forward(hidden_states, img_idx, q_idx) → scores [B, N_I]
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        conv_dim: int = 256,
        proj_dim: int = 256,
        mlp_dim: int = 512,
        num_conv_blocks: int = 2,
        kernel_size: int = 7,
        grid_h: int = 24,
        grid_w: int = 24,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # CNN encoder g_φ
        self.conv_1x1_proj = nn.Conv2d(hidden_dim, conv_dim, 1)
        self.conv_blocks = nn.ModuleList(
            [_ConvNeXtBlock(conv_dim, kernel_size) for _ in range(num_conv_blocks)]
        )

        # token-level projections
        self.W_h = nn.Linear(hidden_dim, proj_dim)  # v_feat
        self.W_q = nn.Linear(hidden_dim, proj_dim)  # q_feat

        # fusion MLP: [v, c, q, v⊙q, c⊙q] → 5*proj_dim → mlp_dim → 1
        self.mlp_head = nn.Sequential(
            nn.Linear(5 * proj_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, D]
        img_idx: torch.Tensor,         # [N_I]  CPU or same device
        q_idx: torch.Tensor,           # [N_Q]  CPU or same device
    ) -> torch.Tensor:                 # [B, N_I]
        B = hidden_states.shape[0]
        img_idx = img_idx.to(hidden_states.device)
        q_idx = q_idx.to(hidden_states.device)

        img_h = hidden_states[:, img_idx, :]          # [B, N_I, D]
        N_I = img_h.shape[1]
        tokens_per_image = self.grid_h * self.grid_w  # 576

        # ── CNN encoder ──────────────────────────────────────────────
        # Process each image chunk separately (multi-image: N_I may be k*576)
        n_chunks = max(1, (N_I + tokens_per_image - 1) // tokens_per_image)
        c_chunks = []
        for ci in range(n_chunks):
            start = ci * tokens_per_image
            end = min(start + tokens_per_image, N_I)
            chunk = img_h[:, start:end, :]               # [B, chunk_size, D]
            chunk_size = chunk.shape[1]
            if chunk_size < tokens_per_image:
                # pad last chunk to full grid size
                pad = img_h.new_zeros(B, tokens_per_image - chunk_size, img_h.shape[2])
                chunk = torch.cat([chunk, pad], dim=1)

            grid = chunk.reshape(B, self.grid_h, self.grid_w, -1)
            grid = grid.permute(0, 3, 1, 2).contiguous()     # [B, D, H, W]
            c_grid = self.conv_1x1_proj(grid)                 # [B, conv_dim, H, W]
            for blk in self.conv_blocks:
                c_grid = blk(c_grid)
            c_out = c_grid.permute(0, 2, 3, 1).reshape(B, tokens_per_image, -1)
            c_chunks.append(c_out[:, :end - start, :])       # trim padding

        c_feat = torch.cat(c_chunks, dim=1)                   # [B, N_I, conv_dim]

        # ── token projections ────────────────────────────────────────
        v_feat = self.W_h(img_h)                        # [B, N_I, proj_dim]

        if q_idx.numel() > 0:
            q_pool = hidden_states[:, q_idx, :].mean(dim=1)  # [B, D]
        else:
            q_pool = hidden_states.mean(dim=1)
        q_feat = self.W_q(q_pool).unsqueeze(1).expand(-1, N_I, -1)  # [B, N_I, proj_dim]

        # ── fusion + MLP ─────────────────────────────────────────────
        fused = torch.cat(
            [v_feat, c_feat, q_feat, v_feat * q_feat, c_feat * q_feat],
            dim=-1,
        )                                               # [B, N_I, 5*proj_dim]
        scores = self.mlp_head(fused).squeeze(-1)       # [B, N_I]
        return scores


class VisualUtilityStudentLlava15(nn.Module):
    """Container holding one VisualUtilityStudentLayer per transformer layer.

    Loads from a checkpoint directory containing:
        config.json  (scope, layer_indices, hidden_dim, conv_dim, …)
        pytorch_model.bin  (state_dict with keys layers.{li}.*)
    """

    def __init__(self, layer_indices: list[int], layer_kwargs: dict):
        super().__init__()
        self.layer_indices: list[int] = layer_indices
        self.layers = nn.ModuleDict(
            {str(li): VisualUtilityStudentLayer(**layer_kwargs) for li in layer_indices}
        )

    @classmethod
    def from_pretrained(cls, ckpt_dir: str) -> "VisualUtilityStudentLlava15":
        path = Path(ckpt_dir)
        with open(path / "config.json") as f:
            cfg = json.load(f)

        layer_indices: list[int] = cfg["layer_indices"]
        layer_kwargs = {
            "hidden_dim":       cfg.get("hidden_dim", 4096),
            "conv_dim":         cfg.get("conv_dim", 256),
            "proj_dim":         cfg.get("proj_dim", 256),
            "mlp_dim":          cfg.get("mlp_dim", 512),
            "num_conv_blocks":  cfg.get("num_conv_blocks", 2),
            "kernel_size":      cfg.get("kernel_size", 7),
            "grid_h":           cfg.get("grid_h", 24),
            "grid_w":           cfg.get("grid_w", 24),
        }

        model = cls(layer_indices, layer_kwargs)

        state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=False)
        # Remap flat keys (layers.0.*) → ModuleDict keys (layers.{str(li)}.*)
        # The checkpoint uses sequential indices 0..N-1 for layer_indices[0..N-1]
        remapped: dict[str, torch.Tensor] = {}
        for raw_key, tensor in state.items():
            parts = raw_key.split(".", 2)   # ["layers", "0", "conv_1x1_proj.weight"]
            if len(parts) == 3 and parts[0] == "layers":
                seq_idx = int(parts[1])
                if seq_idx < len(layer_indices):
                    li = layer_indices[seq_idx]
                    remapped[f"layers.{li}.{parts[2]}"] = tensor
                else:
                    remapped[raw_key] = tensor
            else:
                remapped[raw_key] = tensor

        model.load_state_dict(remapped, strict=True)
        return model
