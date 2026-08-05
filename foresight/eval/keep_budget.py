# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from typing_extensions import assert_never

KeepRatioBasis = Literal["total", "image"]


def normalize_keep_ratio_basis(value: str) -> KeepRatioBasis:
    normalized = value.strip().lower()
    match normalized:
        case "total":
            return "total"
        case "image":
            return "image"
        case _:
            raise ValueError(f"Expected keep_ratio_basis to be 'total' or 'image', got {value!r}")


def image_keep_budget(
    *,
    n_img: int,
    prompt_len: int,
    keep_ratio: float,
    basis: KeepRatioBasis,
) -> int:
    match basis:
        case "total":
            budget = n_img - (1.0 - keep_ratio) * prompt_len
        case "image":
            budget = keep_ratio * n_img
        case unreachable:
            assert_never(unreachable)
    return min(max(1, int(round(budget))), n_img)
