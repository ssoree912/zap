# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache-budget helpers shared by evaluation wrappers."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING


LEGACY_TOTAL_ROUND = "legacy_total_round"
EXACT_TOTAL_CEIL = "exact_total_ceil"
TOTAL_BUDGET_MODES = frozenset({LEGACY_TOTAL_ROUND, EXACT_TOTAL_CEIL})


def validate_total_budget_mode(mode: str) -> str:
    """Return a normalized total-cache budget mode or raise."""
    normalized = str(mode).strip().lower()
    if normalized not in TOTAL_BUDGET_MODES:
        choices = ", ".join(sorted(TOTAL_BUDGET_MODES))
        raise ValueError(f"Unknown keep_budget_mode={mode!r}; expected one of: {choices}")
    return normalized


def requested_total_token_budget(
    *,
    prompt_len: int,
    keep_ratio: float,
    mode: str,
) -> int:
    """Compute the requested total prompt-cache token count."""
    prompt_len = int(prompt_len)
    keep_ratio = float(keep_ratio)
    mode = validate_total_budget_mode(mode)
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be nonnegative, got {prompt_len}")
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    if mode == EXACT_TOTAL_CEIL:
        exact_decimal = Decimal(str(keep_ratio)) * Decimal(prompt_len)
        return int(exact_decimal.to_integral_value(rounding=ROUND_CEILING))
    return int(round(keep_ratio * prompt_len))


def compute_image_keep_count(
    *,
    n_image: int,
    n_text: int,
    keep_ratio: float,
    mode: str,
) -> int:
    """Compute image tokens kept while preserving every text token.

    ``exact_total_ceil`` implements

    ``K_visual = clamp(ceil(keep_ratio * (N_image + N_text)) - N_text, 0, N_image)``.

    ``legacy_total_round`` preserves the original wrapper's minimum-one-image
    and Python ``round`` behavior.
    """
    n_image = int(n_image)
    n_text = int(n_text)
    keep_ratio = float(keep_ratio)
    mode = validate_total_budget_mode(mode)
    if n_image < 0 or n_text < 0:
        raise ValueError(
            f"n_image and n_text must be nonnegative, got {n_image} and {n_text}"
        )
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")

    prompt_len = n_image + n_text
    if mode == EXACT_TOTAL_CEIL:
        requested_total = requested_total_token_budget(
            prompt_len=prompt_len,
            keep_ratio=keep_ratio,
            mode=mode,
        )
        return min(n_image, max(0, requested_total - n_text))

    legacy_keep = max(
        1,
        int(round(n_image - (1.0 - keep_ratio) * prompt_len)),
    )
    return min(n_image, legacy_keep)
