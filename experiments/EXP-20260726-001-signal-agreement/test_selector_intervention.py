# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_llava15_selector_intervention.py")
SPEC = importlib.util.spec_from_file_location("selector_intervention", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_total_keep_budget_matches_original_wrapper_formula() -> None:
    n_visual = 576
    prompt_len = 640
    n_keep = MODULE.compute_n_visual_keep(
        n_visual=n_visual,
        prompt_len=prompt_len,
        visual_keep_ratio=None,
        total_keep_ratio=0.2,
    )
    assert n_keep == round(n_visual - 0.8 * prompt_len)
    assert (prompt_len - n_visual + n_keep) / prompt_len == 0.2


def test_total_keep_budget_preserves_at_least_one_visual_token() -> None:
    assert MODULE.compute_n_visual_keep(
        n_visual=576,
        prompt_len=800,
        visual_keep_ratio=None,
        total_keep_ratio=0.2,
    ) == 1


def test_visual_keep_budget_remains_backward_compatible() -> None:
    assert MODULE.compute_n_visual_keep(
        n_visual=576,
        prompt_len=640,
        visual_keep_ratio=0.2,
        total_keep_ratio=None,
    ) == 115


def test_layer_jaccard_at_k_uses_exact_count() -> None:
    first = MODULE.torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    second = MODULE.torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    assert MODULE.layer_jaccard_at_k(first, second, k=2) == [1.0 / 3.0]
