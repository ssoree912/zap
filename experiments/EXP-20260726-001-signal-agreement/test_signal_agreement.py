# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("analyze_llava15_signal_agreement.py")
SPEC = importlib.util.spec_from_file_location("signal_agreement", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_identical_pair_metrics_are_one() -> None:
    signal = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.3, 0.2]])
    metrics = MODULE.compute_pair_metrics(signal, signal)
    for values in metrics.values():
        assert np.allclose(values, 1.0)


def test_jaccard_uses_same_k_for_both_signals() -> None:
    first = np.arange(10, dtype=np.float64)
    second = first[::-1]
    assert MODULE._jaccard(first, first, 0.2) == 1.0
    assert MODULE._jaccard(first, second, 0.2) == 0.0


def test_smoothing_preserves_constant_grid_and_shape() -> None:
    signal = torch.ones(2, 24 * 24)
    smoothed = MODULE.smooth_visual_grid(signal, grid_h=24, grid_w=24)
    assert smoothed.shape == signal.shape
    assert torch.allclose(smoothed, signal)


def test_normalization_is_layerwise() -> None:
    signal = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    normalized = MODULE.normalize_nonnegative(signal, eps=1e-8)
    assert torch.allclose(normalized.sum(dim=-1), torch.ones(2))
    assert normalized[0, 1] > normalized[0, 0]
