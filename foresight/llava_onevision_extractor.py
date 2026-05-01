# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaVA-OneVision (HF) helpers.

OneVision uses AnyRes image tokenization, so the number of image tokens per
image is variable (depends on chosen image_grid_pinpoint + newline tokens).
Strategy: rely on the processor to fully expand `<image>` placeholders in
`input_ids`; image positions are simply every input_ids slot equal to
`image_token_index`.
"""

from __future__ import annotations

from typing import Any

import torch


def _get_image_token_id(config: Any) -> int:
    for attribute in ("image_token_index", "image_token_id"):
        if hasattr(config, attribute):
            return int(getattr(config, attribute))
    raise AttributeError("Could not find OneVision image token id on the model config")


def configure_onevision_processor(processor: Any, config: Any) -> Any:
    """No-op stub kept for API parity with `configure_llava_processor`.

    OneVision processors already carry all required vision metadata; the
    function exists so collector scripts can stay generic.
    """
    return processor


def infer_onevision_image_positions_no_forward(
    prompt_inputs: dict[str, Any],
    model_config: Any,
    num_images: int,
) -> tuple[torch.Tensor, int]:
    """Return (image_positions, prompt_len_mm) for OneVision prompts.

    The OneVision processor expands `<image>` placeholders into one token per
    visual feature directly inside `input_ids`, so we do not need an extra
    forward pass — every position whose input_ids equals `image_token_index`
    is part of the image span.
    """
    input_ids = prompt_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected `prompt_inputs['input_ids']` to be a [1, T] tensor")
    if num_images <= 0:
        raise ValueError(f"Expected a positive number of images, got {num_images}")

    attention_mask = prompt_inputs.get("attention_mask")
    if attention_mask is not None:
        valid_ids = input_ids[0][attention_mask[0].to(torch.bool)]
    else:
        valid_ids = input_ids[0]

    image_token_id = _get_image_token_id(model_config)
    positions = (valid_ids == image_token_id).nonzero(as_tuple=False).flatten().to(torch.long)

    if positions.numel() == 0:
        raise ValueError("No image placeholders found in OneVision input ids")

    prompt_len_mm = int(valid_ids.shape[0])
    if int(positions.max().item()) >= prompt_len_mm:
        raise ValueError(
            f"Image position {int(positions.max().item())} exceeds prompt length {prompt_len_mm}"
        )
    return positions, prompt_len_mm


def get_onevision_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate `layers` ModuleList inside the OneVision language model."""
    for path in (
        "language_model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
    ):
        obj: Any = model
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "__len__") and len(obj) > 0 and hasattr(obj[0], "self_attn"):
            return obj
    raise AttributeError("Could not locate decoder layers inside the OneVision model")
