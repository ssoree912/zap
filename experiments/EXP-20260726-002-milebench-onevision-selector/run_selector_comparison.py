#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MileBench selector comparison for original LLaVA-OneVision and Q-ViK.

The runner deliberately avoids materializing full prefill attention matrices.
For prefill saliency it reconstructs only user-query rows from each layer's
hidden states. Future utility is captured from one-token autoregressive steps.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half

ZAP_ROOT = Path("/workspace/nips/zap")
QVIK_ROOT = Path("/workspace/nips/Q-ViK")
for _root in (ZAP_ROOT, QVIK_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from kvpress.presses.visual_utility_student_onevision import (  # noqa: E402
    VisualUtilityStudentOneVision,
)
from qvik.llava_onevision.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from qvik.llava_onevision.conversation import (  # noqa: E402
    SeparatorStyle,
    conv_templates,
)
from qvik.llava_onevision.mm_utils import (  # noqa: E402
    process_images,
    tokenizer_image_token,
)
from qvik.llava_onevision.model.builder import load_pretrained_model  # noqa: E402

DATA_ROOT = ZAP_ROOT / "data" / "MileBench"
SELECTORS = ("full", "prefill", "smoothed_prefill", "qvik", "future_oracle")
PAIR_NAMES = (
    "prefill_vs_future",
    "smoothed_prefill_vs_future",
    "qvik_vs_future",
    "qvik_vs_prefill",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/nips/models/llava-onevision-qwen2-7b-ov"),
    )
    parser.add_argument(
        "--student-path",
        type=Path,
        default=ZAP_ROOT / "ckpts" / "student_onevision",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ZAP_ROOT
        / "artifacts"
        / "rebuttal_milebench_onevision_selector_total0p2",
    )
    parser.add_argument("--total-keep-ratio", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--query-chunk-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dataset(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    task_root = DATA_ROOT / dataset
    payload = json.loads((task_root / f"{dataset}.json").read_text())
    return payload["meta_data"], payload["data"], task_root / "combined_1_images"


def build_user_prompt(sample: dict[str, Any], meta: dict[str, Any]) -> str:
    ann = sample["task_instance"]
    task_instructions = meta["task_instruction"]
    instruction_id = int(sample["task_instruction_id"])
    if isinstance(task_instructions, list):
        task_instruction = task_instructions[instruction_id]
    else:
        task_instruction = task_instructions[str(instruction_id)]

    context = ann["context"]
    for index in range(1, len(ann["images_path"]) + 1):
        context = context.replace(f"{{image#{index}}}", f"<Image {index}> ")
        context = context.replace(f"{{table#{index}}}", f"<Image {index}> ")
    if ann.get("choice_list"):
        choices = "\n".join(
            f"{chr(65 + index)}. {choice}"
            for index, choice in enumerate(ann["choice_list"])
        )
        context += f"\nChoice List:\n{choices}\nYour answer is: "
    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}"


def build_prompt(user_prompt: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop


def tokenize_prompt(prompt: str, tokenizer: Any, device: torch.device) -> torch.Tensor:
    return tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)


def image_path(sample: dict[str, Any], combined_root: Path) -> Path:
    return combined_root / sample["task_instance"]["combined_1_images"][0]


def to_image_inputs(image_tensor: Any, device: torch.device) -> Any:
    if isinstance(image_tensor, torch.Tensor):
        return image_tensor.to(device=device, dtype=torch.float16)
    return [
        tensor.to(device=device, dtype=torch.float16) for tensor in image_tensor
    ]


def cache_seq_len(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if hasattr(cache, "key_cache"):
        return int(cache.key_cache[0].shape[-2])
    return int(cache[0][0].shape[-2])


def clone_cache(cache: Any) -> Any:
    if hasattr(cache, "key_cache"):
        cloned = DynamicCache()
        cloned.key_cache = [item.clone() for item in cache.key_cache]
        cloned.value_cache = [item.clone() for item in cache.value_cache]
        if hasattr(cache, "_seen_tokens"):
            cloned._seen_tokens = cache._seen_tokens
        return cloned
    if isinstance(cache, tuple):
        return tuple((key.clone(), value.clone()) for key, value in cache)
    return copy.deepcopy(cache)


def decoder_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers
    raise AttributeError("Could not locate decoder layers")


def infer_image_positions(
    raw_input_ids: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, int, int]:
    raw = raw_input_ids[0].detach().cpu()
    placeholders = torch.where(raw == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholders) != 1:
        raise ValueError(f"Expected one image placeholder, got {len(placeholders)}")
    placeholder = int(placeholders[0])
    feature_len = int(prompt_len - raw.numel() + 1)
    if feature_len <= 0:
        raise ValueError(
            f"Invalid image feature length {feature_len}; "
            f"prompt_len={prompt_len}, raw_len={raw.numel()}"
        )
    positions = torch.arange(placeholder, placeholder + feature_len, dtype=torch.long)
    return positions, placeholder, feature_len


def infer_user_query_positions(
    *,
    input_ids: torch.Tensor,
    image_feature_len: int,
    placeholder: int,
    user_prompt: str,
    conv_template: str,
    tokenizer: Any,
    device: torch.device,
) -> torch.Tensor:
    """Find the actual user-message tokens, excluding assistant/template suffix."""
    image_only_prompt, _ = build_prompt(DEFAULT_IMAGE_TOKEN, conv_template)
    image_only_ids = tokenize_prompt(image_only_prompt, tokenizer, device)[0]
    full_ids = input_ids[0]
    suffix = 0
    max_suffix = min(
        int(full_ids.numel() - placeholder - 1),
        int(image_only_ids.numel()),
    )
    while suffix < max_suffix:
        if int(full_ids[-1 - suffix]) != int(image_only_ids[-1 - suffix]):
            break
        suffix += 1
    raw_end = int(full_ids.numel()) - suffix
    raw_start = placeholder + 1
    if raw_end <= raw_start:
        raise ValueError(
            f"Could not infer user token span for prompt={user_prompt[:80]!r}"
        )
    shift = image_feature_len - 1
    return torch.arange(raw_start + shift, raw_end + shift, dtype=torch.long)


def broad_question_positions(prompt_len: int, image_positions: torch.Tensor) -> torch.Tensor:
    start = int(image_positions[-1]) + 1
    return torch.arange(start, prompt_len, dtype=torch.long)


def normalize_nonnegative(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = values.clamp_min(0)
    sums = values.sum(dim=-1, keepdim=True)
    normalized = values / sums.clamp_min(eps)
    bad = sums.squeeze(-1) <= eps
    if bad.any():
        normalized[bad] = 1.0 / values.shape[-1]
    return normalized


def normalize_student(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


@torch.no_grad()
def restricted_prefill_saliency(
    *,
    model: Any,
    hidden_states: tuple[torch.Tensor, ...],
    query_positions: torch.Tensor,
    visual_positions: torch.Tensor,
    query_chunk_size: int,
) -> torch.Tensor:
    """Reconstruct mean question-to-visual attention without an N x N tensor."""
    layers = decoder_layers(model)
    prompt_len = int(hidden_states[0].shape[1])
    all_position_ids = torch.arange(
        prompt_len, device=hidden_states[0].device, dtype=torch.long
    ).unsqueeze(0)
    query_positions = query_positions.to(hidden_states[0].device)
    visual_positions = visual_positions.to(hidden_states[0].device)
    rows: list[torch.Tensor] = []

    for layer_index, layer in enumerate(layers):
        attn = layer.self_attn
        layer_input = layer.input_layernorm(hidden_states[layer_index])
        batch_size = layer_input.shape[0]
        key_states = attn.k_proj(layer_input)
        key_states = key_states.view(
            batch_size,
            prompt_len,
            attn.num_key_value_heads,
            attn.head_dim,
        ).transpose(1, 2)
        cos_k, sin_k = attn.rotary_emb(key_states, all_position_ids)
        key_states = (
            key_states * cos_k.unsqueeze(1)
            + rotate_half(key_states) * sin_k.unsqueeze(1)
        )
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        visual_sum = torch.zeros(
            visual_positions.numel(), dtype=torch.float32, device=layer_input.device
        )
        count = 0
        for begin in range(0, query_positions.numel(), query_chunk_size):
            q_pos = query_positions[begin : begin + query_chunk_size]
            q_hidden = layer_input.index_select(1, q_pos)
            query_states = attn.q_proj(q_hidden)
            query_states = query_states.view(
                batch_size,
                q_pos.numel(),
                attn.num_heads,
                attn.head_dim,
            ).transpose(1, 2)
            q_position_ids = q_pos.unsqueeze(0)
            cos_q, sin_q = attn.rotary_emb(query_states, q_position_ids)
            query_states = (
                query_states * cos_q.unsqueeze(1)
                + rotate_half(query_states) * sin_q.unsqueeze(1)
            )
            weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(attn.head_dim)
            causal = torch.arange(
                prompt_len, device=weights.device
            ).view(1, 1, 1, -1) > q_pos.view(1, 1, -1, 1)
            weights.masked_fill_(causal, torch.finfo(weights.dtype).min)
            weights = torch.softmax(weights, dim=-1, dtype=torch.float32)
            selected = weights.index_select(-1, visual_positions)
            visual_sum += selected.sum(dim=(0, 1, 2))
            count += batch_size * attn.num_heads * int(q_pos.numel())
            del weights, selected, query_states, q_hidden
        rows.append((visual_sum / max(1, count)).cpu())
        del key_states, layer_input
    return normalize_nonnegative(torch.stack(rows))


@torch.no_grad()
def qvik_scores(
    *,
    student: VisualUtilityStudentOneVision,
    hidden_states: tuple[torch.Tensor, ...],
    visual_positions: torch.Tensor,
    question_positions: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states[0].device
    visual = visual_positions.to(device)
    question = question_positions.to(device)
    rows = []
    for layer_index in student.layer_indices:
        rows.append(
            student.forward_layer(
                layer_index,
                hidden_states[layer_index + 1],
                visual,
                question,
            )[0].float().cpu()
        )
    return normalize_student(torch.stack(rows))


def smooth_prefill_1d(prefill: torch.Tensor) -> torch.Tensor:
    """A local 3-token average for variable-length OneVision AnyRes tokens."""
    smoothed = F.avg_pool1d(
        prefill.unsqueeze(1),
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    ).squeeze(1)
    return normalize_nonnegative(smoothed)


@torch.no_grad()
def full_decode_and_future(
    *,
    model: Any,
    tokenizer: Any,
    base_cache: Any,
    first_token: torch.Tensor,
    prompt_len: int,
    visual_positions: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], torch.Tensor]:
    """Generate with normal SDPA, then replay fixed tokens to collect Future."""
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = 151645

    # Keep the Full Cache comparator on the model's normal SDPA path. Asking
    # Qwen2SdpaAttention for weights forces an eager fallback and can change
    # greedy token choices at near-ties, so attention capture is a replay.
    generation_cache = clone_cache(base_cache)
    answer: list[int] = [int(first_token.item())]
    current = first_token
    if answer[0] != int(eos):
        for step in range(max_new_tokens - 1):
            position = torch.tensor(
                [prompt_len + step],
                dtype=torch.long,
                device=first_token.device,
            )
            output = model(
                input_ids=current,
                past_key_values=generation_cache,
                cache_position=position,
                position_ids=position.unsqueeze(0),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            generation_cache = output.past_key_values
            current = output.logits[:, -1, :].argmax(-1, keepdim=True)
            token = int(current.item())
            answer.append(token)
            if token == int(eos):
                break
    del generation_cache

    cache = clone_cache(base_cache)
    layers = decoder_layers(model)
    visual = visual_positions.to(first_token.device)
    captured: dict[int, torch.Tensor] = {}
    originals: dict[int, Callable[..., Any]] = {}

    def patch(layer_index: int, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs["output_attentions"] = True
            output = original(*args, **kwargs)
            weights = output[1]
            if weights is not None:
                captured[layer_index] = (
                    weights[0, :, -1, :]
                    .index_select(-1, visual)
                    .float()
                    .mean(0)
                    .detach()
                    .cpu()
                )
            return (output[0], None) + output[2:]

        return wrapped

    for layer_index, layer in enumerate(layers):
        originals[layer_index] = layer.self_attn.forward
        layer.self_attn.forward = patch(layer_index, originals[layer_index])

    future_sum = torch.zeros(
        len(layers), visual_positions.numel(), dtype=torch.float32
    )
    attention_mask = torch.ones(
        (1, prompt_len), dtype=torch.long, device=first_token.device
    )
    try:
        for step, token in enumerate(answer):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            captured.clear()
            position = torch.tensor(
                [prompt_len + step],
                dtype=torch.long,
                device=first_token.device,
            )
            replay_token = torch.tensor(
                [[token]], dtype=torch.long, device=first_token.device
            )
            output = model(
                input_ids=replay_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=position,
                position_ids=position.unsqueeze(0),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            cache = output.past_key_values
            if len(captured) != len(layers):
                raise RuntimeError(
                    f"Captured {len(captured)}/{len(layers)} future-attention layers"
                )
            for layer_index in range(len(layers)):
                future_sum[layer_index] += captured[layer_index]
    finally:
        for layer_index, layer in enumerate(layers):
            layer.self_attn.forward = originals[layer_index]
    del cache
    return answer, normalize_nonnegative(future_sum / max(1, len(answer)))


def trim_cache(cache: Any, keep_masks: dict[int, torch.Tensor]) -> Any:
    if not hasattr(cache, "key_cache"):
        cache = DynamicCache.from_legacy_cache(cache)
    for layer_index in range(len(cache.key_cache)):
        mask = keep_masks[layer_index].to(cache.key_cache[layer_index].device)
        cache.key_cache[layer_index] = (
            cache.key_cache[layer_index][:, :, mask, :].contiguous()
        )
        cache.value_cache[layer_index] = (
            cache.value_cache[layer_index][:, :, mask, :].contiguous()
        )
    return cache


def make_keep_masks(
    *,
    scores: torch.Tensor,
    visual_positions: torch.Tensor,
    prompt_len: int,
    n_keep: int,
) -> dict[int, torch.Tensor]:
    masks = {}
    for layer_index in range(scores.shape[0]):
        keep_visual = torch.zeros(visual_positions.numel(), dtype=torch.bool)
        top = torch.topk(scores[layer_index], k=n_keep).indices
        keep_visual[top] = True
        mask = torch.ones(prompt_len, dtype=torch.bool)
        mask[visual_positions] = keep_visual
        masks[layer_index] = mask
    return masks


@torch.no_grad()
def pruned_decode(
    *,
    model: Any,
    tokenizer: Any,
    base_cache: Any,
    first_token: torch.Tensor,
    prompt_len: int,
    keep_masks: dict[int, torch.Tensor],
    max_new_tokens: int,
) -> list[int]:
    cache = trim_cache(clone_cache(base_cache), keep_masks)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = 151645
    output_ids = [int(first_token.item())]
    current = first_token
    if output_ids[0] == int(eos):
        return output_ids
    for step in range(max_new_tokens - 1):
        position = torch.tensor(
            [prompt_len + step],
            dtype=torch.long,
            device=first_token.device,
        )
        output = model(
            input_ids=current,
            past_key_values=cache,
            cache_position=position,
            position_ids=position.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        cache = output.past_key_values
        current = output.logits[:, -1, :].argmax(-1, keepdim=True)
        token = int(current.item())
        output_ids.append(token)
        if token == int(eos):
            break
    del cache
    return output_ids


def decode_text(tokenizer: Any, token_ids: list[int], stop: str) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if stop and text.endswith(stop):
        text = text[: -len(stop)].strip()
    return text


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denom) if denom > 0 else 0.0


def jaccard(first: np.ndarray, second: np.ndarray, ratio: float) -> float:
    k = max(1, min(first.size, int(round(first.size * ratio))))
    a = set(np.argpartition(first, -k)[-k:].tolist())
    b = set(np.argpartition(second, -k)[-k:].tolist())
    return len(a & b) / len(a | b)


def pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, list[float]]:
    result = {
        "spearman": [],
        "cosine": [],
        "jaccard_10": [],
        "jaccard_20": [],
        "jaccard_50": [],
    }
    for layer_index in range(first.shape[0]):
        a = first[layer_index].numpy()
        b = second[layer_index].numpy()
        rho = spearmanr(a, b).statistic
        result["spearman"].append(float(0.0 if np.isnan(rho) else rho))
        result["cosine"].append(cosine(a, b))
        result["jaccard_10"].append(jaccard(a, b, 0.10))
        result["jaccard_20"].append(jaccard(a, b, 0.20))
        result["jaccard_50"].append(jaccard(a, b, 0.50))
    return result


def total_budget(prompt_len: int, n_visual: int, ratio: float) -> int:
    n_keep = int(round(n_visual - (1.0 - ratio) * prompt_len))
    return max(1, min(n_visual, n_keep))


@torch.no_grad()
def evaluate_one(
    *,
    sample: dict[str, Any],
    meta: dict[str, Any],
    combined_root: Path,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudentOneVision,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(sample, meta)
    prompt, stop = build_prompt(user_prompt, args.conv_template)
    input_ids = tokenize_prompt(prompt, tokenizer, device)
    with Image.open(image_path(sample, combined_root)) as raw_image:
        image = raw_image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    image_tensor = to_image_inputs(image_tensor, device)

    prefill = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    base_cache = prefill.past_key_values
    if isinstance(base_cache, tuple):
        base_cache = DynamicCache.from_legacy_cache(base_cache)
    prompt_len = cache_seq_len(base_cache)
    visual_positions, placeholder, feature_len = infer_image_positions(
        input_ids, prompt_len
    )
    user_positions = infer_user_query_positions(
        input_ids=input_ids,
        image_feature_len=feature_len,
        placeholder=placeholder,
        user_prompt=user_prompt,
        conv_template=args.conv_template,
        tokenizer=tokenizer,
        device=device,
    )
    student_questions = broad_question_positions(prompt_len, visual_positions)
    prefill_scores = restricted_prefill_saliency(
        model=model,
        hidden_states=prefill.hidden_states,
        query_positions=user_positions,
        visual_positions=visual_positions,
        query_chunk_size=args.query_chunk_size,
    )
    student_scores = qvik_scores(
        student=student,
        hidden_states=prefill.hidden_states,
        visual_positions=visual_positions,
        question_positions=student_questions,
    )
    smooth_scores = smooth_prefill_1d(prefill_scores)
    first_token = prefill.logits[:, -1, :].argmax(-1, keepdim=True)
    del image_tensor, prefill

    full_ids, future_scores = full_decode_and_future(
        model=model,
        tokenizer=tokenizer,
        base_cache=base_cache,
        first_token=first_token,
        prompt_len=prompt_len,
        visual_positions=visual_positions,
        max_new_tokens=args.max_new_tokens,
    )
    scores_by_selector = {
        "prefill": prefill_scores,
        "smoothed_prefill": smooth_scores,
        "qvik": student_scores,
        "future_oracle": future_scores,
    }
    n_visual = int(visual_positions.numel())
    n_keep = total_budget(prompt_len, n_visual, args.total_keep_ratio)
    predictions = {"full": decode_text(tokenizer, full_ids, stop)}
    for selector, scores in scores_by_selector.items():
        masks = make_keep_masks(
            scores=scores,
            visual_positions=visual_positions,
            prompt_len=prompt_len,
            n_keep=n_keep,
        )
        token_ids = pruned_decode(
            model=model,
            tokenizer=tokenizer,
            base_cache=base_cache,
            first_token=first_token,
            prompt_len=prompt_len,
            keep_masks=masks,
            max_new_tokens=args.max_new_tokens,
        )
        predictions[selector] = decode_text(tokenizer, token_ids, stop)

    pairs = {
        "prefill_vs_future": (prefill_scores, future_scores),
        "smoothed_prefill_vs_future": (smooth_scores, future_scores),
        "qvik_vs_future": (student_scores, future_scores),
        "qvik_vs_prefill": (student_scores, prefill_scores),
    }
    agreement = {
        name: pair_metrics(first, second)
        for name, (first, second) in pairs.items()
    }
    del base_cache
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "sample_id": int(sample["sample_id"]),
        "gt_response": sample["response"],
        "predictions": predictions,
        "agreement": agreement,
        "prompt_len": prompt_len,
        "n_text": prompt_len - n_visual,
        "n_visual": n_visual,
        "n_visual_kept": n_keep,
        "requested_total_keep_ratio": args.total_keep_ratio,
        "actual_total_keep_ratio": (prompt_len - n_visual + n_keep) / prompt_len,
        "answer_tokens": len(full_ids),
        "user_query_tokens": int(user_positions.numel()),
        "student_question_tokens": int(student_questions.numel()),
        "smoothing": "1D average, kernel=3, for variable-length AnyRes sequence",
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.total_keep_ratio <= 1.0:
        raise ValueError("--total-keep-ratio must be in [0, 1]")
    device = torch.device(args.device)
    output_dir = args.output_root / args.dataset
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    meta, samples, combined_root = load_dataset(args.dataset)
    samples = samples[args.start_index :]
    if args.limit is not None:
        samples = samples[: args.limit]

    print(
        f"[load] dataset={args.dataset} n={len(samples)} model={args.model_path} "
        f"student={args.student_path} device={device}",
        flush=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=str(args.model_path),
        model_base=None,
        model_name="llava_qwen",
        device_map=args.device_map,
        attn_implementation="sdpa",
        multimodal=True,
    )
    model.eval()
    device = next(model.parameters()).device
    student = VisualUtilityStudentOneVision.from_pretrained(
        args.student_path, map_location="cpu"
    ).to(device=device, dtype=torch.float16).eval()
    print(
        f"[loaded] model={model.__class__.__name__} layers={len(decoder_layers(model))} "
        f"student_layers={len(student.layer_indices)}",
        flush=True,
    )

    failures = []
    started = time.time()
    for sample in tqdm(samples, desc=args.dataset):
        sample_id = int(sample["sample_id"])
        record_path = records_dir / f"{sample_id:06d}.json"
        if record_path.exists() and not args.overwrite:
            continue
        try:
            result = evaluate_one(
                sample=sample,
                meta=meta,
                combined_root=combined_root,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                student=student,
                args=args,
                device=device,
            )
            atomic_json(record_path, result)
        except Exception as error:  # noqa: BLE001
            failure = {"sample_id": sample_id, "error": repr(error)}
            failures.append(failure)
            print(f"[failure] {failure}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    records = [
        json.loads(path.read_text()) for path in sorted(records_dir.glob("*.json"))
    ]
    for selector in SELECTORS:
        predictions = [
            {
                "sample_id": record["sample_id"],
                "pred_response": record["predictions"][selector],
                "gt_response": record["gt_response"],
            }
            for record in records
        ]
        selector_dir = output_dir / selector
        selector_dir.mkdir(exist_ok=True)
        atomic_json(selector_dir / "pred.json", predictions)
    run_summary = {
        "dataset": args.dataset,
        "n_requested": len(samples),
        "n_completed": len(records),
        "n_failures": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "model_path": str(args.model_path),
        "student_path": str(args.student_path),
        "total_keep_ratio": args.total_keep_ratio,
        "max_new_tokens": args.max_new_tokens,
    }
    atomic_json(output_dir / "run_summary.json", run_summary)
    print(f"[done] {json.dumps(run_summary, ensure_ascii=False)}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
