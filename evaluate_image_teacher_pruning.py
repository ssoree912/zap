#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent
LOOKM_CANDIDATES = [REPO_ROOT.parent / "LOOK-M", REPO_ROOT.parent / "look-m"]
LOOKM_ROOT = next((path for path in LOOKM_CANDIDATES if path.is_dir()), LOOKM_CANDIDATES[0])
if str(LOOKM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOKM_ROOT))

from utils import MileBenchDataset

from kvpress.presses.image_token_press import (
    H2OImageOnlyPress,
    OracleAllTokenPress,
    OracleImageTeacherPress,
    PreselectedImagePress,
    ProbeImageTeacherPress,
    _compute_n_image_keep,
)
from kvzap.image_teacher_utils import build_prompt, load_pt_record, load_vlm_samples, normalize_answer, resolve_teacher_dir
from kvzap.llava_extractor import (
    _get_model_device,
    _get_model_float_dtype,
    _move_batch_to_device,
    _trim_after_eos,
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)
from kvzap.milebench_look_metrics import LookMileBenchEvaluator
from kvzap.phase2_metrics import aggregate_phase2_records, compute_oracle_keep_masks, summarize_phase2_sample

HD_DOCVQA_DATASET_PATH = "/workspace/zap/data/MileBench/DocVQA/DocVQA.json"
HD_DOCVQA_IMAGE_ROOT = "/workspace/zap/data/MileBench/DocVQA/images"
HD_ZAP_ARTIFACT_ROOT = "/workspace/zap/artifacts/combine_prob"
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{image#\d+\}")


def decode_answer(processor: Any, generated_ids: torch.Tensor, prompt_len_text: int) -> str:
    answer_ids = _trim_after_eos(generated_ids, processor.tokenizer.eos_token_id)[prompt_len_text:]
    return processor.tokenizer.decode(
        answer_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def open_images(image_paths: list[str]):
    from PIL import Image

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images[0] if len(images) == 1 else images


def resolve_teacher_image_positions(teacher_record: dict[str, Any]) -> torch.Tensor:
    if "image_pos_mm" in teacher_record:
        return teacher_record["image_pos_mm"].long().flatten()
    if "image_indices_mm" in teacher_record:
        return teacher_record["image_indices_mm"].long().flatten()
    if "is_image_pos_mm" in teacher_record:
        mask = teacher_record["is_image_pos_mm"]
        if not isinstance(mask, torch.Tensor):
            raise TypeError("teacher_record['is_image_pos_mm'] must be a tensor")
        return mask.nonzero(as_tuple=False).flatten().long()
    raise KeyError("Teacher record must contain one of: image_pos_mm, image_indices_mm, is_image_pos_mm")


def extract_att_only_postvision_on_the_fly(
    *,
    model: Any,
    prompt_inputs: dict[str, torch.Tensor],
    image_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute att_only_postvision teacher scores from a full forward pass.

    Returns:
      [n_layers, n_heads, n_image]
    """
    if image_positions.numel() == 0:
        raise ValueError("image_positions is empty")

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    img_idx = image_positions.to(device=device, dtype=torch.long)
    last_image_pos = int(img_idx.max().item())

    all_pos = torch.arange(prompt_len, dtype=torch.long, device=device)
    is_image = torch.zeros(prompt_len, dtype=torch.bool, device=device)
    is_image[img_idx] = True
    is_text = ~is_image

    postvision_text_idx = all_pos[(all_pos > last_image_pos) & is_text]
    if postvision_text_idx.numel() == 0:
        text_idx = all_pos[is_text]
        n_text = int(text_idx.numel())
        postvision_text_idx = text_idx[max(0, n_text - max(1, n_text // 10)):]
    if postvision_text_idx.numel() == 0:
        raise ValueError("Failed to resolve postvision text indices")

    with torch.no_grad():
        out = model(
            **prompt_inputs,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

    if out.attentions is None:
        raise RuntimeError("Model forward returned no attentions")

    layers: list[torch.Tensor] = []
    for layer_attn in out.attentions:
        attn = layer_attn[0].detach() if layer_attn.dim() == 4 else layer_attn.detach()  # [H,S,S]
        block = attn.index_select(1, postvision_text_idx).index_select(2, img_idx)  # [H,Q,I]
        layers.append(block.amax(dim=1).cpu().to(torch.float32))  # [H,I]

    return torch.stack(layers, dim=0)


def _vote_topk(per_layer_scores: list[torch.Tensor], k: int) -> torch.Tensor:
    """Vote-based pool reduction using per-layer top-k.

    Each layer independently casts top-k votes. Final pool: top-k tokens by vote
    count, ties broken by max score across all layers.

    Args:
        per_layer_scores: list of (n_pool,) CPU float tensors, one per probe layer.
                          Each value is the head-aggregated score for that pool token.
        k: target pool size.

    Returns:
        (k,) long tensor of indices into the current pool (CPU).
    """
    n_pool = per_layer_scores[0].numel()
    k = min(k, n_pool)

    vote_counts = torch.zeros(n_pool, dtype=torch.long)
    max_scores = torch.full((n_pool,), float("-inf"))

    for layer_scores in per_layer_scores:
        layer_k = min(k, n_pool)
        topk_idx = torch.topk(layer_scores, k=layer_k).indices
        vote_counts[topk_idx] += 1
        max_scores = torch.maximum(max_scores, layer_scores)

    # Lexicographic sort: primary = vote_count (desc), secondary = max_score (desc).
    # Encode as combined = vote_count * (n_pool + 1) + rank_within_score_bucket.
    score_min, score_max = max_scores.min(), max_scores.max()
    score_range = score_max - score_min
    norm = (max_scores - score_min) / score_range if score_range > 0 else torch.zeros_like(max_scores)
    combined = vote_counts.float() * (n_pool + 1) + norm
    return torch.topk(combined, k=k).indices


def _iterative_probe_preselect(
    *,
    model: Any,
    prompt_inputs: dict[str, Any],
    image_positions: torch.Tensor,
    probe_press: Any,
    n_rounds: int,
    n_image_keep: int,
    device: torch.device,
    float_dtype: Optional[torch.dtype],
    layerwise: bool = False,
) -> torch.Tensor:
    """Run n_rounds LLaVA forward passes, narrowing the image token pool each round.

    Round 1: normal forward on all tokens → probe scores → keep top schedule[0].
    Rounds 2+: inputs_embeds from round-1 embedding output, with evicted image
               positions zeroed in attention_mask → updated hidden states →
               probe re-scores surviving tokens → keep top schedule[i].

    Hidden states genuinely change each round because evicted tokens no longer
    contribute to attention of surviving tokens.

    Returns: (n_image_keep,) long tensor of final surviving mm-space positions (CPU).
    """
    n_image = image_positions.numel()
    n_image_keep = min(max(0, n_image_keep), n_image)

    if n_image_keep >= n_image or n_rounds <= 1:
        return image_positions

    # Linear schedule: n_image → n_image_keep over n_rounds steps
    schedule = [
        max(n_image_keep, int(math.ceil(n_image - (n_image - n_image_keep) * (i + 1) / n_rounds)))
        for i in range(n_rounds)
    ]
    schedule[-1] = n_image_keep

    img_pos_dev = image_positions.to(device=device, dtype=torch.long)
    pool_cpu = torch.arange(n_image, dtype=torch.long)  # indices into image_positions
    probe_model = probe_press.probe_model
    n_probe_layers = len(probe_model.layers)
    merged_embeds: Optional[torch.Tensor] = None

    for round_idx, round_k in enumerate(schedule):
        current_img_pos = img_pos_dev[pool_cpu.to(device)]  # (|pool|,) mm-space on device

        if round_idx == 0:
            with torch.no_grad():
                out = model(
                    **prompt_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # hidden_states[0] = embedding layer output = merged input embeddings [1, mm_len, D]
            merged_embeds = out.hidden_states[0].detach().clone()
        else:
            mm_len = merged_embeds.shape[1]
            # Attention mask: 1 for all non-image positions + surviving image positions, 0 for evicted
            is_image = torch.zeros(mm_len, dtype=torch.bool, device=device)
            is_image[img_pos_dev[img_pos_dev < mm_len]] = True  # guard against out-of-range
            attn_mask = (~is_image).long().unsqueeze(0)  # text positions = 1, all image = 0
            attn_mask[0, current_img_pos] = 1            # surviving image positions = 1

            with torch.no_grad():
                # pixel_values intentionally omitted: merged_embeds already contains vision features.
                # input_ids intentionally omitted: inputs_embeds takes precedence in LLaVA forward.
                out = model(
                    inputs_embeds=merged_embeds,
                    attention_mask=attn_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

        # Score current pool tokens with probe at each transformer layer
        per_layer_scores: list[torch.Tensor] = []
        for layer_idx in range(n_probe_layers):
            hs = out.hidden_states[layer_idx + 1]  # [1, mm_len, D] output of layer layer_idx
            probe_layer = probe_model.layers[layer_idx].to(device=hs.device, dtype=hs.dtype).eval()
            pool_hs = hs[:, current_img_pos, :]            # [1, |pool|, D]
            with torch.no_grad():
                scores = probe_layer(pool_hs).transpose(1, 2)  # [1, n_heads, |pool|]
            agg = scores[0].amax(dim=0).detach().cpu().float()  # (|pool|,) amax across heads
            per_layer_scores.append(agg)

        # Free GPU memory: hidden states from all layers are no longer needed after scoring
        del out

        k = min(round_k, pool_cpu.numel())
        if layerwise:
            # Per-layer vote: each layer votes for its top-k, ties broken by max score.
            local_topk = _vote_topk(per_layer_scores, k=k)
        else:
            # Global consensus: amax across all probe layers → single score per pool token.
            global_scores = torch.stack(per_layer_scores, dim=0).amax(dim=0)  # (|pool|,)
            local_topk = torch.topk(global_scores, k=k).indices
        pool_cpu = pool_cpu[local_topk]

    return image_positions[pool_cpu]  # (n_image_keep,) CPU


def build_press(args: argparse.Namespace):
    # Determine budget: prefer total_keep_ratio (unified basis); fall back to image_keep_ratio
    total_keep_ratio = getattr(args, "total_keep_ratio", None)
    image_keep_ratio = getattr(args, "image_keep_ratio", None)
    if total_keep_ratio is None and image_keep_ratio is None:
        raise ValueError("One of --total_keep_ratio or --image_keep_ratio is required")

    # Positional forced-keep kwargs (EXP-20260412-003)
    forced_kwargs = dict(
        n_initial_keep=getattr(args, "n_initial_keep", 0),
        n_recent_keep=getattr(args, "n_recent_keep", 0),
        n_random_keep=getattr(args, "n_random_keep", 0),
        per_image_forced=getattr(args, "per_image_forced", False),
        n_iterative_rounds=getattr(args, "n_iterative_rounds", 1),
    )

    if args.mode in ("oracle", "oracle_onthefly"):
        return OracleImageTeacherPress(
            total_keep_ratio=total_keep_ratio,
            image_keep_ratio=image_keep_ratio,
            head_reduce=args.head_reduce,
            **forced_kwargs,
        )
    if args.mode == "oracle_all_token":
        if total_keep_ratio is None:
            raise ValueError("--total_keep_ratio is required for oracle_all_token mode")
        return OracleAllTokenPress(total_keep_ratio=total_keep_ratio, head_reduce=args.head_reduce)
    if args.mode == "h2o_image_only":
        return H2OImageOnlyPress(
            total_keep_ratio=total_keep_ratio,
            image_keep_ratio=image_keep_ratio,
            head_reduce=args.head_reduce,
            **forced_kwargs,
        )
    return ProbeImageTeacherPress(
        total_keep_ratio=total_keep_ratio,
        image_keep_ratio=image_keep_ratio,
        head_reduce=args.head_reduce,
        probe_model_name=args.probe_model_name,
        **forced_kwargs,
    )


def load_core_annotation(dataset_path: str) -> Optional[dict[str, Any]]:
    path = Path(dataset_path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("meta_data"), dict):
        return None
    if not isinstance(payload.get("data"), list):
        return None
    return payload


def resolve_task_instruction(task_instructions: Any, task_instruction_id: Any) -> str:
    if isinstance(task_instructions, list):
        try:
            index = int(task_instruction_id)
            if 0 <= index < len(task_instructions):
                return str(task_instructions[index])
        except Exception:  # noqa: BLE001
            pass
    if isinstance(task_instructions, dict):
        if task_instruction_id in task_instructions:
            return str(task_instructions[task_instruction_id])
        key = str(task_instruction_id)
        if key in task_instructions:
            return str(task_instructions[key])
    return ""


def choice_label(index: int) -> str:
    if index < 26:
        return chr(65 + index)
    if index < 52:
        return "A" + chr(65 + index - 26)
    return "B" + chr(65 + index - 52)


def build_choice_block(choice_list: Any, dataset_name: str) -> str:
    if not isinstance(choice_list, list) or not choice_list:
        return ""

    lines: list[str] = []
    for idx, choice in enumerate(choice_list):
        choice_text = str(choice)
        if dataset_name == "GPR1200":
            lines.append(choice_text)
        else:
            lines.append(f"{choice_label(idx)}. {choice_text}")

    return "\nChoice list: \n" + "\n".join(lines) + "\nYour answer is: "


def build_look_question(record: dict[str, Any], core_annotation: dict[str, Any], dataset_name: str) -> str:
    task = record.get("task_instance", {})
    context = str(task.get("context", "")).strip()
    meta = core_annotation.get("meta_data", {})
    task_instruction = resolve_task_instruction(meta.get("task_instruction"), record.get("task_instruction_id", 0))
    choice_block = build_choice_block(task.get("choice_list"), dataset_name=dataset_name)
    combined = f"{task_instruction}\n{context}{choice_block}"
    return combined.strip()


def replace_image_placeholders_for_export(text: str) -> str:
    return IMAGE_PLACEHOLDER_PATTERN.sub("<ImageHere>", text)


def to_int_if_possible(value: Any) -> Any:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def subset_core_annotation(core_annotation: dict[str, Any], prediction_ids: set[Any]) -> dict[str, Any]:
    subset = [item for item in core_annotation["data"] if item.get("sample_id") in prediction_ids]
    return {
        "meta_data": core_annotation["meta_data"],
        "data": subset,
    }


def prepare_lookm_truncated_inputs(
    samples: list[dict[str, Any]],
    *,
    core_annotation: dict[str, Any],
    image_root: str,
    tokenizer: Any,
    dataset_name: str,
    max_context_len: int,
    n_tokens_per_image: int,
    combine_image: int | None = None,
) -> dict[str, dict[str, Any]]:
    raw_annotations: list[dict[str, Any]] = []
    for sample in samples:
        raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else None
        if raw is None:
            raise ValueError("LOOK-M style truncation requires `raw` sample records")
        raw_annotations.append(raw)

    dataset = MileBenchDataset(
        annotation=raw_annotations,
        task_instructions=core_annotation["meta_data"]["task_instruction"],
        img_dir=image_root,
        max_context_len=max_context_len,
        n_tokens_per_image=n_tokens_per_image,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        combine_image=combine_image,
    )

    prepared: dict[str, dict[str, Any]] = {}
    for idx, sample in enumerate(samples):
        item = dataset[idx]
        prepared[str(sample["sample_id"])] = {
            "question": item["context"],
            "image_paths": item["raw_img_list"],
        }
    return prepared


def build_look_prediction_record(
    sample: dict[str, Any],
    question_for_export: str,
    prediction: str,
    gold: Optional[str],
    args: argparse.Namespace,
    image_paths: list[str],
) -> dict[str, Any]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    raw_sample_id = raw.get("sample_id", sample["sample_id"])
    return {
        "sample_id": to_int_if_possible(raw_sample_id),
        "image": image_paths,
        "question": question_for_export,
        "gt_response": "" if gold is None else str(gold),
        "gen_model_id": args.look_model_name,
        "pred_response": prediction,
        "gen_kwargs": {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["oracle", "oracle_onthefly", "probe", "h2o_image_only", "oracle_all_token"], required=True)
    parser.add_argument("--dataset_path", type=str, default=HD_DOCVQA_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=f"{HD_ZAP_ARTIFACT_ROOT}/docvqa_image_pruning")
    parser.add_argument("--implementation_model_name", type=str, default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--image_root", type=str, default=HD_DOCVQA_IMAGE_ROOT)
    parser.add_argument("--image_column", type=str, default="images_path")
    parser.add_argument("--answer_column", type=str, default=None)
    parser.add_argument("--teacher_dir", type=str, default=None)
    parser.add_argument("--teacher_score_name", type=str, default="splus_postvision")
    parser.add_argument("--probe_model_name", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default="USER: <image>\n{question}\nASSISTANT:")
    parser.add_argument("--prompt_style", choices=["default", "look_milebench"], default="look_milebench")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default="none")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--image_keep_ratio", type=float, default=None,
                        help="Fraction of IMAGE tokens to keep (legacy). Use --total_keep_ratio for unified comparison.")
    parser.add_argument("--total_keep_ratio", type=float, default=None,
                        help="Fraction of ALL tokens (text+image) to keep. Makes r_eff_prompt comparable across methods.")
    parser.add_argument("--head_reduce", choices=["amax", "mean"], default="amax")
    parser.add_argument("--n_initial_keep", type=int, default=0,
                        help="Force-keep first N image tokens (global or per-image). EXP-20260412-003.")
    parser.add_argument("--n_recent_keep", type=int, default=0,
                        help="Force-keep last N image tokens (global or per-image). EXP-20260412-003.")
    parser.add_argument("--n_random_keep", type=int, default=0,
                        help="Force-keep N randomly chosen image tokens (control). EXP-20260412-003.")
    parser.add_argument("--per_image_forced", action="store_true", default=False,
                        help="Apply initial/recent forced-keep per image block instead of globally.")
    parser.add_argument("--n_iterative_rounds", type=int, default=1,
                        help=(
                            "Iterative pruning rounds (EXP-20260417-001). "
                            "1 = one-shot (default). 4 = 4-round linear schedule matching PLAN.md."
                        ))
    parser.add_argument("--layerwise_iterative", action="store_true", default=False,
                        help=(
                            "Use per-layer vote-based pool narrowing in each iterative round instead of "
                            "global amax consensus. Each layer independently selects top-k; final pool is "
                            "determined by vote count with max-score tiebreak. Requires n_iterative_rounds > 1."
                        ))
    parser.add_argument("--look_dataset_name", type=str, default="DocVQA")
    parser.add_argument("--look_model_name", type=str, default="zap_docvqa")
    parser.add_argument("--look_result_root", type=str, default=HD_ZAP_ARTIFACT_ROOT)
    parser.add_argument("--save_look_files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate_with_look_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_partial_look_eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--truncate_like_lookm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--look_max_context_len", type=int, default=None)
    parser.add_argument("--look_n_tokens_per_image", type=int, default=None)
    parser.add_argument("--combine_image", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--save_viz_for_first_n",
        type=int,
        default=0,
        help=(
            "Save per-layer score/mask data for the first N samples (0 = disabled). "
            "Data is written as .npz + .json pairs to {output_dir}/viz_data/ (or --viz_output_dir). "
            "Only works for ImageTokenTopKPress subclasses (probe, oracle, h2o_image_only). "
            "Has zero overhead when disabled."
        ),
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default=None,
        help="Directory for visualization data files. Defaults to {output_dir}/viz_data/.",
    )
    parser.add_argument(
        "--collect_phase2_metrics",
        action="store_true",
        default=False,
        help=(
            "Compute Phase 2 auxiliary metrics (oracle overlap ratio, spatial entropy) "
            "for probe mode. Requires --teacher_dir and --save_viz_for_first_n > 0 "
            "(or enables VizCapture automatically for the full run). "
            "Results are written to {output_dir}/phase2_metrics.json."
        ),
    )
    parser.add_argument(
        "--phase2_grid_side",
        type=int,
        default=24,
        help="Grid side length for spatial entropy (default 24 = LLaVA-1.5 336px / patch-14).",
    )
    parser.add_argument(
        "--phase2_n_image_per_image",
        type=int,
        default=576,
        help="Image tokens per single image for spatial entropy grid (default 576).",
    )
    args = parser.parse_args()

    if getattr(args, "total_keep_ratio", None) is None and getattr(args, "image_keep_ratio", None) is None:
        raise ValueError("One of --total_keep_ratio or --image_keep_ratio is required")
    if args.mode in ("oracle", "oracle_all_token") and args.teacher_dir is None:
        raise ValueError("teacher_dir is required for oracle and oracle_all_token modes")
    if args.mode == "probe" and not args.probe_model_name:
        raise ValueError("probe_model_name is required for probe mode")
    if args.mode in ("oracle", "oracle_all_token") and args.truncate_like_lookm:
        raise ValueError("LOOK-M style truncation is currently supported only for probe mode")
    # h2o_image_only and oracle_all_token require output_attentions=True (eager attention)
    needs_output_attentions = args.mode in ("h2o_image_only", "oracle_all_token")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_vlm_samples(
        dataset_path=args.dataset_path,
        image_root=args.image_root,
        image_column=args.image_column,
        answer_column=args.answer_column,
        limit=args.limit,
    )
    core_annotation = load_core_annotation(args.dataset_path)
    teacher_root = resolve_teacher_dir(args.teacher_dir) if args.teacher_dir else None

    processor = AutoProcessor.from_pretrained(args.implementation_model_name, use_fast=False)
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "device_map": None if args.device_map in ("", "none", "None") else args.device_map,
    }
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    model = LlavaForConditionalGeneration.from_pretrained(args.implementation_model_name, **model_kwargs)
    configure_llava_processor(processor, model.config)
    if model_kwargs["device_map"] is None and args.device not in ("", "none", "None"):
        model = model.to(torch.device(args.device))
    model.eval()

    device = _get_model_device(model)
    float_dtype = _get_model_float_dtype(model)
    press = build_press(args)

    prepared_inputs: dict[str, dict[str, Any]] = {}
    if args.truncate_like_lookm:
        if core_annotation is None:
            raise ValueError("LOOK-M style truncation requires a MileBench core annotation JSON")
        max_context_len = args.look_max_context_len
        if max_context_len is None:
            max_context_len = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
        if max_context_len is None:
            max_context_len = getattr(model.config, "max_position_embeddings", None)
        if max_context_len is None:
            max_context_len = 4096

        n_tokens_per_image = args.look_n_tokens_per_image
        if n_tokens_per_image is None:
            n_tokens_per_image = getattr(model.config, "image_seq_length", None)
        if n_tokens_per_image is None:
            n_tokens_per_image = 576

        prepared_inputs = prepare_lookm_truncated_inputs(
            samples,
            core_annotation=core_annotation,
            image_root=args.image_root,
            tokenizer=processor.tokenizer,
            dataset_name=args.look_dataset_name,
            max_context_len=int(max_context_len),
            n_tokens_per_image=int(n_tokens_per_image),
            combine_image=args.combine_image,
        )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    look_predictions: list[dict[str, Any]] = []

    # ── VizCapture setup ──────────────────────────────────────────────────────
    viz_capture = None
    viz_dir: Optional[Path] = None
    if args.save_viz_for_first_n > 0:
        from kvpress.presses.image_token_press import VizCapture
        viz_capture = VizCapture()
        viz_dir = (
            Path(args.viz_output_dir).resolve()
            if args.viz_output_dir
            else output_dir / "viz_data"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(press, "attach_viz_capture"):
            press.attach_viz_capture(viz_capture)
        else:
            print(
                f"Warning: press {type(press).__name__} does not support VizCapture. "
                "Visualization will be skipped."
            )
            viz_capture = None

    viz_sample_idx = 0

    # ── Phase 2 metric collection setup ──────────────────────────────────────
    collect_phase2 = (
        getattr(args, "collect_phase2_metrics", False)
        and args.mode == "probe"
        and teacher_root is not None
    )
    if collect_phase2 and viz_capture is None:
        # Phase 2 needs VizCapture even if --save_viz_for_first_n was not set.
        from kvpress.presses.image_token_press import VizCapture as _VizCapture
        viz_capture = _VizCapture()
        if hasattr(press, "attach_viz_capture"):
            press.attach_viz_capture(viz_capture)
        else:
            print(
                f"Warning: press {type(press).__name__} does not support VizCapture. "
                "Phase 2 metric collection will be skipped."
            )
            collect_phase2 = False
    # Pre-load probe model so _iterative_probe_preselect() can access it before the first
    # press(model) context (which normally triggers post_init_from_model).
    if args.mode == "probe" and getattr(args, "n_iterative_rounds", 1) > 1:
        if hasattr(press, "post_init_from_model"):
            press.post_init_from_model(model)

    phase2_records: list[dict] = []
    active_press = press  # updated per-sample in iterative mode

    for sample in tqdm(samples, desc=f"Evaluating {args.mode} image pruning"):
        try:
            raw_record = sample.get("raw") if isinstance(sample.get("raw"), dict) else None
            prepared = prepared_inputs.get(str(sample["sample_id"]))
            if prepared is not None:
                question_for_prompt = prepared["question"]
                image_paths = prepared["image_paths"]
            else:
                question_for_prompt = sample["question"]
                image_paths = sample["image_paths"]
                if (
                    args.prompt_style == "look_milebench"
                    and core_annotation is not None
                    and raw_record is not None
                    and "task_instance" in raw_record
                ):
                    question_for_prompt = build_look_question(raw_record, core_annotation, dataset_name=args.look_dataset_name)

            image_paths = image_paths or []
            prompt_text = build_prompt(
                question_for_prompt,
                args.prompt_template,
                image_count=len(image_paths),
            )
            if image_paths:
                images = open_images(image_paths)
                prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
            else:
                # LOOK-M style truncation can legitimately drop all images for very long contexts.
                # In that case run text-only tokenization instead of passing an empty image list.
                prompt_inputs = processor(text=prompt_text, return_tensors="pt")

            prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)
            prompt_len_text = int(prompt_inputs["input_ids"].shape[1])
            num_images = int(len(image_paths))
            empty_image_positions = torch.empty(0, dtype=torch.long)

            if args.mode in ("oracle", "oracle_all_token"):
                if num_images == 0:
                    press.set_image_positions(empty_image_positions)
                else:
                    teacher_record = load_pt_record(teacher_root / f"{sample['sample_id']}.pt")
                    image_positions = resolve_teacher_image_positions(teacher_record)
                    press.set_sample_teacher(image_positions, teacher_record[args.teacher_score_name])
            elif args.mode == "oracle_onthefly":
                if num_images == 0:
                    press.set_image_positions(empty_image_positions)
                else:
                    image_positions, _ = infer_llava_image_positions_no_forward(
                        prompt_inputs=prompt_inputs,
                        model_config=model.config,
                        num_images=num_images,
                    )
                    teacher_scores = extract_att_only_postvision_on_the_fly(
                        model=model,
                        prompt_inputs=prompt_inputs,
                        image_positions=image_positions,
                        device=device,
                    )
                    press.set_sample_teacher(image_positions, teacher_scores)
            else:
                if num_images == 0:
                    press.set_image_positions(empty_image_positions)
                else:
                    image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
                        prompt_inputs=prompt_inputs,
                        model_config=model.config,
                        num_images=num_images,
                    )
                    use_iterative = (
                        args.mode == "probe"
                        and getattr(args, "n_iterative_rounds", 1) > 1
                        and isinstance(press, ProbeImageTeacherPress)
                        and press.probe_model is not None
                    )
                    if use_iterative:
                        n_image = image_positions.numel()
                        n_text = prompt_len_mm - n_image
                        n_image_keep_target = _compute_n_image_keep(
                            n_image, n_text, press.image_keep_ratio, press.total_keep_ratio
                        )
                        final_image_positions = _iterative_probe_preselect(
                            model=model,
                            prompt_inputs=prompt_inputs,
                            image_positions=image_positions,
                            probe_press=press,
                            n_rounds=args.n_iterative_rounds,
                            n_image_keep=n_image_keep_target,
                            device=device,
                            float_dtype=float_dtype,
                            layerwise=getattr(args, "layerwise_iterative", False),
                        )
                        active_press = PreselectedImagePress()
                        active_press.set_selection(
                            all_image_positions=image_positions,
                            keep_positions=final_image_positions,
                        )
                    else:
                        press.set_image_positions(image_positions)
                        active_press = press

            generate_kwargs: dict = dict(
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            if needs_output_attentions:
                generate_kwargs["output_attentions"] = True

            with active_press(model):
                with torch.no_grad():
                    generated_ids = model.generate(**prompt_inputs, **generate_kwargs)
            prediction = decode_answer(processor, generated_ids[0], prompt_len_text)
            gold = sample.get("answer")
            exact_match = None if gold is None else normalize_answer(prediction) == normalize_answer(gold)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "question": question_for_prompt,
                    "image_paths": image_paths,
                    "gold_answer": gold,
                    "prediction": prediction,
                    "exact_match": exact_match,
                }
            )

            if args.save_look_files:
                question_for_export = replace_image_placeholders_for_export(question_for_prompt)
                look_predictions.append(
                    build_look_prediction_record(
                        sample=sample,
                        question_for_export=question_for_export,
                        prediction=prediction,
                        gold=gold,
                        args=args,
                        image_paths=image_paths,
                    )
                )

            # ── Phase 2 metrics (probe vs oracle overlap + spatial entropy) ────
            if collect_phase2 and viz_capture is not None and num_images > 0:
                try:
                    p2_teacher_record = load_pt_record(teacher_root / f"{sample['sample_id']}.pt")
                    p2_teacher_scores = p2_teacher_record[args.teacher_score_name].float()  # (L, H, I)
                    probe_masks = np.stack(
                        [m.numpy() for m in viz_capture.keep_masks]
                    ) if viz_capture.keep_masks else None
                    if probe_masks is not None and p2_teacher_scores.shape[0] == len(viz_capture.n_image_keep):
                        oracle_masks = compute_oracle_keep_masks(
                            p2_teacher_scores,
                            viz_capture.n_image_keep,
                            head_reduce=args.head_reduce,
                        )
                        p2 = summarize_phase2_sample(
                            probe_masks,
                            oracle_masks,
                            head_reduce=args.head_reduce,
                            n_image_per_image=args.phase2_n_image_per_image,
                            grid_side=args.phase2_grid_side,
                        )
                        p2["sample_id"] = sample["sample_id"]
                        phase2_records.append(p2)
                except Exception as _p2_exc:  # noqa: BLE001
                    pass  # non-fatal: skip phase2 for this sample
                finally:
                    # Reset here only when the viz-save block below will NOT reset it.
                    if viz_sample_idx >= args.save_viz_for_first_n:
                        viz_capture.reset()

            # ── Save VizCapture data ─────────────────────────────────────────
            if viz_capture is not None and viz_sample_idx < args.save_viz_for_first_n:
                arrays = viz_capture.to_arrays()
                arrays["image_positions"] = (
                    image_positions.cpu().numpy().astype("int64")
                    if image_positions is not None and image_positions.numel() > 0
                    else np.zeros(0, dtype="int64")
                )
                stem = f"{viz_sample_idx:04d}_{sample['sample_id']}"
                np.savez_compressed(viz_dir / f"{stem}_scores.npz", **arrays)
                meta_path = viz_dir / f"{stem}_meta.json"
                with meta_path.open("w", encoding="utf-8") as _mf:
                    json.dump(
                        {
                            "sample_idx": viz_sample_idx,
                            "sample_id": sample["sample_id"],
                            "image_paths": image_paths,
                            "dataset": args.look_dataset_name,
                            "mode": args.mode,
                            "total_keep_ratio": getattr(args, "total_keep_ratio", None),
                            "image_keep_ratio": getattr(args, "image_keep_ratio", None),
                            "n_images": len(image_paths),
                            "question": question_for_prompt,
                            "gold_answer": sample.get("answer"),
                            "prediction": prediction,
                        },
                        _mf,
                        ensure_ascii=False,
                    )
                viz_capture.reset()
                viz_sample_idx += 1

        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sample["sample_id"], "error": repr(exc)})
            if not args.continue_on_error:
                raise
        finally:
            press.clear_sample_context()
            if active_press is not press:
                active_press.clear_sample_context()
            active_press = press  # reset for next iteration

    # ── Phase 2 aggregate save ────────────────────────────────────────────────
    if phase2_records:
        agg = aggregate_phase2_records(phase2_records)
        phase2_output = {"aggregate": agg, "per_sample": phase2_records}
        write_json(output_dir / "phase2_metrics.json", phase2_output)
        print(f"Phase 2 metrics ({len(phase2_records)} samples): overlap={agg.get('overlap_mean', float('nan')):.3f} ± {agg.get('overlap_std', float('nan')):.3f}, entropy={agg.get('entropy_mean', float('nan')):.3f} ± {agg.get('entropy_std', float('nan')):.3f}")

    predictions_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    predictions_df.to_json(output_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    failures_path = output_dir / "failures.csv"
    if failures_df.empty:
        if failures_path.exists():
            failures_path.unlink()
    else:
        failures_df.to_csv(failures_path, index=False)

    look_dataset_dir: Optional[Path] = None
    if args.save_look_files and args.look_result_root:
        look_dataset_dir = Path(args.look_result_root).resolve() / args.look_model_name / args.look_dataset_name
        look_dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.save_look_files:
        write_json(output_dir / "pred.json", look_predictions)
        if look_dataset_dir is not None:
            write_json(look_dataset_dir / "pred.json", look_predictions)

    look_eval_result: Optional[dict[str, Any]] = None
    if args.evaluate_with_look_metrics and args.save_look_files and core_annotation is not None and look_predictions:
        evaluation_ready = len(look_predictions) == len(core_annotation["data"])
        core_for_eval = core_annotation

        if not evaluation_ready and args.allow_partial_look_eval:
            prediction_ids = {item["sample_id"] for item in look_predictions}
            core_for_eval = subset_core_annotation(core_annotation, prediction_ids)
            evaluation_ready = len(look_predictions) == len(core_for_eval["data"])

        if evaluation_ready:
            evaluator = LookMileBenchEvaluator()
            predictions_for_eval = deepcopy(look_predictions)
            predictions_with_extracted, look_eval_result, eval_list = evaluator.evaluate(predictions_for_eval, core_for_eval)

            write_json(output_dir / "eval.json", look_eval_result)
            write_json(output_dir / "eval_score.json", eval_list)
            if predictions_with_extracted is not None:
                write_json(output_dir / "pred_with_extracted.json", predictions_with_extracted)

            if look_dataset_dir is not None:
                write_json(look_dataset_dir / "eval.json", look_eval_result)
                write_json(look_dataset_dir / "eval_score.json", eval_list)
                if predictions_with_extracted is not None:
                    write_json(look_dataset_dir / "pred_with_extracted.json", predictions_with_extracted)
        else:
            print(
                "Skipping LOOK-compatible eval because predictions do not cover the full dataset. "
                "Use --allow_partial_look_eval to score subsets."
            )

    metrics = {
        "mode": args.mode,
        "n_samples": len(samples),
        "n_predictions": int(len(predictions_df)),
        "n_failures": int(len(failures_df)),
        "image_keep_ratio": getattr(args, "image_keep_ratio", None),
        "total_keep_ratio": getattr(args, "total_keep_ratio", None),
        "teacher_score_name": args.teacher_score_name if args.mode in ("oracle", "oracle_onthefly") else None,
        "probe_model_name": args.probe_model_name if args.mode == "probe" else None,
        "prompt_style": args.prompt_style,
        "look_model_name": args.look_model_name if args.save_look_files else None,
        "look_dataset_name": args.look_dataset_name if args.save_look_files else None,
        "look_result_dir": str(look_dataset_dir) if look_dataset_dir is not None else None,
    }
    if not predictions_df.empty and "exact_match" in predictions_df and predictions_df["exact_match"].notna().any():
        metrics["exact_match_accuracy"] = float(predictions_df["exact_match"].dropna().mean())
    if look_eval_result is not None:
        metrics["look_eval"] = look_eval_result

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
