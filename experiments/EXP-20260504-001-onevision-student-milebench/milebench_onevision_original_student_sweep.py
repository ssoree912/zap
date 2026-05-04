#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MileBench sweep for original LLaVA-OneVision + trained OneVision student."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import DynamicCache

REPO_ROOT = Path("/workspace/zap")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(LLAVA_ONEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))

from kvpress.presses.visual_utility_student_onevision import (  # noqa: E402
    VisualUtilityStudentOneVision,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

DATA_ROOT = Path("/workspace/zap/data/MileBench")
DEFAULT_DATASETS = [
    "MovingAttribute",
    "ObjectExistence",
    "ObjectInteraction",
    "ObjectShuffle",
    "EgocentricNavigation",
    "MovingDirection",
    "CharacterOrder",
    "CounterfactualInference",
    "SceneTransition",
    "StateChange",
    "ALFRED",
    "MMCoQA",
]


def patch_siglip_loader_to_local_init() -> None:
    """Avoid a hard-coded external SigLIP path in this LLaVA-OneVision checkout."""
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):  # noqa: ANN001, ARG001
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


def keep_tag(keep_ratio: float) -> str:
    return f"keep{int(round(keep_ratio * 100)):03d}"


def load_dataset(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    data_path = DATA_ROOT / dataset / f"{dataset}.json"
    with data_path.open() as f:
        payload = json.load(f)
    return payload["meta_data"], payload["data"], DATA_ROOT / dataset / "combined_1_images"


def build_user_prompt(sample: dict[str, Any], meta: dict[str, Any]) -> str:
    ann = sample["task_instance"]
    task_instruction = meta["task_instruction"][sample["task_instruction_id"]]

    context = ann["context"]
    n_img = len(ann["images_path"])
    for i in range(1, n_img + 1):
        context = context.replace(f"{{image#{i}}}", f"<Image {i}> ")
        context = context.replace(f"{{table#{i}}}", f"<Image {i}> ")

    if ann.get("choice_list"):
        choice_str = "\nChoice List:\n"
        choice_str += "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(ann["choice_list"]))
        choice_str += "\nYour answer is: "
        context += choice_str

    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}"


def build_prompt_text(user_prompt: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def _to_image_inputs(image_tensor: Any, device: torch.device) -> Any:
    if isinstance(image_tensor, torch.Tensor):
        return image_tensor.to(device=device, dtype=torch.float16)
    return [tensor.to(device=device, dtype=torch.float16) for tensor in image_tensor]


def _cache_seq_len(past_key_values: Any) -> int:
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        return int(past_key_values.key_cache[0].shape[-2])
    return int(past_key_values[0][0].shape[-2])


def _infer_image_positions(input_ids: torch.Tensor, prompt_len_mm: int) -> torch.Tensor:
    raw_ids = input_ids[0].detach().cpu()
    placeholder_positions = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected exactly one image placeholder, found {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    image_feature_len = int(prompt_len_mm - raw_ids.numel() + 1)
    if image_feature_len <= 0:
        raise ValueError(
            f"Invalid image feature length={image_feature_len} prompt_len={prompt_len_mm} raw_len={raw_ids.numel()}"
        )
    return torch.arange(image_start, image_start + image_feature_len, dtype=torch.long)


def _infer_question_positions(prompt_len_mm: int, image_positions: torch.Tensor) -> torch.Tensor:
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


def _resolve_eos_token_id(tokenizer: Any, model_config: Any) -> int:
    eid = tokenizer.eos_token_id
    if eid is None:
        cfg = getattr(model_config, "eos_token_id", None)
        eid = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    return int(eid if eid is not None else 151645)


def _trim_kv_cache_per_layer(past_kv: Any, keep_masks: dict[int, torch.Tensor]) -> Any:
    if hasattr(past_kv, "key_cache"):
        for li in range(len(past_kv.key_cache)):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(past_kv.key_cache[li].device)
            past_kv.key_cache[li] = past_kv.key_cache[li][:, :, mask, :].contiguous()
            past_kv.value_cache[li] = past_kv.value_cache[li][:, :, mask, :].contiguous()
        return past_kv

    if hasattr(past_kv, "layers"):
        for li, layer in enumerate(past_kv.layers):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(layer.keys.device)
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return past_kv

    new_cache = DynamicCache()
    for li, (k, v) in enumerate(past_kv):
        if li in keep_masks:
            mask = keep_masks[li].to(k.device)
            k = k[:, :, mask, :].contiguous()
            v = v[:, :, mask, :].contiguous()
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    return new_cache


@torch.no_grad()
def _greedy_decode_with_kv(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    past_kv: Any,
    first_next_token: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
    stop_str: str,
) -> str:
    eos_token_id = _resolve_eos_token_id(tokenizer, model.config)
    out_tokens: list[int] = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id:
        return ""

    next_token = first_next_token
    pos = int(prompt_len)
    device = next_token.device
    cache_pos = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(max_new_tokens - 1):
        cache_pos[0] = pos
        out = model(
            input_ids=next_token,
            past_key_values=past_kv,
            cache_position=cache_pos,
            position_ids=cache_pos.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(next_token.item())
        out_tokens.append(tok)
        pos += 1
        if tok == eos_token_id:
            break

    text = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


@torch.no_grad()
def _safe_generate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    image_tensor: Any,
    image_size: tuple[int, int],
    max_new_tokens: int,
    stop_str: str,
) -> str:
    out = model.generate(
        inputs=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    sequences = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.decode(sequences[0].tolist(), skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


@torch.no_grad()
def generate_with_student(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudentOneVision,
    prompt_text: str,
    image_path: Path,
    keep_ratio: float,
    device: torch.device,
    conv_template: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    full_prompt, stop_str = build_prompt_text(prompt_text, conv_template)
    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    with Image.open(image_path) as im:
        image = im.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    image_tensor = _to_image_inputs(image_tensor, device)

    if keep_ratio >= 1.0:
        pred = _safe_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            image_tensor=image_tensor,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            stop_str=stop_str,
        )
        return pred, {}

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
    past_kv = prefill.past_key_values
    prompt_len = _cache_seq_len(past_kv)
    image_positions = _infer_image_positions(input_ids, prompt_len)
    question_positions = _infer_question_positions(prompt_len, image_positions)
    n_img = int(image_positions.numel())
    n_text = int(prompt_len) - n_img
    n_keep = max(1, int(round(n_img - (1.0 - keep_ratio) * int(prompt_len))))
    n_keep = min(n_keep, n_img)

    keep_masks: dict[int, torch.Tensor] = {}
    image_idx_dev = image_positions.to(device=device)
    q_idx_dev = question_positions.to(device=device)
    for li in student.layer_indices:
        if li + 1 >= len(prefill.hidden_states):
            continue
        if n_keep >= n_img:
            continue
        scores = student.forward_layer(
            li,
            prefill.hidden_states[li + 1],
            image_idx_dev,
            q_idx_dev,
        ).squeeze(0)
        top = torch.topk(scores, k=n_keep, largest=True).indices
        mask = torch.ones(int(prompt_len), dtype=torch.bool)
        image_keep = torch.zeros(n_img, dtype=torch.bool)
        image_keep[top.detach().cpu()] = True
        mask[image_positions] = image_keep
        keep_masks[li] = mask

    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill
    past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)
    pred = _greedy_decode_with_kv(
        model=model,
        tokenizer=tokenizer,
        past_kv=past_kv,
        first_next_token=next_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        stop_str=stop_str,
    )
    stats = {
        "prompt_len": int(prompt_len),
        "n_image_original": n_img,
        "n_image_kept": n_keep,
        "n_text": n_text,
        "total_keep_ratio": (n_text + n_keep) / max(1, int(prompt_len)),
        "image_keep_ratio": n_keep / max(1, n_img),
    }
    torch.cuda.empty_cache()
    return pred, stats


def sample_image_path(sample: dict[str, Any], combined_img_root: Path) -> Path:
    ann = sample["task_instance"]
    image_name = ann["combined_1_images"][0]
    return combined_img_root / image_name


def evaluate_dataset(
    *,
    dataset: str,
    keep_ratio: float,
    args: argparse.Namespace,
    tokenizer: Any,
    model: torch.nn.Module,
    image_processor: Any,
    student: VisualUtilityStudentOneVision,
    device: torch.device,
) -> None:
    output_dir = Path(args.output_root) / keep_tag(keep_ratio) / dataset
    pred_path = output_dir / "pred.json"
    stats_path = output_dir / "keep_ratio_stats.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if pred_path.exists() and not args.overwrite:
        print(f"[skip] keep={keep_ratio} dataset={dataset}: {pred_path} exists", flush=True)
        return

    meta, samples, combined_img_root = load_dataset(dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    predictions: list[dict[str, Any]] = []
    keep_stats: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    start = time.time()
    desc = f"{keep_tag(keep_ratio)} {dataset}"
    for sample in tqdm(samples, desc=desc):
        sample_id = str(sample.get("sample_id", ""))
        try:
            prompt_text = build_user_prompt(sample, meta)
            pred, stats = generate_with_student(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                student=student,
                prompt_text=prompt_text,
                image_path=sample_image_path(sample, combined_img_root),
                keep_ratio=keep_ratio,
                device=device,
                conv_template=args.conv_template,
                max_new_tokens=args.max_new_tokens,
            )
            if stats:
                stats["sample_id"] = sample_id
                keep_stats.append(stats)
        except Exception as exc:  # noqa: BLE001
            pred = ""
            failures.append({"sample_id": sample_id, "error": repr(exc)})
            gc.collect()
            torch.cuda.empty_cache()

        predictions.append(
            {
                "sample_id": sample_id,
                "pred_response": pred,
                "gt_response": sample["response"],
            }
        )

    with pred_path.open("w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    summary: dict[str, Any] = {
        "dataset": dataset,
        "keep_ratio": keep_ratio,
        "n_samples": len(predictions),
        "n_failures": len(failures),
        "elapsed_seconds": time.time() - start,
        "failures": failures,
    }
    if keep_stats:
        summary.update(
            {
                "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in keep_stats) / len(keep_stats),
                "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in keep_stats) / len(keep_stats),
                "avg_n_image_original": sum(s["n_image_original"] for s in keep_stats) / len(keep_stats),
                "avg_n_image_kept": sum(s["n_image_kept"] for s in keep_stats) / len(keep_stats),
                "samples": keep_stats,
            }
        )
    with stats_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[done] keep={keep_ratio} dataset={dataset} n={len(predictions)} "
        f"fail={len(failures)} -> {pred_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.5, 0.2, 0.1, 0.05])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    patch_siglip_loader_to_local_init()
    print(
        f"[load] model={args.model_path} model_name={args.model_name} "
        f"device_map={args.device_map}",
        flush=True,
    )
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="sdpa",
        multimodal=True,
    )
    model.eval()
    if args.device_map != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    print(f"[load-ok] class={model.__class__.__name__} device={device}", flush=True)

    student = VisualUtilityStudentOneVision.from_pretrained(args.student_path, map_location="cpu")
    student = student.to(device=device, dtype=torch.float16).eval()
    print(f"[student] path={args.student_path} layers={student.layer_indices}", flush=True)

    for keep_ratio in args.keep_ratios:
        for dataset in args.datasets:
            evaluate_dataset(
                dataset=dataset,
                keep_ratio=float(keep_ratio),
                args=args,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                student=student,
                device=device,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
