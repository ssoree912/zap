#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MileBench sweep for original LLaVA-OneVision + student under MULTI-IMAGE protocol.

Same model/student loading + KV pruning logic as
`experiments/EXP-20260504-001/.../milebench_onevision_original_student_sweep.py`,
but the visual input is the multi-frame `images/` directory (matching VFlowOpt /
LOOK-M).  Multi-frame samples are stacked into a single video tensor and passed
with `modalities=["video"]` so OneVision's spatial-pool path produces one
contiguous image-token block (729 tokens/frame) — no anyres patch explosion.
"""

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
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

DATA_ROOT = Path("/workspace/zap/data/MileBench")
DEFAULT_DATASETS = ["ALFRED", "CLEVR-Change", "IEdit", "Spot-the-Diff"]


def patch_siglip_loader_to_local_init() -> None:
    """Avoid hard-coded external SigLIP path in this LLaVA-OneVision checkout."""
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
    return payload["meta_data"], payload["data"], DATA_ROOT / dataset / "images"


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
        choice_str += "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(ann["choice_list"])
        )
        choice_str += "\nYour answer is: "
        context += choice_str
    # Single <image> placeholder — video modality consumes one placeholder
    # and produces one contiguous spatial-pool image-token block in prefill.
    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}"


def build_prompt_text(user_prompt: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def open_images(paths: list[Path]) -> list[Image.Image]:
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def prepare_video_tensor(
    images: list[Image.Image],
    image_processor: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stack MileBench frames into (T, 3, 384, 384) video tensor.

    Matches VFlowOpt's prepare_video_tensor (expand2square + processor preprocess).
    """
    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    processed: list[torch.Tensor] = []
    for im in images:
        w, h = im.size
        if w != h:
            side = max(w, h)
            sq = Image.new(im.mode, (side, side), bg)
            sq.paste(im, ((side - w) // 2, (side - h) // 2))
            im = sq
        out = image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
        processed.append(out)
    return torch.stack(processed, dim=0).to(device=device, dtype=dtype)


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
        raise ValueError(
            f"Expected exactly one image placeholder, found {len(placeholder_positions)}"
        )
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
    video_tensor: torch.Tensor,
    image_size: tuple[int, int],
    max_new_tokens: int,
    stop_str: str,
) -> str:
    out = model.generate(
        inputs=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=[video_tensor],
        image_sizes=[image_size],
        modalities=["video"],
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
    image_paths: list[Path],
    keep_ratio: float,
    device: torch.device,
    conv_template: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    full_prompt, stop_str = build_prompt_text(prompt_text, conv_template)
    input_ids = (
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    pil_images = open_images(image_paths)
    image_size = (pil_images[0].size[0], pil_images[0].size[1])
    video_tensor = prepare_video_tensor(pil_images, image_processor, device, torch.float16)

    if keep_ratio >= 1.0:
        pred = _safe_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            video_tensor=video_tensor,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            stop_str=stop_str,
        )
        return pred, {"n_frames": len(image_paths)}

    prefill = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=[video_tensor],
        image_sizes=[image_size],
        modalities=["video"],
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
        "n_frames": len(image_paths),
        "total_keep_ratio": (n_text + n_keep) / max(1, int(prompt_len)),
        "image_keep_ratio": n_keep / max(1, n_img),
    }
    torch.cuda.empty_cache()
    return pred, stats


def sample_image_paths(sample: dict[str, Any], img_root: Path) -> list[Path]:
    ann = sample["task_instance"]
    paths = ann["images_path"]
    return [img_root / p for p in paths]


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

    meta, samples, img_root = load_dataset(dataset)
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
            pred, stats = generate_with_student(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                student=student,
                prompt_text=build_user_prompt(sample, meta),
                image_paths=sample_image_paths(sample, img_root),
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
        try:
            sid_out: Any = int(sample_id)
        except Exception:
            sid_out = sample_id
        predictions.append(
            {
                "sample_id": sid_out,
                "pred_response": pred,
                "gt_response": sample["response"],
            }
        )
    elapsed = time.time() - start
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    stats_summary = {
        "dataset": dataset,
        "keep_ratio": keep_ratio,
        "n_samples": len(predictions),
        "n_failures": len(failures),
        "elapsed_seconds": elapsed,
        "max_new_tokens": int(args.max_new_tokens),
        "protocol": "multi-image (video modality)",
        "failures": failures[:8],
    }
    if keep_stats:
        n = len(keep_stats)
        with_keep = [s for s in keep_stats if "image_keep_ratio" in s]
        if with_keep:
            m = len(with_keep)
            stats_summary["avg_image_keep_ratio"] = sum(s["image_keep_ratio"] for s in with_keep) / m
            stats_summary["avg_total_keep_ratio"] = sum(s["total_keep_ratio"] for s in with_keep) / m
        stats_summary["avg_n_frames"] = sum(s.get("n_frames", 0) for s in keep_stats) / n
    stats_path.write_text(json.dumps(stats_summary, indent=2))
    print(
        f"[{dataset}] keep={keep_ratio} saved → {pred_path} ({len(predictions)} samples, "
        f"{len(failures)} failures, {elapsed:.1f}s)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[1.0, 0.5, 0.2])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    patch_siglip_loader_to_local_init()
    device = torch.device(args.device)

    print(
        f"[load] model={args.model_path} model_name={args.model_name} "
        f"device_map={args.device_map} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        device_map=args.device_map,
        attn_implementation="sdpa",
        multimodal=True,
    )
    model.eval()
    if args.device_map != "auto":
        model.to(device)
    try:
        model.tie_weights()
    except Exception:
        pass

    student = VisualUtilityStudentOneVision.from_pretrained(args.student_path, map_location="cpu")
    student = student.to(device=device, dtype=torch.float16).eval()
    print(f"[student] path={args.student_path} layers={student.layer_indices}", flush=True)

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for keep_ratio in args.keep_ratios:
        for dataset in args.datasets:
            evaluate_dataset(
                dataset=dataset,
                keep_ratio=keep_ratio,
                args=args,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                student=student,
                device=device,
            )


if __name__ == "__main__":
    main()
