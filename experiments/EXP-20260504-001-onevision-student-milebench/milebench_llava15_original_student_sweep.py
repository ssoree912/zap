#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MileBench sweep for original LLaVA-1.5-7B + trained original student."""

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

REPO_ROOT = Path("/workspace/zap")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(LLAVA_ONEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))

from foresight.eval.kv_decode_utils import (  # noqa: E402
    greedy_decode_with_kv,
    trim_kv_cache_per_layer,
)
from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

DATA_ROOT = Path("/workspace/zap/data/MileBench")
DEFAULT_DATASETS = ["ALFRED", "CLEVR-Change", "IEdit", "Spot-the-Diff"]

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{(?:image|table)#\d+\}")


def keep_tag(keep_ratio: float) -> str:
    return f"keep{int(round(keep_ratio * 100)):03d}"


def load_dataset(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    data_path = DATA_ROOT / dataset / f"{dataset}.json"
    with data_path.open() as f:
        payload = json.load(f)
    return payload["meta_data"], payload["data"], DATA_ROOT / dataset / "combined_1_images"


def _resolve_task_instruction(meta: dict[str, Any], sample: dict[str, Any]) -> str:
    task_instructions = meta.get("task_instruction", "")
    task_instruction_id = sample.get("task_instruction_id", 0)
    if isinstance(task_instructions, list):
        try:
            return str(task_instructions[int(task_instruction_id)])
        except Exception:
            return ""
    if isinstance(task_instructions, dict):
        return str(
            task_instructions.get(
                task_instruction_id,
                task_instructions.get(str(task_instruction_id), ""),
            )
        )
    return str(task_instructions)


def build_user_prompt(sample: dict[str, Any], meta: dict[str, Any]) -> str:
    ann = sample["task_instance"]
    task_instruction = _resolve_task_instruction(meta, sample)
    context = str(ann["context"]).strip()

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        idx_match = re.search(r"#(\d+)", raw)
        idx = idx_match.group(1) if idx_match else "1"
        return f"<Image {idx}> "

    context = IMAGE_PLACEHOLDER_PATTERN.sub(_replace, context)
    if ann.get("choice_list"):
        choice_str = "\nChoice List:\n"
        choice_str += "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(ann["choice_list"]))
        choice_str += "\nYour answer is: "
        context += choice_str
    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}".strip()


def build_prompt_text(user_prompt: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def _to_image_inputs(image_tensor: Any, device: torch.device) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(image_tensor, torch.Tensor):
        return image_tensor.to(device=device, dtype=torch.float16)
    if isinstance(image_tensor, list) and all(isinstance(t, torch.Tensor) and t.dim() == 3 for t in image_tensor):
        return torch.stack(image_tensor, dim=0).to(device=device, dtype=torch.float16)
    return [tensor.to(device=device, dtype=torch.float16) for tensor in image_tensor]


def _resolve_eos_token_id(tokenizer: Any, model_config: Any) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        cfg = getattr(model_config, "eos_token_id", None)
        eos = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    return int(2 if eos is None else eos)


def _infer_image_positions_fixed(input_ids: torch.Tensor, image_feature_len: int) -> tuple[torch.Tensor, int]:
    raw_ids = input_ids[0].detach().cpu()
    placeholder_positions = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected exactly one image placeholder, found {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    image_positions = torch.arange(image_start, image_start + image_feature_len, dtype=torch.long)
    prompt_len = int(raw_ids.numel() - 1 + image_feature_len)
    return image_positions, prompt_len


def _infer_image_positions_dynamic(input_ids: torch.Tensor, prompt_len_mm: int) -> torch.Tensor:
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


def _infer_question_positions(prompt_len: int, image_positions: torch.Tensor) -> torch.Tensor:
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len, dtype=torch.long)


def _decode_generated(tokenizer: Any, sequences: torch.Tensor, input_len: int, max_new_tokens: int) -> str:
    ids = sequences[0].detach().cpu()
    if ids.numel() > max_new_tokens + 1 and ids.numel() > input_len:
        ids = ids[input_len:]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()


@torch.no_grad()
def _safe_generate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    image_tensor: Any,
    image_size: tuple[int, int],
    max_new_tokens: int,
) -> str:
    eos_token_id = _resolve_eos_token_id(tokenizer, model.config)
    out = model.generate(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or eos_token_id,
        eos_token_id=eos_token_id,
    )
    sequences = out.sequences if hasattr(out, "sequences") else out
    return _decode_generated(tokenizer, sequences, input_len=int(input_ids.shape[1]), max_new_tokens=max_new_tokens)


@torch.no_grad()
def generate_with_student(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudent,
    prompt_text: str,
    image_path: Path,
    keep_ratio: float,
    device: torch.device,
    conv_template: str,
    max_new_tokens: int,
    image_feature_len: int,
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
        )
        return pred, {}

    eos_token_id = _resolve_eos_token_id(tokenizer, model.config)
    try:
        prefill = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            modalities=["image"],
            use_cache=True,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
    except Exception as exc:
        print(f"[warn] prefill failed; falling back to full generate: {exc}", file=sys.stderr, flush=True)
        pred = _safe_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            image_tensor=image_tensor,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
        )
        return pred, {}

    past_kv = prefill.past_key_values
    prompt_len = int(prefill.hidden_states[-1].shape[1])
    try:
        image_positions, inferred_prompt_len = _infer_image_positions_fixed(input_ids, image_feature_len)
        if int(inferred_prompt_len) != prompt_len:
            image_positions = _infer_image_positions_dynamic(input_ids, prompt_len)
    except Exception:
        image_positions = _infer_image_positions_dynamic(input_ids, prompt_len)

    question_positions = _infer_question_positions(prompt_len, image_positions)
    n_img = int(image_positions.numel())
    n_text = int(prompt_len) - n_img
    n_keep = max(1, int(round(n_img - (1.0 - keep_ratio) * int(prompt_len))))
    n_keep = min(n_keep, n_img)

    keep_masks: dict[int, torch.Tensor] = {}
    image_idx_dev = image_positions.to(device=device)
    q_idx_dev = question_positions.to(device=device)
    for li in student.layer_indices:
        if li + 1 >= len(prefill.hidden_states) or n_keep >= n_img:
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
    past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
    answer_ids = greedy_decode_with_kv(
        model,
        past_kv,
        next_token,
        prompt_len=int(prompt_len),
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    pred = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
    if stop_str and pred.endswith(stop_str):
        pred = pred[: -len(stop_str)].strip()

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
    image_value = ann.get("combined_1_images")
    if isinstance(image_value, (list, tuple)):
        image_name = str(image_value[0])
    elif image_value:
        image_name = str(image_value)
    else:
        raw = ann.get("images_path")
        image_name = str(raw[0] if isinstance(raw, (list, tuple)) else raw)
    return Path(image_name) if Path(image_name).is_absolute() else combined_img_root / image_name


def evaluate_dataset(
    *,
    dataset: str,
    keep_ratio: float,
    args: argparse.Namespace,
    tokenizer: Any,
    model: torch.nn.Module,
    image_processor: Any,
    student: VisualUtilityStudent,
    device: torch.device,
    image_feature_len: int,
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
            pred, stats = generate_with_student(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                student=student,
                prompt_text=build_user_prompt(sample, meta),
                image_path=sample_image_path(sample, combined_img_root),
                keep_ratio=keep_ratio,
                device=device,
                conv_template=args.conv_template,
                max_new_tokens=args.max_new_tokens,
                image_feature_len=image_feature_len,
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
        "max_new_tokens": int(args.max_new_tokens),
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
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-v1.5-7b")
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.5, 0.1, 0.05])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    print(
        f"[load] model={args.model_path} model_name={args.model_name} "
        f"device_map={args.device_map} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        multimodal=True,
    )
    model.eval()
    if args.device_map != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    vision_tower = model.get_vision_tower()
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    print(
        f"[load-ok] class={model.__class__.__name__} device={device} "
        f"image_feature_len={image_feature_len}",
        flush=True,
    )

    student = VisualUtilityStudent.from_pretrained(args.student_path, map_location="cpu")
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
                image_feature_len=image_feature_len,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
