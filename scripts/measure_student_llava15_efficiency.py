#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from foresight.image_teacher_utils import build_prompt
from kvpress.presses.image_token_press import VisualUtilityStudentPress, _compute_n_image_keep


IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{image#\d+\}")
GIB = 1024**3


@dataclass(frozen=True)
class PooledSample:
    dataset: str
    sample_id: str
    question: str
    image_paths: tuple[str, ...]
    answer: Optional[str]
    raw_prompt_tokens: int
    prompt_mm_tokens: int

    @property
    def sample_key(self) -> str:
        return f"{self.dataset}:{self.sample_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure LLaVA-1.5 full-cache vs visual-utility student KV pruning latency/memory."
    )
    parser.add_argument("--milebench-root", type=str, default="data/MileBench")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="ckpts/llava-1.5-7b-hf")
    parser.add_argument(
        "--student-path",
        type=str,
        default="artifacts/student_llava15_original_future_1800_lr1e4_15ep",
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--prompt-template", type=str, default="USER: <image>\n{question}\nASSISTANT:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        help="Stop decode timing early when EOS is generated. Default is fixed-token timing.",
    )
    parser.add_argument(
        "--max-prompt-mm-tokens",
        type=int,
        default=3900,
        help="Keep the random pool within this multimodal prompt length for 24GB full-cache runs.",
    )
    parser.add_argument(
        "--combined-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MileBench combined_1_images only; skip samples without combined images.",
    )
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_dataset_list(root: Path, requested: Optional[list[str]]) -> list[str]:
    if requested:
        return requested
    datasets: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        json_path = child / f"{child.name}.json"
        if json_path.is_file():
            datasets.append(child.name)
    return datasets


def resolve_task_instruction(task_instructions: Any, task_instruction_id: Any) -> str:
    if isinstance(task_instructions, list):
        try:
            idx = int(task_instruction_id)
            if 0 <= idx < len(task_instructions):
                return str(task_instructions[idx])
        except Exception:
            pass
    if isinstance(task_instructions, dict):
        key = str(task_instruction_id)
        if task_instruction_id in task_instructions:
            return str(task_instructions[task_instruction_id])
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
    lines = []
    for idx, choice in enumerate(choice_list):
        text = str(choice)
        lines.append(text if dataset_name == "GPR1200" else f"{choice_label(idx)}. {text}")
    return "\nChoice list: \n" + "\n".join(lines) + "\nYour answer is:"


def build_milebench_question(record: dict[str, Any], meta: dict[str, Any], dataset_name: str) -> str:
    task = record.get("task_instance", {})
    context = str(task.get("context", "")).strip()
    instruction = resolve_task_instruction(meta.get("task_instruction"), record.get("task_instruction_id", 0))
    choice_block = build_choice_block(task.get("choice_list"), dataset_name)
    return f"{instruction}\n{context}{choice_block}".strip()


def resolve_answer(record: dict[str, Any]) -> Optional[str]:
    for key in ("answer", "response", "target", "label"):
        if record.get(key) is not None:
            return str(record[key])
    task = record.get("task_instance")
    if isinstance(task, dict):
        for key in ("answer", "answers", "label"):
            value = task.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                return str(value[0]) if value else None
            return str(value)
    return None


def sanitize_sample_id(value: Any, fallback_idx: int) -> str:
    text = str(value) if value is not None else f"sample-{fallback_idx:06d}"
    text = text.strip() or f"sample-{fallback_idx:06d}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:128]


def image_seq_length_from_config(config: Any) -> int:
    if getattr(config, "image_seq_length", None) is not None:
        return int(config.image_seq_length)
    vision_config = getattr(config, "vision_config", None)
    image_size = int(getattr(vision_config, "image_size", 336))
    patch_size = int(getattr(vision_config, "patch_size", 14))
    return (image_size // patch_size) ** 2


def infer_llava_image_positions_no_forward(
    *,
    input_ids: torch.Tensor,
    model_config: Any,
    num_images: int,
) -> tuple[torch.Tensor, int]:
    image_token_index = int(getattr(model_config, "image_token_index", 32000))
    image_seq_len = image_seq_length_from_config(model_config)
    raw_ids = input_ids[0].detach().cpu().tolist()
    direct_positions = [idx for idx, token_id in enumerate(raw_ids) if int(token_id) == image_token_index]

    if len(direct_positions) == num_images * image_seq_len:
        return torch.tensor(direct_positions, dtype=torch.long), len(raw_ids)
    if len(direct_positions) not in (0, num_images):
        raise ValueError(
            f"Expected {num_images} image placeholders or {num_images * image_seq_len} expanded image tokens, "
            f"found {len(direct_positions)}"
        )

    positions: list[int] = []
    cursor = 0
    seen = 0
    for token_id in raw_ids:
        if int(token_id) == image_token_index:
            positions.extend(range(cursor, cursor + image_seq_len))
            cursor += image_seq_len
            seen += 1
        else:
            cursor += 1

    if seen != num_images:
        raise ValueError(f"Expected {num_images} image placeholders, found {seen}")
    if not positions and num_images > 0:
        raise ValueError("No image positions inferred")
    return torch.tensor(positions, dtype=torch.long), cursor


def configure_llava_processor(processor: Any, config: Any) -> None:
    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is not None and getattr(processor, "patch_size", None) is None:
        processor.patch_size = int(patch_size)
    strategy = getattr(config, "vision_feature_select_strategy", None)
    if strategy is not None and getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = strategy
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 0


def estimate_mm_prompt_len(raw_input_ids: torch.Tensor, model_config: Any, num_images: int) -> int:
    _, prompt_len = infer_llava_image_positions_no_forward(
        input_ids=raw_input_ids,
        model_config=model_config,
        num_images=num_images,
    )
    return int(prompt_len)


def iter_candidate_records(
    *,
    milebench_root: Path,
    datasets: list[str],
    processor: Any,
    model_config: Any,
    prompt_template: str,
    max_prompt_mm_tokens: int,
    combined_only: bool,
) -> list[PooledSample]:
    candidates: list[PooledSample] = []
    for dataset_name in datasets:
        dataset_dir = milebench_root / dataset_name
        json_path = dataset_dir / f"{dataset_name}.json"
        if not json_path.is_file():
            continue
        payload = load_json(json_path)
        meta = payload.get("meta_data", {}) if isinstance(payload, dict) else {}
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            continue
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            task = record.get("task_instance", {})
            if not isinstance(task, dict):
                continue
            image_values = task.get("combined_1_images")
            image_root = dataset_dir / "combined_1_images"
            if not image_values:
                if combined_only:
                    continue
                image_values = task.get("images_path") or task.get("image_path") or task.get("image")
                image_root = dataset_dir / "images"
            if isinstance(image_values, (list, tuple)):
                raw_paths = [str(value) for value in image_values if str(value)]
            else:
                raw_paths = [str(image_values)] if image_values else []
            if not raw_paths:
                continue
            image_paths = []
            for raw_path in raw_paths:
                path = Path(raw_path)
                if not path.is_absolute():
                    path = image_root / path
                if not path.is_file():
                    image_paths = []
                    break
                image_paths.append(str(path.resolve()))
            if not image_paths:
                continue

            question = build_milebench_question(record, meta, dataset_name)
            prompt = build_prompt(question, prompt_template, image_count=len(image_paths))
            try:
                encoded = processor(text=prompt, return_tensors="pt")
                prompt_mm_tokens = estimate_mm_prompt_len(
                    encoded["input_ids"],
                    model_config=model_config,
                    num_images=len(image_paths),
                )
            except Exception:
                continue
            if prompt_mm_tokens > max_prompt_mm_tokens:
                continue
            candidates.append(
                PooledSample(
                    dataset=dataset_name,
                    sample_id=sanitize_sample_id(record.get("sample_id"), idx),
                    question=question,
                    image_paths=tuple(image_paths),
                    answer=resolve_answer(record),
                    raw_prompt_tokens=int(encoded["input_ids"].shape[1]),
                    prompt_mm_tokens=prompt_mm_tokens,
                )
            )
    return candidates


def sample_manifest(
    *,
    args: argparse.Namespace,
    processor: Any,
    model_config: Any,
) -> list[PooledSample]:
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        payload = load_json(manifest_path)
        samples = []
        for item in payload["samples"]:
            samples.append(
                PooledSample(
                    dataset=item["dataset"],
                    sample_id=str(item["sample_id"]),
                    question=item["question"],
                    image_paths=tuple(item["image_paths"]),
                    answer=item.get("answer"),
                    raw_prompt_tokens=int(item.get("raw_prompt_tokens", 0)),
                    prompt_mm_tokens=int(item.get("prompt_mm_tokens", 0)),
                )
            )
        return samples

    root = Path(args.milebench_root).resolve()
    datasets = normalize_dataset_list(root, args.datasets)
    candidates = iter_candidate_records(
        milebench_root=root,
        datasets=datasets,
        processor=processor,
        model_config=model_config,
        prompt_template=args.prompt_template,
        max_prompt_mm_tokens=args.max_prompt_mm_tokens,
        combined_only=args.combined_only,
    )
    if len(candidates) < args.sample_size:
        raise RuntimeError(f"Only found {len(candidates)} eligible samples, need {args.sample_size}")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    return candidates[: args.sample_size]


def save_manifest(path: Path, samples: list[PooledSample], args: argparse.Namespace) -> None:
    write_json(
        path,
        {
            "seed": args.seed,
            "sample_size": len(samples),
            "milebench_root": str(Path(args.milebench_root).resolve()),
            "combined_only": args.combined_only,
            "max_prompt_mm_tokens": args.max_prompt_mm_tokens,
            "samples": [
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "sample_key": s.sample_key,
                    "question": s.question,
                    "image_paths": list(s.image_paths),
                    "answer": s.answer,
                    "raw_prompt_tokens": s.raw_prompt_tokens,
                    "prompt_mm_tokens": s.prompt_mm_tokens,
                }
                for s in samples
            ],
        },
    )


def open_images(paths: tuple[str, ...]) -> Any:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images[0] if len(images) == 1 else images


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def first_layer_kv(past_kv: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(past_kv, "layers"):
        layer = past_kv.layers[0]
        return layer.keys, layer.values
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return past_kv.key_cache[0], past_kv.value_cache[0]
    return past_kv[0][0], past_kv[0][1]


def iter_kv_tensors(past_kv: Any):
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            yield layer.keys
            yield layer.values
        return
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for key, value in zip(past_kv.key_cache, past_kv.value_cache):
            yield key
            yield value
        return
    for key, value in past_kv:
        yield key
        yield value


def kv_cache_bytes(past_kv: Any) -> int:
    total = 0
    for tensor in iter_kv_tensors(past_kv):
        total += int(tensor.numel() * tensor.element_size())
    return total


def cache_seq_len(past_kv: Any) -> int:
    key, _ = first_layer_kv(past_kv)
    return int(key.shape[2])


def resolve_eos_token_id(processor: Any, model_config: Any) -> int:
    eos = getattr(processor.tokenizer, "eos_token_id", None)
    if eos is None:
        cfg = getattr(model_config, "eos_token_id", None)
        eos = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    return int(2 if eos is None else eos)


@torch.no_grad()
def decode_with_timing(
    *,
    model: Any,
    past_kv: Any,
    first_next_token: torch.Tensor,
    prompt_len: int,
    eos_token_id: int,
    max_new_tokens: int,
    stop_on_eos: bool,
) -> tuple[Any, list[int], list[float]]:
    out_tokens: list[int] = [int(first_next_token.item())]
    step_ms: list[float] = []
    next_token = first_next_token
    pos = int(prompt_len)
    device = next_token.device
    cache_pos = torch.zeros(1, dtype=torch.long, device=device)
    if stop_on_eos and out_tokens[0] == eos_token_id:
        return past_kv, out_tokens, step_ms

    for _ in range(max_new_tokens - 1):
        cache_pos[0] = pos
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(
            input_ids=next_token,
            past_key_values=past_kv,
            cache_position=cache_pos,
            position_ids=cache_pos.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(next_token.item())
        out_tokens.append(tok)
        pos += 1
        if stop_on_eos and tok == eos_token_id:
            break
    return past_kv, out_tokens, step_ms


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_one(
    *,
    sample: PooledSample,
    method: str,
    ratio: float,
    model: Any,
    processor: Any,
    model_config: Any,
    press: Optional[VisualUtilityStudentPress],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_text = build_prompt(sample.question, args.prompt_template, image_count=len(sample.image_paths))
    prompt_inputs = processor(text=prompt_text, images=open_images(sample.image_paths), return_tensors="pt")
    prompt_inputs = move_batch_to_device(prompt_inputs, device, dtype)
    image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
        input_ids=prompt_inputs["input_ids"],
        model_config=model_config,
        num_images=len(sample.image_paths),
    )
    n_img = int(image_positions.numel())
    n_text = int(prompt_len_mm) - n_img
    n_image_keep = n_img
    if press is not None:
        n_image_keep = _compute_n_image_keep(
            n_img,
            n_text,
            image_keep_ratio=None,
            total_keep_ratio=ratio,
        )
        press.set_image_positions(image_positions)
        last_img = int(image_positions.max().item()) if n_img else -1
        question_positions = (
            torch.arange(last_img + 1, int(prompt_len_mm), dtype=torch.long)
            if last_img + 1 < int(prompt_len_mm)
            else torch.empty(0, dtype=torch.long)
        )
        press.set_question_positions(question_positions)
        press.reset_student_timing()

    empty_cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    context = press(model) if press is not None else nullcontext()
    with context:
        prefill = model(
            **prompt_inputs,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000.0

    past_kv = prefill.past_key_values
    prompt_retained_seq_len = cache_seq_len(past_kv)
    first_next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ttft_ms = prefill_ms
    past_kv, generated_ids, decode_step_ms = decode_with_timing(
        model=model,
        past_kv=past_kv,
        first_next_token=first_next_token,
        prompt_len=int(prompt_len_mm),
        eos_token_id=resolve_eos_token_id(processor, model_config),
        max_new_tokens=args.max_new_tokens,
        stop_on_eos=args.stop_on_eos,
    )
    torch.cuda.synchronize()
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / GIB
    final_cache_seq_len = cache_seq_len(past_kv)
    final_kv_gib = kv_cache_bytes(past_kv) / GIB
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    student_ms = None
    if press is not None:
        student_ms = float(getattr(press, "student_score_total_ms", 0.0))
        press.clear_sample_context()

    decode_ms_per_token = mean(decode_step_ms) if decode_step_ms else 0.0
    total_decode_ms = sum(decode_step_ms)
    denominator = max(1, int(prompt_len_mm) + len(decode_step_ms))
    return {
        "method": method,
        "ratio": ratio,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        "prefill_latency_ms": prefill_ms,
        "student_forward_ms": student_ms,
        "ttft_ms": ttft_ms,
        "decode_latency_ms_per_token": decode_ms_per_token,
        "decode_total_ms": total_decode_ms,
        "peak_gpu_memory_gib": peak_alloc_gib,
        "peak_gpu_reserved_gib": peak_reserved_gib,
        "kv_cache_gib": final_kv_gib,
        "prompt_full_seq_len": int(prompt_len_mm),
        "prompt_retained_seq_len": int(prompt_retained_seq_len),
        "final_cache_seq_len": int(final_cache_seq_len),
        "generated_tokens": len(generated_ids),
        "decode_steps": len(decode_step_ms),
        "image_tokens_total": n_img,
        "image_tokens_kept": int(n_image_keep),
        "image_keep_ratio": float(n_image_keep / max(1, n_img)),
        "r_img": float(n_image_keep / max(1, n_img)),
        "r_eff_prompt": float(prompt_retained_seq_len / max(1, int(prompt_len_mm))),
        "r_eff_decode_t_end": float(final_cache_seq_len / denominator),
        "prediction": text,
    }


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "prefill_latency_ms",
        "student_forward_ms",
        "ttft_ms",
        "decode_latency_ms_per_token",
        "decode_total_ms",
        "peak_gpu_memory_gib",
        "peak_gpu_reserved_gib",
        "kv_cache_gib",
        "r_img",
        "r_eff_prompt",
        "r_eff_decode_t_end",
        "prompt_full_seq_len",
        "prompt_retained_seq_len",
        "final_cache_seq_len",
        "generated_tokens",
        "decode_steps",
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    summary: list[dict[str, Any]] = []
    full = by_method.get("full_cache_100", [])
    full_by_key = {row["sample_key"]: row for row in full}
    for method, method_rows in by_method.items():
        item: dict[str, Any] = {
            "method": method,
            "ratio": method_rows[0]["ratio"],
            "n_samples": len(method_rows),
            "datasets": ",".join(sorted({row["dataset"] for row in method_rows})),
        }
        for field in numeric_fields:
            values = [row[field] for row in method_rows if row.get(field) is not None]
            if not values:
                item[f"{field}_mean"] = None
                item[f"{field}_std"] = None
                continue
            item[f"{field}_mean"] = float(mean(values))
            item[f"{field}_std"] = float(stdev(values)) if len(values) > 1 else 0.0

        if method != "full_cache_100" and full_by_key:
            paired = [(row, full_by_key[row["sample_key"]]) for row in method_rows if row["sample_key"] in full_by_key]
            if paired:
                item["paired_n_vs_full"] = len(paired)
                item["decode_latency_pct_of_full"] = float(
                    100.0
                    * mean(row["decode_latency_ms_per_token"] / base["decode_latency_ms_per_token"] for row, base in paired)
                )
                item["peak_gpu_memory_pct_of_full"] = float(
                    100.0 * mean(row["peak_gpu_memory_gib"] / base["peak_gpu_memory_gib"] for row, base in paired)
                )
                item["kv_cache_pct_of_full"] = float(
                    100.0 * mean(row["kv_cache_gib"] / base["kv_cache_gib"] for row, base in paired)
                )
        summary.append(item)
    return sorted(summary, key=lambda x: float(x["ratio"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    extras = sorted({k for row in rows for k in row.keys()} - set(fieldnames))
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU latency/memory measurement")
    device = torch.device(args.device)
    dtype = getattr(torch, args.torch_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_path = str(Path(args.model_path).resolve())
    student_path = str(Path(args.student_path).resolve())
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    configure_llava_processor(processor, config)

    samples = sample_manifest(args=args, processor=processor, model_config=config)
    manifest_path = out_dir / "sample_manifest.json"
    save_manifest(manifest_path, samples, args)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    configure_llava_processor(processor, model.config)
    model = model.to(device).eval()

    ratios = sorted(set(float(r) for r in args.ratios), reverse=True)
    method_specs: list[tuple[str, float]] = [("full_cache_100", 1.0)]
    for ratio in ratios:
        if ratio >= 1.0:
            continue
        method_specs.append((f"student_keep_{int(round(ratio * 100)):03d}", ratio))

    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_model_path": model_path,
            "resolved_student_path": student_path,
            "manifest_path": str(manifest_path),
            "n_samples": len(samples),
        }
    )
    write_json(out_dir / "run_config.json", run_config)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for method, ratio in method_specs:
        press: Optional[VisualUtilityStudentPress] = None
        if ratio < 1.0:
            press = VisualUtilityStudentPress(
                total_keep_ratio=ratio,
                head_reduce="amax",
                student_model_name=student_path,
            )
            press.post_init_from_model(model)
            if press._student is not None:
                press._student = press._student.to(device=device, dtype=dtype).eval()
            empty_cuda()

        if args.warmup_samples > 0:
            for sample in samples[: args.warmup_samples]:
                try:
                    _ = measure_one(
                        sample=sample,
                        method="warmup",
                        ratio=ratio,
                        model=model,
                        processor=processor,
                        model_config=model.config,
                        press=press,
                        device=device,
                        dtype=dtype,
                        args=args,
                    )
                except Exception:
                    pass
                empty_cuda()

        for sample in tqdm(samples, desc=f"Measuring {method}"):
            try:
                row = measure_one(
                    sample=sample,
                    method=method,
                    ratio=ratio,
                    model=model,
                    processor=processor,
                    model_config=model.config,
                    press=press,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                rows.append(row)
                write_csv(out_dir / "per_sample.csv", rows)
            except torch.cuda.OutOfMemoryError as exc:
                failures.append({"sample_key": sample.sample_key, "method": method, "error": f"CUDA OOM: {exc}"})
                empty_cuda()
                if not args.continue_on_error:
                    raise
            except Exception as exc:
                failures.append({"sample_key": sample.sample_key, "method": method, "error": repr(exc)})
                empty_cuda()
                if not args.continue_on_error:
                    raise
            finally:
                if press is not None:
                    press.clear_sample_context()
        if press is not None:
            if press._student is not None:
                press._student = press._student.to("cpu")
            del press
            empty_cuda()

    summary_rows = summarize(rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_json(out_dir / "summary.json", summary_rows)
    write_json(out_dir / "failures.json", failures)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
