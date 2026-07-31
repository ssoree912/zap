#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired full-cache versus Q-ViK post-eviction decode benchmark.

The timed region starts only after multimodal prefill, student scoring, and
per-layer KV eviction have completed. Every sample executes the same number of
cached greedy-decode forward calls; EOS is intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata as importlib_metadata
import json
import math
import pickle
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


WORKSPACE = Path("/workspace/nips")
QVIK_ROOT = WORKSPACE / "Q-ViK"
ZAP_ROOT = WORKSPACE / "zap"
for root in (QVIK_ROOT, ZAP_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _patch_dependency_versions() -> None:
    original_version = importlib_metadata.version

    def version(package_name: str) -> str:
        if package_name == "tokenizers":
            return "0.20.3"
        if package_name == "huggingface-hub":
            return "0.26.5"
        return original_version(package_name)

    importlib_metadata.version = version


def _patch_torch_load() -> None:
    original_load = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        retry = dict(kwargs)
        while True:
            try:
                return original_load(*args, **retry)
            except RuntimeError as exc:
                if retry.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry.pop("mmap", None)
                    continue
                raise
            except pickle.UnpicklingError:
                if retry.get("weights_only") is True:
                    retry["weights_only"] = False
                    retry.pop("mmap", None)
                    continue
                raise

    torch.load = load


_patch_dependency_versions()
_patch_torch_load()

from qvik.eval.kv_decode_utils import trim_kv_cache_per_layer  # noqa: E402


DATASETS = ("ChartQA", "DocVQA", "GQA", "TextVQA")
GIB = 1024**3


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    question: str
    answer: Any
    image_path: str

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.sample_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("llava15", "onevision"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "efficiency_bench/results/vqa_100_seed42/manifest.json",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=25)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Use the first N manifest samples without requiring a balanced VQA manifest.",
    )
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--llava15-model",
        type=Path,
        default=WORKSPACE / "models/llava-v1.5-7b",
    )
    parser.add_argument(
        "--llava15-student",
        type=Path,
        default=(
            ZAP_ROOT
            / "artifacts/rebuttal_tradeoff_llava15_zap_teacher_n600"
            / "checkpoints/base"
        ),
    )
    parser.add_argument("--llava15-total-keep-ratio", type=float, default=0.2)
    parser.add_argument(
        "--onevision-model",
        type=Path,
        default=WORKSPACE / "models/llava-onevision-qwen2-7b-ov",
    )
    parser.add_argument(
        "--onevision-student",
        type=Path,
        default=(
            ZAP_ROOT
            / "artifacts/original_onevision_teacher"
            / "student_onevision_answer_n1800_e15_seed0"
        ),
    )
    parser.add_argument("--onevision-image-keep-ratio", type=float, default=0.1)
    return parser.parse_args()


def load_samples(path: Path, per_dataset: int, sample_limit: int) -> list[Sample]:
    payload = json.loads(path.read_text())
    if sample_limit > 0:
        selected = payload["samples"][:sample_limit]
        if len(selected) != sample_limit:
            raise ValueError(
                f"Manifest has only {len(selected)} samples, requested {sample_limit}"
            )
        return [
            Sample(
                dataset=str(item["dataset"]),
                sample_id=str(item["sample_id"]),
                question=str(item["question"]),
                answer=item.get("answer"),
                image_path=str(item["image_path"]),
            )
            for item in selected
        ]

    groups: dict[str, list[Sample]] = defaultdict(list)
    for item in payload["samples"]:
        dataset = str(item["dataset"])
        if dataset not in DATASETS or len(groups[dataset]) >= per_dataset:
            continue
        groups[dataset].append(
            Sample(
                dataset=dataset,
                sample_id=str(item["sample_id"]),
                question=str(item["question"]),
                answer=item.get("answer"),
                image_path=str(item["image_path"]),
            )
        )
    counts = {dataset: len(groups[dataset]) for dataset in DATASETS}
    if any(count != per_dataset for count in counts.values()):
        raise ValueError(f"Manifest does not provide the requested balanced sample: {counts}")
    return [sample for dataset in DATASETS for sample in groups[dataset]]


def cache_lengths(past_kv: Any) -> list[int]:
    if hasattr(past_kv, "key_cache"):
        return [int(key.shape[-2]) for key in past_kv.key_cache]
    if hasattr(past_kv, "layers"):
        return [int(layer.keys.shape[-2]) for layer in past_kv.layers]
    return [int(layer[0].shape[-2]) for layer in past_kv]


def cache_bytes(past_kv: Any) -> int:
    if hasattr(past_kv, "key_cache"):
        pairs = zip(past_kv.key_cache, past_kv.value_cache)
    elif hasattr(past_kv, "layers"):
        pairs = ((layer.keys, layer.values) for layer in past_kv.layers)
    else:
        pairs = ((layer[0], layer[1]) for layer in past_kv)
    return sum(
        int(key.numel() * key.element_size() + value.numel() * value.element_size())
        for key, value in pairs
    )


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


@torch.inference_mode()
def timed_decode(
    *,
    model: Any,
    past_kv: Any,
    next_token: torch.Tensor,
    logical_prompt_len: int,
    decode_steps: int,
    device: torch.device,
) -> tuple[float, Any]:
    initial_lengths = cache_lengths(past_kv)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for decode_index in range(decode_steps):
        position = torch.full(
            (1,),
            logical_prompt_len + decode_index,
            dtype=torch.long,
            device=device,
        )
        output = model(
            input_ids=next_token,
            past_key_values=past_kv,
            cache_position=position,
            position_ids=position.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_kv = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del output
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    final_lengths = cache_lengths(past_kv)
    if any(
        final - initial != decode_steps
        for initial, final in zip(initial_lengths, final_lengths)
    ):
        raise RuntimeError(
            f"Unexpected cache growth: initial={initial_lengths}, final={final_lengths}"
        )
    return elapsed_ms, past_kv


def infer_llava15_image_positions(
    input_ids: torch.Tensor,
    image_token_index: int,
    image_feature_len: int = 576,
) -> tuple[torch.Tensor, int]:
    positions: list[int] = []
    cursor = 0
    for token_id in input_ids[0].detach().cpu().tolist():
        if int(token_id) == image_token_index:
            positions.extend(range(cursor, cursor + image_feature_len))
            cursor += image_feature_len
        else:
            cursor += 1
    if not positions:
        raise ValueError("No image placeholder in LLaVA-1.5 prompt")
    return torch.tensor(positions, dtype=torch.long), cursor


def load_llava15(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from kvpress.presses.visual_utility_student import VisualUtilityStudent
    from qvik.llava15.mm_utils import get_model_name_from_path
    from qvik.llava15.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(str(args.llava15_model))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.llava15_model),
        None,
        model_name,
        device_map=str(device),
        device=str(device),
    )
    model = model.to(device).eval()
    dtype = next(model.parameters()).dtype
    student = VisualUtilityStudent.from_pretrained(args.llava15_student)
    student = student.to(device=device, dtype=dtype).eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "student": student,
        "dtype": dtype,
    }


@torch.inference_mode()
def measure_llava15(
    *,
    sample: Sample,
    pruned: bool,
    state: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    from qvik.llava15.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from qvik.llava15.conversation import conv_templates
    from qvik.llava15.mm_utils import process_images, tokenizer_image_token

    tokenizer = state["tokenizer"]
    model = state["model"]
    student = state["student"]
    question = f"{DEFAULT_IMAGE_TOKEN}\n{sample.question}"
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    with Image.open(sample.image_path) as image:
        visual = image.convert("RGB")
        image_tensor = process_images([visual], state["image_processor"], model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=state["dtype"])

    prefill = model(
        input_ids=input_ids,
        images=image_tensor,
        use_cache=True,
        output_hidden_states=pruned,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    image_positions, prompt_len = infer_llava15_image_positions(
        input_ids, IMAGE_TOKEN_INDEX
    )
    n_image = int(image_positions.numel())
    n_text = prompt_len - n_image
    n_keep = n_image
    eviction_ms = 0.0

    if pruned:
        n_keep = max(
            1,
            int(
                round(
                    n_image
                    - (1.0 - args.llava15_total_keep_ratio) * prompt_len
                )
            ),
        )
        n_keep = min(n_keep, n_image)
        last_image = int(image_positions.max().item())
        question_positions = torch.arange(
            last_image + 1,
            prompt_len,
            dtype=torch.long,
            device=device,
        )
        image_positions_device = image_positions.to(device)
        keep_masks: dict[int, torch.Tensor] = {}
        torch.cuda.synchronize(device)
        eviction_start = time.perf_counter()
        for layer_index in student.layer_indices:
            scores = student.forward_layer(
                layer_index,
                prefill.hidden_states[layer_index],
                image_positions_device,
                question_positions,
            ).squeeze(0)
            top_indices = torch.topk(scores, k=n_keep, largest=True).indices
            full_mask = torch.ones(prompt_len, dtype=torch.bool)
            image_mask = torch.zeros(n_image, dtype=torch.bool)
            image_mask[top_indices.detach().cpu()] = True
            full_mask[image_positions] = image_mask
            keep_masks[layer_index] = full_mask
        past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
        torch.cuda.synchronize(device)
        eviction_ms = (time.perf_counter() - eviction_start) * 1000.0
        del keep_masks

    initial_lengths = cache_lengths(past_kv)
    del prefill, image_tensor
    clear_cuda(device)
    elapsed_ms, past_kv = timed_decode(
        model=model,
        past_kv=past_kv,
        next_token=next_token,
        logical_prompt_len=prompt_len,
        decode_steps=args.decode_steps,
        device=device,
    )
    row = common_row(
        sample=sample,
        method="qvik" if pruned else "full_cache",
        prompt_len=prompt_len,
        n_image=n_image,
        n_text=n_text,
        n_keep=n_keep,
        initial_lengths=initial_lengths,
        decode_steps=args.decode_steps,
        elapsed_ms=elapsed_ms,
        eviction_ms=eviction_ms,
        past_kv=past_kv,
        device=device,
    )
    del past_kv, next_token, input_ids
    return row


def load_onevision(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    import qvik.llava_onevision  # noqa: F401
    from kvpress.presses.visual_utility_student_onevision import (
        VisualUtilityStudentOneVision,
    )
    from qvik.llava_onevision.mm_utils import get_model_name_from_path
    from qvik.llava_onevision.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(str(args.onevision_model))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.onevision_model),
        None,
        model_name,
        device_map=str(device),
        attn_implementation="sdpa",
        multimodal=True,
    )
    model = model.eval()
    dtype = next(model.parameters()).dtype
    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(device=device, dtype=dtype)
    student = VisualUtilityStudentOneVision.from_pretrained(args.onevision_student)
    student = student.to(device=device, dtype=dtype).eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "student": student,
        "dtype": dtype,
    }


@torch.inference_mode()
def measure_onevision(
    *,
    sample: Sample,
    pruned: bool,
    state: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    from qvik.llava_onevision.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from qvik.llava_onevision.conversation import conv_templates
    from qvik.llava_onevision.mm_utils import process_images, tokenizer_image_token

    tokenizer = state["tokenizer"]
    model = state["model"]
    student = state["student"]
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{sample.question}")
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    attention_mask = input_ids.ne(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    ).to(device)
    with Image.open(sample.image_path) as image:
        visual = image.convert("RGB")
        image_size = visual.size
        image_tensor = process_images(
            [visual], state["image_processor"], model.config
        )
    if isinstance(image_tensor, list):
        image_tensor = [
            tensor.to(device=device, dtype=state["dtype"]) for tensor in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(device=device, dtype=state["dtype"])

    _, _, expanded_attention, _, inputs_embeds, _ = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            attention_mask,
            None,
            None,
            image_tensor,
            ["image"],
            [image_size],
        )
    )
    raw_ids = input_ids[0][attention_mask[0].bool()]
    placeholders = (raw_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if placeholders.numel() != 1:
        raise ValueError(f"Expected one image placeholder, found {placeholders.numel()}")
    n_text = int(raw_ids.shape[0]) - 1
    prompt_len = int(inputs_embeds.shape[1])
    n_image = prompt_len - n_text
    image_start = int(placeholders[0].item())
    image_positions = torch.arange(
        image_start, image_start + n_image, dtype=torch.long
    )

    prefill = model(
        inputs_embeds=inputs_embeds,
        attention_mask=expanded_attention,
        use_cache=True,
        output_hidden_states=pruned,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    n_keep = n_image
    eviction_ms = 0.0

    if pruned:
        n_keep = max(
            1, int(math.ceil(n_image * args.onevision_image_keep_ratio))
        )
        last_image = int(image_positions.max().item())
        question_positions = torch.arange(
            last_image + 1,
            prompt_len,
            dtype=torch.long,
            device=device,
        )
        image_positions_device = image_positions.to(device)
        keep_masks: dict[int, torch.Tensor] = {}
        torch.cuda.synchronize(device)
        eviction_start = time.perf_counter()
        for layer_index in student.layer_indices:
            scores = student.layers[str(layer_index)](
                prefill.hidden_states[layer_index + 1],
                image_positions_device,
                question_positions,
            ).squeeze(0)
            top_indices = torch.topk(scores, k=n_keep, largest=True).indices
            full_mask = torch.ones(prompt_len, dtype=torch.bool)
            image_mask = torch.zeros(n_image, dtype=torch.bool)
            image_mask[top_indices.detach().cpu()] = True
            full_mask[image_positions] = image_mask
            keep_masks[layer_index] = full_mask
        past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
        torch.cuda.synchronize(device)
        eviction_ms = (time.perf_counter() - eviction_start) * 1000.0
        del keep_masks

    initial_lengths = cache_lengths(past_kv)
    del prefill, inputs_embeds, image_tensor
    clear_cuda(device)
    elapsed_ms, past_kv = timed_decode(
        model=model,
        past_kv=past_kv,
        next_token=next_token,
        logical_prompt_len=prompt_len,
        decode_steps=args.decode_steps,
        device=device,
    )
    row = common_row(
        sample=sample,
        method="qvik" if pruned else "full_cache",
        prompt_len=prompt_len,
        n_image=n_image,
        n_text=n_text,
        n_keep=n_keep,
        initial_lengths=initial_lengths,
        decode_steps=args.decode_steps,
        elapsed_ms=elapsed_ms,
        eviction_ms=eviction_ms,
        past_kv=past_kv,
        device=device,
    )
    del past_kv, next_token, input_ids
    return row


def common_row(
    *,
    sample: Sample,
    method: str,
    prompt_len: int,
    n_image: int,
    n_text: int,
    n_keep: int,
    initial_lengths: list[int],
    decode_steps: int,
    elapsed_ms: float,
    eviction_ms: float,
    past_kv: Any,
    device: torch.device,
) -> dict[str, Any]:
    ms_per_token = elapsed_ms / decode_steps
    return {
        "method": method,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "sample_key": sample.key,
        "prompt_tokens_full": prompt_len,
        "text_tokens": n_text,
        "image_tokens_full": n_image,
        "image_tokens_kept": n_keep,
        "effective_image_keep_ratio": n_keep / max(1, n_image),
        "effective_total_keep_ratio": (n_text + n_keep) / max(1, prompt_len),
        "initial_cache_tokens_mean": statistics.mean(initial_lengths),
        "initial_cache_tokens_min": min(initial_lengths),
        "initial_cache_tokens_max": max(initial_lengths),
        "decode_steps": decode_steps,
        "decode_total_ms": elapsed_ms,
        "decode_ms_per_token": ms_per_token,
        "decode_tokens_per_second": 1000.0 / ms_per_token,
        "eviction_ms_excluded": eviction_ms,
        "decode_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "decode_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "final_kv_gib": cache_bytes(past_kv) / GIB,
    }


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["decode_ms_per_token"]) for row in rows]
    throughputs = [float(row["decode_tokens_per_second"]) for row in rows]
    return {
        "n": len(rows),
        "decode_ms_per_token_mean": statistics.mean(latencies),
        "decode_ms_per_token_std": (
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        ),
        "decode_ms_per_token_median": statistics.median(latencies),
        "decode_total_ms_mean": statistics.mean(
            float(row["decode_total_ms"]) for row in rows
        ),
        "decode_tokens_per_second_mean": statistics.mean(throughputs),
        "initial_cache_tokens_mean": statistics.mean(
            float(row["initial_cache_tokens_mean"]) for row in rows
        ),
        "effective_image_keep_ratio_mean": statistics.mean(
            float(row["effective_image_keep_ratio"]) for row in rows
        ),
        "effective_total_keep_ratio_mean": statistics.mean(
            float(row["effective_total_keep_ratio"]) for row in rows
        ),
        "eviction_ms_mean_excluded": statistics.mean(
            float(row["eviction_ms_excluded"]) for row in rows
        ),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_dataset_method[(row["dataset"], row["method"])].append(row)
    full = {row["sample_key"]: row for row in by_method["full_cache"]}
    qvik = {row["sample_key"]: row for row in by_method["qvik"]}
    paired_keys = sorted(set(full) & set(qvik))
    latency_ratios = [
        qvik[key]["decode_ms_per_token"] / full[key]["decode_ms_per_token"]
        for key in paired_keys
    ]
    speedups = [
        full[key]["decode_ms_per_token"] / qvik[key]["decode_ms_per_token"]
        for key in paired_keys
    ]

    datasets: dict[str, Any] = {}
    dataset_names = sorted({row["dataset"] for row in rows})
    for dataset in dataset_names:
        dataset_full = {
            row["sample_key"]: row
            for row in by_dataset_method[(dataset, "full_cache")]
        }
        dataset_qvik = {
            row["sample_key"]: row
            for row in by_dataset_method[(dataset, "qvik")]
        }
        keys = sorted(set(dataset_full) & set(dataset_qvik))
        ratios = [
            dataset_qvik[key]["decode_ms_per_token"]
            / dataset_full[key]["decode_ms_per_token"]
            for key in keys
        ]
        datasets[dataset] = {
            "full_cache": stats(list(dataset_full.values())),
            "qvik": stats(list(dataset_qvik.values())),
            "paired_n": len(keys),
            "latency_reduction_percent": (
                100.0 * (1.0 - statistics.mean(ratios)) if ratios else None
            ),
            "decode_speedup_x": (
                statistics.mean(
                    dataset_full[key]["decode_ms_per_token"]
                    / dataset_qvik[key]["decode_ms_per_token"]
                    for key in keys
                )
                if keys
                else None
            ),
        }

    return {
        "backbone": args.backbone,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "batch_size": 1,
        "decode_steps_per_sample": args.decode_steps,
        "timed_region": (
            "CUDA-synchronized cached autoregressive decode only; multimodal "
            "prefill, student scoring, and KV eviction are completed before timing"
        ),
        "eos_policy": "ignored; fixed decode step count",
        "keep_ratio_basis": (
            "total_prompt"
            if args.backbone == "llava15"
            else "image_only"
        ),
        "requested_keep_ratio": (
            args.llava15_total_keep_ratio
            if args.backbone == "llava15"
            else args.onevision_image_keep_ratio
        ),
        "full_cache": stats(by_method["full_cache"]),
        "qvik": stats(by_method["qvik"]),
        "paired_n": len(paired_keys),
        "latency_pct_of_full": (
            100.0 * statistics.mean(latency_ratios) if latency_ratios else None
        ),
        "latency_reduction_percent": (
            100.0 * (1.0 - statistics.mean(latency_ratios))
            if latency_ratios
            else None
        ),
        "decode_speedup_x": (
            statistics.mean(speedups) if speedups else None
        ),
        "datasets": datasets,
        "failures": failures,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.decode_steps <= 0 or args.samples_per_dataset <= 0:
        raise ValueError("decode steps and samples per dataset must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(
        args.manifest,
        args.samples_per_dataset,
        args.sample_limit,
    )
    measure = measure_llava15 if args.backbone == "llava15" else measure_onevision
    state = load_llava15(args, device) if args.backbone == "llava15" else load_onevision(args, device)
    print(
        f"[load-ok] backbone={args.backbone} dtype={state['dtype']} "
        f"samples={len(samples)} decode_steps={args.decode_steps} "
        f"gpu={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    for sample in samples[: args.warmup_samples]:
        for pruned in (False, True):
            warmup_row = measure(
                sample=sample,
                pruned=pruned,
                state=state,
                args=args,
                device=device,
            )
            print(
                f"[warmup] {sample.key} {warmup_row['method']} "
                f"{warmup_row['decode_ms_per_token']:.3f} ms/token",
                flush=True,
            )
            clear_cuda(device)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, 1):
        order = (False, True) if sample_index % 2 else (True, False)
        for pruned in order:
            try:
                row = measure(
                    sample=sample,
                    pruned=pruned,
                    state=state,
                    args=args,
                    device=device,
                )
                rows.append(row)
                write_csv(args.output_dir / "per_sample.csv", rows)
                print(
                    f"[{sample_index:03d}/{len(samples):03d}] {sample.key} "
                    f"{row['method']} {row['decode_ms_per_token']:.3f} ms/token "
                    f"({row['decode_tokens_per_second']:.2f} tok/s) "
                    f"cache={row['initial_cache_tokens_mean']:.1f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "sample_key": sample.key,
                        "method": "qvik" if pruned else "full_cache",
                        "error": repr(exc),
                    }
                )
                print(
                    f"[error] {sample.key} "
                    f"{'qvik' if pruned else 'full_cache'}: {exc!r}",
                    flush=True,
                )
                clear_cuda(device)

    summary = summarize(rows, args=args, failures=failures)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "manifest": str(args.manifest.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "llava15_model": str(args.llava15_model.resolve()),
                "llava15_student": str(args.llava15_student.resolve()),
                "onevision_model": str(args.onevision_model.resolve()),
                "onevision_student": str(args.onevision_student.resolve()),
                "samples": [asdict(sample) for sample in samples],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["paired_n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
