#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure cached decode efficiency for the ZAP LLaVA-1.5 base student.

The implementation follows ``lmms_llava15_original_student.py``: prefill with
all hidden states, score image tokens independently at every decoder layer,
trim every layer's KV cache, and decode with absolute multimodal positions.
Only the fixed cached-decode loop is timed and included in peak CUDA memory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata as importlib_metadata
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
ZAP_ROOT = WORKSPACE / "zap"
QVIK_ROOT = WORKSPACE / "Q-ViK"
for _path in (str(QVIK_ROOT), str(ZAP_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _patch_transformers_dependency_versions() -> None:
    original_version = importlib_metadata.version

    def version(package_name: str) -> str:
        if package_name == "tokenizers":
            return "0.20.3"
        if package_name == "huggingface-hub":
            return "0.26.5"
        return original_version(package_name)

    importlib_metadata.version = version


def _patch_torch_load_legacy_bin_mmap() -> None:
    original_torch_load = torch.load

    def load(*args, **kwargs):
        retry_kwargs = dict(kwargs)
        while True:
            try:
                return original_torch_load(*args, **retry_kwargs)
            except RuntimeError as exc:
                if retry_kwargs.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry_kwargs.pop("mmap", None)
                    continue
                raise
            except pickle.UnpicklingError:
                if retry_kwargs.get("weights_only") is True:
                    retry_kwargs["weights_only"] = False
                    retry_kwargs.pop("mmap", None)
                    continue
                raise

    torch.load = load


_patch_transformers_dependency_versions()
_patch_torch_load_legacy_bin_mmap()

from foresight.eval.kv_decode_utils import trim_kv_cache_per_layer  # noqa: E402
from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402
from qvik.llava15.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from qvik.llava15.conversation import conv_templates  # noqa: E402
from qvik.llava15.mm_utils import (  # noqa: E402
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from qvik.llava15.model.builder import load_pretrained_model  # noqa: E402


DATASETS = ("ChartQA", "DocVQA", "GQA", "TextVQA")
MODEL_PATH = WORKSPACE / "models" / "llava-v1.5-7b"
STUDENT_PATH = (
    ZAP_ROOT
    / "artifacts"
    / "rebuttal_tradeoff_llava15_zap_teacher_n600"
    / "checkpoints"
    / "base"
)
OUTPUT_ROOT = WORKSPACE / "efficiency_bench" / "results"
MANIFEST_PATH = OUTPUT_ROOT / "vqa_100_seed42" / "manifest.json"
RUN_NAME = "vqa_decode_100_seed42"
METHOD = "zap_student_base"
KEEP_RATIO = 0.2
IMAGE_FEATURE_LEN = 576
GRID_H = 24
GRID_W = 24
GIB = 1024**3
TFLOPS = 1e12


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    question: str
    answer: Any
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--student-path", type=Path, default=STUDENT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_manifest(path: Path, count: int, seed: int) -> list[Sample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["samples_per_dataset"]) != count or int(payload["seed"]) != seed:
        raise ValueError(
            f"Manifest settings differ: {path} has count="
            f"{payload['samples_per_dataset']}, seed={payload['seed']}"
        )
    samples = [Sample(**item) for item in payload["samples"]]
    counts = {dataset: sum(item.dataset == dataset for item in samples) for dataset in DATASETS}
    if any(value != count for value in counts.values()):
        raise ValueError(f"Unexpected manifest dataset counts: {counts}")
    return samples


def infer_image_positions(input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Map the raw image placeholder to its 576 multimodal positions."""
    image_positions: list[int] = []
    cursor = 0
    for token_id in input_ids[0].detach().cpu().tolist():
        if int(token_id) == IMAGE_TOKEN_INDEX:
            image_positions.extend(range(cursor, cursor + IMAGE_FEATURE_LEN))
            cursor += IMAGE_FEATURE_LEN
        else:
            cursor += 1
    if not image_positions:
        raise ValueError("No image placeholder found.")
    return torch.tensor(image_positions, dtype=torch.long), cursor


def build_inputs(
    sample: Sample,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        image_tensor = process_images([visual], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    return input_ids, image_tensor.to(device=device, dtype=model_dtype)


def cache_lengths(past_key_values: Any) -> list[int]:
    if hasattr(past_key_values, "key_cache"):
        return [int(key.shape[-2]) for key in past_key_values.key_cache]
    if hasattr(past_key_values, "layers"):
        return [int(layer.keys.shape[-2]) for layer in past_key_values.layers]
    return [int(layer[0].shape[-2]) for layer in past_key_values]


def decoder_step_flops(cache_lens: list[int], decode_index: int) -> int:
    """Dense matmul FLOPs for one cached LLaMA decode token."""
    hidden, intermediate, vocab = 4096, 11008, 32000
    total = 0
    for initial_cache_len in cache_lens:
        kv_len = initial_cache_len + 1 + decode_index
        total += (
            8 * hidden * hidden
            + 4 * hidden * kv_len
            + 6 * hidden * intermediate
        )
    return total + 2 * hidden * vocab


@torch.inference_mode()
def measure_one(
    *,
    sample: Sample,
    tokenizer: Any,
    model: Any,
    student: VisualUtilityStudent,
    image_processor: Any,
    device: torch.device,
    model_dtype: torch.dtype,
    decode_tokens: int,
) -> dict[str, Any]:
    input_ids, image_tensor = build_inputs(
        sample, tokenizer, model, image_processor, device, model_dtype
    )
    prefill = model(
        input_ids=input_ids,
        images=image_tensor,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    hidden_states = prefill.hidden_states
    past_key_values = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    image_positions, inferred_prompt_len = infer_image_positions(input_ids)
    prompt_len = int(hidden_states[-1].shape[1])
    if inferred_prompt_len != prompt_len:
        raise RuntimeError(
            f"Multimodal prompt mismatch: inferred={inferred_prompt_len}, actual={prompt_len}"
        )
    n_image = int(image_positions.numel())
    n_text = prompt_len - n_image
    n_keep = max(1, int(round(n_image - (1.0 - KEEP_RATIO) * prompt_len)))
    n_keep = min(n_keep, n_image)

    last_image_position = int(image_positions.max().item())
    question_positions = torch.arange(
        last_image_position + 1, prompt_len, dtype=torch.long, device=device
    )
    image_positions_device = image_positions.to(device)
    keep_masks: dict[int, torch.Tensor] = {}
    for layer_idx in student.layer_indices:
        scores = student.layers[str(layer_idx)](
            hidden_states[layer_idx + 1],
            image_positions_device,
            question_positions,
            GRID_H,
            GRID_W,
        ).squeeze(0)
        if n_keep >= n_image:
            continue
        top_indices = torch.topk(scores, k=n_keep, largest=True).indices
        full_mask = torch.ones(prompt_len, dtype=torch.bool)
        image_mask = torch.zeros(n_image, dtype=torch.bool)
        image_mask[top_indices.detach().cpu()] = True
        full_mask[image_positions] = image_mask
        keep_masks[layer_idx] = full_mask

    past_key_values = trim_kv_cache_per_layer(past_key_values, keep_masks)
    initial_cache_lens = cache_lengths(past_key_values)
    expected_cache_len = n_text + n_keep
    if len(initial_cache_lens) != int(model.config.num_hidden_layers):
        raise RuntimeError(f"Unexpected cache layer count: {len(initial_cache_lens)}")
    if any(length != expected_cache_len for length in initial_cache_lens):
        raise RuntimeError(
            f"Unexpected trimmed lengths: expected={expected_cache_len}, "
            f"actual={initial_cache_lens}"
        )

    del (
        hidden_states,
        prefill,
        image_tensor,
        scores,
        top_indices,
        full_mask,
        image_mask,
        keep_masks,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for decode_index in range(decode_tokens):
        cache_position = torch.full(
            (1,), prompt_len + decode_index, dtype=torch.long, device=device
        )
        output = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del output
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    final_cache_lens = cache_lengths(past_key_values)
    if any(
        final - initial != decode_tokens
        for initial, final in zip(initial_cache_lens, final_cache_lens)
    ):
        raise RuntimeError(
            f"Unexpected KV growth: initial={initial_cache_lens}, final={final_cache_lens}"
        )
    decode_flops = [
        decoder_step_flops(initial_cache_lens, decode_index)
        for decode_index in range(decode_tokens)
    ]
    row = {
        "method": METHOD,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "raw_prompt_tokens": int(input_ids.shape[1]),
        "multimodal_prompt_tokens": prompt_len,
        "text_tokens": n_text,
        "image_tokens_original": n_image,
        "image_tokens_kept": n_keep,
        "effective_total_keep_ratio": expected_cache_len / prompt_len,
        "effective_image_keep_ratio": n_keep / n_image,
        "decode_tokens": decode_tokens,
        "decode_latency_ms_per_token": elapsed_ms / decode_tokens,
        "decode_latency_ms_total": elapsed_ms,
        "decode_peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "decode_peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "decode_tflops_per_token": statistics.mean(decode_flops) / TFLOPS,
        "decode_tflops_total": sum(decode_flops) / TFLOPS,
        "initial_cache_tokens_per_layer": initial_cache_lens,
        "initial_cache_tokens_layer_mean": statistics.mean(initial_cache_lens),
        "initial_cache_tokens_layer_min": min(initial_cache_lens),
        "initial_cache_tokens_layer_max": max(initial_cache_lens),
        "precision": str(model_dtype).removeprefix("torch."),
    }
    del past_key_values, next_token, input_ids
    return row


def summarize(
    rows: list[dict[str, Any]],
    decode_tokens: int,
    model_path: Path,
    student_path: Path,
    student_variant: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        latency = [float(item["decode_latency_ms_per_token"]) for item in items]
        return {
            "n": len(items),
            "decode_latency_ms_per_token_mean": statistics.mean(latency),
            "decode_latency_ms_per_token_std": (
                statistics.stdev(latency) if len(latency) > 1 else 0.0
            ),
            "decode_latency_ms_per_token_median": statistics.median(latency),
            "decode_peak_memory_allocated_gib_max": max(
                float(item["decode_peak_memory_allocated_gib"]) for item in items
            ),
            "decode_peak_memory_reserved_gib_max": max(
                float(item["decode_peak_memory_reserved_gib"]) for item in items
            ),
            "decode_tflops_per_token_mean": statistics.mean(
                float(item["decode_tflops_per_token"]) for item in items
            ),
            "initial_cache_tokens_layer_mean": statistics.mean(
                float(item["initial_cache_tokens_layer_mean"]) for item in items
            ),
            "effective_total_keep_ratio_mean": statistics.mean(
                float(item["effective_total_keep_ratio"]) for item in items
            ),
            "effective_image_keep_ratio_mean": statistics.mean(
                float(item["effective_image_keep_ratio"]) for item in items
            ),
            "image_tokens_kept_mean": statistics.mean(
                float(item["image_tokens_kept"]) for item in items
            ),
        }

    return {
        "method": METHOD,
        "model": str(model_path.resolve()),
        "student": str(student_path.resolve()),
        "student_variant": student_variant,
        "precision": rows[0]["precision"],
        "batch_size": 1,
        "keep_ratio": KEEP_RATIO,
        "keep_ratio_basis": "total multimodal prompt KV entries per layer",
        "decode_tokens_per_sample": decode_tokens,
        "latency_definition": (
            "CUDA-synchronized cached autoregressive decode after prefill and "
            "student selection; mean over fixed decode steps, EOS ignored"
        ),
        "peak_memory_definition": (
            "PyTorch CUDA peak reset after prefill/selection and inactive-cache "
            "cleanup; includes backbone, resident student, retained prefill KV, "
            "and decode workspace"
        ),
        "flops_definition": (
            "Analytical dense-matmul FLOPs per cached decode token; vision, "
            "projector, prefill, student selection, gather, and elementwise ops excluded"
        ),
        "datasets": {dataset: stats(groups[dataset]) for dataset in DATASETS},
        "overall": stats(rows),
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "dataset",
        "n",
        "decode_latency_ms_per_token_mean",
        "decode_latency_ms_per_token_std",
        "decode_latency_ms_per_token_median",
        "decode_peak_memory_allocated_gib_max",
        "decode_peak_memory_reserved_gib_max",
        "decode_tflops_per_token_mean",
        "initial_cache_tokens_layer_mean",
        "effective_total_keep_ratio_mean",
        "effective_image_keep_ratio_mean",
        "image_tokens_kept_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in (*DATASETS, "overall"):
            values = (
                summary["overall"]
                if dataset == "overall"
                else summary["datasets"][dataset]
            )
            writer.writerow(
                {
                    field: dataset if field == "dataset" else values.get(field, "")
                    for field in fields
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive.")

    samples = load_manifest(args.manifest, args.samples_per_dataset, args.seed)
    run_dir = args.output_root / args.run_name / METHOD
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "samples.jsonl"
    if rows_path.exists() and not args.overwrite:
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows and any(int(row["decode_tokens"]) != args.decode_tokens for row in rows):
            raise ValueError("Existing rows use another decode length; pass --overwrite.")
    else:
        rows = []
        rows_path.write_text("", encoding="utf-8")

    completed = {(row["dataset"], str(row["sample_id"])) for row in rows}
    device = torch.device(args.device)
    model_name = get_model_name_from_path(str(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path),
        None,
        model_name,
        device_map=args.device,
        device=args.device,
    )
    model.eval()
    model.to(device)
    try:
        model.tie_weights()
    except Exception:
        pass
    model_dtype = next(model.parameters()).dtype
    student = VisualUtilityStudent.from_pretrained(args.student_path)
    student = student.to(device=device, dtype=model_dtype).eval()
    print(
        f"Loaded {model_name} ({model_dtype}) and {student.config['variant']} "
        f"student from {args.student_path}",
        flush=True,
    )

    pending = [
        sample
        for sample in samples
        if (sample.dataset, str(sample.sample_id)) not in completed
    ]
    for sample in pending[: args.warmup]:
        measure_one(
            sample=sample,
            tokenizer=tokenizer,
            model=model,
            student=student,
            image_processor=image_processor,
            device=device,
            model_dtype=model_dtype,
            decode_tokens=args.decode_tokens,
        )

    with rows_path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(pending, 1):
            row = measure_one(
                sample=sample,
                tokenizer=tokenizer,
                model=model,
                student=student,
                image_processor=image_processor,
                device=device,
                model_dtype=model_dtype,
                decode_tokens=args.decode_tokens,
            )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index:04d}/{len(pending):04d}] "
                f"{sample.dataset}/{sample.sample_id}: "
                f"{row['decode_latency_ms_per_token']:.3f} ms/token, "
                f"{row['decode_peak_memory_allocated_gib']:.3f} GiB, "
                f"{row['decode_tflops_per_token']:.4f} TFLOPs/token",
                flush=True,
            )

    summary = summarize(
        rows,
        args.decode_tokens,
        args.model_path,
        args.student_path,
        str(student.config["variant"]),
    )
    json_dump(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
