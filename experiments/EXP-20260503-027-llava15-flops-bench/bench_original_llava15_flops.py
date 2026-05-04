#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaVA-1.5-7B (original LLaVA-OneVision repo format) inference compute bench.

Loads the original `LlavaLlamaForCausalLM` checkpoint (not the HF-converted
version) through VFlowOpt's `llava.model.builder.load_pretrained_model`, runs
`generate()` on N random MileBench samples to capture the actual prompt length
and number of decoded tokens, then reports analytical FLOPS for:

  * full_cache_100   : keep all KV
  * student_keep_050 : student-driven KV pruning at total_keep_ratio=0.5
  * student_keep_020 : student-driven KV pruning at total_keep_ratio=0.2

Pruning compute is computed analytically: the press preserves all text-token
KV and only evicts image tokens, so the retained cache size is
    L_kept(ratio) = max(L_p_text, ceil(ratio * L_p))
which determines per-step decode cost during generation. Prefill compute is
identical across configs because eviction happens AFTER each layer's attention.

We do not run the actual student press here — the script focuses on compute
(FLOPS) and end-to-end full-cache latency. Pruning latency is reported as a
roofline `decode_compute_pct_of_full` derived from FLOPS.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VFLOWOPT_LLAVA_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from foresight.image_teacher_utils import build_prompt  # noqa: E402  (kept for compatibility, not used)


GIB = 1024 ** 3
TFLOPS = 1e12


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


# ----------------------------------------------------------------------
# FLOPS calculators
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaFlops:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    def per_layer_decode(self, cache_len: int) -> int:
        D, I = self.hidden_size, self.intermediate_size
        proj = 8 * D * D                # Q,K,V,O linears (no GQA in Llama-7B)
        attn = 4 * D * cache_len        # Q@K^T + A@V (per query token)
        ffn = 6 * D * I                 # SwiGLU (gate, up, down)
        return proj + attn + ffn

    def decode_step(self, cache_len: int) -> int:
        layer = self.per_layer_decode(cache_len)
        lm_head = 2 * self.hidden_size * self.vocab_size
        return self.num_layers * layer + lm_head

    def prefill(self, T: int) -> int:
        D, I, V = self.hidden_size, self.intermediate_size, self.vocab_size
        proj = 8 * D * D * T
        attn = 2 * D * T * (T + 1)      # causal: sum_q (q+1) for QK and AV
        ffn = 6 * D * I * T
        layer = proj + attn + ffn
        lm_head = 2 * D * V * T
        return self.num_layers * layer + lm_head

    def decode_total(self, prompt_len: int, t_decode: int, retained_prompt_len: Optional[int] = None) -> int:
        """Sum decode FLOPS over t = 1..t_decode.

        Cache length at step t is `cache_after_t = base + t`, where
        base = retained_prompt_len if pruning is applied, else prompt_len.
        """
        base = prompt_len if retained_prompt_len is None else retained_prompt_len
        return sum(self.decode_step(base + t) for t in range(1, t_decode + 1))


@dataclass(frozen=True)
class VitFlops:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    image_size: int
    patch_size: int

    def total(self) -> int:
        Dv, Iv = self.hidden_size, self.intermediate_size
        P = (self.image_size // self.patch_size) ** 2 + 1   # +1 cls token
        patch_embed = 2 * 3 * (self.patch_size ** 2) * Dv * (P - 1)
        proj = 8 * Dv * Dv * P
        attn = 2 * Dv * P * P
        ffn = 4 * Dv * Iv * P
        layer = proj + attn + ffn
        return patch_embed + self.num_layers * layer


def make_llama_flops(model_config: Any) -> LlamaFlops:
    return LlamaFlops(
        num_layers=int(getattr(model_config, "num_hidden_layers", 32)),
        hidden_size=int(getattr(model_config, "hidden_size", 4096)),
        intermediate_size=int(getattr(model_config, "intermediate_size", 11008)),
        vocab_size=int(getattr(model_config, "vocab_size", 32064)),
    )


def make_vit_flops_default() -> VitFlops:
    """LLaVA-1.5-7B uses CLIP-ViT-L/14 at 336."""
    return VitFlops(num_layers=24, hidden_size=1024, intermediate_size=4096, image_size=336, patch_size=14)


def compute_n_image_keep(n_img: int, n_text: int, total_keep_ratio: float) -> int:
    total_tokens = n_text + n_img
    total_keep = int(math.ceil(total_keep_ratio * total_tokens))
    n_image_keep = total_keep - n_text
    return min(n_img, max(0, n_image_keep))


# ----------------------------------------------------------------------
# MileBench sampling
# ----------------------------------------------------------------------


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


def build_vicuna_prompt(question: str, conv_template: str = "vicuna_v1") -> str:
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def estimate_prompt_mm_len(input_ids: torch.Tensor, image_feature_len: int) -> tuple[int, int, int]:
    """Returns (prompt_mm_len, image_start, n_text) for a single-image prompt.

    `input_ids` should contain exactly one IMAGE_TOKEN_INDEX placeholder which
    the model expands to `image_feature_len` patch tokens during forward.
    """
    raw = input_ids[0].detach().cpu().tolist()
    placeholder_positions = [i for i, t in enumerate(raw) if int(t) == IMAGE_TOKEN_INDEX]
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected exactly one image placeholder, found {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    prompt_mm_len = len(raw) - 1 + image_feature_len
    n_text = prompt_mm_len - image_feature_len
    return prompt_mm_len, image_start, n_text


def iter_candidate_records(
    *,
    milebench_root: Path,
    datasets: list[str],
    tokenizer: Any,
    image_feature_len: int,
    conv_template: str,
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
            # Use only the first image for full-cache LLaVA-1.5 (single image).
            image_paths = image_paths[:1]

            question = build_milebench_question(record, meta, dataset_name)
            prompt = build_vicuna_prompt(question, conv_template)
            try:
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                prompt_mm_len, _, _ = estimate_prompt_mm_len(input_ids, image_feature_len)
            except Exception:
                continue
            if prompt_mm_len > max_prompt_mm_tokens:
                continue
            candidates.append(
                PooledSample(
                    dataset=dataset_name,
                    sample_id=sanitize_sample_id(record.get("sample_id"), idx),
                    question=question,
                    image_paths=tuple(image_paths),
                    answer=resolve_answer(record),
                    raw_prompt_tokens=int(input_ids.shape[1]),
                    prompt_mm_tokens=prompt_mm_len,
                )
            )
    return candidates


def sample_manifest(*, args: argparse.Namespace, tokenizer: Any, image_feature_len: int) -> list[PooledSample]:
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
        tokenizer=tokenizer,
        image_feature_len=image_feature_len,
        conv_template=args.conv_template,
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
            "conv_template": args.conv_template,
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


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------
# Per-sample measurement (full-cache run + analytical FLOPS for all configs)
# ----------------------------------------------------------------------


@torch.no_grad()
def run_full_cache_one(
    *,
    sample: PooledSample,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    image_feature_len: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with Image.open(sample.image_paths[0]) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    prompt = build_vicuna_prompt(sample.question, args.conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    prompt_mm_len, image_start, n_text = estimate_prompt_mm_len(input_ids, image_feature_len)

    empty_cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_attentions=False,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    end_to_end_ms = (time.perf_counter() - t0) * 1000.0

    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / GIB

    sequences = out.sequences
    seq_len = int(sequences.shape[1])
    input_len = int(input_ids.shape[1])
    # LLaVA-OneVision generate may return either (a) input_ids ++ generated or
    # (b) generated-only. Detect by length comparison.
    n_generated = seq_len - input_len if seq_len > input_len else seq_len
    n_generated = max(0, n_generated)
    if n_generated > 0:
        decoded = tokenizer.decode(sequences[0, -n_generated:].tolist(), skip_special_tokens=True).strip()
    else:
        decoded = ""

    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        "image_path": sample.image_paths[0],
        "prompt_mm_len": int(prompt_mm_len),
        "n_text": int(n_text),
        "n_img": int(image_feature_len),
        "image_start": int(image_start),
        "t_decode": int(n_generated),
        "end_to_end_ms": float(end_to_end_ms),
        "peak_gpu_memory_gib": float(peak_alloc_gib),
        "peak_gpu_reserved_gib": float(peak_reserved_gib),
        "prediction": decoded,
    }


# ----------------------------------------------------------------------
# Per-sample FLOPS rows (one row per (sample, method))
# ----------------------------------------------------------------------


def build_rows_from_run(
    *,
    run: dict[str, Any],
    method_specs: list[tuple[str, float]],
    llama_flops: LlamaFlops,
    vit_flops_per_image: int,
    n_images: int,
) -> list[dict[str, Any]]:
    L_p = int(run["prompt_mm_len"])
    n_text = int(run["n_text"])
    n_img = int(run["n_img"])
    T = int(run["t_decode"])

    full_decode_total = llama_flops.decode_total(L_p, T)
    rows: list[dict[str, Any]] = []
    for method, ratio in method_specs:
        n_image_keep = compute_n_image_keep(n_img, n_text, ratio) if ratio < 1.0 else n_img
        retained_prompt_len = n_text + n_image_keep
        prefill_flops = llama_flops.prefill(L_p)
        decode_flops = llama_flops.decode_total(L_p, T, retained_prompt_len=retained_prompt_len)
        vision_flops = vit_flops_per_image * n_images
        total_llm = prefill_flops + decode_flops
        total_all = total_llm + vision_flops
        decode_pct = (decode_flops / full_decode_total * 100.0) if full_decode_total > 0 else 100.0
        end_to_end_ms = float(run["end_to_end_ms"]) if ratio >= 1.0 else None
        peak_gib = float(run["peak_gpu_memory_gib"]) if ratio >= 1.0 else None
        kv_cache_full_gib = (2 * llama_flops.num_layers * llama_flops.hidden_size * L_p * 2) / GIB  # fp16
        kv_cache_kept_gib = (2 * llama_flops.num_layers * llama_flops.hidden_size * retained_prompt_len * 2) / GIB
        rows.append(
            {
                "method": method,
                "ratio": ratio,
                "dataset": run["dataset"],
                "sample_id": run["sample_id"],
                "sample_key": run["sample_key"],
                "prompt_mm_len": L_p,
                "n_text": n_text,
                "n_img": n_img,
                "n_image_keep": int(n_image_keep),
                "retained_prompt_len": int(retained_prompt_len),
                "t_decode": T,
                "prefill_flops": int(prefill_flops),
                "decode_flops": int(decode_flops),
                "vision_flops": int(vision_flops),
                "total_llm_flops": int(total_llm),
                "total_flops": int(total_all),
                "prefill_tflops": prefill_flops / TFLOPS,
                "decode_tflops": decode_flops / TFLOPS,
                "vision_tflops": vision_flops / TFLOPS,
                "total_llm_tflops": total_llm / TFLOPS,
                "total_tflops": total_all / TFLOPS,
                "decode_flops_pct_of_full": float(decode_pct),
                "kv_cache_full_gib": float(kv_cache_full_gib),
                "kv_cache_kept_gib": float(kv_cache_kept_gib),
                "kv_cache_pct_of_full": float(retained_prompt_len / max(1, L_p) * 100.0),
                "end_to_end_ms": end_to_end_ms,
                "peak_gpu_memory_gib": peak_gib,
                "prediction": run.get("prediction") if ratio >= 1.0 else None,
            }
        )
    return rows


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "prompt_mm_len",
        "n_text",
        "n_img",
        "n_image_keep",
        "retained_prompt_len",
        "t_decode",
        "prefill_flops",
        "decode_flops",
        "vision_flops",
        "total_llm_flops",
        "total_flops",
        "prefill_tflops",
        "decode_tflops",
        "vision_tflops",
        "total_llm_tflops",
        "total_tflops",
        "decode_flops_pct_of_full",
        "kv_cache_full_gib",
        "kv_cache_kept_gib",
        "kv_cache_pct_of_full",
        "end_to_end_ms",
        "peak_gpu_memory_gib",
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
                ratios_decode = [row["decode_flops"] / base["decode_flops"] for row, base in paired if base["decode_flops"] > 0]
                ratios_llm = [row["total_llm_flops"] / base["total_llm_flops"] for row, base in paired if base["total_llm_flops"] > 0]
                ratios_all = [row["total_flops"] / base["total_flops"] for row, base in paired if base["total_flops"] > 0]
                ratios_kv = [row["retained_prompt_len"] / base["prompt_mm_len"] for row, base in paired if base["prompt_mm_len"] > 0]
                item["decode_flops_pct_of_full"] = float(100.0 * mean(ratios_decode)) if ratios_decode else None
                item["total_llm_flops_pct_of_full"] = float(100.0 * mean(ratios_llm)) if ratios_llm else None
                item["total_flops_pct_of_full"] = float(100.0 * mean(ratios_all)) if ratios_all else None
                item["kv_cache_pct_of_full_paired"] = float(100.0 * mean(ratios_kv)) if ratios_kv else None
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


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original LLaVA-1.5-7B inference compute bench (FLOPS).")
    parser.add_argument("--milebench-root", type=str, default="data/MileBench")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/workspace/zap/ckpts/llava-v1.5-7b")
    parser.add_argument("--model-name", type=str, default="llava-v1.5-7b")
    parser.add_argument("--conv-template", type=str, default="vicuna_v1")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-mm-tokens", type=int, default=3900)
    parser.add_argument("--combined-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU latency/memory measurement")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"[load] model={args.model_path} model_name={args.model_name} "
        f"device_map={args.device_map} attn={args.attn_implementation}",
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
    vision_tower = model.get_vision_tower()
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"image_feature_len={image_feature_len}",
        flush=True,
    )

    llama_flops = make_llama_flops(model.config)
    vit_flops = make_vit_flops_default()
    print(
        f"[flops] LLM N={llama_flops.num_layers} D={llama_flops.hidden_size} "
        f"I={llama_flops.intermediate_size} V={llama_flops.vocab_size} | "
        f"ViT N={vit_flops.num_layers} D={vit_flops.hidden_size} I={vit_flops.intermediate_size} "
        f"img={vit_flops.image_size} patch={vit_flops.patch_size} | "
        f"vision_tflops_per_image={vit_flops.total() / TFLOPS:.4f}",
        flush=True,
    )

    samples = sample_manifest(args=args, tokenizer=tokenizer, image_feature_len=image_feature_len)
    manifest_path = out_dir / "sample_manifest.json"
    save_manifest(manifest_path, samples, args)

    ratios = sorted(set(float(r) for r in args.ratios), reverse=True)
    method_specs: list[tuple[str, float]] = []
    seen_full = False
    for ratio in ratios:
        if math.isclose(ratio, 1.0):
            method_specs.append(("full_cache_100", 1.0))
            seen_full = True
        else:
            method_specs.append((f"student_keep_{int(round(ratio * 100)):03d}", ratio))
    if not seen_full:
        method_specs.insert(0, ("full_cache_100", 1.0))

    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_model_path": str(Path(args.model_path).resolve()),
            "manifest_path": str(manifest_path),
            "n_samples": len(samples),
            "image_feature_len": image_feature_len,
            "llama_flops": llama_flops.__dict__,
            "vit_flops": vit_flops.__dict__,
            "vision_tflops_per_image": vit_flops.total() / TFLOPS,
            "method_specs": method_specs,
            "note": (
                "FLOPS for keep_050/keep_020 are analytical and assume the "
                "VisualUtilityStudentPress design: text KV is preserved, only "
                "image KV is evicted, retained_prompt_len = n_text + n_image_keep "
                "with n_image_keep = ceil(ratio*L_p) - n_text. End-to-end latency "
                "and peak memory are reported only for full_cache (actual runs)."
            ),
        }
    )
    write_json(out_dir / "run_config.json", run_config)

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if args.warmup_samples > 0:
        for sample in samples[: args.warmup_samples]:
            try:
                _ = run_full_cache_one(
                    sample=sample,
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image_feature_len=image_feature_len,
                    device=device,
                    args=args,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warmup-skip] {sample.sample_key}: {exc}", flush=True)
            empty_cuda()

    for sample in tqdm(samples, desc="Measuring full_cache"):
        try:
            run = run_full_cache_one(
                sample=sample,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_feature_len=image_feature_len,
                device=device,
                args=args,
            )
            runs.append(run)
        except torch.cuda.OutOfMemoryError as exc:
            failures.append({"sample_key": sample.sample_key, "error": f"CUDA OOM: {exc}"})
            empty_cuda()
            if not args.continue_on_error:
                raise
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_key": sample.sample_key, "error": repr(exc)})
            empty_cuda()
            if not args.continue_on_error:
                raise

    rows: list[dict[str, Any]] = []
    vit_flops_per_image = vit_flops.total()
    for run in runs:
        rows.extend(
            build_rows_from_run(
                run=run,
                method_specs=method_specs,
                llama_flops=llama_flops,
                vit_flops_per_image=vit_flops_per_image,
                n_images=1,
            )
        )

    write_csv(out_dir / "per_sample.csv", rows)
    write_json(out_dir / "per_sample_runs.json", runs)

    summary_rows = summarize(rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_json(out_dir / "summary.json", summary_rows)
    write_json(out_dir / "failures.json", failures)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
