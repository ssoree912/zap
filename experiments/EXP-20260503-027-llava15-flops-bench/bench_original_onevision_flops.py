#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaVA-OneVision-Qwen2-7B (original repo format) inference compute bench.

Loads `LlavaQwenForCausalLM` (Qwen2-7B + SigLIP-SO400m@384, AnyRes) through
VFlowOpt's `llava.model.builder.load_pretrained_model`, runs `generate()` on
N random MileBench samples to capture the actual prompt length and number of
decoded tokens, then reports analytical FLOPS for full_cache vs student
keep_050 / keep_020.

Differences from the LLaVA-1.5 bench:
- Qwen2-7B uses GQA (4 KV heads vs 28 query heads), so Q/K/V projection FLOPS
  are reduced.
- AnyRes image tokenization → image_feature_len varies per sample.
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


def patch_siglip_loader(local_siglip_path: str) -> None:
    """The vendored LLaVA-OneVision hardcodes /mnt/petrelfs/.../siglip-so400m-...

    Patch ``SigLipVisionTower.load_model`` to load from a local directory
    we control.
    """
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel.from_pretrained(
            local_siglip_path, device_map=device_map
        )
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


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

    @property
    def sample_key(self) -> str:
        return f"{self.dataset}:{self.sample_id}"


# ----------------------------------------------------------------------
# FLOPS calculators (Qwen2-7B with GQA + SigLIP vision tower)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen2Flops:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def proj_factor(self) -> float:
        # Sum of in_dim*out_dim across Q, K, V, O projections, normalized by D^2.
        # Q: D*D, K: D*(K_h*Dh) = D*D*K/H, V: D*D*K/H, O: D*D
        # FLOPS (×2 for MAC->FLOP): 2*(D^2 + 2*D^2*K/H + D^2) = 4*D^2*(1 + K/H)
        return 4.0 * (1.0 + self.num_key_value_heads / self.num_attention_heads)

    def per_layer_decode(self, cache_len: int) -> int:
        D, I = self.hidden_size, self.intermediate_size
        proj = int(self.proj_factor * D * D)
        attn = 4 * D * cache_len  # Q@K^T + A@V; sum over H query heads, head_dim cancels
        ffn = 6 * D * I            # SwiGLU (gate, up, down)
        return proj + attn + ffn

    def decode_step(self, cache_len: int) -> int:
        layer = self.per_layer_decode(cache_len)
        lm_head = 2 * self.hidden_size * self.vocab_size
        return self.num_layers * layer + lm_head

    def prefill(self, T: int) -> int:
        D, I, V = self.hidden_size, self.intermediate_size, self.vocab_size
        proj = int(self.proj_factor * D * D * T)
        attn = 2 * D * T * (T + 1)
        ffn = 6 * D * I * T
        layer = proj + attn + ffn
        lm_head = 2 * D * V * T
        return self.num_layers * layer + lm_head

    def decode_total(self, prompt_len: int, t_decode: int, retained_prompt_len: Optional[int] = None) -> int:
        base = prompt_len if retained_prompt_len is None else retained_prompt_len
        return sum(self.decode_step(base + t) for t in range(1, t_decode + 1))


@dataclass(frozen=True)
class SiglipFlops:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    image_size: int
    patch_size: int

    def patches_per_image(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def total_per_crop(self) -> int:
        Dv, Iv = self.hidden_size, self.intermediate_size
        P = self.patches_per_image()
        patch_embed = 2 * 3 * (self.patch_size ** 2) * Dv * P
        proj = 8 * Dv * Dv * P
        attn = 2 * Dv * P * P
        ffn = 4 * Dv * Iv * P
        layer = proj + attn + ffn
        return patch_embed + self.num_layers * layer


def make_qwen2_flops(model_config: Any) -> Qwen2Flops:
    return Qwen2Flops(
        num_layers=int(getattr(model_config, "num_hidden_layers", 28)),
        hidden_size=int(getattr(model_config, "hidden_size", 3584)),
        intermediate_size=int(getattr(model_config, "intermediate_size", 18944)),
        num_attention_heads=int(getattr(model_config, "num_attention_heads", 28)),
        num_key_value_heads=int(getattr(model_config, "num_key_value_heads", 4)),
        vocab_size=int(getattr(model_config, "vocab_size", 152064)),
    )


def make_siglip_flops_default() -> SiglipFlops:
    """LLaVA-OneVision uses SigLIP-SO400M-patch14-384."""
    return SiglipFlops(num_layers=27, hidden_size=1152, intermediate_size=4304, image_size=384, patch_size=14)


def compute_n_image_keep(n_img: int, n_text: int, total_keep_ratio: float) -> int:
    total_tokens = n_text + n_img
    total_keep = int(math.ceil(total_keep_ratio * total_tokens))
    n_image_keep = total_keep - n_text
    return min(n_img, max(0, n_image_keep))


# ----------------------------------------------------------------------
# MileBench sampling helpers (largely reused, parameterized by tokenizer)
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


def build_qwen_prompt(question: str, conv_template: str = "qwen_1_5") -> str:
    import copy
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"<image>\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def iter_candidate_records(
    *,
    milebench_root: Path,
    datasets: list[str],
    tokenizer: Any,
    conv_template: str,
    max_raw_prompt_tokens: int,
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
            image_paths = image_paths[:1]

            question = build_milebench_question(record, meta, dataset_name)
            prompt = build_qwen_prompt(question, conv_template)
            try:
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                raw_len = int(input_ids.shape[1])
            except Exception:
                continue
            if raw_len > max_raw_prompt_tokens:
                continue
            candidates.append(
                PooledSample(
                    dataset=dataset_name,
                    sample_id=sanitize_sample_id(record.get("sample_id"), idx),
                    question=question,
                    image_paths=tuple(image_paths),
                    answer=resolve_answer(record),
                    raw_prompt_tokens=raw_len,
                )
            )
    return candidates


def sample_manifest(*, args: argparse.Namespace, tokenizer: Any) -> list[PooledSample]:
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
                )
            )
        return samples

    root = Path(args.milebench_root).resolve()
    datasets = normalize_dataset_list(root, args.datasets)
    candidates = iter_candidate_records(
        milebench_root=root,
        datasets=datasets,
        tokenizer=tokenizer,
        conv_template=args.conv_template,
        max_raw_prompt_tokens=args.max_raw_prompt_tokens,
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
            "max_raw_prompt_tokens": args.max_raw_prompt_tokens,
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
# Per-sample full-cache run, with image_feature_len capture
# ----------------------------------------------------------------------


@torch.no_grad()
def run_full_cache_one(
    *,
    sample: PooledSample,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with Image.open(sample.image_paths[0]) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        # AnyRes returns a list of crops
        image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]
        n_crops = sum(t.shape[0] if t.dim() == 4 else 1 for t in image_tensor)
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)
        n_crops = image_tensor.shape[0] if image_tensor.dim() == 4 else 1

    prompt = build_qwen_prompt(sample.question, args.conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

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
    n_generated = seq_len - input_len if seq_len > input_len else seq_len
    n_generated = max(0, n_generated)
    if n_generated > 0:
        decoded = tokenizer.decode(sequences[0, -n_generated:].tolist(), skip_special_tokens=True).strip()
    else:
        decoded = ""

    # Recover image_feature_len by running the multimodal preparation once.
    # The attention shape is not exposed (we ran with output_attentions=False
    # for speed), so we use prepare_inputs_labels_for_multimodal directly.
    raw_ids = input_ids[0].detach().cpu().tolist()
    placeholder_positions = [i for i, t in enumerate(raw_ids) if int(t) == IMAGE_TOKEN_INDEX]
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected 1 IMAGE_TOKEN_INDEX, got {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    # Use prepare_inputs_labels_for_multimodal to get the expanded prompt_len_mm.
    with torch.no_grad():
        (
            _new_input_ids,
            _,
            _,
            _,
            new_input_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            images=image_tensor,
            modalities=["image"],
            image_sizes=[image_size],
        )
    prompt_mm_len = int(new_input_embeds.shape[1])
    image_feature_len = int(prompt_mm_len - input_len + 1)
    n_text = prompt_mm_len - image_feature_len

    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        "image_path": sample.image_paths[0],
        "image_pixel_size": list(image_size),
        "n_crops": int(n_crops),
        "raw_prompt_tokens": int(input_len),
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


def build_rows_from_run(
    *,
    run: dict[str, Any],
    method_specs: list[tuple[str, float]],
    qwen_flops: Qwen2Flops,
    siglip_per_crop: int,
    n_crops: int,
) -> list[dict[str, Any]]:
    L_p = int(run["prompt_mm_len"])
    n_text = int(run["n_text"])
    n_img = int(run["n_img"])
    T = int(run["t_decode"])

    full_decode_total = qwen_flops.decode_total(L_p, T)
    rows: list[dict[str, Any]] = []
    for method, ratio in method_specs:
        n_image_keep = compute_n_image_keep(n_img, n_text, ratio) if ratio < 1.0 else n_img
        retained_prompt_len = n_text + n_image_keep
        prefill_flops = qwen_flops.prefill(L_p)
        decode_flops = qwen_flops.decode_total(L_p, T, retained_prompt_len=retained_prompt_len)
        vision_flops = siglip_per_crop * n_crops
        total_llm = prefill_flops + decode_flops
        total_all = total_llm + vision_flops
        decode_pct = (decode_flops / full_decode_total * 100.0) if full_decode_total > 0 else 100.0
        end_to_end_ms = float(run["end_to_end_ms"]) if ratio >= 1.0 else None
        peak_gib = float(run["peak_gpu_memory_gib"]) if ratio >= 1.0 else None
        # Qwen2-7B GQA: KV cache per token = 2 * num_kv_heads * head_dim * num_layers * 2 bytes (fp16)
        kv_per_token = 2 * qwen_flops.num_key_value_heads * qwen_flops.head_dim * qwen_flops.num_layers * 2
        kv_cache_full_gib = kv_per_token * L_p / GIB
        kv_cache_kept_gib = kv_per_token * retained_prompt_len / GIB
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
                "n_crops": n_crops,
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "prompt_mm_len", "n_text", "n_img", "n_crops", "n_image_keep", "retained_prompt_len", "t_decode",
        "prefill_flops", "decode_flops", "vision_flops", "total_llm_flops", "total_flops",
        "prefill_tflops", "decode_tflops", "vision_tflops", "total_llm_tflops", "total_tflops",
        "decode_flops_pct_of_full", "kv_cache_full_gib", "kv_cache_kept_gib", "kv_cache_pct_of_full",
        "end_to_end_ms", "peak_gpu_memory_gib",
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
                d_ratios = [row["decode_flops"] / base["decode_flops"] for row, base in paired if base["decode_flops"] > 0]
                llm_ratios = [row["total_llm_flops"] / base["total_llm_flops"] for row, base in paired if base["total_llm_flops"] > 0]
                tot_ratios = [row["total_flops"] / base["total_flops"] for row, base in paired if base["total_flops"] > 0]
                kv_ratios = [row["retained_prompt_len"] / base["prompt_mm_len"] for row, base in paired if base["prompt_mm_len"] > 0]
                item["decode_flops_pct_of_full"] = float(100.0 * mean(d_ratios)) if d_ratios else None
                item["total_llm_flops_pct_of_full"] = float(100.0 * mean(llm_ratios)) if llm_ratios else None
                item["total_flops_pct_of_full"] = float(100.0 * mean(tot_ratios)) if tot_ratios else None
                item["kv_cache_pct_of_full_paired"] = float(100.0 * mean(kv_ratios)) if kv_ratios else None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA-OneVision-Qwen2-7B inference compute bench (FLOPS).")
    parser.add_argument("--milebench-root", type=str, default="data/MileBench")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-name", type=str, default="llava_qwen")
    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-raw-prompt-tokens", type=int, default=2000,
                        help="Skip pre-expansion prompts longer than this many tokens.")
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

    patch_siglip_loader("/workspace/zap/ckpts/siglip-so400m-patch14-384")

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
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"hidden={model.config.hidden_size} n_kv={model.config.num_key_value_heads}",
        flush=True,
    )

    qwen_flops = make_qwen2_flops(model.config)
    siglip = make_siglip_flops_default()
    print(
        f"[flops] LLM N={qwen_flops.num_layers} D={qwen_flops.hidden_size} I={qwen_flops.intermediate_size} "
        f"H={qwen_flops.num_attention_heads} K={qwen_flops.num_key_value_heads} V={qwen_flops.vocab_size} "
        f"proj_factor={qwen_flops.proj_factor:.3f} | "
        f"SigLIP N={siglip.num_layers} D={siglip.hidden_size} I={siglip.intermediate_size} "
        f"img={siglip.image_size} patch={siglip.patch_size} | "
        f"vision_tflops_per_crop={siglip.total_per_crop() / TFLOPS:.4f}",
        flush=True,
    )

    samples = sample_manifest(args=args, tokenizer=tokenizer)
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
            "qwen_flops": qwen_flops.__dict__,
            "siglip": siglip.__dict__,
            "vision_tflops_per_crop": siglip.total_per_crop() / TFLOPS,
            "method_specs": method_specs,
            "note": (
                "FLOPS for keep_050/keep_020 are analytical and assume the "
                "VisualUtilityStudentOneVision design: text KV is preserved, only "
                "image KV is evicted. End-to-end latency / peak memory are reported "
                "only for full_cache (actual runs)."
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
                    device=device,
                    args=args,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warmup-skip] {sample.sample_key}: {exc}", flush=True)
            empty_cuda()

    for sample in tqdm(samples, desc="Measuring full_cache (OneVision)"):
        try:
            run = run_full_cache_one(
                sample=sample,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
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
    siglip_per_crop = siglip.total_per_crop()
    for run in runs:
        rows.extend(
            build_rows_from_run(
                run=run,
                method_specs=method_specs,
                qwen_flops=qwen_flops,
                siglip_per_crop=siglip_per_crop,
                n_crops=int(run["n_crops"]),
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
