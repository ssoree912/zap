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
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


REPO_ROOT = Path(__file__).resolve().parents[1]
LOOKM_CANDIDATES = [REPO_ROOT.parent / "LOOK-M", REPO_ROOT.parent / "look-m"]
LOOKM_ROOT = next((path for path in LOOKM_CANDIDATES if path.is_dir()), LOOKM_CANDIDATES[0])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOOKM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOKM_ROOT))

from kvzap.image_teacher_utils import build_prompt, load_pt_record, load_vlm_samples, resolve_teacher_dir  # noqa: E402
from utils import MileBenchDataset, get_worker_class  # noqa: E402


DEFAULT_DATASETS = ["DocVQA", "Spot-the-Diff", "CLEVR-Change", "IEdit"]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{image#\d+\}")


def _lazy_import_zap_runtime():
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from kvpress.presses.image_token_press import H2OImageOnlyPress, OracleAllTokenPress, OracleImageTeacherPress, ProbeImageTeacherPress
    from kvzap.llava_extractor import (
        _get_model_device,
        _get_model_float_dtype,
        _move_batch_to_device,
        configure_llava_processor,
        infer_llava_image_positions_no_forward,
    )

    runtime = {
        "AutoProcessor": AutoProcessor,
        "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
        "H2OImageOnlyPress": H2OImageOnlyPress,
        "OracleAllTokenPress": OracleAllTokenPress,
        "OracleImageTeacherPress": OracleImageTeacherPress,
        "ProbeImageTeacherPress": ProbeImageTeacherPress,
        "_get_model_device": _get_model_device,
        "_get_model_float_dtype": _get_model_float_dtype,
        "_move_batch_to_device": _move_batch_to_device,
        "configure_llava_processor": configure_llava_processor,
        "infer_llava_image_positions_no_forward": infer_llava_image_positions_no_forward,
    }
    globals().update(runtime)
    return runtime


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


def load_core_annotation(dataset_path: str) -> Optional[dict[str, Any]]:
    path = Path(dataset_path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
    except Exception:
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
        except Exception:
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

DEFAULT_TEACHER_DIRS = {
    "DocVQA": "/workspace/hd/artifacts/oracle/llava_docvqa_full_multi_teacher4",
    "Spot-the-Diff": "/workspace/hd/artifacts/oracle/llava_spot_the_diff_full_multi_teacher4",
    "CLEVR-Change": "/workspace/hd/artifacts/oracle/llava_clevr_change_full_multi_teacher4",
    "IEdit": "/workspace/hd/artifacts/oracle/llava_iedit_full_multi_teacher4",
    "SlideVQA": "/workspace/hd/artifacts/oracle/llava_slidevqa_full_multi_teacher4",
    "OCR-VQA": "/workspace/hd/artifacts/oracle/llava_ocr_vqa_full_multi_teacher4",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_seq_length(past_key_values: Any) -> Optional[int]:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except TypeError:
            return int(past_key_values.get_seq_length(0))
    if isinstance(past_key_values, (list, tuple)) and past_key_values:
        layer0 = past_key_values[0]
        if isinstance(layer0, (list, tuple)) and layer0:
            key = layer0[0]
            if isinstance(key, torch.Tensor):
                return int(key.shape[-2])
    return None


class ForwardTrace:
    def __init__(self, model: Any, device: torch.device):
        self.model = model
        self.device = device
        self._handles = []
        self._start_time = None
        self.prefill_duration_ms: Optional[float] = None
        # first_decode_ms: time for the very first decode step (prefill→first token).
        # TTFT = prefill_duration_ms + first_decode_ms
        self.first_decode_ms: Optional[float] = None
        # decode_total_ms / decode_call_count includes ALL decode steps.
        # Steady-state TBT = (decode_total_ms - first_decode_ms) / (decode_call_count - 1)
        self.decode_total_ms: float = 0.0
        self.decode_call_count: int = 0
        self.prefill_full_seq_len: Optional[int] = None
        self.prefill_cache_seq_len: Optional[int] = None
        self.final_cache_seq_len: Optional[int] = None

    def _pre_hook(self, module, args, kwargs):
        _sync(self.device)
        self._start_time = time.perf_counter()

    def _post_hook(self, module, args, kwargs, output):
        _sync(self.device)
        start_time = self._start_time if self._start_time is not None else time.perf_counter()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logits = getattr(output, "logits", None)
        cache_len = _cache_seq_length(getattr(output, "past_key_values", None))
        if self.prefill_duration_ms is None:
            self.prefill_duration_ms = elapsed_ms
            self.prefill_full_seq_len = None if logits is None else int(logits.shape[1])
            self.prefill_cache_seq_len = cache_len
        else:
            if self.first_decode_ms is None:
                self.first_decode_ms = elapsed_ms
            self.decode_total_ms += elapsed_ms
            self.decode_call_count += 1
        if cache_len is not None:
            self.final_cache_seq_len = cache_len
        return output

    def __enter__(self):
        self._handles.append(self.model.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        self._handles.append(self.model.register_forward_hook(self._post_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


@dataclass
class PooledSample:
    dataset_name: str
    sample_id: str
    question_raw: str
    question_look: str
    image_paths: list[str]
    raw: dict[str, Any]
    teacher_dir: Optional[Path]

    @property
    def sample_key(self) -> str:
        return f"{self.dataset_name}:{self.sample_id}"


def _dataset_json_path(milebench_root: Path, dataset_name: str) -> Path:
    return milebench_root / dataset_name / f"{dataset_name}.json"


def _dataset_image_root(milebench_root: Path, dataset_name: str) -> Path:
    return milebench_root / dataset_name / "images"


def collect_pooled_samples(
    milebench_root: Path,
    dataset_names: list[str],
    teacher_dir_map: Optional[dict[str, str]] = None,
) -> list[PooledSample]:
    pooled: list[PooledSample] = []
    for dataset_name in dataset_names:
        dataset_path = _dataset_json_path(milebench_root, dataset_name)
        image_root = _dataset_image_root(milebench_root, dataset_name)
        core_annotation = load_core_annotation(str(dataset_path))
        if core_annotation is None:
            raise ValueError(f"Failed to load MileBench core annotation from {dataset_path}")
        teacher_dir = None
        if teacher_dir_map is not None and dataset_name in teacher_dir_map:
            teacher_dir = resolve_teacher_dir(teacher_dir_map[dataset_name])
        samples = load_vlm_samples(
            dataset_path=str(dataset_path),
            image_root=str(image_root),
            image_column="images_path",
        )
        for sample in samples:
            raw = sample["raw"]
            question_raw = build_look_question(raw, core_annotation, dataset_name=dataset_name)
            question_look = replace_image_placeholders_for_export(question_raw)
            pooled.append(
                PooledSample(
                    dataset_name=dataset_name,
                    sample_id=sample["sample_id"],
                    question_raw=question_raw,
                    question_look=question_look,
                    image_paths=sample["image_paths"],
                    raw=raw,
                    teacher_dir=teacher_dir,
                )
            )
    return pooled


def select_random_samples(pool: list[PooledSample], sample_size: int, seed: int) -> list[PooledSample]:
    rng = random.Random(seed)
    if sample_size >= len(pool):
        picked = list(pool)
    else:
        picked = rng.sample(pool, sample_size)
    picked.sort(key=lambda sample: (sample.dataset_name, sample.sample_id))
    return picked


def select_samples_from_manifest(pool: list[PooledSample], manifest_path: Path) -> list[PooledSample]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_key = {sample.sample_key: sample for sample in pool}
    picked: list[PooledSample] = []
    missing: list[str] = []
    for row in manifest:
        sample_key = row.get("sample_key") or f"{row['dataset']}:{row['sample_id']}"
        sample = by_key.get(sample_key)
        if sample is None:
            missing.append(sample_key)
            continue
        picked.append(sample)
    if missing:
        raise ValueError(f"Missing samples from manifest: {missing[:5]}")
    picked.sort(key=lambda sample: (sample.dataset_name, sample.sample_id))
    return picked


def _resolve_language_config(model_config: Any) -> Any:
    if hasattr(model_config, "text_config") and model_config.text_config is not None:
        return model_config.text_config
    if hasattr(model_config, "language_config") and model_config.language_config is not None:
        return model_config.language_config
    return model_config


def _theoretical_kv_cache_gib(model_config: Any, seq_len: int, dtype: torch.dtype) -> float:
    """Compute theoretical KV cache size in GiB for a given retained sequence length.

    Formula: n_layers * 2 (K+V) * n_kv_heads * head_dim * seq_len * bytes_per_element

    This is the "true" cache footprint, comparable across methods regardless of
    activation/fragmentation overhead captured by peak_gpu_memory_gib.
    """
    language_config = _resolve_language_config(model_config)
    n_layers = int(language_config.num_hidden_layers)
    n_heads = int(language_config.num_attention_heads)
    n_kv_heads = int(getattr(language_config, "num_key_value_heads", n_heads))
    hidden_size = int(language_config.hidden_size)
    head_dim = hidden_size // n_heads
    bytes_per_element = torch.finfo(dtype).bits // 8
    total_bytes = n_layers * 2 * n_kv_heads * head_dim * seq_len * bytes_per_element
    return total_bytes / (1024**3)


def _compute_image_retention(
    image_positions: torch.Tensor,
    keep_ratio: float,
    *,
    total_keep_ratio: Optional[float] = None,
    prompt_full_seq_len: Optional[int] = None,
) -> tuple[int, int, float]:
    """Return (n_image_total, n_image_kept, r_img).

    If total_keep_ratio is given (unified basis), n_image_kept is derived so that
    the total retained tokens = ceil(total_keep_ratio * prompt_full_seq_len).
    Otherwise keep_ratio is applied directly to image token count (legacy).
    """
    total = int(image_positions.numel())
    if total == 0:
        return 0, 0, 1.0

    if total_keep_ratio is not None and prompt_full_seq_len is not None:
        n_text = max(0, prompt_full_seq_len - total)
        total_keep = int(math.ceil(total_keep_ratio * prompt_full_seq_len))
        kept = min(total, max(0, total_keep - n_text))
    else:
        kept = int(math.ceil(total * keep_ratio))
        kept = min(total, max(kept, 0))

    return total, kept, kept / total if total > 0 else 1.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_teacher_cache(samples: list[PooledSample]) -> dict[str, dict[str, Any]]:
    teacher_cache: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if sample.teacher_dir is None:
            continue
        teacher_path = sample.teacher_dir / f"{sample.sample_id}.pt"
        teacher_cache[sample.sample_key] = load_pt_record(teacher_path)
    return teacher_cache


def _prepare_lookm_inputs(
    samples: list[PooledSample],
    *,
    milebench_root: Path,
    worker: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[PooledSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.dataset_name, []).append(sample)

    prepared: dict[str, dict[str, Any]] = {}
    for dataset_name, dataset_samples in grouped.items():
        dataset_path = _dataset_json_path(milebench_root, dataset_name)
        image_root = _dataset_image_root(milebench_root, dataset_name)
        core_annotation = load_core_annotation(str(dataset_path))
        if core_annotation is None:
            raise ValueError(f"Failed to load MileBench core annotation from {dataset_path}")
        lc_dataset = MileBenchDataset(
            annotation=[sample.raw for sample in dataset_samples],
            task_instructions=core_annotation["meta_data"]["task_instruction"],
            img_dir=str(image_root),
            max_context_len=worker.max_context_len,
            n_tokens_per_image=worker.n_tokens_per_image,
            tokenizer=worker.tokenizer,
            dataset_name=dataset_name,
            combine_image=getattr(worker, "combine_image", None),
        )
        for index, sample in enumerate(dataset_samples):
            item = lc_dataset[index]
            prepared[sample.sample_key] = {
                "question": item["context"],
                "image_paths": item["raw_img_list"],
            }
    return prepared


def measure_zap_method_for_sample(
    *,
    sample: PooledSample,
    method_name: str,
    model: Any,
    processor: Any,
    device: torch.device,
    float_dtype: torch.dtype,
    prompt_template: str,
    max_new_tokens: int,
    press: Optional[Any],
    teacher_record: Optional[dict[str, Any]],
    include_probe_position_inference: bool,
    image_keep_ratio: float,
    total_keep_ratio: Optional[float] = None,
    output_attentions: bool = False,
    prepared_input: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    # Use truncated input if provided (same truncation as LOOK-M)
    import re as _re
    if prepared_input is not None:
        image_paths = prepared_input["image_paths"]
        raw_q = prepared_input["question"]
    else:
        image_paths = sample.image_paths
        raw_q = sample.question_raw
    # Strip ALL image placeholder tokens (<ImageHere> and {image#N}) so that
    # build_prompt's fallback path prepends the correct number of <image> tokens.
    # This is safe for the efficiency script because we only measure latency, not accuracy.
    # It also prevents ValueError when placeholder count != image count (e.g. MMCoQA, TextNeedle).
    question_raw = _re.sub(r"<ImageHere>|\{image#\d+\}", "", raw_q).strip()
    num_images = len(image_paths)
    prompt_text = build_prompt(question_raw, prompt_template, image_count=num_images)
    if image_paths:
        images = open_images(image_paths)
        prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    else:
        # LOOK-M truncation can drop all images for very long contexts; run text-only.
        prompt_inputs = processor(text=prompt_text, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)

    _empty_cuda_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    setup_prefill_ms = 0.0
    image_positions = torch.empty(0, dtype=torch.long)
    image_positions_dropped: Optional[int] = None
    prompt_full_seq_len: Optional[int] = None
    if teacher_record is not None:
        if num_images > 0:
            _, prompt_full_seq_len = infer_llava_image_positions_no_forward(
                prompt_inputs=prompt_inputs,
                model_config=model.config,
                num_images=num_images,
            )
            image_positions = resolve_teacher_image_positions(teacher_record)
        press.set_sample_teacher(image_positions, teacher_record["att_only_postvision"])
    elif isinstance(press, ProbeImageTeacherPress):
        if num_images > 0:
            start = time.perf_counter()
            image_positions, prompt_full_seq_len = infer_llava_image_positions_no_forward(
                prompt_inputs=prompt_inputs,
                model_config=model.config,
                num_images=num_images,
            )
            setup_prefill_ms += (time.perf_counter() - start) * 1000.0
        press.set_image_positions(image_positions)
        press.reset_probe_timing()
    elif isinstance(press, (H2OImageOnlyPress,)):
        if num_images > 0:
            start = time.perf_counter()
            image_positions, prompt_full_seq_len = infer_llava_image_positions_no_forward(
                prompt_inputs=prompt_inputs,
                model_config=model.config,
                num_images=num_images,
            )
            setup_prefill_ms += (time.perf_counter() - start) * 1000.0
        press.set_image_positions(image_positions)

    generate_kwargs: dict = dict(do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    if output_attentions:
        generate_kwargs["output_attentions"] = True

    trace = ForwardTrace(model, device)
    press_ctx = press(model) if press is not None else nullcontext()
    try:
        with trace, press_ctx:
            with torch.no_grad():
                generated_ids = model.generate(**prompt_inputs, **generate_kwargs)
    finally:
        probe_forward_ms: Optional[float] = None
        if isinstance(press, ProbeImageTeacherPress):
            probe_forward_ms = press.probe_score_total_ms
        if press is not None:
            press.clear_sample_context()

    generated_tokens = max(int(generated_ids.shape[1]) - int(prompt_inputs["input_ids"].shape[1]), 0)

    # TTFT = prefill + first decode step (time until first output token ready)
    # TBT  = average of decode steps 2..N (steady-state inter-token latency)
    ttft_ms: Optional[float] = None
    tbt_ms_per_token: Optional[float] = None
    if trace.prefill_duration_ms is not None:
        ttft_ms = float(trace.prefill_duration_ms) + float(trace.first_decode_ms or 0.0)
    if trace.decode_call_count > 1 and trace.first_decode_ms is not None:
        steady_ms = trace.decode_total_ms - trace.first_decode_ms
        tbt_ms_per_token = steady_ms / (trace.decode_call_count - 1)
    elif trace.decode_call_count == 1:
        tbt_ms_per_token = trace.decode_total_ms  # only one step, use it as-is

    if press is None and teacher_record is None:
        if trace.prefill_cache_seq_len is not None:
            prompt_full_seq_len = int(trace.prefill_cache_seq_len)
        elif trace.final_cache_seq_len is not None:
            prompt_full_seq_len = max(int(trace.final_cache_seq_len) - generated_tokens, 0)
        else:
            raise ValueError("Failed to recover the full-cache prompt length without image position inference")

        prompt_retained_seq_len = prompt_full_seq_len
        final_cache_seq_len = int(trace.final_cache_seq_len) if trace.final_cache_seq_len is not None else (prompt_full_seq_len + generated_tokens)
        total_image_tokens = None
        kept_image_tokens = None
        r_img = 1.0
        full_final_cache_len = final_cache_seq_len
        r_eff_prompt = 1.0 if prompt_full_seq_len > 0 else None
        r_eff_decode = 1.0 if full_final_cache_len > 0 else None
    else:
        if prompt_full_seq_len is None:
            raise ValueError("Prompt multimodal length was not recovered for the pruned run")
        prompt_full_seq_len = int(prompt_full_seq_len)

        # Use the count of positions that actually fall within the processed prompt.
        # image_positions from infer_llava_image_positions_no_forward are within [0, prompt_full_seq_len)
        # by construction, but clamp here to match compress() behaviour exactly.
        image_positions_valid = image_positions[
            (image_positions >= 0) & (image_positions < prompt_full_seq_len)
        ]
        image_positions_dropped = int(image_positions.numel()) - int(image_positions_valid.numel())

        total_image_tokens, kept_image_tokens, r_img = _compute_image_retention(
            image_positions_valid, image_keep_ratio,
            total_keep_ratio=total_keep_ratio,
            prompt_full_seq_len=prompt_full_seq_len,
        )
        non_image_prompt_tokens = max(prompt_full_seq_len - total_image_tokens, 0)
        prompt_retained_seq_len = non_image_prompt_tokens + kept_image_tokens
        final_cache_seq_len = prompt_retained_seq_len + generated_tokens
        full_final_cache_len = prompt_full_seq_len + generated_tokens

        r_eff_prompt = (prompt_retained_seq_len / prompt_full_seq_len) if prompt_full_seq_len > 0 else None
        r_eff_decode = (final_cache_seq_len / full_final_cache_len) if full_final_cache_len > 0 else None

    peak_gpu_memory_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else None
    )
    kv_cache_gib = _theoretical_kv_cache_gib(model.config, final_cache_seq_len, float_dtype)

    return {
        "implementation": "zap_hf",
        "method": method_name,
        "dataset": sample.dataset_name,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        # ── latency ──────────────────────────────────────────────────────────
        "prefill_latency_ms": setup_prefill_ms + float(trace.prefill_duration_ms or 0.0),
        "prefill_setup_latency_ms": setup_prefill_ms if include_probe_position_inference else 0.0,
        "probe_forward_ms": probe_forward_ms,
        "ttft_ms": ttft_ms,
        "tbt_ms_per_token": tbt_ms_per_token,
        "decode_latency_ms_per_token": (
            trace.decode_total_ms / trace.decode_call_count if trace.decode_call_count > 0 else None
        ),
        # ── memory ───────────────────────────────────────────────────────────
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "kv_cache_gib": kv_cache_gib,
        # ── compression ratios ───────────────────────────────────────────────
        "r_img": r_img,
        "r_eff_prompt": r_eff_prompt,
        "r_eff_decode_t_end": r_eff_decode,
        # ── sequence lengths ─────────────────────────────────────────────────
        "prompt_full_seq_len": prompt_full_seq_len,
        "prompt_retained_seq_len": prompt_retained_seq_len,
        "final_cache_seq_len": final_cache_seq_len,
        "generated_tokens": generated_tokens,
        "decode_steps": trace.decode_call_count,
        # ── image token counts ───────────────────────────────────────────────
        "image_tokens_total": total_image_tokens,
        "image_tokens_kept": kept_image_tokens,
        "image_keep_ratio": image_keep_ratio,
        "image_positions_dropped": image_positions_dropped,
    }


def run_zap_measurements(args: argparse.Namespace, samples: list[PooledSample]) -> list[dict[str, Any]]:
    _lazy_import_zap_runtime()
    processor = AutoProcessor.from_pretrained(args.implementation_model_name)
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "device_map": None,
    }
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    model = LlavaForConditionalGeneration.from_pretrained(args.implementation_model_name, **model_kwargs)
    configure_llava_processor(processor, model.config)
    model = model.to(torch.device(args.device))
    model.eval()

    device = _get_model_device(model)
    float_dtype = _get_model_float_dtype(model)
    teacher_cache = _build_teacher_cache(samples) if args.include_oracle else {}

    oracle_press = None
    if args.include_oracle:
        oracle_press = OracleImageTeacherPress(
            total_keep_ratio=args.total_keep_ratio,
            head_reduce=args.head_reduce,
        )
    probe_press = ProbeImageTeacherPress(
        total_keep_ratio=args.total_keep_ratio,
        head_reduce=args.head_reduce,
        probe_model_name=args.probe_model_name,
    )
    probe_press.post_init_from_model(model)

    h2o_press = None
    if args.include_h2o_ablation:
        h2o_press = H2OImageOnlyPress(total_keep_ratio=args.total_keep_ratio, head_reduce=args.head_reduce)

    oracle_all_token_press = None
    if args.include_oracle_all_token:
        oracle_all_token_press = OracleAllTokenPress(total_keep_ratio=args.total_keep_ratio, head_reduce=args.head_reduce)

    # Prepare truncated inputs for all methods if requested (same truncation as LOOK-M)
    zap_prepared_inputs: dict[str, dict[str, Any]] = {}
    if getattr(args, "truncate_like_lookm", False):
        # Build a lightweight stub using ZAP's already-loaded processor.
        # Avoids loading the full LOOK-M LlavaLlamaForCausalLM (which requires
        # torch >= 2.6 due to CVE-2025-32434). _prepare_lookm_inputs only needs
        # tokenizer, max_context_len, n_tokens_per_image, and combine_image.
        class _TruncationStub:
            pass
        _stub = _TruncationStub()
        _stub.tokenizer = processor.tokenizer
        _stub.max_context_len = 4096
        _stub.n_tokens_per_image = 576
        _stub.combine_image = None
        zap_prepared_inputs = _prepare_lookm_inputs(
            samples,
            milebench_root=Path(args.milebench_root),
            worker=_stub,
        )

    rows: list[dict[str, Any]] = []
    # Warmup: try a quick full_cache pass to prime CUDA; skip on OOM
    try:
        _ = measure_zap_method_for_sample(
            sample=samples[0],
            method_name="warmup_full_cache",
            model=model,
            processor=processor,
            device=device,
            float_dtype=float_dtype,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            press=None,
            teacher_record=None,
            include_probe_position_inference=False,
            image_keep_ratio=1.0,
            prepared_input=None,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
        if "out of memory" not in str(_e).lower() and not isinstance(_e, torch.cuda.OutOfMemoryError):
            raise
        print(f"[OOM] warmup skipped (sample too large for full_cache): {_e}")
    _empty_cuda_cache()

    def _run_safe(method_name: str, **kwargs) -> Optional[dict]:
        """Run measurement, returning None on OOM and continuing."""
        try:
            return measure_zap_method_for_sample(method_name=method_name, **kwargs)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] {method_name} sample={kwargs.get('sample', '?')} — skipping")
            _empty_cuda_cache()
            return None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] {method_name} — skipping: {e}")
                _empty_cuda_cache()
                return None
            raise

    for sample in samples:
        prepared = zap_prepared_inputs.get(sample.sample_key)
        # full_cache: always original input (no truncation) — true baseline
        row = _run_safe(
            "full_cache",
            sample=sample,
            model=model,
            processor=processor,
            device=device,
            float_dtype=float_dtype,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            press=None,
            teacher_record=None,
            include_probe_position_inference=False,
            image_keep_ratio=1.0,
            prepared_input=None,
        )
        if row is not None:
            rows.append(row)
        _empty_cuda_cache()
        if args.include_oracle:
            row = _run_safe(
                "oracle_att_only_postvision",
                sample=sample,
                model=model,
                processor=processor,
                device=device,
                float_dtype=float_dtype,
                prompt_template=args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                press=oracle_press,
                teacher_record=teacher_cache[sample.sample_key],
                include_probe_position_inference=False,
                image_keep_ratio=args.oracle_image_keep_ratio,
                prepared_input=prepared,
            )
            if row is not None:
                rows.append(row)
            _empty_cuda_cache()
        row = _run_safe(
            "probe_att_only_postvision_mlp",
            sample=sample,
            model=model,
            processor=processor,
            device=device,
            float_dtype=float_dtype,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            press=probe_press,
            teacher_record=None,
            include_probe_position_inference=True,
            image_keep_ratio=args.total_keep_ratio,
            total_keep_ratio=args.total_keep_ratio,
            output_attentions=False,
            prepared_input=prepared,
        )
        if row is not None:
            rows.append(row)
        _empty_cuda_cache()

        if args.include_h2o_ablation:
            row = _run_safe(
                "h2o_image_only",
                sample=sample,
                model=model,
                processor=processor,
                device=device,
                float_dtype=float_dtype,
                prompt_template=args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                press=h2o_press,
                teacher_record=None,
                include_probe_position_inference=True,
                image_keep_ratio=args.total_keep_ratio,
                total_keep_ratio=args.total_keep_ratio,
                output_attentions=True,
                prepared_input=prepared,
            )
            if row is not None:
                rows.append(row)
            _empty_cuda_cache()

        if args.include_oracle_all_token and sample.sample_key in teacher_cache:
            row = _run_safe(
                "oracle_all_token",
                sample=sample,
                model=model,
                processor=processor,
                device=device,
                float_dtype=float_dtype,
                prompt_template=args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                press=oracle_all_token_press,
                teacher_record=teacher_cache[sample.sample_key],
                include_probe_position_inference=False,
                image_keep_ratio=args.total_keep_ratio,
                total_keep_ratio=args.total_keep_ratio,
                output_attentions=True,
                prepared_input=prepared,
            )
            if row is not None:
                rows.append(row)
            _empty_cuda_cache()

    del model
    _empty_cuda_cache()
    return rows


def build_lookm_worker(args: argparse.Namespace):
    # Monkey-patch transformers' torch.load security check (CVE-2025-32434) so that
    # LOOK-M can load its .bin weights with the current torch version in the kv env.
    # This is acceptable in a controlled research environment.
    # Patch the check in both the source module and wherever it was imported into.
    import transformers.utils.import_utils as _tfu
    import transformers.modeling_utils as _tmu
    _orig_check_tfu = _tfu.check_torch_load_is_safe
    _orig_check_tmu = getattr(_tmu, "check_torch_load_is_safe", _orig_check_tfu)
    _noop = lambda: None
    _tfu.check_torch_load_is_safe = _noop
    _tmu.check_torch_load_is_safe = _noop
    try:
        worker_class = get_worker_class("llava-v1.5-7b")
        config = AttrDict(
            model_name="llava-v1.5",
            model_dir="liuhaotian/llava-v1.5-7b",
            gen_kwargs=AttrDict(
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                temperature=0.0,
            ),
            max_context_len=4096,
            n_tokens_per_image=576,
            kv_mode=args.look_kv_mode,
            hh_ratio=args.look_hh_ratio,
            recent_ratio=args.look_recent_ratio,
            device=args.device,
        )
        worker = worker_class.from_config(config=config)
        worker.max_context_len = config.max_context_len
        worker.n_tokens_per_image = config.n_tokens_per_image
        worker.combine_image = None
    finally:
        _tfu.check_torch_load_is_safe = _orig_check_tfu
        _tmu.check_torch_load_is_safe = _orig_check_tmu
    return worker


def measure_lookm_for_sample(
    *,
    sample: PooledSample,
    worker: Any,
    device: torch.device,
    method_name: str,
    prepared_question: str,
    prepared_image_paths: list[str],
) -> dict[str, Any]:
    _empty_cuda_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    trace = ForwardTrace(worker.model, device)
    with trace:
        _ = worker.forward(
            questions=[prepared_question],
            image_paths=[prepared_image_paths],
            device=device,
            gen_kwargs=worker.gen_kwargs,
        )

    prompt_full_seq_len = int(trace.prefill_full_seq_len or 0)
    prompt_retained_seq_len = int(trace.prefill_cache_seq_len or prompt_full_seq_len)
    final_cache_seq_len = int(
        trace.final_cache_seq_len if trace.final_cache_seq_len is not None else prompt_retained_seq_len
    )
    generated_tokens = 1 + trace.decode_call_count if trace.prefill_duration_ms is not None else 0
    full_final_cache_len = prompt_full_seq_len + generated_tokens
    peak_gpu_memory_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else None
    )

    ttft_ms: Optional[float] = None
    tbt_ms_per_token: Optional[float] = None
    if trace.prefill_duration_ms is not None:
        ttft_ms = float(trace.prefill_duration_ms) + float(trace.first_decode_ms or 0.0)
    if trace.decode_call_count > 1 and trace.first_decode_ms is not None:
        steady_ms = trace.decode_total_ms - trace.first_decode_ms
        tbt_ms_per_token = steady_ms / (trace.decode_call_count - 1)
    elif trace.decode_call_count == 1:
        tbt_ms_per_token = trace.decode_total_ms

    model_config = getattr(worker, "model", None)
    model_config = getattr(model_config, "config", None)
    kv_cache_gib: Optional[float] = None
    if model_config is not None and hasattr(model_config, "num_hidden_layers"):
        try:
            model_dtype = next(worker.model.parameters()).dtype
            kv_cache_gib = _theoretical_kv_cache_gib(model_config, final_cache_seq_len, model_dtype)
        except Exception:
            pass

    return {
        "implementation": "look_m",
        "method": method_name,
        "dataset": sample.dataset_name,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        # ── latency ──────────────────────────────────────────────────────────
        "prefill_latency_ms": float(trace.prefill_duration_ms or 0.0),
        "prefill_setup_latency_ms": 0.0,
        "probe_forward_ms": None,
        "ttft_ms": ttft_ms,
        "tbt_ms_per_token": tbt_ms_per_token,
        "decode_latency_ms_per_token": (
            trace.decode_total_ms / trace.decode_call_count if trace.decode_call_count > 0 else None
        ),
        # ── memory ───────────────────────────────────────────────────────────
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "kv_cache_gib": kv_cache_gib,
        # ── compression ratios ───────────────────────────────────────────────
        "r_img": None,
        "r_eff_prompt": (prompt_retained_seq_len / prompt_full_seq_len) if prompt_full_seq_len > 0 else None,
        "r_eff_decode_t_end": (final_cache_seq_len / full_final_cache_len) if full_final_cache_len > 0 else None,
        # ── sequence lengths ─────────────────────────────────────────────────
        "prompt_full_seq_len": prompt_full_seq_len,
        "prompt_retained_seq_len": prompt_retained_seq_len,
        "final_cache_seq_len": final_cache_seq_len,
        "generated_tokens": generated_tokens,
        "decode_steps": trace.decode_call_count,
        # ── image token counts ───────────────────────────────────────────────
        "image_tokens_total": None,
        "image_tokens_kept": None,
        "image_keep_ratio": None,
        "image_positions_dropped": None,
    }


def run_lookm_measurements(args: argparse.Namespace, samples: list[PooledSample]) -> list[dict[str, Any]]:
    worker = build_lookm_worker(args)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    prepared_inputs = _prepare_lookm_inputs(
        samples,
        milebench_root=Path(args.milebench_root).resolve(),
        worker=worker,
    )

    warmup_input = prepared_inputs[samples[0].sample_key]
    _ = measure_lookm_for_sample(
        sample=samples[0],
        worker=worker,
        device=device,
        method_name="warmup_look_m",
        prepared_question=warmup_input["question"],
        prepared_image_paths=warmup_input["image_paths"],
    )
    _empty_cuda_cache()

    for sample in samples:
        prepared = prepared_inputs[sample.sample_key]
        rows.append(
            measure_lookm_for_sample(
                sample=sample,
                worker=worker,
                device=device,
                method_name="look_m",
                prepared_question=prepared["question"],
                prepared_image_paths=prepared["image_paths"],
            )
        )
        _empty_cuda_cache()

    del worker
    _empty_cuda_cache()
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "prefill_latency_ms",
        "probe_forward_ms",
        "ttft_ms",
        "tbt_ms_per_token",
        "decode_latency_ms_per_token",
        "peak_gpu_memory_gib",
        "kv_cache_gib",
        "r_img",
        "r_eff_prompt",
        "r_eff_decode_t_end",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["implementation"], row["method"])
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (implementation, method), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "implementation": implementation,
            "method": method,
            "n_samples": len(group_rows),
            "datasets": ",".join(sorted({str(row["dataset"]) for row in group_rows})),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = math.sqrt(variance)
            else:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milebench_root", type=str, default="/workspace/zap/data/MileBench")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/workspace/zap/artifacts/combine_prob/efficiency_random20")
    parser.add_argument("--implementation_model_name", type=str, default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prompt_template", type=str, default="USER: <image>\n{question}\nASSISTANT:")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--head_reduce", choices=["amax", "mean"], default="amax")
    # ── Unified compression budget (preferred) ───────────────────────────────
    parser.add_argument(
        "--total_keep_ratio", type=float, default=0.20,
        help="Fraction of ALL tokens (text+image) to retain. Applied to all ZAP methods and ablations "
             "so that r_eff_prompt is comparable across methods. LOOK-M uses --look_hh_ratio + "
             "--look_recent_ratio for its own budget.",
    )
    # Legacy per-method ratios (kept for backward compat; total_keep_ratio takes precedence)
    parser.add_argument("--oracle_image_keep_ratio", type=float, default=None)
    parser.add_argument("--probe_image_keep_ratio", type=float, default=None)
    parser.add_argument(
        "--probe_model_name",
        type=str,
        default="/workspace/zap/ckpts/image_probe_combined_v1/mlp",
    )
    parser.add_argument("--look_kv_mode", type=str, default="text_prior_pivot_merge")
    parser.add_argument("--look_hh_ratio", type=float, default=0.10)
    parser.add_argument("--look_recent_ratio", type=float, default=0.10)
    parser.add_argument("--include_zap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_lookm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_h2o_ablation", action=argparse.BooleanOptionalAction, default=False,
                        help="Include Ablation A: H2O score + image-only eviction.")
    parser.add_argument("--include_oracle_all_token", action=argparse.BooleanOptionalAction, default=False,
                        help="Include Ablation B: oracle att_only_postvision score + all-token eviction.")
    parser.add_argument("--sample_manifest_path", type=str, default=None)
    parser.add_argument(
        "--truncate_like_lookm", action=argparse.BooleanOptionalAction, default=False,
        help="Apply LOOK-M style truncation (max_context_len=4096, n_tokens_per_image=576) "
             "to ALL methods (full_cache, probe, oracle, h2o) so they see the same input as LOOK-M. "
             "Required for fair comparison on high-image-count datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_dir_map: Optional[dict[str, str]] = None
    if args.include_oracle or args.include_oracle_all_token:
        missing = [name for name in args.datasets if name not in DEFAULT_TEACHER_DIRS]
        if missing:
            raise ValueError(f"Missing teacher dirs for datasets: {missing}")
        teacher_dir_map = {name: DEFAULT_TEACHER_DIRS[name] for name in args.datasets}
    pooled = collect_pooled_samples(
        milebench_root=Path(args.milebench_root).resolve(),
        dataset_names=args.datasets,
        teacher_dir_map=teacher_dir_map,
    )
    if args.sample_manifest_path:
        samples = select_samples_from_manifest(pooled, Path(args.sample_manifest_path).resolve())
    else:
        samples = select_random_samples(pooled, sample_size=args.sample_size, seed=args.seed)

    manifest = [
        {
            "dataset": sample.dataset_name,
            "sample_id": sample.sample_id,
            "sample_key": sample.sample_key,
            "image_paths": sample.image_paths,
            "teacher_dir": "" if sample.teacher_dir is None else str(sample.teacher_dir),
        }
        for sample in samples
    ]
    _write_json(output_dir / "sample_manifest.json", manifest)
    _write_json(output_dir / "run_config.json", vars(args))

    rows = []
    if args.include_zap:
        rows.extend(run_zap_measurements(args, samples))
    if args.include_lookm:
        rows.extend(run_lookm_measurements(args, samples))
    summaries = summarize_rows(rows)

    per_sample_fields = [
        "implementation",
        "method",
        "dataset",
        "sample_id",
        "sample_key",
        # latency
        "prefill_latency_ms",
        "prefill_setup_latency_ms",
        "probe_forward_ms",
        "ttft_ms",
        "tbt_ms_per_token",
        "decode_latency_ms_per_token",
        # memory
        "peak_gpu_memory_gib",
        "kv_cache_gib",
        # compression ratios
        "r_img",
        "r_eff_prompt",
        "r_eff_decode_t_end",
        # sequence lengths
        "prompt_full_seq_len",
        "prompt_retained_seq_len",
        "final_cache_seq_len",
        "generated_tokens",
        "decode_steps",
        # image token counts
        "image_tokens_total",
        "image_tokens_kept",
        "image_keep_ratio",
        "image_positions_dropped",
    ]
    summary_fields = [
        "implementation",
        "method",
        "n_samples",
        "datasets",
        # latency
        "prefill_latency_ms_mean",
        "prefill_latency_ms_std",
        "probe_forward_ms_mean",
        "probe_forward_ms_std",
        "ttft_ms_mean",
        "ttft_ms_std",
        "tbt_ms_per_token_mean",
        "tbt_ms_per_token_std",
        "decode_latency_ms_per_token_mean",
        "decode_latency_ms_per_token_std",
        # memory
        "peak_gpu_memory_gib_mean",
        "peak_gpu_memory_gib_std",
        "kv_cache_gib_mean",
        "kv_cache_gib_std",
        # compression ratios
        "r_img_mean",
        "r_img_std",
        "r_eff_prompt_mean",
        "r_eff_prompt_std",
        "r_eff_decode_t_end_mean",
        "r_eff_decode_t_end_std",
    ]

    _write_csv(output_dir / "per_sample.csv", rows, per_sample_fields)
    _write_csv(output_dir / "summary.csv", summaries, summary_fields)
    _write_json(output_dir / "summary.json", summaries)

    print(json.dumps({"output_dir": str(output_dir), "n_samples": len(samples), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
