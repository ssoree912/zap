#!/usr/bin/env python3
"""
Probe-only efficiency re-measurement with total_keep_ratio=0.20.

Removes existing probe rows from per_sample.csv, re-runs probe for all
135 manifest samples using total_keep_ratio (not image_keep_ratio), appends
corrected rows.

Usage:
  conda run -n kv --no-capture-output \
    python scripts/rerun_probe_efficiency.py \
      --per_sample_csv /workspace/hd/artifacts/probe_global/efficiency_all_datasets/per_sample.csv \
      --manifest /workspace/hd/artifacts/probe_global/efficiency_all_datasets/sample_manifest.json \
      --milebench_root /workspace/hd/data/MileBench \
      --total_keep_ratio 0.20 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kvzap.image_teacher_utils import build_prompt
from kvzap.llava_extractor import (
    _get_model_float_dtype,
    _move_batch_to_device,
    infer_llava_image_positions_no_forward,
)

PROBE_METHOD_NAME = "probe_att_only_postvision_mlp"


# ── helpers ──────────────────────────────────────────────────────────────────

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_seq_length(past_key_values) -> Optional[int]:
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


def _theoretical_kv_cache_gib(model_config, seq_len: int, dtype: torch.dtype) -> float:
    cfg = getattr(model_config, "text_config", model_config)
    n_layers = cfg.num_hidden_layers
    n_heads  = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    bpe = torch.finfo(dtype).bits // 8
    return 2 * n_layers * n_heads * head_dim * seq_len * bpe / (1024 ** 3)


def open_images(paths: list[str]):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return imgs[0] if len(imgs) == 1 else imgs


class ForwardTrace:
    def __init__(self, model, device: torch.device):
        self.model, self.device = model, device
        self._handles = []
        self._start = None
        self.prefill_duration_ms: Optional[float] = None
        self.first_decode_ms: Optional[float]  = None
        self.decode_total_ms: float = 0.0
        self.decode_call_count: int = 0
        self.final_cache_seq_len: Optional[int] = None

    def _pre(self, module, args, kwargs):
        _sync(self.device); self._start = time.perf_counter()

    def _post(self, module, args, kwargs, output):
        _sync(self.device)
        elapsed = (time.perf_counter() - self._start) * 1000.0
        cache_len = _cache_seq_length(getattr(output, "past_key_values", None))
        if self.prefill_duration_ms is None:
            self.prefill_duration_ms = elapsed
        else:
            if self.first_decode_ms is None:
                self.first_decode_ms = elapsed
            self.decode_total_ms += elapsed
            self.decode_call_count += 1
        if cache_len is not None:
            self.final_cache_seq_len = cache_len
        return output

    def __enter__(self):
        self._handles.append(self.model.register_forward_pre_hook(self._pre, with_kwargs=True))
        self._handles.append(self.model.register_forward_hook(self._post, with_kwargs=True))
        return self

    def __exit__(self, *_):
        for h in self._handles: h.remove()
        self._handles.clear()


# ── probe measurement ─────────────────────────────────────────────────────────

def measure_probe_for_sample(
    *,
    sample: dict,
    model,
    processor,
    probe_press,
    device: torch.device,
    float_dtype: torch.dtype,
    prompt_template: str,
    max_new_tokens: int,
    total_keep_ratio: float,
) -> Optional[dict]:

    image_paths  = sample["image_paths"]
    question_raw = sample["question"]

    images = open_images(image_paths)
    prompt_text = build_prompt(question_raw, prompt_template, image_count=len(image_paths))
    prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)

    # Probe position inference (timed as setup)
    t0 = time.perf_counter()
    image_positions, prompt_full_seq_len = infer_llava_image_positions_no_forward(
        prompt_inputs=prompt_inputs,
        model_config=model.config,
        num_images=len(image_paths),
    )
    setup_ms = (time.perf_counter() - t0) * 1000.0
    image_positions = image_positions.to(device)

    _empty_cuda_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    probe_press.set_image_positions(image_positions)
    if hasattr(probe_press, "reset_probe_timing"):
        probe_press.reset_probe_timing()

    try:
        with ForwardTrace(model, device) as trace:
            with probe_press(model):
                outputs = model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f"  [OOM] probe: {sample['dataset']} sid={sample['sample_id']}")
            _empty_cuda_cache()
            return None
        raise
    finally:
        probe_press.clear_sample_context()

    generated_tokens = max(0, outputs.shape[1] - prompt_inputs["input_ids"].shape[1])

    ttft_ms = tbt_ms = None
    if trace.prefill_duration_ms is not None:
        ttft_ms = float(trace.prefill_duration_ms) + float(trace.first_decode_ms or 0.0)
    if trace.decode_call_count > 1 and trace.first_decode_ms is not None:
        tbt_ms = (trace.decode_total_ms - trace.first_decode_ms) / (trace.decode_call_count - 1)
    elif trace.decode_call_count == 1:
        tbt_ms = trace.decode_total_ms

    # Compute token counts with total_keep_ratio
    n_image = int(image_positions.numel())
    n_text  = max(0, prompt_full_seq_len - n_image)
    total_keep   = int(math.ceil(total_keep_ratio * prompt_full_seq_len))
    n_img_keep   = min(n_image, max(0, total_keep - n_text))
    prompt_retained_seq_len = n_text + n_img_keep
    final_cache_seq_len     = prompt_retained_seq_len + generated_tokens
    full_final_cache_len    = prompt_full_seq_len + generated_tokens

    r_eff_prompt = prompt_retained_seq_len / prompt_full_seq_len if prompt_full_seq_len > 0 else None
    r_eff_decode = final_cache_seq_len / full_final_cache_len if full_final_cache_len > 0 else None

    peak_mem_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else None
    )
    kv_gib = _theoretical_kv_cache_gib(model.config, final_cache_seq_len, float_dtype)

    return {
        "implementation":         "zap_hf",
        "method":                 PROBE_METHOD_NAME,
        "dataset":                sample["dataset"],
        "sample_id":              sample["sample_id"],
        "sample_key":             sample["sample_key"],
        "prefill_latency_ms":     setup_ms + float(trace.prefill_duration_ms or 0.0),
        "prefill_setup_latency_ms": setup_ms,
        "probe_forward_ms":       None,
        "ttft_ms":                ttft_ms,
        "tbt_ms_per_token":       tbt_ms,
        "decode_latency_ms_per_token": (
            trace.decode_total_ms / trace.decode_call_count if trace.decode_call_count > 0 else None
        ),
        "peak_gpu_memory_gib":    peak_mem_gib,
        "kv_cache_gib":           kv_gib,
        "r_img":                  n_img_keep / n_image if n_image > 0 else None,
        "r_eff_prompt":           r_eff_prompt,
        "r_eff_decode_t_end":     r_eff_decode,
        "prompt_full_seq_len":    prompt_full_seq_len,
        "prompt_retained_seq_len": prompt_retained_seq_len,
        "final_cache_seq_len":    final_cache_seq_len,
        "generated_tokens":       generated_tokens,
        "decode_steps":           trace.decode_call_count,
        "image_tokens_total":     n_image,
        "image_tokens_kept":      n_img_keep,
        "image_keep_ratio":       n_img_keep / n_image if n_image > 0 else None,
        "image_positions_dropped": None,
        "num_images":             len(image_paths),
    }


# ── data loading ──────────────────────────────────────────────────────────────

def load_manifest_samples(manifest_path: Path, milebench_root: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    samples = []
    for entry in manifest:
        ds  = entry["dataset"]
        sid = entry["sample_id"]
        ds_path = milebench_root / ds / f"{ds}.json"
        try:
            data = json.loads(ds_path.read_text())
        except Exception:
            print(f"[WARN] Cannot load {ds_path}")
            continue
        samples_list = data if isinstance(data, list) else data.get("data", [])
        raw = next((s for s in samples_list if str(s.get("sample_id", "")) == str(sid)), None)
        if raw is None:
            print(f"[WARN] sample_id={sid} not found in {ds}")
            continue
        q_raw = raw.get("context") or raw.get("question") or ""
        if isinstance(q_raw, dict):
            q_raw = q_raw.get("question", "")
        q_raw = re.sub(r"<ImageHere>|\{image#\d+\}", "", str(q_raw)).strip()
        samples.append({
            "dataset":    ds,
            "sample_id":  str(sid),
            "sample_key": entry["sample_key"],
            "question":   q_raw,
            "image_paths": entry["image_paths"],
        })
    print(f"Loaded {len(samples)} samples from manifest")
    return samples


def filter_and_backup_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    """Remove probe rows from CSV, return (non-probe rows, fieldnames)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            # Drop malformed overflow columns (DictReader stores them under key None).
            row.pop(None, None)
            rows.append({k: row.get(k, "") for k in fieldnames})
    probe_count = sum(1 for r in rows if r.get("method") == PROBE_METHOD_NAME)
    kept = [r for r in rows if r.get("method") != PROBE_METHOD_NAME]
    print(f"Removed {probe_count} old probe rows from {csv_path}")
    print(f"Keeping {len(kept)} non-probe rows")
    # Rewrite CSV without probe rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    return kept, fieldnames


def append_rows(rows: list[dict], csv_path: Path, fieldnames: list[str]):
    if not rows:
        return

    # Read existing rows first so we can safely rewrite with a unified header.
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            if not fieldnames:
                fieldnames.extend(existing_fields)
            for row in reader:
                row.pop(None, None)
                existing_rows.append(row)

    # Add any new fields from incoming rows.
    for r in rows:
        for k in r:
            if k is None:
                continue
            if k not in fieldnames:
                fieldnames.append(k)

    # Normalize both existing + new rows to the same schema, then rewrite.
    def normalize(row: dict) -> dict:
        return {k: row.get(k, "") for k in fieldnames}

    merged_rows = [normalize(r) for r in existing_rows] + [normalize(r) for r in rows]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"Appended {len(rows)} probe rows → {csv_path} (total rows={len(merged_rows)})")


# ── LOOK-M style truncation (reuse logic from measure_milebench_efficiency.py) ──

def truncate_sample_like_lookm(
    sample: dict,
    processor,
    max_context_len: int = 4096,
    n_tokens_per_image: int = 576,
    prompt_template: str = "USER: <image>\n{question}\nASSISTANT:",
) -> dict:
    """Truncate multi-image sample to LOOK-M budget (same as the original ZAP run)."""
    image_paths = sample["image_paths"]
    question = sample["question"]

    n_imgs = len(image_paths)
    text_tokens_budget = max_context_len - n_imgs * n_tokens_per_image
    if text_tokens_budget < 0:
        # Too many images — truncate to as many images as fit
        max_imgs = max_context_len // n_tokens_per_image
        n_imgs = min(n_imgs, max_imgs)
        image_paths = image_paths[:n_imgs]
        text_tokens_budget = max_context_len - n_imgs * n_tokens_per_image

    return {
        **sample,
        "image_paths": image_paths,
        "question": question,  # kept as-is; build_prompt will construct prompt
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_sample_csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--milebench_root", default="/workspace/hd/data/MileBench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total_keep_ratio", type=float, default=0.20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--probe_model_name", default="att_only_postvision_mlp")
    parser.add_argument("--head_reduce", default="amax")
    parser.add_argument("--prompt_template",
                        default="USER: <image>\n{question}\nASSISTANT:")
    parser.add_argument("--attn_impl", default="flash_attention_2",
                        choices=["flash_attention_2", "eager", "sdpa"])
    args = parser.parse_args()

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from kvpress.presses.image_token_press import ProbeImageTeacherPress

    csv_path = Path(args.per_sample_csv)
    device   = torch.device(args.device)
    milebench_root = Path(args.milebench_root)

    # Step 1: remove old probe rows from CSV
    kept_rows, fieldnames = filter_and_backup_csv(csv_path)

    # Step 2: load model
    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation=args.attn_impl,
    ).to(device)
    model.eval()
    float_dtype = torch.float16
    print("Model loaded.")

    # Step 3: build probe press
    probe_press = ProbeImageTeacherPress(
        total_keep_ratio=args.total_keep_ratio,
        head_reduce=args.head_reduce,
        probe_model_name=args.probe_model_name,
    )
    probe_press.post_init_from_model(model)
    print(f"ProbeImageTeacherPress ready (total_keep_ratio={args.total_keep_ratio})")

    # Step 4: load manifest samples
    samples = load_manifest_samples(Path(args.manifest), milebench_root)

    # Step 5: measure
    new_rows = []
    for i, sample in enumerate(samples):
        sample_for_run = truncate_sample_like_lookm(
            sample=sample,
            processor=processor,
            max_context_len=4096,
            n_tokens_per_image=576,
            prompt_template=args.prompt_template,
        )
        n_imgs = len(sample_for_run["image_paths"])
        print(f"[{i+1}/{len(samples)}] {sample_for_run['dataset']} sid={sample_for_run['sample_id']} n_imgs={n_imgs}")
        row = measure_probe_for_sample(
            sample=sample_for_run,
            model=model,
            probe_press=probe_press,
            processor=processor,
            device=device,
            float_dtype=float_dtype,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            total_keep_ratio=args.total_keep_ratio,
        )
        if row is not None:
            new_rows.append(row)
            tbt_s = f"{row['tbt_ms_per_token']:.1f}" if row["tbt_ms_per_token"] else "N/A"
            print(f"  prefill={row['prefill_latency_ms']:.0f}ms  tbt={tbt_s}ms/tok  "
                  f"mem={row['peak_gpu_memory_gib']:.2f}GiB  r_eff={row['r_eff_prompt']:.3f}")
        _empty_cuda_cache()

    # Step 6: append new probe rows
    append_rows(new_rows, csv_path, fieldnames)
    print(f"\nDone. {len(new_rows)}/{len(samples)} probe rows written.")


if __name__ == "__main__":
    main()
