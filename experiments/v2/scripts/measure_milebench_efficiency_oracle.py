#!/usr/bin/env python3
"""
Oracle efficiency measurement on matched samples (full_cache 성공 77개).

사용법:
  conda run -n kv --no-capture-output \
    python scripts/measure_milebench_efficiency_oracle.py \
      --per_sample_csv /workspace/hd/artifacts/probe_global/efficiency_all_datasets/per_sample.csv \
      --manifest /workspace/hd/artifacts/probe_global/efficiency_all_datasets/sample_manifest.json \
      --output_dir /workspace/hd/artifacts/probe_global/efficiency_all_datasets \
      --milebench_root /workspace/hd/data/MileBench \
      --total_keep_ratio 0.20 --device cuda:0

동작:
  Pass 1: full forward (output_attentions=True) → att_only_postvision 추출
  Pass 2: OracleImageTeacherPress로 압축 forward + generate → latency/memory

주의:
  기본 모드(pass2_only)는 기존과 동일하게 Pass 2만 primary latency/memory로 기록합니다.
  공정 비교가 필요하면 --oracle_timing_mode two_pass_total 을 사용하세요.
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
    # LlavaConfig wraps text_config; fall back to top-level if already a language model config
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


# ── teacher extraction (Pass 1, not timed) ───────────────────────────────────

def extract_att_only_postvision(
    model,
    processor,
    prompt_inputs: dict,
    image_positions: torch.Tensor,   # [n_image] — merged-prompt positions
    device: torch.device,
    float_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    full forward (output_attentions=True) → att_only_postvision [n_layers, n_heads, n_image]
    """
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    n_image = int(image_positions.numel())

    if n_image == 0:
        return None

    # postvision text positions: text positions AFTER the last image token
    last_image_pos = int(image_positions.max().item())
    all_pos = torch.arange(prompt_len, dtype=torch.long, device=device)
    img_set = image_positions.to(device)

    # boolean mask: is each position an image position?
    is_image = torch.zeros(prompt_len, dtype=torch.bool, device=device)
    is_image[img_set] = True
    is_text = ~is_image

    postvision_text_idx = all_pos[(all_pos > last_image_pos) & is_text]
    if postvision_text_idx.numel() == 0:
        # fallback: use last 10% of text tokens if no postvision text
        n_text = int(is_text.sum().item())
        text_idx = all_pos[is_text]
        postvision_text_idx = text_idx[max(0, n_text - max(1, n_text // 10)):]

    if postvision_text_idx.numel() == 0:
        return None

    try:
        with torch.no_grad():
            out = model(
                **prompt_inputs,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            return None
        raise

    if out.attentions is None:
        return None

    img_idx_dev = img_set.to(torch.long)
    pv_idx_dev  = postvision_text_idx.to(torch.long)

    layers = []
    for layer_attn in out.attentions:
        # layer_attn: [1, H, S, S] or [H, S, S]
        attn = layer_attn[0].detach() if layer_attn.dim() == 4 else layer_attn.detach()
        # attn: [H, S, S]
        block = attn.index_select(1, pv_idx_dev).index_select(2, img_idx_dev)  # [H, Q_pv, n_image]
        layers.append(block.amax(dim=1).cpu().to(torch.float16))               # [H, n_image]

    # [n_layers, n_heads, n_image]
    return torch.stack(layers, dim=0)


# ── oracle compressed forward (Pass 2, timed) ────────────────────────────────

def measure_oracle_for_sample(
    *,
    sample: dict,
    model,
    processor,
    device: torch.device,
    float_dtype: torch.dtype,
    prompt_template: str,
    max_new_tokens: int,
    oracle_press,
    total_keep_ratio: float,
    oracle_timing_mode: str = "pass2_only",
) -> Optional[dict]:

    image_paths  = sample["image_paths"]
    question_raw = sample["question"]   # already stripped of {image#N}

    # ── 공통 prompt 빌드 (Pass 1 & 2 동일) ────────────────────────────────
    images = open_images(image_paths)
    prompt_text = build_prompt(question_raw, prompt_template, image_count=len(image_paths))
    prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)

    # image positions (no forward needed)
    image_positions, prompt_full_seq_len = infer_llava_image_positions_no_forward(
        prompt_inputs=prompt_inputs,
        model_config=model.config,
        num_images=len(image_paths),
    )
    image_positions = image_positions.to(device)

    # ── Pass 1: att_only_postvision 추출 ───────────────────────────────────
    _empty_cuda_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    pass1_start = time.perf_counter()
    teacher_scores = extract_att_only_postvision(
        model, processor, prompt_inputs, image_positions, device, float_dtype
    )
    _sync(device)
    pass1_teacher_ms = (time.perf_counter() - pass1_start) * 1000.0
    pass1_peak_mem_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else None
    )
    if teacher_scores is None:
        print(f"  [SKIP] teacher extraction failed: {sample['dataset']} sid={sample['sample_id']}")
        return None

    # ── Pass 2: oracle 압축 forward (시간 측정) ───────────────────────────
    _empty_cuda_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    oracle_press.set_sample_teacher(image_positions, teacher_scores.to(device))

    try:
        with ForwardTrace(model, device) as trace:
            with oracle_press(model):
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
            print(f"  [OOM] oracle pass: {sample['dataset']} sid={sample['sample_id']}")
            _empty_cuda_cache()
            return None
        raise

    # ── metrics ────────────────────────────────────────────────────────────
    generated_tokens = max(0, outputs.shape[1] - prompt_inputs["input_ids"].shape[1])

    ttft_ms = tbt_ms = None
    if trace.prefill_duration_ms is not None:
        ttft_ms = float(trace.prefill_duration_ms) + float(trace.first_decode_ms or 0.0)
    if trace.decode_call_count > 1 and trace.first_decode_ms is not None:
        tbt_ms = (trace.decode_total_ms - trace.first_decode_ms) / (trace.decode_call_count - 1)
    elif trace.decode_call_count == 1:
        tbt_ms = trace.decode_total_ms

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
    two_pass_peak_mem_gib = (
        max(pass1_peak_mem_gib, peak_mem_gib)
        if pass1_peak_mem_gib is not None and peak_mem_gib is not None
        else (pass1_peak_mem_gib if peak_mem_gib is None else peak_mem_gib)
    )
    kv_gib = _theoretical_kv_cache_gib(model.config, final_cache_seq_len, float_dtype)

    pass2_prefill_latency_ms = float(trace.prefill_duration_ms or 0.0)
    two_pass_prefill_latency_ms = pass1_teacher_ms + pass2_prefill_latency_ms
    two_pass_ttft_ms = (pass1_teacher_ms + ttft_ms) if ttft_ms is not None else None

    if oracle_timing_mode == "two_pass_total":
        method_name = "oracle_att_only_postvision_2pass_total"
        prefill_latency_ms_primary = two_pass_prefill_latency_ms
        ttft_ms_primary = two_pass_ttft_ms
        peak_mem_gib_primary = two_pass_peak_mem_gib
    else:
        method_name = "oracle_att_only_postvision"
        prefill_latency_ms_primary = pass2_prefill_latency_ms
        ttft_ms_primary = ttft_ms
        peak_mem_gib_primary = peak_mem_gib

    return {
        "implementation":         "zap_hf",
        "method":                 method_name,
        "dataset":                sample["dataset"],
        "sample_id":              sample["sample_id"],
        "sample_key":             sample["sample_key"],
        "prefill_latency_ms":     prefill_latency_ms_primary,
        "prefill_setup_latency_ms": 0.0,
        "probe_forward_ms":       None,
        "ttft_ms":                ttft_ms_primary,
        "tbt_ms_per_token":       tbt_ms,
        "decode_latency_ms_per_token": (
            trace.decode_total_ms / trace.decode_call_count if trace.decode_call_count > 0 else None
        ),
        "peak_gpu_memory_gib":    peak_mem_gib_primary,
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
        # transparency fields for fair reporting
        "oracle_timing_mode": oracle_timing_mode,
        "oracle_pass1_teacher_ms": pass1_teacher_ms,
        "oracle_pass1_peak_gpu_memory_gib": pass1_peak_mem_gib,
        "oracle_pass2_prefill_latency_ms": pass2_prefill_latency_ms,
        "oracle_pass2_ttft_ms": ttft_ms,
        "oracle_pass2_peak_gpu_memory_gib": peak_mem_gib,
        "oracle_two_pass_prefill_latency_ms": two_pass_prefill_latency_ms,
        "oracle_two_pass_ttft_ms": two_pass_ttft_ms,
        "oracle_two_pass_peak_gpu_memory_gib": two_pass_peak_mem_gib,
    }


# ── data loading ──────────────────────────────────────────────────────────────

def load_matched_samples(per_sample_csv: Path, manifest_path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(per_sample_csv)))
    fc_keys = {r["sample_key"] for r in rows if r["method"] == "full_cache"}
    manifest = json.loads(manifest_path.read_text())
    matched = [m for m in manifest if m["sample_key"] in fc_keys]
    print(f"Matched samples: {len(matched)} (full_cache 성공 기준)")
    return matched


def load_milebench_question(entry: dict, milebench_root: Path) -> dict:
    ds  = entry["dataset"]
    sid = entry["sample_id"]
    ds_path = milebench_root / ds / f"{ds}.json"

    data = json.loads(ds_path.read_text())
    samples_list = data if isinstance(data, list) else data.get("data", [])
    raw = next((s for s in samples_list if str(s.get("sample_id", "")) == str(sid)), None)
    if raw is None:
        raise ValueError(f"sample_id={sid} not found in {ds}")

    q_raw = raw.get("context") or raw.get("question") or ""
    if isinstance(q_raw, dict):
        q_raw = q_raw.get("question", "")
    # strip image placeholders — build_prompt will re-inject correct count
    q_raw = re.sub(r"<ImageHere>|\{image#\d+\}", "", str(q_raw)).strip()

    # image_paths in manifest are already absolute
    image_paths = entry["image_paths"]

    return {
        "dataset":    ds,
        "sample_id":  str(sid),
        "sample_key": entry["sample_key"],
        "question":   q_raw,
        "image_paths": image_paths,
    }


def append_rows(rows: list[dict], csv_path: Path):
    if not rows:
        return

    existing_rows = []
    fieldnames: list[str] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row.pop(None, None)
                existing_rows.append(row)

    if not fieldnames:
        fieldnames = [k for k in rows[0].keys() if k is not None]

    for r in rows:
        for k in r.keys():
            if k is None:
                continue
            if k not in fieldnames:
                fieldnames.append(k)

    def normalize(row: dict) -> dict:
        return {k: row.get(k, "") for k in fieldnames}

    merged_rows = [normalize(r) for r in existing_rows] + [normalize(r) for r in rows]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"Appended {len(rows)} rows → {csv_path} (total rows={len(merged_rows)})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_sample_csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--milebench_root", default="/workspace/hd/data/MileBench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total_keep_ratio", type=float, default=0.20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--head_reduce", default="amax")
    parser.add_argument(
        "--oracle_timing_mode",
        choices=("pass2_only", "two_pass_total"),
        default="pass2_only",
        help="pass2_only: 기존 oracle 측정(teacher 추출 제외), two_pass_total: Pass1+Pass2를 primary latency/memory로 기록",
    )
    parser.add_argument("--prompt_template",
                        default="USER: <image>\n{question}\nASSISTANT:")
    args = parser.parse_args()

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from kvpress.presses.image_token_press import OracleImageTeacherPress

    output_dir    = Path(args.output_dir)
    device        = torch.device(args.device)
    milebench_root = Path(args.milebench_root)

    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float16, attn_implementation="eager",
    ).to(device)
    model.eval()
    float_dtype = torch.float16
    print("Model loaded.")

    oracle_press = OracleImageTeacherPress(
        total_keep_ratio=args.total_keep_ratio,
        head_reduce=args.head_reduce,
    )

    matched = load_matched_samples(Path(args.per_sample_csv), Path(args.manifest))

    rows = []
    for i, entry in enumerate(matched):
        try:
            sample = load_milebench_question(entry, milebench_root)
        except Exception as e:
            print(f"[{i+1}/{len(matched)}] SKIP load: {e}")
            continue

        n_imgs = len(sample["image_paths"])
        print(f"[{i+1}/{len(matched)}] {sample['dataset']} sid={sample['sample_id']} n_imgs={n_imgs}")

        row = measure_oracle_for_sample(
            sample=sample,
            model=model,
            processor=processor,
            device=device,
            float_dtype=float_dtype,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            oracle_press=oracle_press,
            total_keep_ratio=args.total_keep_ratio,
            oracle_timing_mode=args.oracle_timing_mode,
        )
        if row is not None:
            rows.append(row)
            tbt_s = f"{row['tbt_ms_per_token']:.1f}" if row["tbt_ms_per_token"] else "N/A"
            print(f"  prefill={row['prefill_latency_ms']:.0f}ms  tbt={tbt_s}ms/tok  "
                  f"mem={row['peak_gpu_memory_gib']:.2f}GiB  r_eff={row['r_eff_prompt']:.3f}")
        _empty_cuda_cache()

    append_rows(rows, output_dir / "per_sample.csv")
    print(f"\nDone. {len(rows)} oracle rows written.")


if __name__ == "__main__":
    main()
