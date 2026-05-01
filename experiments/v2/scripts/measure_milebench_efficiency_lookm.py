#!/usr/bin/env python3
"""
LOOK-M efficiency measurement — runs in the `look` conda environment.

Reads sample_manifest.json produced by measure_milebench_efficiency.py (ZAP run),
runs LOOK-M forward pass on the same samples, and appends results to per_sample.csv.

Usage (look env):
  cd /workspace/zap
  conda run -n look --no-capture-output python scripts/measure_milebench_efficiency_lookm.py \
    --manifest /workspace/hd/artifacts/probe_global/efficiency_all_datasets/sample_manifest.json \
    --output_dir /workspace/hd/artifacts/probe_global/efficiency_all_datasets \
    --milebench_root /workspace/hd/data/MileBench \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

LOOKM_ROOT = Path(__file__).resolve().parents[2] / "LOOK-M"
sys.path.insert(0, str(LOOKM_ROOT))
sys.path.insert(0, str(LOOKM_ROOT / "LLaVA-mix_merge_v1"))

from utils import MileBenchDataset, get_worker_class  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    try:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        kv = past_key_values
        while isinstance(kv, (list, tuple)):
            kv = kv[0]
        if hasattr(kv, "shape"):
            return int(kv.shape[2])
    except Exception:
        pass
    return None


def _theoretical_kv_cache_gib(model_config, seq_len: int, dtype: torch.dtype) -> float:
    n_layers = model_config.num_hidden_layers
    n_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    bytes_per_elem = torch.finfo(dtype).bits // 8
    total_bytes = 2 * n_layers * n_heads * head_dim * seq_len * bytes_per_elem
    return total_bytes / (1024 ** 3)


class ForwardTrace:
    def __init__(self, model: Any, device: torch.device):
        self.model = model
        self.device = device
        self._handles = []
        self._start_time = None
        self.prefill_duration_ms: Optional[float] = None
        self.first_decode_ms: Optional[float] = None
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
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
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
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_core_annotation(dataset_path: str) -> Optional[dict]:
    p = Path(dataset_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"meta_data": {"task_instruction": ""}, "data": data}
    return data


def load_samples_for_manifest(
    manifest: list[dict],
    milebench_root: Path,
) -> list[dict]:
    """Reload full sample data (including question text) for manifest entries."""
    # Group by dataset
    by_dataset: dict[str, list[dict]] = {}
    for entry in manifest:
        by_dataset.setdefault(entry["dataset"], []).append(entry)

    result: dict[str, dict] = {}
    for dataset_name, entries in by_dataset.items():
        dataset_path = milebench_root / dataset_name / f"{dataset_name}.json"
        image_root = milebench_root / dataset_name / "images"
        core = load_core_annotation(str(dataset_path))
        if core is None:
            print(f"[WARN] Cannot load {dataset_path}, skipping {dataset_name}")
            continue
        raw_data = core["data"] if "data" in core else core
        task_instruction = core.get("meta_data", {}).get("task_instruction", "")

        # Build sample_id → raw mapping
        raw_by_id: dict[str, dict] = {}
        for item in raw_data:
            sid = str(item.get("sample_id", ""))
            raw_by_id[sid] = item

        for entry in entries:
            sid = entry["sample_id"]
            raw = raw_by_id.get(sid)
            if raw is None:
                print(f"[WARN] sample_id={sid} not found in {dataset_name}")
                continue
            result[entry["sample_key"]] = {
                "dataset": dataset_name,
                "sample_id": sid,
                "sample_key": entry["sample_key"],
                "image_paths": entry["image_paths"],
                "raw": raw,
                "task_instruction": task_instruction,
            }
    return list(result.values())


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure_lookm(
    *,
    sample: dict,
    worker: Any,
    device: torch.device,
    method_name: str,
    prepared_question: str,
    prepared_image_paths: list[str],
) -> Optional[dict]:
    _empty_cuda_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        trace = ForwardTrace(worker.model, device)
        with trace:
            _ = worker.forward(
                questions=[prepared_question],
                image_paths=[prepared_image_paths],
                device=device,
                gen_kwargs=worker.gen_kwargs,
            )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {method_name} ds={sample['dataset']} sid={sample['sample_id']}: {e}")
        _empty_cuda_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] {method_name} ds={sample['dataset']} sid={sample['sample_id']}: {e}")
            _empty_cuda_cache()
            return None
        raise

    prompt_full_seq_len = int(trace.prefill_full_seq_len or 0)
    prompt_retained_seq_len = int(trace.prefill_cache_seq_len or prompt_full_seq_len)
    final_cache_seq_len = int(
        trace.final_cache_seq_len if trace.final_cache_seq_len is not None else prompt_retained_seq_len
    )
    generated_tokens = 1 + trace.decode_call_count if trace.prefill_duration_ms is not None else 0
    full_final_cache_len = prompt_full_seq_len + generated_tokens
    peak_gpu_memory_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else None
    )

    ttft_ms = None
    tbt_ms_per_token = None
    if trace.prefill_duration_ms is not None:
        ttft_ms = float(trace.prefill_duration_ms) + float(trace.first_decode_ms or 0.0)
    if trace.decode_call_count > 1 and trace.first_decode_ms is not None:
        tbt_ms_per_token = (trace.decode_total_ms - trace.first_decode_ms) / (trace.decode_call_count - 1)
    elif trace.decode_call_count == 1:
        tbt_ms_per_token = trace.decode_total_ms

    model_config = getattr(getattr(worker, "model", None), "config", None)
    kv_cache_gib = None
    if model_config is not None and hasattr(model_config, "num_hidden_layers"):
        try:
            dtype = next(worker.model.parameters()).dtype
            kv_cache_gib = _theoretical_kv_cache_gib(model_config, final_cache_seq_len, dtype)
        except Exception:
            pass

    return {
        "implementation": "look_m",
        "method": method_name,
        "dataset": sample["dataset"],
        "sample_id": sample["sample_id"],
        "sample_key": sample["sample_key"],
        "prefill_latency_ms": float(trace.prefill_duration_ms or 0.0),
        "prefill_setup_latency_ms": 0.0,
        "probe_forward_ms": None,
        "ttft_ms": ttft_ms,
        "tbt_ms_per_token": tbt_ms_per_token,
        "decode_latency_ms_per_token": (
            trace.decode_total_ms / trace.decode_call_count if trace.decode_call_count > 0 else None
        ),
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "kv_cache_gib": kv_cache_gib,
        "r_img": None,
        "r_eff_prompt": (prompt_retained_seq_len / prompt_full_seq_len) if prompt_full_seq_len > 0 else None,
        "r_eff_decode_t_end": (final_cache_seq_len / full_final_cache_len) if full_final_cache_len > 0 else None,
        "prompt_full_seq_len": prompt_full_seq_len,
        "prompt_retained_seq_len": prompt_retained_seq_len,
        "final_cache_seq_len": final_cache_seq_len,
        "generated_tokens": generated_tokens,
        "decode_steps": trace.decode_call_count,
        "image_tokens_total": None,
        "image_tokens_kept": None,
        "image_keep_ratio": None,
        "image_positions_dropped": None,
        "num_images": len(prepared_image_paths),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def build_lookm_worker(args):
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
    return worker


def prepare_inputs(samples: list[dict], milebench_root: Path, worker) -> dict[str, dict]:
    from utils import MileBenchDataset

    # Group by dataset
    by_dataset: dict[str, list[dict]] = {}
    for s in samples:
        by_dataset.setdefault(s["dataset"], []).append(s)

    prepared: dict[str, dict] = {}
    for dataset_name, ds_samples in by_dataset.items():
        dataset_path = milebench_root / dataset_name / f"{dataset_name}.json"
        image_root = milebench_root / dataset_name / "images"
        core = load_core_annotation(str(dataset_path))
        if core is None:
            continue
        task_instruction = core.get("meta_data", {}).get("task_instruction", "")
        lc_dataset = MileBenchDataset(
            annotation=[s["raw"] for s in ds_samples],
            task_instructions=task_instruction,
            img_dir=str(image_root),
            max_context_len=worker.max_context_len,
            n_tokens_per_image=worker.n_tokens_per_image,
            tokenizer=worker.tokenizer,
            dataset_name=dataset_name,
            combine_image=getattr(worker, "combine_image", None),
        )
        for i, s in enumerate(ds_samples):
            item = lc_dataset[i]
            prepared[s["sample_key"]] = {
                "question": item["context"],
                "image_paths": item["raw_img_list"],
            }
    return prepared


def append_rows_to_csv(rows: list[dict], csv_path: Path):
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
    print(f"Appended {len(rows)} rows to {csv_path} (total rows={len(merged_rows)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--milebench_root", default="/workspace/hd/data/MileBench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--look_kv_mode", default="text_prior_pivot_merge")
    parser.add_argument("--look_hh_ratio", type=float, default=0.10)
    parser.add_argument("--look_recent_ratio", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    milebench_root = Path(args.milebench_root)
    device = torch.device(args.device)

    manifest = json.loads(Path(args.manifest).read_text())
    print(f"Manifest: {len(manifest)} samples")

    samples = load_samples_for_manifest(manifest, milebench_root)
    print(f"Loaded {len(samples)} samples with question text")

    print("Loading LOOK-M worker...")
    worker = build_lookm_worker(args)
    print("LOOK-M worker loaded")

    print("Preparing truncated inputs...")
    prepared = prepare_inputs(samples, milebench_root, worker)
    print(f"Prepared {len(prepared)} samples")

    rows = []
    for i, sample in enumerate(samples):
        key = sample["sample_key"]
        p = prepared.get(key)
        if p is None:
            print(f"[SKIP] No prepared input for {key}")
            continue

        print(f"[{i+1}/{len(samples)}] {sample['dataset']} sid={sample['sample_id']} "
              f"n_imgs={len(p['image_paths'])}")
        row = measure_lookm(
            sample=sample,
            worker=worker,
            device=device,
            method_name="look_m_text_prior_pivot_merge",
            prepared_question=p["question"],
            prepared_image_paths=p["image_paths"],
        )
        if row is not None:
            rows.append(row)
            print(f"  prefill={row['prefill_latency_ms']:.0f}ms  "
                  f"tbt={row['tbt_ms_per_token']:.1f}ms/tok  "
                  f"mem={row['peak_gpu_memory_gib']:.2f}GiB  "
                  f"r_eff={row['r_eff_prompt']:.2f}")
        _empty_cuda_cache()

    csv_path = output_dir / "per_sample.csv"
    append_rows_to_csv(rows, csv_path)
    print(f"\nDone. {len(rows)} LOOK-M rows written.")


if __name__ == "__main__":
    main()
