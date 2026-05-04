#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step decode latency + peak GPU memory bench for LLaVA-OneVision-Qwen2-7B
with synthetic KV-cache truncation at total_keep_ratio=1.0/0.5/0.2.

For each sample we:
  1. Prefill manually (model.forward with use_cache=True), record prefill_ms.
  2. If ratio < 1.0: truncate every layer's K/V to retain all text positions
     plus the first ceil(ratio*L_p)-n_text image positions. The selection is
     deterministic (positional) -- it determines the SAME cache size that the
     trained student would produce, which is what governs decode latency.
  3. Run a manual decode loop calling model.forward(input_ids=last,
     past_key_values=cache) for max_new_tokens steps (or until EOS), measuring
     per-step latency and tracking peak memory.

This decouples actual student scoring from the cost model; FLOPS analysis
is already in `bench_original_onevision_flops.py`.
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
from typing import Any, Iterable, Optional

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

# Reuse FLOPS calculators from the analytical bench
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_original_onevision_flops import (  # noqa: E402
    Qwen2Flops, SiglipFlops, make_qwen2_flops, make_siglip_flops_default,
)

GIB = 1024 ** 3
TFLOPS = 1e12


def patch_siglip_loader(local_siglip_path: str) -> None:
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


@dataclass(frozen=True)
class PooledSample:
    dataset: str
    sample_id: str
    question: str
    image_paths: tuple[str, ...]

    @property
    def sample_key(self) -> str:
        return f"{self.dataset}:{self.sample_id}"


# ----------------------------------------------------------------------
# KV cache helpers (DynamicCache layout from transformers >=4.36 OR legacy tuple)
# ----------------------------------------------------------------------


def iter_kv_pairs(past_kv: Any):
    """Yield (k, v) tensors for each layer in order.

    Supports DynamicCache (.layers[i].keys/.values), older Cache objects
    (.key_cache/.value_cache lists), and legacy tuple-of-tuples format.
    """
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            yield layer.keys, layer.values
        return
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for k, v in zip(past_kv.key_cache, past_kv.value_cache):
            yield k, v
        return
    if isinstance(past_kv, (tuple, list)):
        for layer in past_kv:
            yield layer[0], layer[1]
        return
    raise RuntimeError(f"Unsupported KV cache layout: {type(past_kv)}")


def cache_seq_len(past_kv: Any) -> int:
    for k, _ in iter_kv_pairs(past_kv):
        return int(k.shape[2])
    return 0


def kv_cache_bytes(past_kv: Any) -> int:
    total = 0
    for k, v in iter_kv_pairs(past_kv):
        total += int(k.numel() * k.element_size() + v.numel() * v.element_size())
    return total


def truncate_kv(past_kv: Any, keep_positions: torch.Tensor) -> Any:
    """Return a NEW past_kv with K/V sliced at `keep_positions` (works for both
    Cache objects and legacy tuple format)."""
    new_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in iter_kv_pairs(past_kv):
        keep_dev = keep_positions.to(k.device)
        new_k = k.index_select(dim=2, index=keep_dev).contiguous()
        new_v = v.index_select(dim=2, index=keep_dev).contiguous()
        new_pairs.append((new_k, new_v))

    if hasattr(past_kv, "layers"):
        for layer, (nk, nv) in zip(past_kv.layers, new_pairs):
            layer.keys = nk
            layer.values = nv
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = int(nk.shape[2])
        if hasattr(past_kv, "_seen_tokens"):
            past_kv._seen_tokens = int(new_pairs[0][0].shape[2])
        return past_kv
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i, (nk, nv) in enumerate(new_pairs):
            past_kv.key_cache[i] = nk
            past_kv.value_cache[i] = nv
        if hasattr(past_kv, "_seen_tokens"):
            past_kv._seen_tokens = int(new_pairs[0][0].shape[2])
        return past_kv
    # legacy tuple of tuples
    return tuple(new_pairs)


# ----------------------------------------------------------------------
# Sampling reuse: read the 100-sample manifest written by the FLOPS bench
# ----------------------------------------------------------------------


def load_samples_from_manifest(manifest_path: Path) -> list[PooledSample]:
    payload = json.loads(manifest_path.read_text())
    samples: list[PooledSample] = []
    for item in payload["samples"]:
        samples.append(
            PooledSample(
                dataset=item["dataset"],
                sample_id=str(item["sample_id"]),
                question=item["question"],
                image_paths=tuple(item["image_paths"]),
            )
        )
    return samples


def build_qwen_prompt(question: str, conv_template: str = "qwen_1_5") -> str:
    import copy
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"<image>\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------
# Per-sample measurement: prefill -> truncate (optional) -> decode loop
# ----------------------------------------------------------------------


def compute_n_image_keep(n_img: int, n_text: int, total_keep_ratio: float) -> int:
    total_tokens = n_text + n_img
    total_keep = int(math.ceil(total_keep_ratio * total_tokens))
    n_image_keep = total_keep - n_text
    return min(n_img, max(0, n_image_keep))


@torch.no_grad()
def measure_one(
    *,
    sample: PooledSample,
    method: str,
    ratio: float,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # ---- Build inputs ----
    with Image.open(sample.image_paths[0]) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    prompt = build_qwen_prompt(sample.question, args.conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    raw_ids = input_ids[0].detach().cpu().tolist()
    placeholders = [i for i, t in enumerate(raw_ids) if int(t) == IMAGE_TOKEN_INDEX]
    if len(placeholders) != 1:
        raise ValueError(f"Expected 1 image placeholder, found {len(placeholders)}")
    image_start_raw = int(placeholders[0])

    # ---- Prefill via prepare_inputs_labels_for_multimodal + model.model forward ----
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
        max_new_tokens=1,                 # only the first token; we'll re-decode manually after pruning
        use_cache=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    if not hasattr(out, "past_key_values") or out.past_key_values is None:
        raise RuntimeError("generate() did not expose past_key_values; cannot continue")
    past_kv = out.past_key_values

    L_p = cache_seq_len(past_kv)
    image_feature_len = L_p - (input_ids.shape[1] - 1)
    image_start = image_start_raw  # placeholder stays at the same offset before image expansion
    n_text = L_p - image_feature_len
    n_img = image_feature_len

    # ---- Optional KV truncation ----
    n_image_keep = n_img
    retained_seq_len = L_p
    if ratio < 1.0:
        n_image_keep = compute_n_image_keep(n_img, n_text, ratio)
        keep = torch.zeros(L_p, dtype=torch.bool, device=device)
        keep[:image_start] = True
        keep[image_start + image_feature_len:] = True
        if n_image_keep > 0:
            keep[image_start:image_start + n_image_keep] = True
        keep_positions = keep.nonzero(as_tuple=False).squeeze(-1).contiguous()
        retained_seq_len = int(keep_positions.numel())
        past_kv = truncate_kv(past_kv, keep_positions)

    # ---- Manual decode loop ----
    first_token = out.sequences[:, -1:] if out.sequences.shape[1] > input_ids.shape[1] else out.sequences[:, -1:]
    out_ids = [int(first_token.item())]
    next_token = first_token.to(device)
    eos_id = int(getattr(tokenizer, "eos_token_id", 2) or 2)
    pos = retained_seq_len
    cache_pos = torch.zeros(1, dtype=torch.long, device=device)
    decode_step_ms: list[float] = []
    decode_cache_lens: list[int] = []

    if args.stop_on_eos and out_ids[0] == eos_id:
        pass
    else:
        for _ in range(args.max_new_tokens - 1):
            cache_pos[0] = pos
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step = model(
                input_ids=next_token,
                past_key_values=past_kv,
                cache_position=cache_pos,
                position_ids=cache_pos.unsqueeze(0),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            torch.cuda.synchronize()
            decode_step_ms.append((time.perf_counter() - t1) * 1000.0)
            past_kv = step.past_key_values
            decode_cache_lens.append(cache_seq_len(past_kv))
            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tok = int(next_token.item())
            out_ids.append(tok)
            pos += 1
            if args.stop_on_eos and tok == eos_id:
                break

    torch.cuda.synchronize()
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / GIB

    final_kv_bytes = kv_cache_bytes(past_kv)
    decoded = tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    # ---- Analytical FLOPS using actual L_p, retained_prompt_len, T_decode ----
    qwen_flops: Qwen2Flops = args._qwen_flops
    vit_flops: SiglipFlops = args._siglip_flops
    T = len(out_ids)
    prefill_flops = qwen_flops.prefill(L_p)
    decode_flops_full = qwen_flops.decode_total(L_p, T)
    decode_flops_kept = qwen_flops.decode_total(L_p, T, retained_prompt_len=retained_seq_len)
    vision_flops = vit_flops.total_per_crop()  # 1 image × n_crops collapsed in OneVision config

    return {
        "method": method,
        "ratio": ratio,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        "prompt_full_seq_len": int(L_p),
        "n_text": int(n_text),
        "n_img": int(n_img),
        "n_image_keep": int(n_image_keep),
        "retained_prompt_len": int(retained_seq_len),
        "t_decode": len(out_ids),
        "decode_steps": len(decode_step_ms),
        "prefill_latency_ms": float(prefill_ms),
        "decode_latency_ms_per_token": float(mean(decode_step_ms)) if decode_step_ms else 0.0,
        "decode_total_ms": float(sum(decode_step_ms)),
        "decode_step_ms": decode_step_ms,
        "decode_cache_lens": decode_cache_lens,
        "peak_gpu_memory_gib": float(peak_alloc_gib),
        "peak_gpu_reserved_gib": float(peak_reserved_gib),
        "kv_cache_gib": final_kv_bytes / GIB,
        "prefill_flops": int(prefill_flops),
        "decode_flops": int(decode_flops_kept),
        "decode_flops_full_ref": int(decode_flops_full),
        "vision_flops": int(vision_flops),
        "total_llm_flops": int(prefill_flops + decode_flops_kept),
        "total_flops": int(prefill_flops + decode_flops_kept + vision_flops),
        "prefill_tflops": prefill_flops / TFLOPS,
        "decode_tflops": decode_flops_kept / TFLOPS,
        "vision_tflops": vision_flops / TFLOPS,
        "total_tflops": (prefill_flops + decode_flops_kept + vision_flops) / TFLOPS,
        "prediction": decoded,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OneVision decode latency / memory bench")
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    p.add_argument("--model-name", default="llava_qwen")
    p.add_argument("--conv-template", default="qwen_1_5")
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--stop-on-eos", action=argparse.BooleanOptionalAction, default=False,
                   help="If False (default), force fixed-T decoding to compare latency at equal step counts.")
    p.add_argument("--warmup-samples", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    drop_keys = {"decode_step_ms", "decode_cache_lens"}
    keep_rows = [{k: v for k, v in r.items() if k not in drop_keys} for r in rows]
    fieldnames = list(keep_rows[0].keys())
    extras = sorted({k for r in keep_rows for k in r.keys()} - set(fieldnames))
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keep_rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "prompt_full_seq_len", "n_text", "n_img", "n_image_keep", "retained_prompt_len",
        "t_decode", "decode_steps",
        "prefill_latency_ms", "decode_latency_ms_per_token", "decode_total_ms",
        "peak_gpu_memory_gib", "peak_gpu_reserved_gib", "kv_cache_gib",
        "prefill_tflops", "decode_tflops", "vision_tflops", "total_tflops",
    ]
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    full = by_method.get("full_cache_100", [])
    full_by_key = {r["sample_key"]: r for r in full}
    out: list[dict[str, Any]] = []
    for method, mrows in by_method.items():
        item: dict[str, Any] = {
            "method": method,
            "ratio": mrows[0]["ratio"],
            "n_samples": len(mrows),
        }
        for f in fields:
            vals = [r[f] for r in mrows if r.get(f) is not None]
            item[f"{f}_mean"] = float(mean(vals)) if vals else None
            item[f"{f}_std"] = float(stdev(vals)) if len(vals) > 1 else 0.0
        if method != "full_cache_100" and full_by_key:
            paired = [(r, full_by_key[r["sample_key"]]) for r in mrows if r["sample_key"] in full_by_key]
            if paired:
                item["paired_n_vs_full"] = len(paired)
                d = [r["decode_latency_ms_per_token"] / b["decode_latency_ms_per_token"]
                     for r, b in paired if b["decode_latency_ms_per_token"] > 0]
                m = [r["peak_gpu_memory_gib"] / b["peak_gpu_memory_gib"]
                     for r, b in paired if b["peak_gpu_memory_gib"] > 0]
                kvr = [r["kv_cache_gib"] / b["kv_cache_gib"] for r, b in paired if b["kv_cache_gib"] > 0]
                d_flops = [r["decode_flops"] / b["decode_flops"] for r, b in paired if b["decode_flops"] > 0]
                t_flops = [r["total_flops"] / b["total_flops"] for r, b in paired if b["total_flops"] > 0]
                item["decode_latency_pct_of_full"] = float(100.0 * mean(d)) if d else None
                item["peak_gpu_memory_pct_of_full"] = float(100.0 * mean(m)) if m else None
                item["kv_cache_pct_of_full"] = float(100.0 * mean(kvr)) if kvr else None
                item["decode_flops_pct_of_full"] = float(100.0 * mean(d_flops)) if d_flops else None
                item["total_flops_pct_of_full"] = float(100.0 * mean(t_flops)) if t_flops else None
        out.append(item)
    return sorted(out, key=lambda x: float(x["ratio"]), reverse=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    patch_siglip_loader("/workspace/zap/ckpts/siglip-so400m-patch14-384")

    print(
        f"[load] model={args.model_path} model_name={args.model_name} attn={args.attn_implementation}",
        flush=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
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

    samples = load_samples_from_manifest(Path(args.manifest_path))
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"[data] {len(samples)} samples from {args.manifest_path}", flush=True)

    # Build FLOPS calculators once and stash on args for measure_one to read
    args._qwen_flops = make_qwen2_flops(model.config)
    args._siglip_flops = make_siglip_flops_default()
    print(
        f"[flops] LLM N={args._qwen_flops.num_layers} D={args._qwen_flops.hidden_size} "
        f"K={args._qwen_flops.num_key_value_heads} | siglip per_crop={args._siglip_flops.total_per_crop() / TFLOPS:.4f} TFLOPS",
        flush=True,
    )

    ratios = sorted(set(float(r) for r in args.ratios), reverse=True)
    method_specs: list[tuple[str, float]] = []
    for r in ratios:
        if math.isclose(r, 1.0):
            method_specs.append(("full_cache_100", 1.0))
        else:
            method_specs.append((f"keep_{int(round(r * 100)):03d}", r))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for method, r in method_specs:
        if args.warmup_samples > 0:
            for s in samples[: args.warmup_samples]:
                try:
                    measure_one(
                        sample=s, method="warmup", ratio=r,
                        model=model, tokenizer=tokenizer, image_processor=image_processor,
                        device=device, args=args,
                    )
                except Exception:
                    pass
                empty_cuda()
        for s in tqdm(samples, desc=f"Measuring {method}"):
            try:
                row = measure_one(
                    sample=s, method=method, ratio=r,
                    model=model, tokenizer=tokenizer, image_processor=image_processor,
                    device=device, args=args,
                )
                rows.append(row)
                write_csv(out_dir / "per_sample.csv", rows)
            except torch.cuda.OutOfMemoryError as exc:
                failures.append({"sample_key": s.sample_key, "method": method, "error": f"CUDA OOM: {exc}"})
                empty_cuda()
                if not args.continue_on_error:
                    raise
            except Exception as exc:  # noqa: BLE001
                failures.append({"sample_key": s.sample_key, "method": method, "error": repr(exc)})
                empty_cuda()
                if not args.continue_on_error:
                    raise

    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
