#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Teacher-forcing PPL evaluation for zap KV eviction methods, using the PrefixKV
protocol (detail_1k.json + mm-vet.json, 1000 / 218 samples, ratio sweep).

Mirrors PrefixKV/eval_ppl.py but runs on zap's HuggingFace LlavaForConditionalGeneration
stack with kvpress presses (Future probe, H2O-prefill, full-cache).

Output: a single JSON at {output_dir}/result.json containing PPL, args, per-sample NLL,
and metadata. Also writes a one-line summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.image_token_press import (
    FutureAllTokenPress,
    FutureSupervisedImagePress,
    H2OAllTokenPress,
    H2OImageOnlyPress,
    HybridH2OFutureAllTokenPress,
    QuadrantEvictionPress,
    VisualUtilityStudentPress,
)

METHODS_NEED_ATTN = {"h2o_image_only", "h2o_all_token", "hybrid_h2o_future_all_token", "quadrant_eviction"}
METHODS_ALL_TOKEN = {"h2o_all_token", "future_all_token", "hybrid_h2o_future_all_token"}
from foresight.image_teacher_utils import DEFAULT_PROMPT_TEMPLATE, build_prompt
from foresight.llava_15b_extractor import (
    _get_model_device,
    _get_model_float_dtype,
    _move_batch_to_device,
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)


def load_samples(data_path: str, image_path: str, eval_samples: int) -> list[dict[str, Any]]:
    """Load mm-vet- or detail_1k-style JSON and normalize to {id, image_file, question, answer}.

    Supported layouts:
      * detail_1k (list): [{id, image, conversations:[{value}, {value}]}, ...]
      * mm-vet v1 (dict): {sample_id: {imagename, question, answer, ...}}
      * mm-vet flat (list): [{id, image, question, answer}, ...]
    """
    with open(data_path) as f:
        raw = json.load(f)
    is_mmvet = "mm-vet" in data_path

    if isinstance(raw, dict):
        items = [{"id": k, **v} for k, v in raw.items()]
    else:
        items = list(raw)

    samples = []
    for item in items:
        if is_mmvet:
            question = item["question"]
            answer = item["answer"]
            image_rel = item.get("image") or item.get("imagename")
            if image_rel and not image_rel.startswith("images/"):
                image_rel = os.path.join("images", image_rel)
        elif "conversations" in item:
            convs = item["conversations"]
            assert len(convs) >= 2, f"Expected ≥2 conv turns, got {len(convs)}"
            question = convs[0]["value"]
            answer = convs[1]["value"]
            image_rel = item["image"]
        else:
            # Flat {image, question, answer} schema (e.g. PrefixKV's
            # rouge-llava-v1.5-7b-detail_1k.json with full-cache predictions).
            question = item["question"]
            answer = item["answer"]
            image_rel = item["image"]
        # Strip any <image> markers from the question — the prompt template adds exactly one.
        question = question.replace("<image>", "").replace("\n\n", "\n").strip()
        image_file = os.path.join(image_path, image_rel)
        samples.append({
            "id": item.get("id", item.get("sample_id")),
            "image_file": image_file,
            "question": question,
            "answer": answer,
        })
    return samples[:eval_samples]


def _resolve_ratio_kwargs(args: argparse.Namespace) -> dict:
    if args.image_keep_ratio is None and args.total_keep_ratio is None:
        raise ValueError(
            "Exactly one of --image-keep-ratio or --total-keep-ratio must be set"
        )
    if args.image_keep_ratio is not None and args.total_keep_ratio is not None:
        raise ValueError(
            "Exactly one of --image-keep-ratio or --total-keep-ratio must be set, not both"
        )
    if args.image_keep_ratio is not None:
        return {"image_keep_ratio": args.image_keep_ratio}
    return {"total_keep_ratio": args.total_keep_ratio}


def build_press(args: argparse.Namespace):
    if args.method == "full":
        return None
    if args.method in ("random_image_only", "random_all_token"):
        # Lazy import: figure-only ablation (EXP-20260426-001-figure).
        import sys as _sys
        _sys.path.insert(0, "/workspace/zap/experiments/EXP-20260426-001-figure")
        from random_press import RandomImageOnlyPress, RandomAllTokenPress
        ratio_kwargs = _resolve_ratio_kwargs(args)
        if args.method == "random_image_only":
            return RandomImageOnlyPress(
                head_reduce=args.head_reduce,
                **ratio_kwargs,
            )
        if "total_keep_ratio" not in ratio_kwargs:
            raise ValueError("--total-keep-ratio is required for random_all_token")
        return RandomAllTokenPress(
            total_keep_ratio=ratio_kwargs["total_keep_ratio"],
            head_reduce=args.head_reduce,
        )
    if args.method == "quadrant_eviction":
        if not args.future_probe_name:
            raise ValueError("--future-probe-name is required for quadrant_eviction")
        quadrant = getattr(args, "quadrant", None)
        if not quadrant:
            raise ValueError("--quadrant is required for quadrant_eviction")
        evict_ratio = getattr(args, "evict_ratio", 0.25) or 0.25
        return QuadrantEvictionPress(
            future_probe_name=args.future_probe_name,
            quadrant=quadrant,
            evict_ratio=evict_ratio,
            head_reduce=args.head_reduce,
        )
    ratio_kwargs = _resolve_ratio_kwargs(args)
    if args.method == "future":
        if not args.future_probe_name:
            raise ValueError("--future-probe-name is required for method=future")
        selected = tuple(args.selected_layer_indices) if args.selected_layer_indices else ()
        if args.strict_selected and not selected:
            raise ValueError(
                "Empty --selected-layer-indices with --strict-selected. "
                "Pass e.g. --selected-layer-indices 24 25 26 27 28 29 30 31."
            )
        return FutureSupervisedImagePress(
            probe_model_name=args.future_probe_name,
            selected_layer_indices=selected,
            head_reduce=args.head_reduce,
            **ratio_kwargs,
        )
    if args.method == "h2o_image_only":
        return H2OImageOnlyPress(
            head_reduce=args.head_reduce,
            **ratio_kwargs,
        )
    if args.method == "h2o_all_token":
        if "total_keep_ratio" not in ratio_kwargs:
            raise ValueError("--total-keep-ratio is required for h2o_all_token")
        return H2OAllTokenPress(
            total_keep_ratio=ratio_kwargs["total_keep_ratio"],
            head_reduce=args.head_reduce,
        )
    if args.method == "future_all_token":
        if "total_keep_ratio" not in ratio_kwargs:
            raise ValueError("--total-keep-ratio is required for future_all_token")
        if not args.future_probe_name:
            raise ValueError("--future-probe-name is required for future_all_token")
        return FutureAllTokenPress(
            total_keep_ratio=ratio_kwargs["total_keep_ratio"],
            head_reduce=args.head_reduce,
            future_probe_name=args.future_probe_name,
        )
    if args.method == "hybrid_h2o_future_all_token":
        if "total_keep_ratio" not in ratio_kwargs:
            raise ValueError("--total-keep-ratio is required for hybrid_h2o_future_all_token")
        if not args.future_probe_name:
            raise ValueError("--future-probe-name is required for hybrid_h2o_future_all_token")
        return HybridH2OFutureAllTokenPress(
            total_keep_ratio=ratio_kwargs["total_keep_ratio"],
            head_reduce=args.head_reduce,
            future_probe_name=args.future_probe_name,
            alpha=args.alpha,
            future_blend_layers=tuple(args.future_blend_layers),
        )
    if args.method == "visual_utility_student":
        if not args.student_model_name:
            raise ValueError("--student-model-name is required for visual_utility_student")
        return VisualUtilityStudentPress(
            student_model_name=args.student_model_name,
            head_reduce=args.head_reduce,
            **ratio_kwargs,
        )
    raise ValueError(f"Unknown method: {args.method}")


@torch.no_grad()
def compute_sample_nll(
    *,
    model: LlavaForConditionalGeneration,
    processor: Any,
    press: Any,
    sample: dict[str, Any],
    prompt_template: str,
    device: torch.device,
    float_dtype: torch.dtype,
    needs_output_attentions: bool,
    max_answer_tokens: int | None,
) -> list[float]:
    """Teacher-force the ground-truth answer and return per-token NLLs (one sync per sample)."""
    prompt_text = build_prompt(sample["question"], prompt_template, image_count=1)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    prompt_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)

    # Drop BOS the same way PrefixKV does (`[:, 1:]`). LLaMA slow tokenizer always prepends it.
    answer_ids = processor.tokenizer.encode(sample["answer"], return_tensors="pt").to(device)[:, 1:]
    if max_answer_tokens is not None:
        answer_ids = answer_ids[:, :max_answer_tokens]
    n_answer = int(answer_ids.shape[1])
    if n_answer == 0:
        return []

    image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
        prompt_inputs=prompt_inputs,
        model_config=model.config,
        num_images=1,
    )
    if press is not None:
        press.set_image_positions(image_positions)
        if hasattr(press, "set_question_positions"):
            last_img = int(image_positions.max().item())
            q_positions = torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)
            press.set_question_positions(q_positions)

    press_ctx = press(model) if press is not None else nullcontext()
    nll_tensors: list[torch.Tensor] = []

    with press_ctx:
        # Prefill: logits at last position predict the first answer token.
        prefill_kwargs: dict[str, Any] = dict(use_cache=True, return_dict=True)
        if needs_output_attentions:
            prefill_kwargs["output_attentions"] = True
        outputs = model(**prompt_inputs, **prefill_kwargs)
        logits_last = outputs.logits[0, -1, :].float()
        nll_tensors.append(F.cross_entropy(
            logits_last.unsqueeze(0), answer_ids[0, 0].unsqueeze(0), reduction="none"
        ))
        cache = outputs.past_key_values

    # Decode runs outside the press context: the forward_hook only fires at prefill
    # (cache_position[-1] > q_len guard), and exiting the context is safe because the
    # cache has already been compressed in-place. Explicit cache_position keeps RoPE
    # aligned with the ORIGINAL prompt length after eviction shortens the cache.
    for i in range(n_answer - 1):
        cache_position = torch.tensor([prompt_len_mm + i], dtype=torch.long, device=device)
        out = model(
            input_ids=answer_ids[:, i:i+1],
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[0, -1, :].float()
        nll_tensors.append(F.cross_entropy(
            logits.unsqueeze(0), answer_ids[0, i+1].unsqueeze(0), reduction="none"
        ))
        cache = out.past_key_values

    # Single GPU→CPU sync per sample.
    return torch.cat(nll_tensors).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    # Core data / model
    parser.add_argument("--model-path", type=str, default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True,
                        help="Root prepended to each sample's `image` field.")
    parser.add_argument("--eval-samples", type=int, default=218)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max-answer-tokens", type=int, default=None,
                        help="Cap per-sample answer length (default: no cap, use full ground truth).")
    parser.add_argument("--output-dir", type=str, required=True)

    # Method / press
    parser.add_argument("--method",
                        choices=["full", "future", "h2o_image_only",
                                 "h2o_all_token", "future_all_token",
                                 "hybrid_h2o_future_all_token",
                                 "quadrant_eviction",
                                 "visual_utility_student",
                                 "random_image_only", "random_all_token"],
                        required=True)
    parser.add_argument("--student-model-name", type=str, default=None,
                        help="Path to VisualUtilityStudent ckpt dir; required for visual_utility_student.")
    parser.add_argument("--quadrant", type=str, default=None,
                        choices=["HH", "HL", "LH", "LL"],
                        help="Quadrant to evict. Only used by quadrant_eviction.")
    parser.add_argument("--evict-ratio", type=float, default=0.25,
                        help="Fraction of ALL tokens to evict per layer. Only used by quadrant_eviction.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid blend weight for H2O (1-alpha for Future). "
                             "Only used by hybrid_h2o_future_all_token.")
    parser.add_argument("--future-blend-layers", type=int, nargs="*", default=[],
                        help="Per-layer gating for hybrid: blend Future only at these layer "
                             "indices (others fall back to H2O-only). Empty = blend every layer.")
    parser.add_argument("--image-keep-ratio", type=float, default=None,
                        help="Fraction of IMAGE tokens kept (text untouched). "
                             "Mutually exclusive with --total-keep-ratio.")
    parser.add_argument("--total-keep-ratio", type=float, default=None,
                        help="Fraction of ALL tokens (text+image) kept. "
                             "Mutually exclusive with --image-keep-ratio.")
    parser.add_argument("--future-probe-name", type=str, default=None)
    parser.add_argument("--selected-layer-indices", type=int, nargs="+",
                        default=None,
                        help="[future mode only] model-layer indices where the probe was trained. "
                             "None = all layers prune (correct for full-model checkpoints).")
    parser.add_argument("--strict-selected", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail if --selected-layer-indices is empty (prevents silent untrained-layer eviction).")
    parser.add_argument("--head-reduce", choices=["amax", "mean"], default="amax")

    # Runtime
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for stochastic eviction baselines.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.method in METHODS_NEED_ATTN and args.attn_implementation != "eager":
        print(f"[WARN] method={args.method} forces --attn-implementation=eager "
              f"(was {args.attn_implementation})")
        args.attn_implementation = "eager"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load samples before heavy model init (fails fast on bad paths).
    samples = load_samples(args.data_path, args.image_path, args.eval_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {args.data_path}")

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    torch_dtype = getattr(torch, args.torch_dtype)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    configure_llava_processor(processor, model.config)
    model = model.to(torch.device(args.device))
    model.eval()

    device = _get_model_device(model)
    float_dtype = _get_model_float_dtype(model)

    press = build_press(args)
    needs_output_attentions = args.method in METHODS_NEED_ATTN

    total_nll = 0.0
    total_tokens = 0
    per_sample_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    t_start = time.time()
    ratio_tag = (f"img_k={args.image_keep_ratio}" if args.image_keep_ratio is not None
                 else f"tot_k={args.total_keep_ratio}" if args.total_keep_ratio is not None
                 else "full")
    iterator = tqdm(samples, desc=f"PPL [{args.method} {ratio_tag}]")
    for sample in iterator:
        try:
            sample_nlls = compute_sample_nll(
                model=model,
                processor=processor,
                press=press,
                sample=sample,
                prompt_template=args.prompt_template,
                device=device,
                float_dtype=float_dtype,
                needs_output_attentions=needs_output_attentions,
                max_answer_tokens=args.max_answer_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sample["id"], "error": repr(exc)})
            continue

        sum_nll = math.fsum(sample_nlls)
        total_nll += sum_nll
        total_tokens += len(sample_nlls)
        per_sample_records.append({
            "id": sample["id"],
            "n_answer_tokens": len(sample_nlls),
            "sum_nll": sum_nll,
        })
        if total_tokens > 0:
            iterator.set_postfix(ppl=f"{math.exp(total_nll / total_tokens):.3f}")

    if total_tokens == 0:
        raise RuntimeError("No answer tokens scored — all samples failed")

    ppl = math.exp(total_nll / total_tokens)
    elapsed = time.time() - t_start

    result = {
        "ppl": ppl,
        "n_samples": len(per_sample_records),
        "n_failures": len(failures),
        "n_total_answer_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "sec_per_sample": elapsed / max(len(per_sample_records), 1),
        "args": vars(args),
        "per_sample": per_sample_records,
        "failures": failures,
    }

    out_path = output_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[DONE] method={args.method} {ratio_tag} "
          f"ds={Path(args.data_path).name} ppl={ppl:.4f} "
          f"n={len(per_sample_records)} fail={len(failures)} "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
