#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Free-generation ROUGE-L evaluation for zap KV eviction methods, using the PrefixKV
protocol (detail_1k.json + mm-vet.json, ratio sweep).

Mirrors PrefixKV/eval_rouge.py but runs on zap's HF LlavaForConditionalGeneration
stack with kvpress presses (Future probe, H2O-prefill, full-cache).

For each sample:
  1. Prefill with the press → cache is compressed in-place.
  2. Greedy-decode up to --max-new-tokens using the compressed cache.
  3. Compute ROUGE-L F1 vs. the reference answer.

Output: {output_dir}/result.json with mean ROUGE-L, per-sample predictions, args.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_ppl import build_press, load_samples, METHODS_NEED_ATTN
from kvzap.image_teacher_utils import DEFAULT_PROMPT_TEMPLATE, build_prompt
from kvzap.llava_extractor import (
    _get_model_device,
    _get_model_float_dtype,
    _move_batch_to_device,
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)


@torch.no_grad()
def generate_sample(
    *,
    model: LlavaForConditionalGeneration,
    processor: Any,
    press: Any,
    sample: dict[str, Any],
    prompt_template: str,
    device: torch.device,
    float_dtype: torch.dtype,
    needs_output_attentions: bool,
    max_new_tokens: int,
) -> str:
    """Greedy-decode with a compressed cache, return the decoded answer string."""
    prompt_text = build_prompt(sample["question"], prompt_template, image_count=1)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    prompt_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)

    image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
        prompt_inputs=prompt_inputs,
        model_config=model.config,
        num_images=1,
    )
    if press is not None:
        press.set_image_positions(image_positions)

    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id

    press_ctx = press(model) if press is not None else nullcontext()
    with press_ctx:
        prefill_kwargs: dict[str, Any] = dict(use_cache=True, return_dict=True)
        if needs_output_attentions:
            prefill_kwargs["output_attentions"] = True
        outputs = model(**prompt_inputs, **prefill_kwargs)
        cache = outputs.past_key_values
        next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    generated: list[int] = []
    for step in range(max_new_tokens):
        tok_int = int(next_token.item())
        if tok_int == eos_id:
            break
        generated.append(tok_int)
        cache_position = torch.tensor([prompt_len_mm + step], dtype=torch.long, device=device)
        out = model(
            input_ids=next_token,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--eval-samples", type=int, default=218)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--method",
                        choices=["full", "future", "h2o_image_only",
                                 "h2o_all_token", "future_all_token",
                                 "hybrid_h2o_future_all_token",
                                 "quadrant_eviction"],
                        required=True)
    parser.add_argument("--quadrant", type=str, default=None,
                        choices=["HH", "HL", "LH", "LL"])
    parser.add_argument("--evict-ratio", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--future-blend-layers", type=int, nargs="*", default=[])
    parser.add_argument("--image-keep-ratio", type=float, default=None)
    parser.add_argument("--total-keep-ratio", type=float, default=None)
    parser.add_argument("--future-probe-name", type=str, default=None)
    parser.add_argument("--selected-layer-indices", type=int, nargs="+",
                        default=[24, 25, 26, 27, 28, 29, 30, 31])
    parser.add_argument("--strict-selected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head-reduce", choices=["amax", "mean"], default="amax")

    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")

    args = parser.parse_args()

    if args.method in METHODS_NEED_ATTN and args.attn_implementation != "eager":
        print(f"[WARN] method={args.method} forces --attn-implementation=eager "
              f"(was {args.attn_implementation})")
        args.attn_implementation = "eager"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    rouge = Rouge()
    per_sample_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scores_f: list[float] = []

    t_start = time.time()
    ratio_tag = (f"img_k={args.image_keep_ratio}" if args.image_keep_ratio is not None
                 else f"tot_k={args.total_keep_ratio}" if args.total_keep_ratio is not None
                 else "full")
    iterator = tqdm(samples, desc=f"ROUGE [{args.method} {ratio_tag}]")
    for sample in iterator:
        try:
            pred = generate_sample(
                model=model,
                processor=processor,
                press=press,
                sample=sample,
                prompt_template=args.prompt_template,
                device=device,
                float_dtype=float_dtype,
                needs_output_attentions=needs_output_attentions,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sample["id"], "error": repr(exc)})
            continue

        # Rouge lib fails on empty strings — use sentinel token.
        pred_for_rouge = pred if pred.strip() else "<empty>"
        ref_for_rouge = sample["answer"] if sample["answer"].strip() else "<empty>"
        try:
            score = rouge.get_scores(pred_for_rouge, ref_for_rouge)[0]["rouge-l"]["f"]
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sample["id"], "error": f"rouge:{exc!r}", "pred": pred})
            continue

        scores_f.append(score)
        per_sample_records.append({
            "id": sample["id"],
            "pred": pred,
            "answer": sample["answer"],
            "rouge_l_f": score,
        })
        if scores_f:
            iterator.set_postfix(rouge=f"{sum(scores_f)/len(scores_f):.4f}")

    if not scores_f:
        raise RuntimeError("No samples scored — all failed")

    mean_rouge = sum(scores_f) / len(scores_f)
    elapsed = time.time() - t_start
    result = {
        "rouge_l_f_mean": mean_rouge,
        "n_samples": len(per_sample_records),
        "n_failures": len(failures),
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
          f"ds={Path(args.data_path).name} rouge_l={mean_rouge:.4f} "
          f"n={len(per_sample_records)} fail={len(failures)} "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
