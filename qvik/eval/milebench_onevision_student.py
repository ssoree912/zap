#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone MileBench evaluation using LlavaOnevisionStudent.

Outputs pred.json compatible with /workspace/look-m/evaluate.py and score.py.

Usage:
    python qvik/eval/milebench_onevision_student.py \
        --dataset ActionLocalization \
        --keep_ratio 0.5 \
        --output_dir /workspace/zap/experiments/.../outputs/keep050 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/workspace/zap")

DATA_ROOT = "/workspace/zap/data/MileBench"
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_NEW_TOKENS = 32


def build_prompt(sample: dict, meta: dict) -> str:
    ann = sample["task_instance"]
    task_instruction = meta["task_instruction"][sample["task_instruction_id"]]

    context = ann["context"]
    n_img = len(ann["images_path"])
    for i in range(1, n_img + 1):
        context = context.replace(f"{{image#{i}}}", f"<Image {i}> ")
        context = context.replace(f"{{table#{i}}}", f"<Image {i}> ")

    if ann.get("choice_list"):
        choice_str = "\nChoice List:\n"
        choice_str += "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ann["choice_list"]))
        choice_str += "\nYour answer is: "
        context += choice_str

    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrained", default="/workspace/zap/model/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--student_path", default="/workspace/zap/ckpts/student_onevision_A_ep20")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_out = os.path.join(args.output_dir, args.dataset)
    pred_path = os.path.join(task_out, "pred.json")
    os.makedirs(task_out, exist_ok=True)

    if os.path.exists(pred_path) and not args.overwrite:
        print(f"[skip] {args.dataset}: {pred_path} exists")
        return

    # Load data
    data_path = os.path.join(DATA_ROOT, args.dataset, f"{args.dataset}.json")
    data = json.load(open(data_path))
    meta = data["meta_data"]
    samples = data["data"]
    if args.limit is not None:
        samples = samples[:args.limit]
    combined_img_root = os.path.join(DATA_ROOT, args.dataset, "combined_1_images")

    print(
        f"[{args.dataset}] {len(samples)} samples | keep_ratio={args.keep_ratio} "
        f"keep_ratio_basis=image max_new_tokens={args.max_new_tokens}"
    )

    # Load student model (reuse existing class — no duplication)
    from qvik.eval.lmms_onevision_student import LlavaOnevisionStudent
    model_wrapper = LlavaOnevisionStudent(
        pretrained=args.pretrained,
        student_path=args.student_path,
        keep_ratio=args.keep_ratio,
        device=args.device,
        stats_output_dir=task_out,
        model_format="hf",
    )

    predictions = []

    for sample in tqdm(samples, desc=args.dataset):
        ann = sample["task_instance"]
        prompt_text = build_prompt(sample, meta)

        img_file = ann["combined_1_images"][0]
        img_path = os.path.join(combined_img_root, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {img_path}: {e}", file=sys.stderr)
            predictions.append({
                "sample_id": sample["sample_id"],
                "pred_response": "",
                "gt_response": sample["response"],
            })
            continue

        conversation = [{"role": "user", "content": prompt_text}]
        text = model_wrapper._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = model_wrapper._processor(
            images=[image],
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=model_wrapper.max_length,
        ).to(args.device, torch.float16)

        try:
            answer = model_wrapper._generate_with_student(inputs, [image], args.max_new_tokens)
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"[warn] OOM on {args.dataset} sample {sample['sample_id']}: {e}",
                file=sys.stderr,
                flush=True,
            )
            torch.cuda.empty_cache()
            answer = ""

        predictions.append({
            "sample_id": sample["sample_id"],
            "pred_response": answer,
            "gt_response": sample["response"],
        })
        del inputs, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save pred.json
    json.dump(predictions, open(pred_path, "w"), ensure_ascii=False, indent=2)
    print(f"[{args.dataset}] saved → {pred_path}")

    # Save keep ratio stats
    model_wrapper._save_keep_stats(args.dataset)


if __name__ == "__main__":
    main()
