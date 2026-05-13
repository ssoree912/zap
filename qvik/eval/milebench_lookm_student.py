#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MileBench eval using original LLaVA repo (LLaVA-mix_merge_v1) + student KV pruning.

Same model/data pipeline as LOOK-M (multi-image, MileBenchDataset truncation),
with our student scoring image tokens and trimming the KV cache.

Usage:
    CUDA_VISIBLE_DEVICES=2 python qvik/eval/milebench_lookm_student.py \
        --dataset CharacterOrder \
        --keep_ratio 1.0 0.5 0.2 \
        --output_dir /workspace/zap/experiments/.../outputs \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
from tqdm import tqdm

LOOKM_LLAVA_ROOT = "/workspace/look-m/LLaVA-mix_merge_v1"
LOOKM_ROOT = "/workspace/look-m"
ZAP_ROOT = "/workspace/zap"
for p in [LOOKM_LLAVA_ROOT, LOOKM_ROOT, ZAP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_ROOT = "/workspace/zap/data/MileBench"
LOOKM_MODEL_DIR = "/workspace/zap/model/llava-1.5-7b-hf"
MAX_CONTEXT_LEN = 4096
N_TOKENS_PER_IMAGE = 576
MAX_NEW_TOKENS = 512

MILEBENCH_DATASETS = [
    "ALFRED", "ActionLocalization", "ActionPrediction", "ActionSequence",
    "CLEVR-Change", "CharacterOrder", "CounterfactualInference", "DocVQA",
    "EgocentricNavigation", "GPR1200", "IEdit", "ImageNeedleInAHaystack",
    "MMCoQA", "MovingAttribute", "MovingDirection", "MultiModalQA",
    "OCR-VQA", "ObjectExistence", "ObjectInteraction", "ObjectShuffle",
    "SceneTransition", "SlideVQA", "Spot-the-Diff", "StateChange",
    "TQA", "TextNeedleInAHaystack", "WebQA", "WikiVQA",
]


def to_int_if_possible(value: Any) -> Any:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return str(value)


def build_look_prediction_record(
    *,
    sample: dict[str, Any],
    question_for_export: str,
    prediction: str,
    image_paths: list[str],
    look_model_name: str,
) -> dict[str, Any]:
    return {
        "sample_id": to_int_if_possible(sample["sample_id"]),
        "image": image_paths,
        "question": question_for_export,
        "gt_response": str(sample.get("response", "")),
        "gen_model_id": look_model_name,
        "pred_response": prediction,
        "gen_kwargs": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.0,
        },
    }


def load_lookm_model(model_dir: str, device: str):
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_dir,
        model_base=None,
        model_name=model_dir,
        device_map=None,
        kv_mode="origin",
    )
    model = model.to(torch.device(device), torch.float16).eval()
    return tokenizer, model, image_processor


def build_prompt(question: str, tokenizer, model) -> torch.Tensor:
    """Build input_ids using LOOK-M's conv template and tokenizer_image_token."""
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    if mm_use_im_start_end:
        from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        single_img_tok = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        single_img_tok = DEFAULT_IMAGE_TOKEN

    # Replace <ImageHere> with the LLaVA image token
    input_prompt = question.replace('<ImageHere>', single_img_tok)
    # Handle CLEVR-Change edge case
    input_prompt = input_prompt.replace(
        single_img_tok + single_img_tok,
        single_img_tok + '\n' + single_img_tok
    )

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors='pt',
    ).unsqueeze(0)
    return input_ids


def get_image_positions_from_input_ids(
    input_ids: torch.Tensor,
    n_img_tokens: int,
    n_tokens_per_image: int = N_TOKENS_PER_IMAGE,
) -> torch.Tensor:
    """Compute image token positions in the merged (multimodal) sequence.

    input_ids has IMAGE_TOKEN_INDEX placeholders. Each expands to n_tokens_per_image.
    Returns a 1-D LongTensor of all image token positions in the merged sequence.
    """
    from llava.constants import IMAGE_TOKEN_INDEX

    ids = input_ids[0]  # [L]
    img_placeholder_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()

    # Build merged position list
    all_image_positions = []
    offset = 0  # extra tokens added so far due to image expansion
    for placeholder_pos in img_placeholder_positions:
        merged_start = placeholder_pos + offset
        positions = list(range(merged_start, merged_start + n_tokens_per_image))
        all_image_positions.extend(positions)
        offset += n_tokens_per_image - 1  # placeholder was 1 token → now n_tokens_per_image

    return torch.tensor(all_image_positions, dtype=torch.long)


@torch.no_grad()
def generate_with_student(
    model,
    tokenizer,
    image_processor,
    student,
    input_ids: torch.Tensor,
    image_paths: list[str],
    keep_ratio: float,
    device: str,
) -> tuple[str, dict]:
    from llava.mm_utils import process_images
    from PIL import Image

    device_t = torch.device(device)
    eos_token_id = tokenizer.eos_token_id

    # Process images
    if image_paths:
        pil_images = [Image.open(p).convert('RGB') for p in image_paths]
        image_tensor = process_images(pil_images, image_processor, model.config).to(device_t, torch.float16)
    else:
        image_tensor = None

    input_ids = input_ids.to(device_t)

    # No pruning: full generate
    if keep_ratio >= 1.0 or image_tensor is None:
        out = model.generate(
            input_ids,
            images=image_tensor,
            use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=eos_token_id,
        )
        # Original LLaVA generate() passes inputs_embeds to super().generate(),
        # so output contains ONLY newly generated tokens (not input).
        answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return answer, {}

    # Prefill with hidden states
    try:
        prefill = model(
            input_ids,
            images=image_tensor,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
    except Exception as e:
        print(f"[warn] prefill failed ({e}), falling back to full generate", file=sys.stderr)
        out = model.generate(input_ids, images=image_tensor, use_cache=True,
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=eos_token_id)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip(), {}

    H_all = prefill.hidden_states    # tuple of [1, seq_len_mm, D], len = n_layers + 1
    past_kv = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    prompt_len_mm = int(prefill.logits.shape[1])  # merged sequence length

    # Compute image positions in the merged sequence
    n_img_placeholders = int((input_ids[0] == -200).sum().item())  # IMAGE_TOKEN_INDEX = -200
    image_positions = get_image_positions_from_input_ids(input_ids, n_img_placeholders)
    image_positions = image_positions.to(device_t)

    n_img = image_positions.numel()
    n_text = prompt_len_mm - n_img

    # How many image tokens to keep (total-token basis)
    total_keep = int(torch.ceil(torch.tensor(keep_ratio * prompt_len_mm)).item())
    n_keep = min(n_img, max(0, total_keep - n_text))

    stats = {
        "n_image_original": n_img,
        "n_image_kept": n_keep,
        "n_text": n_text,
        "prompt_len": prompt_len_mm,
        "image_keep_ratio": n_keep / max(1, n_img),
        "total_keep_ratio": (n_text + n_keep) / max(1, prompt_len_mm),
    }

    if n_keep >= n_img:
        # No pruning needed — just decode
        answer_ids = _greedy_decode(model, past_kv, next_token, eos_token_id, MAX_NEW_TOKENS)
        return tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip(), stats

    # Question positions (text tokens AFTER last image token)
    last_img = int(image_positions.max().item())
    q_positions = (
        torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long, device=device_t)
        if last_img + 1 < prompt_len_mm else torch.empty(0, dtype=torch.long, device=device_t)
    )

    # Score & build keep masks per student layer
    keep_masks: dict[int, torch.Tensor] = {}
    for li in student.layer_indices:
        H_l = H_all[li + 1]  # [1, seq_len_mm, D]
        scores = student.forward_layer(li, H_l, image_positions, q_positions).squeeze(0)  # [n_img]
        top = torch.topk(scores, k=n_keep, largest=True).indices
        image_keep = torch.zeros(n_img, dtype=torch.bool, device=device_t)
        image_keep[top] = True
        mask = torch.ones(prompt_len_mm, dtype=torch.bool, device=device_t)
        mask[image_positions] = image_keep
        keep_masks[li] = mask.cpu()

    del H_all

    # Trim KV cache
    from qvik.eval.vlmeval_onevision_student import _trim_kv_cache_per_layer
    past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

    answer_ids = _greedy_decode(model, past_kv, next_token, eos_token_id, MAX_NEW_TOKENS)
    torch.cuda.empty_cache()
    return tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip(), stats


@torch.no_grad()
def _greedy_decode(model, past_kv, next_token, eos_token_id, max_new_tokens):
    generated = []
    cur_token = next_token
    for _ in range(max_new_tokens):
        tok_id = int(cur_token[0, 0].item())
        if eos_token_id is not None and tok_id == eos_token_id:
            break
        generated.append(tok_id)
        out = model(
            input_ids=cur_token,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out.past_key_values
        cur_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return torch.tensor(generated, dtype=torch.long)


def run_dataset(
    dataset: str,
    tokenizer,
    model,
    image_processor,
    student,
    output_dir: str,
    keep_ratio: float,
    device: str,
    overwrite: bool,
    limit: int | None,
    look_model_name: str,
):
    from utils import MileBenchDataset
    from torch.utils.data import DataLoader

    task_out = os.path.join(output_dir, dataset)
    pred_path = os.path.join(task_out, "pred.json")
    os.makedirs(task_out, exist_ok=True)

    if os.path.exists(pred_path) and not overwrite:
        print(f"[skip] {dataset}: already done")
        return

    data_path = os.path.join(DATA_ROOT, dataset, f"{dataset}.json")
    core = json.load(open(data_path))
    samples_raw = core["data"]
    if limit is not None:
        samples_raw = samples_raw[:limit]

    img_dir = os.path.join(DATA_ROOT, dataset, "images")

    # Group by image count (same as LOOK-M generate.py)
    from collections import defaultdict
    groups: dict[int, list] = defaultdict(list)
    for s in samples_raw:
        n = len(s["task_instance"]["images_path"])
        groups[n].append(s)

    predictions = []
    keep_stats = []

    for n_img, sub_data in sorted(groups.items()):
        print(f"[{dataset}] {n_img}-image samples: {len(sub_data)}")
        mb_dataset = MileBenchDataset(
            annotation=sub_data,
            task_instructions=core["meta_data"]["task_instruction"],
            img_dir=img_dir,
            max_context_len=MAX_CONTEXT_LEN,
            n_tokens_per_image=N_TOKENS_PER_IMAGE,
            tokenizer=tokenizer,
            dataset_name=dataset,
            combine_image=None,
        )

        for idx in tqdm(range(len(mb_dataset)), desc=f"{dataset}(n={n_img})"):
            item = mb_dataset[idx]
            sample = sub_data[idx]
            question = item["context"]
            image_paths = item["raw_img_list"]

            try:
                input_ids = build_prompt(question, tokenizer, model)
            except Exception as e:
                print(f"[warn] build_prompt failed for {sample['sample_id']}: {e}", file=sys.stderr)
                predictions.append(build_look_prediction_record(
                    sample=sample,
                    question_for_export=question,
                    prediction="",
                    image_paths=image_paths,
                    look_model_name=look_model_name,
                ))
                continue

            try:
                answer, stats = generate_with_student(
                    model, tokenizer, image_processor, student,
                    input_ids, image_paths, keep_ratio, device,
                )
            except Exception as e:
                print(f"[warn] generate failed for {sample['sample_id']}: {e}", file=sys.stderr)
                answer, stats = "", {}

            if stats:
                keep_stats.append(stats)
            predictions.append(build_look_prediction_record(
                sample=sample,
                question_for_export=question,
                prediction=answer,
                image_paths=image_paths,
                look_model_name=look_model_name,
            ))

    json.dump(predictions, open(pred_path, "w"), ensure_ascii=False, indent=2)
    print(f"[{dataset}] saved → {pred_path} ({len(predictions)} samples)")

    # Run LOOK-M evaluate.py
    import subprocess
    subprocess.run([
        "/opt/conda/envs/vflowopt_chartqa_eval/bin/python",
        "/workspace/look-m/evaluate.py",
        "--data-dir", DATA_ROOT,
        "--dataset", dataset,
        "--result-dir", output_dir,
    ], check=False)

    if keep_stats:
        n = len(keep_stats)
        summary = {
            "task": dataset, "keep_ratio": keep_ratio, "n_samples": n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in keep_stats) / n,
        }
        stats_path = os.path.join(task_out, "keep_ratio_stats.json")
        json.dump(summary, open(stats_path, "w"), indent=2)
        print(f"[{dataset}] avg_image_keep_ratio={summary['avg_image_keep_ratio']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CharacterOrder")
    parser.add_argument("--model_dir", default=LOOKM_MODEL_DIR)
    parser.add_argument("--student_path", default="/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4")
    parser.add_argument("--keep_ratio", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--look_model_name", default="foresight_lookm_student")
    args = parser.parse_args()

    print(f"[load] LOOK-M LLaVA model from {args.model_dir}")
    tokenizer, model, image_processor = load_lookm_model(args.model_dir, args.device)

    from kvpress.presses.visual_utility_student import VisualUtilityStudent
    student = VisualUtilityStudent.from_pretrained(args.student_path)
    student = student.to(device=args.device, dtype=torch.float16).eval()
    print(f"[load] student={args.student_path} layers={student.layer_indices}")

    datasets = MILEBENCH_DATASETS if args.dataset == "all" else [args.dataset]

    for keep_ratio in args.keep_ratio:
        ratio_tag = f"keep{str(keep_ratio).replace('.', '')}"
        out_dir = os.path.join(args.output_dir, ratio_tag)
        look_model_name = f"{args.look_model_name}_keep{keep_ratio:g}"
        print(f"\n=== keep_ratio={keep_ratio} → {out_dir} ===")
        for dataset in datasets:
            try:
                run_dataset(
                    dataset, tokenizer, model, image_processor, student,
                    out_dir, keep_ratio, args.device, args.overwrite, args.limit,
                    look_model_name,
                )
            except Exception as e:
                print(f"[error] {dataset}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
