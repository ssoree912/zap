#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MileBench evaluation for LLaVA-1.5-7B with student KV pruning.

Usage:
    python foresight/eval/milebench_llava15_student.py \
        --dataset ActionLocalization \
        --student_path /workspace/zap/ckpts/student_llava15_mmvet \
        --keep_ratio 0.5 \
        --output_dir /workspace/zap/experiments/.../outputs/llava15_keep050 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/workspace/zap")

from kvpress.presses.visual_utility_student import VisualUtilityStudent
from kvpress.presses.image_token_press import VisualUtilityStudentPress
from foresight.llava_15b_extractor import (
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)
from foresight.eval.vlmeval_onevision_student import (
    _greedy_decode_with_kv,
    _trim_kv_cache_per_layer,
)

DATA_ROOT = "/workspace/zap/data/MileBench"
LLAVA15_CKPT = "/workspace/zap/ckpts/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 32
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{(?:image|table)#\d+\}")
LEGACY_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"<ImageHere>", re.IGNORECASE)

LOOKM_ROOT = "/workspace/look-m"
LOOKM_MAX_CONTEXT_LEN = 4096
LOOKM_N_TOKENS_PER_IMAGE = 576


def to_int_if_possible(value):
    try:
        return int(value)
    except Exception:
        return str(value)


def replace_image_placeholders_for_export(text: str) -> str:
    return IMAGE_PLACEHOLDER_PATTERN.sub("<ImageHere>", text)


def build_look_prediction_record(
    *,
    sample: dict,
    question_for_export: str,
    prediction: str,
    image_paths: list[str],
    look_model_name: str,
) -> dict:
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


def _choice_label(index: int) -> str:
    if index < 26:
        return chr(65 + index)
    if index < 52:
        return "A" + chr(65 + index - 26)
    return "B" + chr(65 + index - 52)


def _build_choice_block(choice_list, dataset_name: str) -> str:
    if not isinstance(choice_list, list) or not choice_list:
        return ""
    lines = []
    for idx, choice in enumerate(choice_list):
        if dataset_name == "GPR1200":
            lines.append(str(choice))
        else:
            lines.append(f"{_choice_label(idx)}. {choice}")
    return "\nChoice list: \n" + "\n".join(lines) + "\nYour answer is: "


def _resolve_task_instruction_str(task_instructions, task_instruction_id) -> str:
    if isinstance(task_instructions, list):
        try:
            return str(task_instructions[int(task_instruction_id)])
        except Exception:
            return ""
    if isinstance(task_instructions, dict):
        return str(task_instructions.get(task_instruction_id, task_instructions.get(str(task_instruction_id), "")))
    if task_instructions:
        return str(task_instructions)
    return ""


def build_look_question(record: dict, meta: dict, dataset_name: str) -> str:
    """Build the LOOK-M style question string (no image tokens injected yet)."""
    task = record.get("task_instance", {})
    context = str(task.get("context", "")).strip()
    task_instruction = _resolve_task_instruction_str(
        meta.get("task_instruction", ""),
        record.get("task_instruction_id", 0),
    )
    choice_block = _build_choice_block(task.get("choice_list"), dataset_name=dataset_name)
    combined = f"{task_instruction}\n{context}{choice_block}"
    return combined.strip()


def prepare_lookm_truncated_inputs(
    samples: list[dict],
    *,
    dataset: str,
    meta: dict,
    image_root: str,
    tokenizer,
    max_context_len: int = LOOKM_MAX_CONTEXT_LEN,
    n_tokens_per_image: int = LOOKM_N_TOKENS_PER_IMAGE,
) -> dict[str, dict]:
    """Use MileBenchDataset from LOOK-M to produce truncated inputs per sample.

    Returns a dict mapping str(sample_id) → {question, image_paths}.
    """
    if LOOKM_ROOT not in sys.path:
        sys.path.insert(0, LOOKM_ROOT)
    from utils import MileBenchDataset  # noqa: PLC0415

    annotations = [s["task_instance"] if "task_instance" not in s else s for s in samples]
    raw_annotations = []
    for s in samples:
        raw = s if "task_instance" in s else s.get("raw", s)
        raw_annotations.append(raw)

    task_instructions = meta.get("task_instruction", "")
    mb_dataset = MileBenchDataset(
        annotation=raw_annotations,
        task_instructions=task_instructions,
        img_dir=image_root,
        max_context_len=max_context_len,
        n_tokens_per_image=n_tokens_per_image,
        tokenizer=tokenizer,
        dataset_name=dataset,
        combine_image=None,
    )

    prepared: dict[str, dict] = {}
    for idx, sample in enumerate(samples):
        item = mb_dataset[idx]
        prepared[str(sample["sample_id"])] = {
            "question": item["context"],
            "image_paths": item["raw_img_list"],
        }
    return prepared


def _inject_image_tokens(question: str, image_count: int) -> str:
    question = str(question).strip()
    image_count = max(0, int(image_count))
    marker = "__KVZAP_IMAGE_MARKER__"

    question = IMAGE_PLACEHOLDER_PATTERN.sub(marker, question)
    question = LEGACY_IMAGE_PLACEHOLDER_PATTERN.sub(marker, question)
    question = question.replace("<image>", marker)

    marker_count = question.count(marker)
    if marker_count < image_count:
        prefix = "\n".join([marker] * (image_count - marker_count))
        question = f"{prefix}\n{question}" if question else prefix
    elif marker_count > image_count:
        question = question.replace(marker, "", marker_count - image_count)

    question = question.replace(marker, "<image>\n")
    if image_count > 0 and "<image>" not in question:
        prefix = "\n".join(["<image>"] * image_count)
        question = f"{prefix}\n{question}" if question else prefix

    question = re.sub(r"[ \t]+\n", "\n", question)
    question = re.sub(r"\n{3,}", "\n\n", question)
    return question.strip()


def _build_look_prompt(question: str, image_count: int) -> str:
    """Wrap a LOOK-M question while preserving placeholder image positions."""
    question = _inject_image_tokens(question, image_count=image_count)
    return f"USER: {question}\nASSISTANT:"


def _resolve_task_instruction(meta: dict, sample: dict) -> str:
    task_instructions = meta.get("task_instruction", "")
    task_instruction_id = sample.get("task_instruction_id", 0)
    if isinstance(task_instructions, list):
        try:
            return str(task_instructions[int(task_instruction_id)])
        except Exception:
            return ""
    if isinstance(task_instructions, dict):
        return str(task_instructions.get(task_instruction_id, task_instructions.get(str(task_instruction_id), "")))
    return str(task_instructions)


def build_prompt(sample: dict, meta: dict, image_count: int, image_column: str) -> str:
    ann = sample["task_instance"]
    task_instruction = _resolve_task_instruction(meta, sample)

    context = str(ann["context"]).strip()

    question = f"{task_instruction}\n{context}".strip()

    if ann.get("choice_list"):
        choice_str = "\nChoice list: \n"
        choice_str += "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ann["choice_list"]))
        choice_str += "\nYour answer is: "
        question += choice_str

    question = _inject_image_tokens(question, image_count=image_count)
    return f"USER: {question}\nASSISTANT:"


def resolve_image_paths(dataset: str, ann: dict, image_column: str) -> list[str]:
    image_value = ann.get(image_column)
    if not image_value:
        image_value = ann.get("images_path") or ann.get("combined_1_images")
    if not image_value:
        raise KeyError("sample has no image path field")

    if isinstance(image_value, (list, tuple)):
        raw_paths = [str(path).strip() for path in image_value if str(path).strip()]
    else:
        raw_paths = [str(image_value).strip()]

    image_root_name = "combined_1_images" if image_column == "combined_1_images" else "images"
    image_root = os.path.join(DATA_ROOT, dataset, image_root_name)
    return [
        raw_path if os.path.isabs(raw_path) else os.path.join(image_root, raw_path)
        for raw_path in raw_paths
    ]


def open_images(image_paths: list[str]):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images[0] if len(images) == 1 else images


def compute_n_image_keep(n_img: int, n_text: int, total_keep_ratio: float) -> int:
    total_keep = int(torch.ceil(torch.tensor(float(total_keep_ratio) * float(n_img + n_text))).item())
    return min(n_img, max(0, total_keep - n_text))


def question_positions_after_images(image_positions: torch.Tensor, prompt_len: int) -> torch.Tensor:
    if image_positions.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len, dtype=torch.long)


def build_keep_stats(image_positions: torch.Tensor, prompt_len: int, keep_ratio: float, num_images: int) -> dict:
    n_img = int(image_positions.numel())
    n_text = int(prompt_len) - n_img
    n_keep = compute_n_image_keep(n_img, n_text, keep_ratio)
    return {
        "n_image_original": n_img,
        "n_image_kept": n_keep,
        "n_images": num_images,
        "n_text": n_text,
        "prompt_len": int(prompt_len),
        "image_token_ratio": n_img / max(1, int(prompt_len)),
        "text_token_ratio": n_text / max(1, int(prompt_len)),
        "total_keep_ratio": (n_text + n_keep) / max(1, int(prompt_len)),
        "image_keep_ratio": n_keep / max(1, n_img),
    }


@torch.no_grad()
def generate_answer(
    model,
    processor,
    student,
    press,
    inputs,
    keep_ratio: float,
    num_images: int,
    device: str,
) -> tuple[str, dict]:
    prompt_len_text = int(inputs["input_ids"].shape[1])
    eos_token_id = processor.tokenizer.eos_token_id or 2

    if press is not None:
        try:
            image_positions, prompt_len = infer_llava_image_positions_no_forward(
                prompt_inputs={"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                model_config=model.config,
                num_images=num_images,
            )
            q_positions = question_positions_after_images(image_positions, prompt_len)
            stats = build_keep_stats(image_positions, prompt_len, keep_ratio, num_images)
            press.set_image_positions(image_positions)
            press.set_question_positions(q_positions)
            try:
                with press(model):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=eos_token_id,
                    )
            finally:
                press.clear_sample_context()
            return processor.tokenizer.decode(out[0][prompt_len_text:], skip_special_tokens=True).strip(), stats
        except Exception as exc:
            press.clear_sample_context()
            print(f"[warn] press generate failed; falling back to full generate: {exc}", file=sys.stderr)
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=eos_token_id,
            )
            return processor.tokenizer.decode(out[0][prompt_len_text:], skip_special_tokens=True).strip(), {}

    if keep_ratio >= 1.0:
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            use_cache=True, pad_token_id=eos_token_id,
        )
        return processor.tokenizer.decode(out[0][prompt_len_text:], skip_special_tokens=True).strip(), {}

    try:
        prefill = model(
            **inputs, use_cache=True, output_hidden_states=True,
            output_attentions=False, return_dict=True,
        )
    except Exception:
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        return processor.tokenizer.decode(out[0][prompt_len_text:], skip_special_tokens=True).strip(), {}

    H_all = prefill.hidden_states
    past_kv = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prompt_len_mm = int(prefill.logits.shape[1])

    try:
        image_positions, prompt_len = infer_llava_image_positions_no_forward(
            prompt_inputs={"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
            model_config=model.config,
            num_images=num_images,
        )
    except Exception:
        answer_ids = _greedy_decode_with_kv(model, past_kv, next_token, prompt_len=prompt_len_mm,
                                             eos_token_id=eos_token_id, max_new_tokens=MAX_NEW_TOKENS)
        return processor.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip(), {}

    n_img = int(image_positions.numel())
    n_text = int(prompt_len) - n_img
    stats = build_keep_stats(image_positions, prompt_len, keep_ratio, num_images)
    n_keep = stats["n_image_kept"]

    image_idx_dev = image_positions.to(device)
    q_positions = question_positions_after_images(image_positions, prompt_len)
    q_idx_dev = q_positions.to(device)

    keep_masks: dict[int, torch.Tensor] = {}
    for li in student.layer_indices:
        H_l = H_all[li + 1]
        scores = student.forward_layer(li, H_l, image_idx_dev, q_idx_dev).squeeze(0)
        if n_keep >= n_img:
            continue
        if n_keep <= 0:
            mask = torch.ones(prompt_len, dtype=torch.bool)
            mask[image_positions.cpu()] = False
            keep_masks[li] = mask
            continue
        top = torch.topk(scores, k=n_keep, largest=True).indices
        mask = torch.ones(prompt_len, dtype=torch.bool)
        image_keep = torch.zeros(n_img, dtype=torch.bool)
        image_keep[top.cpu()] = True
        mask[image_positions.cpu()] = image_keep
        keep_masks[li] = mask

    del H_all
    past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

    # Match the original kvpress/direct-generate semantics: the first answer
    # token comes from prefill logits; the pruned KV affects subsequent decode.
    answer_ids = _greedy_decode_with_kv(model, past_kv, next_token, prompt_len=int(prompt_len),
                                         eos_token_id=eos_token_id, max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.empty_cache()

    return processor.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip(), stats


MILEBENCH_DATASETS = [
    "ALFRED", "ActionLocalization", "ActionPrediction", "ActionSequence",
    "CLEVR-Change", "CharacterOrder", "CounterfactualInference", "DocVQA",
    "EgocentricNavigation", "GPR1200", "IEdit", "ImageNeedleInAHaystack",
    "MMCoQA", "MovingAttribute", "MovingDirection", "MultiModalQA",
    "OCR-VQA", "ObjectExistence", "ObjectInteraction", "ObjectShuffle",
    "SceneTransition", "SlideVQA", "Spot-the-Diff", "StateChange",
    "TQA", "TextNeedleInAHaystack", "WebQA", "WikiVQA",
]


def run_dataset(
    dataset: str,
    model,
    processor,
    student,
    output_dir: str,
    keep_ratio: float,
    device: str,
    overwrite: bool,
    image_column: str,
    limit: int | None,
    press,
    truncate_like_lookm: bool = False,
    prompt_style: str = "look_milebench",
    eval_backend: str = "internal",
    allow_partial_eval: bool = False,
    look_model_name: str = "foresight_llava15_student",
):
    task_out = os.path.join(output_dir, dataset)
    pred_path = os.path.join(task_out, "pred.json")
    stats_path = os.path.join(task_out, "keep_ratio_stats.json")
    os.makedirs(task_out, exist_ok=True)

    if os.path.exists(pred_path) and not overwrite:
        print(f"[skip] {dataset}: {pred_path} exists")
        return

    data_path = os.path.join(DATA_ROOT, dataset, f"{dataset}.json")
    data = json.load(open(data_path))
    meta = data["meta_data"]
    samples = data["data"]
    if limit is not None:
        samples = samples[:limit]

    print(f"[{dataset}] {len(samples)} samples | total_keep_ratio={keep_ratio} image_column={image_column} truncate={truncate_like_lookm} prompt_style={prompt_style}")

    # Prepare LOOK-M truncated inputs if requested (uses raw images_path + context truncation)
    prepared_lookm: dict[str, dict] = {}
    if truncate_like_lookm:
        image_root = os.path.join(DATA_ROOT, dataset, "images")
        try:
            prepared_lookm = prepare_lookm_truncated_inputs(
                samples,
                dataset=dataset,
                meta=meta,
                image_root=image_root,
                tokenizer=processor.tokenizer,
            )
            print(f"[{dataset}] LOOK-M truncated inputs prepared for {len(prepared_lookm)} samples")
        except Exception as e:
            print(f"[{dataset}] WARNING: LOOK-M truncation failed ({e}), falling back to direct image loading", file=sys.stderr)

    keep_stats = []
    predictions = []

    for sample in tqdm(samples, desc=dataset):
        ann = sample["task_instance"]
        sample_id_str = str(sample["sample_id"])
        try:
            if sample_id_str in prepared_lookm:
                # LOOK-M truncated: context has <ImageHere> markers; replace with <image> then wrap
                prepared = prepared_lookm[sample_id_str]
                image_paths = prepared["image_paths"]
                look_question = prepared["question"]
                question_for_export = look_question
                # MileBenchDataset puts <ImageHere> in context; replace with <image>
                q_with_imgs = LEGACY_IMAGE_PLACEHOLDER_PATTERN.sub("<image>", look_question)
                prompt_text = f"USER: {q_with_imgs}\nASSISTANT:"
            else:
                image_paths = resolve_image_paths(dataset, ann, image_column=image_column)
                if prompt_style == "look_milebench":
                    look_question = build_look_question(sample, meta, dataset_name=dataset)
                    question_for_export = replace_image_placeholders_for_export(look_question)
                    prompt_text = _build_look_prompt(look_question, len(image_paths))
                else:
                    question_for_export = replace_image_placeholders_for_export(str(ann.get("context", "")))
                    prompt_text = build_prompt(sample, meta, image_count=len(image_paths), image_column=image_column)
            if image_paths:
                images = open_images(image_paths)
            else:
                images = None
        except Exception as e:
            print(f"[warn] cannot prepare sample {sample.get('sample_id')}: {e}", file=sys.stderr)
            predictions.append(build_look_prediction_record(
                sample=sample,
                question_for_export="",
                prediction="",
                image_paths=[],
                look_model_name=look_model_name,
            ))
            continue

        if images is not None:
            inputs = processor(images=images, text=prompt_text, return_tensors="pt")
        else:
            inputs = processor(text=prompt_text, return_tensors="pt")
        inputs = inputs.to(device, torch.float16)

        answer, stats = generate_answer(model, processor, student, press, inputs, keep_ratio, len(image_paths), device)
        if stats:
            keep_stats.append(stats)

        predictions.append(build_look_prediction_record(
            sample=sample,
            question_for_export=question_for_export,
            prediction=answer,
            image_paths=image_paths,
            look_model_name=look_model_name,
        ))

    json.dump(predictions, open(pred_path, "w"), ensure_ascii=False, indent=2)
    print(f"[{dataset}] saved → {pred_path}")

    eval_path = os.path.join(task_out, "eval.json")
    should_eval = eval_backend != "none" and (limit is None or allow_partial_eval)
    if should_eval and (not os.path.exists(eval_path) or overwrite):
        if eval_backend == "internal":
            from foresight.eval.score_milebench_predictions import score_dataset

            score_dataset(
                data_root=DATA_ROOT,
                result_dir=output_dir,
                dataset=dataset,
                allow_partial=allow_partial_eval,
                overwrite=True,
            )
        elif eval_backend == "lookm":
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
            "image_column": image_column,
            "avg_image_token_ratio": sum(s["image_token_ratio"] for s in keep_stats) / n,
            "avg_text_token_ratio": sum(s["text_token_ratio"] for s in keep_stats) / n,
            "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in keep_stats) / n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in keep_stats) / n,
            "avg_n_images": sum(s["n_images"] for s in keep_stats) / n,
            "samples": keep_stats,
        }
        json.dump(summary, open(stats_path, "w"), indent=2)
        print(f"[{dataset}] keep_ratio stats → {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="dataset name or 'all' to run all MileBench tasks")
    parser.add_argument("--pretrained", default=LLAVA15_CKPT)
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--keep_ratio", type=float, nargs="+", default=None,
                        help="one or more total keep ratios, e.g. --keep_ratio 0.2 0.1")
    parser.add_argument("--total_keep_ratio", type=float, nargs="+", default=None,
                        help="alias for --keep_ratio; kept for older experiment scripts")
    parser.add_argument("--image_column", default="combined_1_images",
                        help="MileBench task_instance image field; 'combined_1_images' for pre-merged single image (default, required for LLaVA-1.5 which doesn't support native multi-image), 'images_path' for raw multi-image")
    parser.add_argument("--prompt_style", choices=["look_milebench", "default"], default="look_milebench",
                        help="look_milebench matches the LOOK-M prompt format used in old good experiments")
    parser.add_argument("--truncate_like_lookm", action="store_true", default=False,
                        help="Apply LOOK-M-style context truncation (max_context_len=4096, n_tokens_per_image=576) using MileBenchDataset. Required for fair comparison with LOOK-M.")
    parser.add_argument("--generation_backend", choices=["press", "manual"], default="press",
                        help="press matches the older kvpress/model.generate evaluation path")
    parser.add_argument("--eval_backend", choices=["internal", "lookm", "none"], default="internal",
                        help="internal uses foresight LOOK-M-compatible scorer; lookm calls /workspace/look-m/evaluate.py")
    parser.add_argument("--allow_partial_eval", action="store_true", default=False,
                        help="Score limited/subset runs with the internal evaluator")
    parser.add_argument("--look_model_name", default="foresight_llava15_student")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = MILEBENCH_DATASETS if args.dataset == "all" else [args.dataset]
    keep_ratios = args.total_keep_ratio if args.total_keep_ratio is not None else args.keep_ratio
    if keep_ratios is None:
        keep_ratios = [0.2]

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    print(f"[load] model={args.pretrained} device={args.device}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.pretrained, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.pretrained)
    processor = configure_llava_processor(processor, model.config)

    student = None
    if args.generation_backend == "manual":
        student = VisualUtilityStudent.from_pretrained(args.student_path)
        student = student.to(device=args.device, dtype=torch.float16).eval()
        print(f"[load] student={args.student_path}")
    else:
        print(f"[load] student via VisualUtilityStudentPress={args.student_path}")

    for keep_ratio in keep_ratios:
        ratio_tag = f"keep{str(keep_ratio).replace('.', '')}"
        output_dir = os.path.join(args.output_dir, ratio_tag)
        look_model_name = f"{args.look_model_name}_keep{keep_ratio:g}"
        print(f"\n=== total_keep_ratio={keep_ratio} → {output_dir} ===")
        press = None
        if args.generation_backend == "press":
            press = VisualUtilityStudentPress(total_keep_ratio=float(keep_ratio), student_model_name=args.student_path)
            press.post_init_from_model(model)
        for dataset in datasets:
            try:
                run_dataset(
                    dataset,
                    model,
                    processor,
                    student,
                    output_dir,
                    keep_ratio,
                    args.device,
                    args.overwrite,
                    args.image_column,
                    args.limit,
                    press,
                    truncate_like_lookm=args.truncate_like_lookm,
                    prompt_style=args.prompt_style,
                    eval_backend=args.eval_backend,
                    allow_partial_eval=args.allow_partial_eval,
                    look_model_name=look_model_name,
                )
            except Exception as e:
                print(f"[error] {dataset}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
