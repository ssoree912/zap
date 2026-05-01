#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent
LOOKM_CANDIDATES = [REPO_ROOT.parent / "LOOK-M", REPO_ROOT.parent / "look-m"]
LOOKM_ROOT = next((path for path in LOOKM_CANDIDATES if path.is_dir()), LOOKM_CANDIDATES[0])
if str(LOOKM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOKM_ROOT))

from kvpress.presses.image_token_press import VisualUtilityStudentPress
from kvzap.image_teacher_utils import build_prompt, load_vlm_samples, normalize_answer
from kvzap.llava_extractor import (
    _get_model_device,
    _move_batch_to_device,
    _trim_after_eos,
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)
from kvzap.milebench_look_metrics import LookMileBenchEvaluator

HD_DOCVQA_DATASET_PATH = "/workspace/zap/data/MileBench/DocVQA/DocVQA.json"
HD_DOCVQA_IMAGE_ROOT = "/workspace/zap/data/MileBench/DocVQA/images"
HD_ZAP_ARTIFACT_ROOT = "/workspace/zap/artifacts/combine_prob"
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{image#\d+\}")


def decode_answer(processor: Any, generated_ids: torch.Tensor, prompt_len_text: int) -> str:
    answer_ids = _trim_after_eos(generated_ids, processor.tokenizer.eos_token_id)[prompt_len_text:]
    return processor.tokenizer.decode(
        answer_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def open_images(image_paths: list[str]):
    from PIL import Image

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images[0] if len(images) == 1 else images


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


def to_int_if_possible(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def subset_core_annotation(core_annotation: dict[str, Any], prediction_ids: set[Any]) -> dict[str, Any]:
    subset = [item for item in core_annotation["data"] if item.get("sample_id") in prediction_ids]
    return {"meta_data": core_annotation["meta_data"], "data": subset}


def build_look_prediction_record(
    sample: dict[str, Any],
    question_for_export: str,
    prediction: str,
    gold: Optional[str],
    args: argparse.Namespace,
    image_paths: list[str],
) -> dict[str, Any]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    raw_sample_id = raw.get("sample_id", sample["sample_id"])
    return {
        "sample_id": to_int_if_possible(raw_sample_id),
        "image": image_paths,
        "question": question_for_export,
        "gt_response": "" if gold is None else str(gold),
        "gen_model_id": args.look_model_name,
        "pred_response": prediction,
        "gen_kwargs": {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=HD_DOCVQA_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=f"{HD_ZAP_ARTIFACT_ROOT}/docvqa_student")
    parser.add_argument("--implementation_model_name", type=str, default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--image_root", type=str, default=HD_DOCVQA_IMAGE_ROOT)
    parser.add_argument("--image_column", type=str, default="combined_1_images")
    parser.add_argument("--answer_column", type=str, default=None)
    parser.add_argument("--student_model_name", type=str, required=True)
    parser.add_argument("--total_keep_ratio", type=float, required=True)
    parser.add_argument("--head_reduce", choices=["amax", "mean"], default="amax")
    parser.add_argument("--prompt_template", type=str, default="USER: <image>\n{question}\nASSISTANT:")
    parser.add_argument("--prompt_style", choices=["default", "look_milebench"], default="look_milebench")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--look_dataset_name", type=str, default="DocVQA")
    parser.add_argument("--look_model_name", type=str, default="student")
    parser.add_argument("--look_result_root", type=str, default=HD_ZAP_ARTIFACT_ROOT)
    parser.add_argument("--save_look_files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate_with_look_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_partial_look_eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_question_positions", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_vlm_samples(
        dataset_path=args.dataset_path,
        image_root=args.image_root,
        image_column=args.image_column,
        answer_column=args.answer_column,
        limit=args.limit,
    )
    core_annotation = load_core_annotation(args.dataset_path)

    processor = AutoProcessor.from_pretrained(args.implementation_model_name, use_fast=False)
    model_kwargs = {"attn_implementation": args.attn_implementation, "device_map": None}
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    model = LlavaForConditionalGeneration.from_pretrained(args.implementation_model_name, **model_kwargs)
    configure_llava_processor(processor, model.config)
    model = model.to(torch.device(args.device))
    model.eval()

    device = _get_model_device(model)

    press = VisualUtilityStudentPress(
        total_keep_ratio=args.total_keep_ratio,
        head_reduce=args.head_reduce,
        student_model_name=args.student_model_name,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    look_predictions: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="student pruning"):
        try:
            raw_record = sample.get("raw") if isinstance(sample.get("raw"), dict) else None

            question_for_prompt = sample["question"]
            image_paths = sample["image_paths"] or []
            if (
                args.prompt_style == "look_milebench"
                and core_annotation is not None
                and raw_record is not None
                and "task_instance" in raw_record
            ):
                question_for_prompt = build_look_question(raw_record, core_annotation, dataset_name=args.look_dataset_name)

            prompt_text = build_prompt(
                question_for_prompt,
                args.prompt_template,
                image_count=len(image_paths),
            )

            if image_paths:
                images = open_images(image_paths)
                prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
            else:
                prompt_inputs = processor(text=prompt_text, return_tensors="pt")

            prompt_inputs = _move_batch_to_device(prompt_inputs, device, torch.float16)
            prompt_len_text = int(prompt_inputs["input_ids"].shape[1])
            num_images = len(image_paths)

            if num_images == 0:
                press.set_image_positions(torch.empty(0, dtype=torch.long))
            else:
                image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
                    prompt_inputs=prompt_inputs,
                    model_config=model.config,
                    num_images=num_images,
                )
                press.set_image_positions(image_positions)
                if not args.no_question_positions:
                    img_set = set(image_positions.tolist())
                    q_positions = torch.tensor(
                        [i for i in range(prompt_len_mm) if i not in img_set],
                        dtype=torch.long,
                    )
                    press.set_question_positions(q_positions)

            with press(model):
                with torch.no_grad():
                    generated_ids = model.generate(
                        **prompt_inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

            prediction = decode_answer(processor, generated_ids[0], prompt_len_text)
            gold = sample.get("answer")
            exact_match = None if gold is None else normalize_answer(prediction) == normalize_answer(gold)

            rows.append({
                "sample_id": sample["sample_id"],
                "question": question_for_prompt,
                "image_paths": image_paths,
                "gold_answer": gold,
                "prediction": prediction,
                "exact_match": exact_match,
            })

            if args.save_look_files:
                question_for_export = replace_image_placeholders_for_export(question_for_prompt)
                look_predictions.append(build_look_prediction_record(
                    sample=sample,
                    question_for_export=question_for_export,
                    prediction=prediction,
                    gold=gold,
                    args=args,
                    image_paths=image_paths,
                ))

        except Exception as exc:
            failures.append({"sample_id": sample["sample_id"], "error": repr(exc)})
            if not args.continue_on_error:
                raise
        finally:
            press.clear_sample_context()

    predictions_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    predictions_df.to_json(output_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    if not failures_df.empty:
        failures_df.to_csv(output_dir / "failures.csv", index=False)
    elif (output_dir / "failures.csv").exists():
        (output_dir / "failures.csv").unlink()

    look_dataset_dir: Optional[Path] = None
    if args.save_look_files and args.look_result_root:
        look_dataset_dir = Path(args.look_result_root).resolve() / args.look_model_name / args.look_dataset_name
        look_dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.save_look_files:
        write_json(output_dir / "pred.json", look_predictions)
        if look_dataset_dir is not None:
            write_json(look_dataset_dir / "pred.json", look_predictions)

    look_eval_result: Optional[dict[str, Any]] = None
    if args.evaluate_with_look_metrics and args.save_look_files and core_annotation is not None and look_predictions:
        evaluation_ready = len(look_predictions) == len(core_annotation["data"])
        core_for_eval = core_annotation

        if not evaluation_ready and args.allow_partial_look_eval:
            prediction_ids = {item["sample_id"] for item in look_predictions}
            core_for_eval = subset_core_annotation(core_annotation, prediction_ids)
            evaluation_ready = len(look_predictions) == len(core_for_eval["data"])

        if evaluation_ready:
            evaluator = LookMileBenchEvaluator()
            predictions_with_extracted, look_eval_result, eval_list = evaluator.evaluate(
                deepcopy(look_predictions), core_for_eval
            )
            write_json(output_dir / "eval.json", look_eval_result)
            write_json(output_dir / "eval_score.json", eval_list)
            if predictions_with_extracted is not None:
                write_json(output_dir / "pred_with_extracted.json", predictions_with_extracted)
            if look_dataset_dir is not None:
                write_json(look_dataset_dir / "eval.json", look_eval_result)
                write_json(look_dataset_dir / "eval_score.json", eval_list)
                if predictions_with_extracted is not None:
                    write_json(look_dataset_dir / "pred_with_extracted.json", predictions_with_extracted)
        else:
            print("Skipping LOOK eval: predictions do not cover the full dataset. Use --allow_partial_look_eval to score subsets.")

    metrics: dict[str, Any] = {
        "n_samples": len(samples),
        "n_predictions": int(len(predictions_df)),
        "n_failures": int(len(failures_df)),
        "total_keep_ratio": args.total_keep_ratio,
        "student_model_name": args.student_model_name,
        "prompt_style": args.prompt_style,
        "look_model_name": args.look_model_name if args.save_look_files else None,
        "look_dataset_name": args.look_dataset_name if args.save_look_files else None,
        "look_result_dir": str(look_dataset_dir) if look_dataset_dir is not None else None,
    }
    if not predictions_df.empty and "exact_match" in predictions_df and predictions_df["exact_match"].notna().any():
        metrics["exact_match_accuracy"] = float(predictions_df["exact_match"].dropna().mean())
    if look_eval_result is not None:
        metrics["look_eval"] = look_eval_result

    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_config.json", vars(args))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
