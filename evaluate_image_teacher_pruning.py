#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from kvpress import OracleImageTeacherPress, ProbeImageTeacherPress
from kvzap.image_teacher_utils import build_prompt, load_pt_record, load_vlm_samples, normalize_answer, resolve_teacher_dir
from kvzap.llava_extractor import (
    _get_model_device,
    _get_model_float_dtype,
    _move_batch_to_device,
    _resolve_prompt_image_mask,
    _trim_after_eos,
    disable_merge_trace,
    enable_merge_trace,
)
from kvzap.milebench_look_metrics import LookMileBenchEvaluator

HD_DOCVQA_DATASET_PATH = "/workspace/hd/data/MileBench/DocVQA/DocVQA.json"
HD_DOCVQA_IMAGE_ROOT = "/workspace/hd/data/MileBench/DocVQA/images"
HD_ZAP_ARTIFACT_ROOT = "/workspace/hd/artifacts/zap"
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{image#\d+\}")


def infer_prompt_image_positions(model: LlavaForConditionalGeneration, prompt_inputs: dict[str, Any]) -> torch.Tensor:
    prompt_input_ids = prompt_inputs["input_ids"]
    trace_ctx = enable_merge_trace(model)
    try:
        with torch.no_grad():
            prompt_out = model(
                **prompt_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        disable_merge_trace(trace_ctx)
    prompt_len_mm = int(prompt_out.logits.shape[1])
    mask = _resolve_prompt_image_mask(model, prompt_input_ids, prompt_len_mm, trace_ctx)
    return mask.nonzero(as_tuple=False).flatten().cpu()


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


def resolve_teacher_image_positions(teacher_record: dict[str, Any]) -> torch.Tensor:
    if "image_pos_mm" in teacher_record:
        return teacher_record["image_pos_mm"].long().flatten()
    if "image_indices_mm" in teacher_record:
        return teacher_record["image_indices_mm"].long().flatten()
    if "is_image_pos_mm" in teacher_record:
        mask = teacher_record["is_image_pos_mm"]
        if not isinstance(mask, torch.Tensor):
            raise TypeError("teacher_record['is_image_pos_mm'] must be a tensor")
        return mask.nonzero(as_tuple=False).flatten().long()
    raise KeyError("Teacher record must contain one of: image_pos_mm, image_indices_mm, is_image_pos_mm")


def build_press(args: argparse.Namespace):
    if args.mode == "oracle":
        return OracleImageTeacherPress(image_keep_ratio=args.image_keep_ratio, head_reduce=args.head_reduce)
    return ProbeImageTeacherPress(
        image_keep_ratio=args.image_keep_ratio,
        head_reduce=args.head_reduce,
        probe_model_name=args.probe_model_name,
    )


def load_core_annotation(dataset_path: str) -> Optional[dict[str, Any]]:
    path = Path(dataset_path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def subset_core_annotation(core_annotation: dict[str, Any], prediction_ids: set[Any]) -> dict[str, Any]:
    subset = [item for item in core_annotation["data"] if item.get("sample_id") in prediction_ids]
    return {
        "meta_data": core_annotation["meta_data"],
        "data": subset,
    }


def build_look_prediction_record(
    sample: dict[str, Any],
    question_for_export: str,
    prediction: str,
    gold: Optional[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    raw_sample_id = raw.get("sample_id", sample["sample_id"])
    return {
        "sample_id": to_int_if_possible(raw_sample_id),
        "image": sample["image_paths"],
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
    parser.add_argument("--mode", choices=["oracle", "probe"], required=True)
    parser.add_argument("--dataset_path", type=str, default=HD_DOCVQA_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=f"{HD_ZAP_ARTIFACT_ROOT}/docvqa_image_pruning")
    parser.add_argument("--implementation_model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image_root", type=str, default=HD_DOCVQA_IMAGE_ROOT)
    parser.add_argument("--image_column", type=str, default="images_path")
    parser.add_argument("--answer_column", type=str, default=None)
    parser.add_argument("--teacher_dir", type=str, default=None)
    parser.add_argument("--teacher_score_name", type=str, default="splus_postvision")
    parser.add_argument("--probe_model_name", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default="USER: <image>\n{question}\nASSISTANT:")
    parser.add_argument("--prompt_style", choices=["default", "look_milebench"], default="look_milebench")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--image_keep_ratio", type=float, required=True)
    parser.add_argument("--head_reduce", choices=["amax", "mean"], default="amax")
    parser.add_argument("--look_dataset_name", type=str, default="DocVQA")
    parser.add_argument("--look_model_name", type=str, default="zap_docvqa")
    parser.add_argument("--look_result_root", type=str, default=HD_ZAP_ARTIFACT_ROOT)
    parser.add_argument("--save_look_files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate_with_look_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_partial_look_eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    if args.mode == "oracle" and args.teacher_dir is None:
        raise ValueError("teacher_dir is required for oracle mode")
    if args.mode == "probe" and not args.probe_model_name:
        raise ValueError("probe_model_name is required for probe mode")

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
    teacher_root = resolve_teacher_dir(args.teacher_dir) if args.teacher_dir else None

    processor = AutoProcessor.from_pretrained(args.implementation_model_name)
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "device_map": None if args.device_map in ("", "none", "None") else args.device_map,
    }
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    model = LlavaForConditionalGeneration.from_pretrained(args.implementation_model_name, **model_kwargs)

    device = _get_model_device(model)
    float_dtype = _get_model_float_dtype(model)
    press = build_press(args)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    look_predictions: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc=f"Evaluating {args.mode} image pruning"):
        try:
            raw_record = sample.get("raw") if isinstance(sample.get("raw"), dict) else None
            question_for_prompt = sample["question"]
            if (
                args.prompt_style == "look_milebench"
                and core_annotation is not None
                and raw_record is not None
                and "task_instance" in raw_record
            ):
                question_for_prompt = build_look_question(raw_record, core_annotation, dataset_name=args.look_dataset_name)

            images = open_images(sample["image_paths"])
            prompt_text = build_prompt(
                question_for_prompt,
                args.prompt_template,
                image_count=len(sample["image_paths"]),
            )
            prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
            prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)
            prompt_len_text = int(prompt_inputs["input_ids"].shape[1])

            if args.mode == "oracle":
                teacher_record = load_pt_record(teacher_root / f"{sample['sample_id']}.pt")
                image_positions = resolve_teacher_image_positions(teacher_record)
                press.set_sample_teacher(image_positions, teacher_record[args.teacher_score_name])
            else:
                image_positions = infer_prompt_image_positions(model, prompt_inputs)
                press.set_image_positions(image_positions)

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
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "question": question_for_prompt,
                    "image_paths": sample["image_paths"],
                    "gold_answer": gold,
                    "prediction": prediction,
                    "exact_match": exact_match,
                }
            )

            if args.save_look_files:
                question_for_export = replace_image_placeholders_for_export(question_for_prompt)
                look_predictions.append(
                    build_look_prediction_record(
                        sample=sample,
                        question_for_export=question_for_export,
                        prediction=prediction,
                        gold=gold,
                        args=args,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sample["sample_id"], "error": repr(exc)})
            if not args.continue_on_error:
                raise
        finally:
            press.clear_sample_context()

    predictions_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    predictions_df.to_json(output_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    failures_df.to_csv(output_dir / "failures.csv", index=False)

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
            predictions_for_eval = deepcopy(look_predictions)
            predictions_with_extracted, look_eval_result, eval_list = evaluator.evaluate(predictions_for_eval, core_for_eval)

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
            print(
                "Skipping LOOK-compatible eval because predictions do not cover the full dataset. "
                "Use --allow_partial_look_eval to score subsets."
            )

    metrics = {
        "mode": args.mode,
        "n_samples": len(samples),
        "n_predictions": int(len(predictions_df)),
        "n_failures": int(len(failures_df)),
        "image_keep_ratio": args.image_keep_ratio,
        "teacher_score_name": args.teacher_score_name if args.mode == "oracle" else None,
        "probe_model_name": args.probe_model_name if args.mode == "probe" else None,
        "prompt_style": args.prompt_style,
        "look_model_name": args.look_model_name if args.save_look_files else None,
        "look_dataset_name": args.look_dataset_name if args.save_look_files else None,
        "look_result_dir": str(look_dataset_dir) if look_dataset_dir is not None else None,
    }
    if not predictions_df.empty and "exact_match" in predictions_df and predictions_df["exact_match"].notna().any():
        metrics["exact_match_accuracy"] = float(predictions_df["exact_match"].dropna().mean())
    if look_eval_result is not None:
        metrics["look_eval"] = look_eval_result

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
