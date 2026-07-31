#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data adapter for the 7 standard tasks plus 29 local MileBench tasks.

The seven standard datasets delegate to the already-audited eval-700 loader.
MileBench is exposed with collision-free names of the form
``milebench__<original-task-name>`` and uses the local ``combined_1_images``
rendering used by the existing LLaVA-1.5/OneVision experiments.

MileBench prompt truncation follows the LOOK-M policy relevant to the combined
single-image setting:

* retain the task instruction;
* retain the rightmost (question/answer-adjacent) part of the task context;
* discard context tokens from the left until the *expanded* LLaVA-1.5 prompt,
  including the 576-token visual block and Vicuna template, is at most 4096
  tokens.

The module contains no model inference and is safe to use for manifest checks.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
WORKSPACE_ROOT = ZAP_ROOT.parent
BASE_EXP = ZAP_ROOT / "experiments" / "EXP-20260727-001-eval700-q2"
DEFAULT_EVAL_ROOT = WORKSPACE_ROOT / "data" / "eval"
DEFAULT_MILEBENCH_ROOT = ZAP_ROOT / "data" / "MileBench"
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models" / "llava-v1.5-7b"

MILEBENCH_PREFIX = "milebench__"
DEFAULT_MAX_PROMPT_TOKENS = 4096
DEFAULT_IMAGE_FEATURE_LEN = 576
DEFAULT_MILEBENCH_MAX_NEW_TOKENS = 64
LONG_GENERATION_MILEBENCH_TASKS = {
    "IEdit",
    "Spot-the-Diff",
    "ImageNeedleInAHaystack",
    "TextNeedleInAHaystack",
    "MMCoQA",
}

OFFICIAL_MILEBENCH_TASKS = (
    "ALFRED",
    "ActionLocalization",
    "ActionPrediction",
    "ActionSequence",
    "CLEVR-Change",
    "CharacterOrder",
    "CounterfactualInference",
    "DocVQA",
    "EgocentricNavigation",
    "GPR1200",
    "IEdit",
    "ImageNeedleInAHaystack",
    "MMCoQA",
    "MovingAttribute",
    "MovingDirection",
    "MultiModalQA",
    "OCR-VQA",
    "ObjectExistence",
    "ObjectInteraction",
    "ObjectShuffle",
    "SceneTransition",
    "SlideVQA",
    "Spot-the-Diff",
    "StateChange",
    "TQA",
    "TextNeedleInAHaystack",
    "WebQA",
    "WikiVQA",
)
LOCAL_EXTENSION_TASKS = ("nuscenes",)
LOCAL_MILEBENCH_TASKS = OFFICIAL_MILEBENCH_TASKS + LOCAL_EXTENSION_TASKS
MILEBENCH_DATASETS = tuple(
    f"{MILEBENCH_PREFIX}{task}" for task in LOCAL_MILEBENCH_TASKS
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module("eval7200_base_data", BASE_EXP / "eval700_data.py")
EvalSample = base.EvalSample
STANDARD_DATASETS = tuple(base.DATASETS)
DATASETS = STANDARD_DATASETS + MILEBENCH_DATASETS


def is_milebench_dataset(name: str) -> bool:
    return str(name).startswith(MILEBENCH_PREFIX)


def milebench_task_name(name: str) -> str:
    value = str(name)
    if not is_milebench_dataset(value):
        raise ValueError(
            f"{name!r} is not a MileBench adapter name; expected "
            f"{MILEBENCH_PREFIX}<task>"
        )
    task = value[len(MILEBENCH_PREFIX) :]
    if task not in LOCAL_MILEBENCH_TASKS:
        raise ValueError(
            f"Unknown local MileBench task {task!r}; expected one of "
            f"{LOCAL_MILEBENCH_TASKS}"
        )
    return task


def _resolve_instruction(meta: dict[str, Any], sample: dict[str, Any]) -> str:
    instructions = meta.get("task_instruction", "")
    instruction_id = sample.get("task_instruction_id", 0)
    if isinstance(instructions, list):
        return str(instructions[int(instruction_id)])
    if isinstance(instructions, dict):
        if instruction_id in instructions:
            return str(instructions[instruction_id])
        return str(instructions[str(instruction_id)])
    return str(instructions)


def _render_task_context(task: str, sample: dict[str, Any]) -> str:
    annotation = sample["task_instance"]
    context = str(annotation.get("context", ""))
    image_count = len(annotation.get("images_path", ()))
    for index in range(1, image_count + 1):
        label = f"<Image {index}> "
        context = context.replace(f"{{image#{index}}}", label)
        context = context.replace(f"{{table#{index}}}", label)

    choices = annotation.get("choice_list")
    if choices:
        # This exactly follows LOOK-M's special handling: GPR1200 has many
        # options and deliberately omits alphabetic choice labels.
        if task == "GPR1200":
            rendered_choices = "\n".join(str(choice) for choice in choices)
        else:
            rendered_choices = "\n".join(
                f"{chr(65 + index)}. {choice}"
                for index, choice in enumerate(choices)
            )
        context += (
            "\nChoice list: \n"
            + rendered_choices
            + "\nYour answer is: "
        )
    return context


def _build_vicuna_prompt(context: str) -> str:
    from qvik.llava15.conversation import conv_templates

    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(
        conversation.roles[0],
        f"<image>\n{context.strip()}",
    )
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def expanded_prompt_length(
    context: str,
    tokenizer: Any,
    *,
    image_feature_len: int = DEFAULT_IMAGE_FEATURE_LEN,
) -> tuple[int, int]:
    """Return ``(raw_token_count, expanded_multimodal_token_count)``."""

    from qvik.llava15.constants import IMAGE_TOKEN_INDEX
    from qvik.llava15.mm_utils import tokenizer_image_token

    prompt = _build_vicuna_prompt(context)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    placeholder_count = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
    if placeholder_count != 1:
        raise ValueError(
            f"Expected exactly one combined-image placeholder, found "
            f"{placeholder_count}"
        )
    raw_tokens = int(input_ids.numel())
    expanded_tokens = raw_tokens - 1 + int(image_feature_len)
    return raw_tokens, expanded_tokens


def truncate_milebench_context(
    *,
    instruction: str,
    task_context: str,
    tokenizer: Any,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    image_feature_len: int = DEFAULT_IMAGE_FEATURE_LEN,
) -> tuple[str, dict[str, int | bool]]:
    """Keep the instruction and the longest fitting suffix of task context."""

    instruction = str(instruction).strip()
    task_context = str(task_context).strip()
    complete = (
        f"{instruction}\n{task_context}" if task_context else instruction
    )
    _, original_expanded = expanded_prompt_length(
        complete,
        tokenizer,
        image_feature_len=image_feature_len,
    )
    context_ids = tokenizer(
        task_context,
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    original_context_tokens = len(context_ids)
    if original_expanded <= max_prompt_tokens:
        return complete, {
            "prompt_was_left_truncated": False,
            "original_expanded_prompt_tokens": original_expanded,
            "expanded_prompt_tokens": original_expanded,
            "original_task_context_tokens": original_context_tokens,
            "kept_task_context_tokens": original_context_tokens,
            "removed_task_context_tokens": 0,
        }

    _, instruction_only_expanded = expanded_prompt_length(
        instruction,
        tokenizer,
        image_feature_len=image_feature_len,
    )
    if instruction_only_expanded > max_prompt_tokens:
        raise ValueError(
            "MileBench task instruction alone exceeds the expanded prompt cap: "
            f"instruction={instruction_only_expanded}, cap={max_prompt_tokens}"
        )

    def candidate(keep: int) -> tuple[str, int]:
        if keep <= 0:
            rendered = instruction
        else:
            suffix = tokenizer.decode(
                context_ids[-keep:],
                skip_special_tokens=True,
            ).lstrip()
            rendered = f"{instruction}\n{suffix}" if suffix else instruction
        _, expanded = expanded_prompt_length(
            rendered,
            tokenizer,
            image_feature_len=image_feature_len,
        )
        return rendered, expanded

    low = 0
    high = original_context_tokens
    best_context, best_expanded = candidate(0)
    best_keep = 0
    # Removing one stored SentencePiece token changes the re-tokenized prompt
    # length by approximately one.  Start from that estimate and then retain
    # the same exact monotonic-bound search contract as binary search.  This
    # avoids ~14 repeated 4K-token encodes for every long Needle/Wiki sample.
    guess = max(
        low,
        min(
            high,
            original_context_tokens
            - max(0, original_expanded - max_prompt_tokens),
        ),
    )
    while low <= high:
        rendered, expanded = candidate(guess)
        if expanded <= max_prompt_tokens:
            best_context = rendered
            best_expanded = expanded
            best_keep = guess
            low = guess + 1
            direction = max(1, max_prompt_tokens - expanded)
        else:
            high = guess - 1
            direction = -max(1, expanded - max_prompt_tokens)
        if low <= high:
            guess = max(low, min(high, guess + direction))

    # SentencePiece decode/encode boundaries are almost monotonic but verify
    # the final contract explicitly instead of relying on that property.
    _, verified_expanded = expanded_prompt_length(
        best_context,
        tokenizer,
        image_feature_len=image_feature_len,
    )
    if verified_expanded > max_prompt_tokens:
        raise AssertionError(
            f"Left truncation produced {verified_expanded} tokens, above "
            f"the cap {max_prompt_tokens}"
        )
    return best_context, {
        "prompt_was_left_truncated": True,
        "original_expanded_prompt_tokens": original_expanded,
        "expanded_prompt_tokens": verified_expanded,
        "original_task_context_tokens": original_context_tokens,
        "kept_task_context_tokens": best_keep,
        "removed_task_context_tokens": original_context_tokens - best_keep,
    }


@lru_cache(maxsize=2)
def load_tokenizer(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        legacy=True,
    )
    # The checkpoint tokenizer advertises 2048 although the model config and
    # local MileBench protocol use 4096. This only suppresses false warnings;
    # the explicit expanded-length check above remains authoritative.
    tokenizer.model_max_length = max(
        int(getattr(tokenizer, "model_max_length", 0)),
        DEFAULT_MAX_PROMPT_TOKENS,
    )
    return tokenizer


def _load_milebench_samples(
    name: str,
    root: Path,
    n: int,
    seed: int,
    *,
    tokenizer: Any,
    load_images: bool,
    max_prompt_tokens: int,
    image_feature_len: int,
    max_new_tokens: int | None,
) -> list[EvalSample]:
    task = milebench_task_name(name)
    task_root = Path(root) / task
    annotation_path = task_root / f"{task}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing MileBench annotation {annotation_path}")
    payload = json.loads(annotation_path.read_text())
    rows = payload.get("data", ())
    if n < 0:
        raise ValueError(f"n must be nonnegative, got {n}")
    if len(rows) < n:
        raise ValueError(
            f"{task} has only {len(rows)} samples, fewer than requested n={n}"
        )

    rng = np.random.default_rng(int(seed))
    selected_indices = [
        int(index)
        for index in rng.choice(len(rows), size=n, replace=False).tolist()
    ]
    task_max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else (128 if task in LONG_GENERATION_MILEBENCH_TASKS else 64)
    )

    samples: list[EvalSample] = []
    for row_index in selected_indices:
        row = rows[row_index]
        annotation = row["task_instance"]
        instruction = _resolve_instruction(payload["meta_data"], row)
        task_context = _render_task_context(task, row)
        context, truncation = truncate_milebench_context(
            instruction=instruction,
            task_context=task_context,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            image_feature_len=image_feature_len,
        )

        combined_values = annotation.get("combined_1_images") or ()
        if not combined_values:
            raise ValueError(
                f"{task}/{row.get('sample_id')} has no combined_1_images"
            )
        image_path = task_root / "combined_1_images" / str(combined_values[0])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing combined image for {task}/{row.get('sample_id')}: "
                f"{image_path}"
            )
        image = None
        if load_images:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB").copy()

        question_type = str(
            payload.get("meta_data", {}).get("question_type", "")
        ).lower()
        choices = tuple(str(value) for value in annotation.get("choice_list", ()))
        metadata: dict[str, Any] = {
            "task_type": "milebench",
            "type": question_type,
            "milebench_task": task,
            "milebench_question_type": question_type,
            "choice_list": list(choices),
            "original_image_paths": list(annotation.get("images_path", ())),
            "combined_image_path": str(image_path),
            "selection_protocol": (
                "numpy_default_rng_choice_without_replacement"
            ),
            "selection_seed": int(seed),
            "selection_size_requested": int(n),
            "max_prompt_tokens": int(max_prompt_tokens),
            "image_feature_len": int(image_feature_len),
            **truncation,
        }
        samples.append(
            EvalSample(
                dataset=name,
                sample_id=str(row.get("sample_id")),
                row_index=row_index,
                image=image,
                # The complete retained user content is the semantic-question
                # span for the question-only diagnostic.
                raw_question=context,
                context=context,
                references=(str(row.get("response", "")),),
                max_new_tokens=task_max_new_tokens,
                metadata=metadata,
            )
        )
    return samples


def load_samples(
    name: str,
    eval_root: Path = DEFAULT_EVAL_ROOT,
    n: int = 200,
    seed: int = 42,
    *,
    milebench_root: Path = DEFAULT_MILEBENCH_ROOT,
    tokenizer: Any | None = None,
    load_images: bool = True,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    image_feature_len: int = DEFAULT_IMAGE_FEATURE_LEN,
    max_new_tokens: int | None = None,
) -> list[EvalSample]:
    if not is_milebench_dataset(name):
        return base.load_samples(
            name,
            Path(eval_root),
            int(n),
            int(seed),
            load_images=load_images,
        )
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return _load_milebench_samples(
        name,
        Path(milebench_root),
        int(n),
        int(seed),
        tokenizer=tokenizer,
        load_images=load_images,
        max_prompt_tokens=int(max_prompt_tokens),
        image_feature_len=int(image_feature_len),
        max_new_tokens=max_new_tokens,
    )


@lru_cache(maxsize=1)
def _milebench_evaluator() -> Any:
    from foresight.eval.look_milebench_metrics import LookMileBenchEvaluator

    return LookMileBenchEvaluator()


def score_milebench_prediction(sample: EvalSample, prediction: str) -> float:
    """Return the decomposable LOOK-M-compatible score for one prediction."""

    if not is_milebench_dataset(sample.dataset):
        raise ValueError(f"{sample.dataset} is not a MileBench adapter sample")
    evaluator = _milebench_evaluator()
    task = str(sample.metadata["milebench_task"])
    question_type = str(sample.metadata["milebench_question_type"]).lower()
    ground_truth = str(sample.references[0] if sample.references else "")
    prediction = str(prediction)

    if "NeedleInAHaystack" in task:
        target = evaluator.process(ground_truth)
        predicted = evaluator.process(prediction)
        return float(target in predicted.split())
    if task == "MMCoQA":
        target = evaluator.process(ground_truth)
        predicted = evaluator.process(prediction)
        return float(target in predicted)
    if question_type in {"open-ended", "open_ended", "openended"}:
        # Match LOOK-M's official per-sample implementation exactly.  In
        # particular this is the python ``rouge`` package's Rouge-L F score,
        # after LOOK-M text processing, rather than our helper LCS function.
        from rouge import Rouge

        processed_prediction = evaluator.process(prediction)
        processed_ground_truth = evaluator.process(ground_truth)
        if not processed_prediction:
            return 0.0
        return float(
            Rouge()
            .get_scores(processed_prediction, processed_ground_truth)[0][
                "rouge-l"
            ]["f"]
        )
    if question_type == "multi-choice":
        scored = {
            "sample_id": sample.sample_id,
            "gt_response": ground_truth,
            "pred_response": prediction,
            "choice_list": list(sample.metadata.get("choice_list", ())),
        }
        evaluator.process_sample(scored)
        score, _extracted = evaluator.judge_multi_choice(scored)
        return float(score)
    raise ValueError(
        f"Unsupported MileBench question_type={question_type!r} for {task}"
    )


def score_prediction(sample: EvalSample, prediction: str) -> float:
    if is_milebench_dataset(sample.dataset):
        return score_milebench_prediction(sample, prediction)
    return float(base.score_prediction(sample, prediction))


def write_manifest(samples: Sequence[EvalSample], path: Path) -> None:
    """Write a deterministic manifest without the base dataset-name filter."""

    records = [base.sample_manifest_record(sample) for sample in samples]
    counts = Counter(sample.dataset for sample in samples)
    payload = {
        "schema_version": 1,
        "sample_count": len(records),
        "datasets": dict(sorted(counts.items())),
        "samples": records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def validate_milebench_manifest(
    *,
    root: Path = DEFAULT_MILEBENCH_ROOT,
    tokenizer: Any | None = None,
    n: int = 200,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> dict[str, Any]:
    if tokenizer is None:
        tokenizer = load_tokenizer()
    task_counts: dict[str, int] = {}
    truncated_counts: dict[str, int] = {}
    max_expanded = 0
    for dataset in MILEBENCH_DATASETS:
        samples = load_samples(
            dataset,
            n=n,
            seed=42,
            milebench_root=root,
            tokenizer=tokenizer,
            load_images=False,
            max_prompt_tokens=max_prompt_tokens,
        )
        ids = {(sample.row_index, sample.sample_id) for sample in samples}
        if len(samples) != n or len(ids) != n:
            raise AssertionError(
                f"{dataset}: expected {n} unique samples, got "
                f"{len(samples)}/{len(ids)}"
            )
        expanded = [
            int(sample.metadata["expanded_prompt_tokens"])
            for sample in samples
        ]
        if max(expanded) > max_prompt_tokens:
            raise AssertionError(
                f"{dataset}: expanded prompt exceeds {max_prompt_tokens}"
            )
        task_counts[dataset] = len(samples)
        truncated_counts[dataset] = sum(
            bool(sample.metadata["prompt_was_left_truncated"])
            for sample in samples
        )
        max_expanded = max(max_expanded, max(expanded))
    return {
        "n_tasks": len(MILEBENCH_DATASETS),
        "n_samples": sum(task_counts.values()),
        "n_per_task": n,
        "task_counts": task_counts,
        "truncated_counts": truncated_counts,
        "max_expanded_prompt_tokens": max_expanded,
    }


__all__ = [
    "DATASETS",
    "DEFAULT_EVAL_ROOT",
    "DEFAULT_IMAGE_FEATURE_LEN",
    "DEFAULT_MAX_PROMPT_TOKENS",
    "DEFAULT_MILEBENCH_MAX_NEW_TOKENS",
    "DEFAULT_MILEBENCH_ROOT",
    "EvalSample",
    "LOCAL_EXTENSION_TASKS",
    "LOCAL_MILEBENCH_TASKS",
    "MILEBENCH_DATASETS",
    "MILEBENCH_PREFIX",
    "OFFICIAL_MILEBENCH_TASKS",
    "STANDARD_DATASETS",
    "expanded_prompt_length",
    "is_milebench_dataset",
    "load_samples",
    "load_tokenizer",
    "milebench_task_name",
    "score_milebench_prediction",
    "score_prediction",
    "truncate_milebench_context",
    "validate_milebench_manifest",
    "write_manifest",
]
