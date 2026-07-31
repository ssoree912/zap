# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run matched-budget selector interventions on the held-out teacher samples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache


EXP_DIR = Path(__file__).resolve().parent
AGREEMENT_PATH = EXP_DIR / "analyze_llava15_signal_agreement.py"
SPEC = importlib.util.spec_from_file_location("signal_agreement", AGREEMENT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import {AGREEMENT_PATH}")
agreement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agreement)

from foresight.eval.kv_decode_utils import (  # noqa: E402
    greedy_decode_with_kv,
    trim_kv_cache_per_layer,
)


SELECTORS = ("full_cache", "prefill", "smoothed_prefill", "qvik", "future_oracle")
ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, default=agreement.DEFAULT_TEACHER_ROOT)
    parser.add_argument("--student-dir", type=Path, default=agreement.DEFAULT_STUDENT_DIR)
    parser.add_argument("--model-path", type=Path, default=agreement.DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=agreement.DEFAULT_OUTPUT_DIR / "intervention")
    parser.add_argument("--device", default="cuda:0")
    keep_group = parser.add_mutually_exclusive_group()
    keep_group.add_argument("--visual-keep-ratio", type=float, default=None)
    keep_group.add_argument("--total-keep-ratio", type=float, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(agreement.DEFAULT_DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip().replace("\n", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    punctuation = string.punctuation.replace(".", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    words = []
    for word in text.split():
        word = NUMBER_WORDS.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(words)


def prediction_variants(prediction: str) -> set[str]:
    first_line = next(
        (line.strip() for line in str(prediction).splitlines() if line.strip()),
        "",
    )
    stripped = re.sub(
        r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|option|choice)"
        r"\s*(?:is|:)?\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip(" \t\"'`")
    values = {
        normalize_answer(prediction),
        normalize_answer(first_line),
        normalize_answer(stripped),
    }
    return {value for value in values if value}


def scienceqa_letter(prediction: str, choice_count: int) -> int | None:
    first_line = next(
        (line.strip() for line in str(prediction).splitlines() if line.strip()),
        "",
    )
    for pattern in (
        r"^\s*\(?([a-z])\)?(?:[\s.:\-]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([a-z])\)?"
        r"(?:[\s.:\-]|$)",
    ):
        match = re.search(pattern, first_line, flags=re.IGNORECASE)
        if match:
            index = ord(match.group(1).lower()) - ord("a")
            return index if 0 <= index < choice_count else None
    return None


def score_prediction(
    dataset: str,
    prediction: str,
    *,
    answers: list[str],
    answer_index: int | None,
    choices: list[str],
) -> float:
    predicted = prediction_variants(prediction)
    if dataset == "scienceqa":
        if answer_index is None:
            return 0.0
        predicted_index = scienceqa_letter(prediction, len(choices))
        if predicted_index is not None:
            return float(predicted_index == answer_index)
        expected = normalize_answer(choices[answer_index])
        return float(expected in predicted)
    normalized_answers = [normalize_answer(answer) for answer in answers]
    if dataset == "gqa":
        return float(bool(predicted & set(normalized_answers)))
    if dataset == "textvqa":
        # Standard VQA soft accuracy for the best normalized prediction variant.
        return max(
            (
                min(
                    sum(candidate == answer for answer in normalized_answers) / 3.0,
                    1.0,
                )
                for candidate in predicted
            ),
            default=0.0,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_ground_truth() -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    textvqa_path = agreement.WORKSPACE_ROOT / "data/train/textvqa/train/data.json"
    textvqa_payload = json.loads(textvqa_path.read_text())
    textvqa_rows = (
        textvqa_payload["data"]
        if isinstance(textvqa_payload, dict)
        else textvqa_payload
    )
    textvqa = {
        str(row.get("question_id", row.get("id"))): [
            str(answer) for answer in row.get("answers", [])
        ]
        for row in textvqa_rows
    }
    scienceqa_path = agreement.WORKSPACE_ROOT / "data/train/scienceqa/problems.json"
    scienceqa = json.loads(scienceqa_path.read_text())
    return textvqa, scienceqa


def ground_truth_for(
    record: dict[str, Any],
    textvqa: dict[str, list[str]],
    scienceqa: dict[str, dict[str, Any]],
) -> tuple[list[str], int | None, list[str]]:
    dataset = str(record["dataset"])
    sample_id = str(record["sample_id"])
    if dataset == "textvqa":
        return textvqa[sample_id], None, []
    if dataset == "scienceqa":
        problem = scienceqa[sample_id]
        return [], int(problem["answer"]), [str(choice) for choice in problem["choices"]]
    return [str(record["answer"])], None, []


def clone_cache(cache: Any) -> Any:
    if hasattr(cache, "key_cache"):
        cloned = DynamicCache()
        cloned.key_cache = [keys.clone() for keys in cache.key_cache]
        cloned.value_cache = [values.clone() for values in cache.value_cache]
        return cloned
    if hasattr(cache, "layers"):
        raise TypeError("Cache layers API is not supported by this intervention script")
    return tuple(
        tuple(tensor.clone() for tensor in layer)
        for layer in cache
    )


def make_keep_masks(
    scores: torch.Tensor,
    *,
    image_positions: torch.Tensor,
    prompt_len: int,
    n_keep: int,
) -> dict[int, torch.Tensor]:
    masks: dict[int, torch.Tensor] = {}
    n_visual = int(image_positions.numel())
    for layer in range(scores.shape[0]):
        top = torch.topk(scores[layer], k=n_keep, largest=True).indices
        image_keep = torch.zeros(n_visual, dtype=torch.bool)
        image_keep[top.detach().cpu()] = True
        mask = torch.ones(prompt_len, dtype=torch.bool)
        mask[image_positions] = image_keep
        masks[layer] = mask
    return masks


def compute_n_visual_keep(
    *,
    n_visual: int,
    prompt_len: int,
    visual_keep_ratio: float | None,
    total_keep_ratio: float | None,
) -> int:
    if (visual_keep_ratio is None) == (total_keep_ratio is None):
        raise ValueError("Specify exactly one keep-ratio basis")
    if visual_keep_ratio is not None:
        if not 0.0 <= visual_keep_ratio <= 1.0:
            raise ValueError("visual_keep_ratio must be in [0, 1]")
        n_keep = int(round(n_visual * visual_keep_ratio))
    else:
        assert total_keep_ratio is not None
        if not 0.0 <= total_keep_ratio <= 1.0:
            raise ValueError("total_keep_ratio must be in [0, 1]")
        # Match the paper-style original Q-ViK wrapper exactly: preserve every
        # text token, then fill the remaining total-prompt budget with image KV.
        n_keep = int(round(n_visual - (1.0 - total_keep_ratio) * prompt_len))
    return max(1, min(n_visual, n_keep))


def layer_jaccard_at_k(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    k: int,
) -> list[float]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Jaccard inputs must have the same [layers, tokens] shape")
    k = max(1, min(int(first.shape[-1]), int(k)))
    values: list[float] = []
    for first_layer, second_layer in zip(first, second, strict=True):
        first_top = set(torch.topk(first_layer, k=k).indices.tolist())
        second_top = set(torch.topk(second_layer, k=k).indices.tolist())
        values.append(len(first_top & second_top) / len(first_top | second_top))
    return values


@torch.inference_mode()
def run_one(
    *,
    record_path: Path,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    student: Any,
    image_feature_len: int,
    device: torch.device,
    visual_keep_ratio: float | None,
    total_keep_ratio: float | None,
    textvqa_ground_truth: dict[str, list[str]],
    scienceqa_ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = torch.load(record_path, weights_only=False, map_location="cpu")
    with Image.open(agreement._resolve_image_path(record["image_path"])) as image:
        image_tensor = image_processor.preprocess(
            image.convert("RGB"),
            return_tensors="pt",
        )["pixel_values"]
    image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)
    input_ids = agreement._tokenize_prompt(
        record["prompt_text"],
        tokenizer,
    ).unsqueeze(0).to(device)
    image_positions, prompt_len, placeholder = agreement.infer_image_positions(
        input_ids,
        image_feature_len,
    )
    user_question_positions = agreement.infer_user_question_positions(
        prompt=record["prompt_text"],
        question=record["question"],
        tokenizer=tokenizer,
        full_input_ids=input_ids,
        image_placeholder=placeholder,
        image_feature_len=image_feature_len,
    )
    student_question_positions = record["question_token_indices"].to(dtype=torch.long)

    prefill = model(
        input_ids=input_ids,
        images=image_tensor,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    if int(prefill.logits.shape[1]) != prompt_len:
        raise ValueError("Prefill length mismatch")
    first_next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    image_idx = image_positions.to(device)
    question_idx = user_question_positions.to(device)
    student_q_idx = student_question_positions.to(device)

    prefill_scores: list[torch.Tensor] = []
    qvik_scores: list[torch.Tensor] = []
    for layer, attention in enumerate(prefill.attentions):
        selected = attention[0, :, question_idx, :].index_select(
            dim=-1,
            index=image_idx,
        )
        prefill_scores.append(selected.float().mean(dim=(0, 1)).cpu())
        qvik_scores.append(
            student.forward_layer(
                layer,
                prefill.hidden_states[layer + 1].float(),
                image_idx,
                student_q_idx,
            ).squeeze(0).cpu()
        )
    prefill_tensor = torch.stack(prefill_scores)
    selector_scores = {
        "prefill": prefill_tensor,
        "smoothed_prefill": agreement.smooth_visual_grid(
            prefill_tensor,
            grid_h=student.grid_h,
            grid_w=student.grid_w,
        ),
        "qvik": torch.stack(qvik_scores),
    }

    n_visual = int(image_positions.numel())
    n_keep = compute_n_visual_keep(
        n_visual=n_visual,
        prompt_len=prompt_len,
        visual_keep_ratio=visual_keep_ratio,
        total_keep_ratio=total_keep_ratio,
    )
    n_text = prompt_len - n_visual
    keep_ratio_basis = "total" if total_keep_ratio is not None else "visual"
    requested_keep_ratio = (
        total_keep_ratio
        if total_keep_ratio is not None
        else visual_keep_ratio
    )
    assert requested_keep_ratio is not None
    eos_token_id = int(tokenizer.eos_token_id or model.config.eos_token_id or 2)
    predictions: dict[str, str] = {}
    base_cache = prefill.past_key_values
    max_new_tokens = int(record["max_new_tokens"])
    full_answer_ids = greedy_decode_with_kv(
        model,
        clone_cache(base_cache),
        first_next_token,
        prompt_len=prompt_len,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    predictions["full_cache"] = tokenizer.decode(
        full_answer_ids.tolist(),
        skip_special_tokens=True,
    ).strip()

    # Recompute the privileged selector from the answer trajectory produced by
    # this exact model invocation. The saved teacher remains the correct target
    # for agreement analysis, but can differ when generation code has changed.
    answer_ids_device = full_answer_ids.to(device=device).unsqueeze(0)
    full_input_ids = torch.cat([input_ids, answer_ids_device], dim=1)
    future_output = model(
        input_ids=full_input_ids,
        images=image_tensor,
        use_cache=False,
        output_hidden_states=False,
        output_attentions=True,
        return_dict=True,
    )
    answer_positions = torch.arange(
        prompt_len,
        prompt_len + int(full_answer_ids.numel()),
        dtype=torch.long,
        device=device,
    )
    future_scores: list[torch.Tensor] = []
    for attention in future_output.attentions:
        selected = attention[0, :, answer_positions, :].index_select(
            dim=-1,
            index=image_idx,
        )
        future_scores.append(selected.float().mean(dim=(0, 1)).cpu())
    selector_scores["future_oracle"] = torch.stack(future_scores)
    del future_output, full_input_ids, answer_ids_device
    saved_future_scores = record["teacher_norm"].to(dtype=torch.float32)
    exact_budget_agreement = {
        "prefill_vs_future": layer_jaccard_at_k(
            selector_scores["prefill"],
            saved_future_scores,
            k=n_keep,
        ),
        "smoothed_prefill_vs_future": layer_jaccard_at_k(
            selector_scores["smoothed_prefill"],
            saved_future_scores,
            k=n_keep,
        ),
        "qvik_vs_future": layer_jaccard_at_k(
            selector_scores["qvik"],
            saved_future_scores,
            k=n_keep,
        ),
        "qvik_vs_prefill": layer_jaccard_at_k(
            selector_scores["qvik"],
            selector_scores["prefill"],
            k=n_keep,
        ),
    }

    for selector in SELECTORS[1:]:
        cache = clone_cache(base_cache)
        masks = make_keep_masks(
            selector_scores[selector],
            image_positions=image_positions,
            prompt_len=prompt_len,
            n_keep=n_keep,
        )
        cache = trim_kv_cache_per_layer(cache, masks)
        answer_ids = greedy_decode_with_kv(
            model,
            cache,
            first_next_token,
            prompt_len=prompt_len,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        predictions[selector] = tokenizer.decode(
            answer_ids.tolist(),
            skip_special_tokens=True,
        ).strip()
        del cache

    answers, answer_index, choices = ground_truth_for(
        record,
        textvqa_ground_truth,
        scienceqa_ground_truth,
    )
    scores = {
        selector: score_prediction(
            str(record["dataset"]),
            prediction,
            answers=answers,
            answer_index=answer_index,
            choices=choices,
        )
        for selector, prediction in predictions.items()
    }
    return {
        "schema_version": 2,
        "dataset": str(record["dataset"]),
        "sample_id": str(record["sample_id"]),
        "source_record": str(record_path.resolve()),
        "keep_ratio_basis": keep_ratio_basis,
        "requested_keep_ratio": requested_keep_ratio,
        "total_keep_ratio_actual": (n_text + n_keep) / prompt_len,
        "visual_keep_ratio_actual": n_keep / n_visual,
        "prompt_len": prompt_len,
        "n_text": n_text,
        "n_visual": n_visual,
        "n_visual_kept": n_keep,
        "future_oracle_source": "current_full_cache_answer_attention",
        "exact_budget_jaccard": exact_budget_agreement,
        "exact_budget_jaccard_future_source": "saved_training_teacher",
        "predictions": predictions,
        "scores": scores,
    }


def bootstrap_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    cursor = 0
    while cursor < replicates:
        chunk = min(512, replicates - cursor)
        indices = rng.integers(0, len(values), size=(chunk, len(values)))
        means[cursor : cursor + chunk] = values[indices].mean(axis=1)
        cursor += chunk
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    replicates: int,
    seed: int,
) -> None:
    rows: list[dict[str, Any]] = []
    groups = [("all", records)] + [
        (
            dataset,
            [record for record in records if record["dataset"] == dataset],
        )
        for dataset in agreement.DEFAULT_DATASETS
    ]
    for group, selected in groups:
        if not selected:
            continue
        for selector_index, selector in enumerate(SELECTORS):
            values = np.asarray(
                [record["scores"][selector] for record in selected],
                dtype=np.float64,
            )
            mean, low, high = bootstrap_ci(
                values,
                replicates=replicates,
                seed=seed + selector_index,
            )
            rows.append(
                {
                    "group": group,
                    "n_samples": len(selected),
                    "selector": selector,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        prefill = np.asarray(
            [record["scores"]["prefill"] for record in selected],
            dtype=np.float64,
        )
        for selector_index, selector in enumerate(("smoothed_prefill", "qvik", "future_oracle")):
            values = np.asarray(
                [record["scores"][selector] for record in selected],
                dtype=np.float64,
            )
            mean, low, high = bootstrap_ci(
                values - prefill,
                replicates=replicates,
                seed=seed + 100 + selector_index,
            )
            rows.append(
                {
                    "group": group,
                    "n_samples": len(selected),
                    "selector": f"{selector}_minus_prefill",
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    import csv

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(row["group"], row["selector"]): row for row in rows}
    table_groups = [
        group
        for group in ("all", "textvqa", "scienceqa", "gqa")
        if (group, "full_cache") in lookup
    ]
    group_labels = {
        "all": "Overall",
        "textvqa": "TextVQA",
        "scienceqa": "ScienceQA",
        "gqa": "GQA",
    }
    keep_ratio_basis = records[0]["keep_ratio_basis"]
    requested_keep_ratio = records[0]["requested_keep_ratio"]
    mean_total_keep_ratio = float(
        np.mean([record["total_keep_ratio_actual"] for record in records])
    )
    mean_visual_keep_ratio = float(
        np.mean([record["visual_keep_ratio_actual"] for record in records])
    )
    lines = [
        "# Held-out selector intervention",
        "",
        f"Requested {keep_ratio_basis}-token keep ratio: {requested_keep_ratio:.0%}. "
        f"Mean actual total-token keep ratio: {mean_total_keep_ratio:.2%}. "
        f"Mean actual visual-token keep ratio: {mean_visual_keep_ratio:.2%}. "
        f"Sample bootstrap: {replicates:,} replicates.",
        "",
        "| Selector | " + " | ".join(group_labels[group] for group in table_groups) + " |",
        "|---|" + "---:|" * len(table_groups),
    ]
    labels = {
        "full_cache": "Full cache",
        "prefill": "Prefill attention",
        "smoothed_prefill": "Smoothed prefill",
        "qvik": "Q-ViK",
        "future_oracle": "Future Oracle",
    }
    for selector in SELECTORS:
        cells = []
        for group in table_groups:
            row = lookup[(group, selector)]
            cells.append(
                f"{row['mean']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            )
        lines.append(f"| {labels[selector]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Overall is the micro-average of dataset-native scores. TextVQA uses "
            "VQA soft accuracy; ScienceQA and GQA use accuracy.",
        ]
    )
    (output_dir / "table.md").write_text("\n".join(lines) + "\n")

    agreement_rows: list[dict[str, Any]] = []
    agreement_pairs = (
        "prefill_vs_future",
        "smoothed_prefill_vs_future",
        "qvik_vs_future",
        "qvik_vs_prefill",
    )
    for group, selected in groups:
        if not selected:
            continue
        sample_means: dict[str, np.ndarray] = {}
        for pair_index, pair in enumerate(agreement_pairs):
            values = np.asarray(
                [
                    np.mean(record["exact_budget_jaccard"][pair])
                    for record in selected
                ],
                dtype=np.float64,
            )
            sample_means[pair] = values
            mean, low, high = bootstrap_ci(
                values,
                replicates=replicates,
                seed=seed + 1_000 + pair_index,
            )
            agreement_rows.append(
                {
                    "group": group,
                    "n_samples": len(selected),
                    "pair": pair,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        delta = sample_means["qvik_vs_future"] - sample_means["prefill_vs_future"]
        mean, low, high = bootstrap_ci(
            delta,
            replicates=replicates,
            seed=seed + 2_000,
        )
        agreement_rows.append(
            {
                "group": group,
                "n_samples": len(selected),
                "pair": "qvik_minus_prefill_vs_future",
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    with (output_dir / "agreement_at_budget.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]))
        writer.writeheader()
        writer.writerows(agreement_rows)
    agreement_lookup = {
        (row["group"], row["pair"]): row for row in agreement_rows
    }
    agreement_lines = [
        "# Exact-budget visual top-K agreement",
        "",
        f"Requested {keep_ratio_basis}-token keep ratio: {requested_keep_ratio:.0%}. "
        f"Mean actual visual-token keep ratio: {mean_visual_keep_ratio:.2%}.",
        "Future is the saved teacher target used to train Q-ViK.",
        "",
        "| Compared pair | Jaccard (95% CI) |",
        "|---|---:|",
    ]
    agreement_labels = {
        "prefill_vs_future": "Prefill vs. Future",
        "smoothed_prefill_vs_future": "Smoothed Prefill vs. Future",
        "qvik_vs_future": "Q-ViK vs. Future",
        "qvik_vs_prefill": "Q-ViK vs. Prefill",
        "qvik_minus_prefill_vs_future": "Q-ViK − Prefill vs. Future",
    }
    for pair in (*agreement_pairs, "qvik_minus_prefill_vs_future"):
        row = agreement_lookup[("all", pair)]
        agreement_lines.append(
            f"| {agreement_labels[pair]} | "
            f"{row['mean']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    (output_dir / "agreement_at_budget.md").write_text(
        "\n".join(agreement_lines) + "\n"
    )


def main() -> int:
    args = parse_args()
    if args.visual_keep_ratio is None and args.total_keep_ratio is None:
        args.visual_keep_ratio = 0.2
    for attr in ("teacher_root", "student_dir", "model_path", "output_dir"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    validation_paths = agreement.recover_validation_paths(args.student_dir)
    files = [
        path
        for dataset in args.datasets
        for path in sorted((args.teacher_root / dataset).glob("*.pt"))
        if path.resolve() in validation_paths
    ]
    if args.limit is not None:
        files = files[: args.limit]
    textvqa_ground_truth, scienceqa_ground_truth = load_ground_truth()

    print(
        f"[load] model={args.model_path} student={args.student_dir} "
        f"heldout_samples={len(files)} device={args.device}",
        flush=True,
    )
    tokenizer, model, image_processor, student, image_feature_len = (
        agreement._load_model_and_student(args)
    )
    print("[load-ok]", flush=True)

    failures: list[dict[str, str]] = []
    start = time.time()
    for index, path in enumerate(files, start=1):
        output_path = sample_dir / path.parent.name / f"{path.stem}.json"
        if output_path.exists() and not args.overwrite:
            continue
        try:
            result = run_one(
                record_path=path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                student=student,
                image_feature_len=image_feature_len,
                device=torch.device(args.device),
                visual_keep_ratio=args.visual_keep_ratio,
                total_keep_ratio=args.total_keep_ratio,
                textvqa_ground_truth=textvqa_ground_truth,
                scienceqa_ground_truth=scienceqa_ground_truth,
            )
            agreement._dump_json(output_path, result)
        except Exception as exc:  # noqa: BLE001
            failures.append({"record": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[failed] {path}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            torch.cuda.empty_cache()
        if index == 1 or index % 10 == 0 or index == len(files):
            print(
                f"[progress] {index}/{len(files)} failures={len(failures)} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )

    records = [
        json.loads(path.read_text())
        for path in sorted(sample_dir.glob("*/*.json"))
    ]
    if failures:
        agreement._dump_json(args.output_dir / "failures.json", failures)
        return 1
    if len(records) != len(files):
        raise RuntimeError(f"Expected {len(files)} results, found {len(records)}")
    summarize(
        records,
        output_dir=args.output_dir,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(f"[done] output={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
