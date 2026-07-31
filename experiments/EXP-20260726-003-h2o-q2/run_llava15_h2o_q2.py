#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Corrected Q2 analysis: H2O prefill accumulation versus future utility.

The primary H2O signal is accumulated over *all* causal prefill query rows and
keeps a separate Top-K set for every attention/KV head, matching
``H2OImageOnlyPress`` for LLaVA-1.5 (32 attention heads == 32 KV heads).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache

EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
LEGACY_EXP = ZAP_ROOT / "experiments" / "EXP-20260726-001-signal-agreement"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


agreement = load_module(
    "q2_legacy_agreement",
    LEGACY_EXP / "analyze_llava15_signal_agreement.py",
)
intervention = load_module(
    "q2_legacy_intervention",
    LEGACY_EXP / "run_llava15_selector_intervention.py",
)

from qvik.llava15.mm_utils import process_images  # noqa: E402
from qvik.llava15.model.builder import load_pretrained_model  # noqa: E402

DEFAULT_OUTPUT = (
    ZAP_ROOT
    / "artifacts"
    / "rebuttal_h2o_q2_llava15_base_n600_total0p2"
)
SELECTORS = (
    "full_cache_manual",
    "full_cache",
    "h2o_prefill",
    "question_prefill",
    "qvik",
    "future_oracle",
)
VECTOR_PAIRS = (
    "h2o_vs_future",
    "question_prefill_vs_future",
    "qvik_vs_future",
    "qvik_vs_h2o",
    "captured_future_vs_saved_teacher",
)
TOKEN_METHODS = ("h2o", "question", "qvik")
TOKEN_METRICS = (
    "future_topk_recall",
    "jaccard_vs_future",
    "visual_future_mass_retained",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, default=agreement.DEFAULT_TEACHER_ROOT)
    parser.add_argument("--student-dir", type=Path, default=agreement.DEFAULT_STUDENT_DIR)
    parser.add_argument("--model-path", type=Path, default=agreement.DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=list(agreement.DEFAULT_DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-keep-ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_exact_model_and_student(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, Any, int]:
    """Load through the same Q-ViK backend used to create the n600 artifact."""
    agreement._patch_generation_config_nested_dicts()
    agreement._patch_llava_arch_for_dynamic_cache()
    tokenizer, model, image_processor, _context_length = load_pretrained_model(
        model_path=str(args.model_path),
        model_base=None,
        model_name="llava-v1.5-7b",
        device_map=args.device,
    )
    model.eval()
    model_dtype = next(model.parameters()).dtype
    student = agreement.VisualUtilityStudent.from_pretrained(
        args.student_dir,
        map_location="cpu",
    ).to(device=torch.device(args.device), dtype=model_dtype).eval()
    vision_tower = model.get_vision_tower()
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    return tokenizer, model, image_processor, student, image_feature_len


def normalize_rows(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values = torch.nan_to_num(
        values.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0)
    sums = values.sum(dim=-1, keepdim=True)
    normalized = values / sums.clamp_min(eps)
    bad = sums.squeeze(-1) <= eps
    if bad.any():
        normalized[bad] = 1.0 / values.shape[-1]
    return normalized


def topk_mask(values: torch.Tensor, k: int) -> torch.Tensor:
    k = max(0, min(int(k), int(values.shape[-1])))
    if k == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    indices = torch.topk(values, k=k, dim=-1).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def exact_total_budget(
    *,
    n_visual: int,
    prompt_len: int,
    total_keep_ratio: float,
) -> int:
    """Match ImageTokenTopKPress' total-budget calculation exactly."""
    n_text = prompt_len - n_visual
    total_keep = int(math.ceil(total_keep_ratio * prompt_len))
    return min(n_visual, max(0, total_keep - n_text))


def exact_head_token_metrics(
    *,
    h2o_heads: torch.Tensor,
    question_scores: torch.Tensor,
    qvik_scores: torch.Tensor,
    future_heads: torch.Tensor,
    k: int,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    """Compare actual per-head keep decisions at the exact eviction budget."""
    if h2o_heads.ndim != 3 or future_heads.shape != h2o_heads.shape:
        raise ValueError(
            f"Expected matching [L,H,N] H2O/Future, got "
            f"{tuple(h2o_heads.shape)} and {tuple(future_heads.shape)}"
        )
    expected_shared_shape = (h2o_heads.shape[0], h2o_heads.shape[2])
    if question_scores.shape != expected_shared_shape:
        raise ValueError("Question score shape does not match H2O token dimensions")
    if qvik_scores.shape != expected_shared_shape:
        raise ValueError("Q-ViK score shape does not match H2O token dimensions")
    layers, heads, tokens = h2o_heads.shape
    k = max(0, min(k, tokens))
    if k == 0:
        raise ValueError(
            "The requested total budget leaves zero visual tokens; "
            "head-resolved Top-K agreement is undefined at K=0"
        )
    h2o_keep = topk_mask(h2o_heads, k)
    question_keep_shared = topk_mask(question_scores, k)
    question_keep = question_keep_shared.unsqueeze(1).expand(-1, heads, -1)
    qvik_keep_shared = topk_mask(qvik_scores, k)
    qvik_keep = qvik_keep_shared.unsqueeze(1).expand(-1, heads, -1)
    future_keep = topk_mask(future_heads, k)
    future_norm = normalize_rows(future_heads)

    output: dict[str, list[float]] = {
        f"{method}_{metric}": []
        for method in TOKEN_METHODS
        for metric in TOKEN_METRICS
    }
    output.update(
        {
            "qvik_minus_h2o_future_recall": [],
            "qvik_minus_h2o_jaccard": [],
            "qvik_minus_h2o_future_mass": [],
            "qvik_useful_additions_per_k": [],
            "qvik_harmful_removals_per_k": [],
            "qvik_head_win_fraction": [],
            "qvik_head_tie_fraction": [],
            "qvik_head_loss_fraction": [],
        }
    )

    for layer in range(layers):
        future_layer = future_keep[layer]
        future_mass = future_norm[layer]
        hit_counts: dict[str, torch.Tensor] = {}
        jaccards: dict[str, torch.Tensor] = {}
        retained_mass: dict[str, torch.Tensor] = {}
        for method, keep in (
            ("h2o", h2o_keep[layer]),
            ("question", question_keep[layer]),
            ("qvik", qvik_keep[layer]),
        ):
            intersection = (keep & future_layer).sum(dim=-1).float()
            union = (keep | future_layer).sum(dim=-1).float()
            mass = (future_mass * keep.float()).sum(dim=-1)
            hit_counts[method] = intersection
            jaccards[method] = intersection / union.clamp_min(1)
            retained_mass[method] = mass
            output[f"{method}_future_topk_recall"].append(
                float((intersection / k).mean())
            )
            output[f"{method}_jaccard_vs_future"].append(
                float((intersection / union.clamp_min(1)).mean())
            )
            output[f"{method}_visual_future_mass_retained"].append(
                float(mass.mean())
            )

        h_keep = h2o_keep[layer]
        q_keep = qvik_keep[layer]
        q_only = q_keep & ~h_keep
        h_only = h_keep & ~q_keep
        useful_add = (q_only & future_layer).sum(dim=-1).float()
        useful_remove = (h_only & future_layer).sum(dim=-1).float()
        delta_hits = hit_counts["qvik"] - hit_counts["h2o"]
        output["qvik_minus_h2o_future_recall"].append(
            float((delta_hits / k).mean())
        )
        output["qvik_minus_h2o_jaccard"].append(
            float((jaccards["qvik"] - jaccards["h2o"]).mean())
        )
        output["qvik_minus_h2o_future_mass"].append(
            float((retained_mass["qvik"] - retained_mass["h2o"]).mean())
        )
        output["qvik_useful_additions_per_k"].append(
            float((useful_add / k).mean())
        )
        output["qvik_harmful_removals_per_k"].append(
            float((useful_remove / k).mean())
        )
        output["qvik_head_win_fraction"].append(
            float((delta_hits > 0).float().mean())
        )
        output["qvik_head_tie_fraction"].append(
            float((delta_hits == 0).float().mean())
        )
        output["qvik_head_loss_fraction"].append(
            float((delta_hits < 0).float().mean())
        )
    return output, {
        "h2o_keep": h2o_keep,
        "question_keep": question_keep,
        "qvik_keep": qvik_keep,
        "future_keep": future_keep,
    }


def trim_cache_per_head(
    cache: Any,
    *,
    head_scores: torch.Tensor,
    image_positions: torch.Tensor,
    prompt_len: int,
    n_keep: int,
) -> Any:
    """Apply H2O's separate visual Top-K positions to each KV head."""
    image_positions = image_positions.to(dtype=torch.long)
    all_positions = torch.arange(prompt_len, dtype=torch.long)
    is_text = torch.ones(prompt_len, dtype=torch.bool)
    is_text[image_positions] = False
    text_positions = all_positions[is_text]

    def trim_one(
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads = int(keys.shape[1])
        if layer_scores.shape[0] != heads:
            raise ValueError(
                f"H2O score/cache head mismatch: {layer_scores.shape[0]} vs {heads}"
            )
        top = torch.topk(layer_scores, k=n_keep, dim=-1).indices.cpu()
        visual_absolute = image_positions[top]
        text = text_positions.unsqueeze(0).expand(heads, -1)
        positions = torch.cat([text, visual_absolute], dim=-1).sort(dim=-1).values
        gather = positions.to(keys.device).unsqueeze(0).unsqueeze(-1).expand(
            keys.shape[0],
            heads,
            positions.shape[-1],
            keys.shape[-1],
        )
        return (
            keys.gather(2, gather).contiguous(),
            values.gather(2, gather).contiguous(),
        )

    if hasattr(cache, "key_cache"):
        for layer in range(len(cache.key_cache)):
            keys, values = trim_one(
                cache.key_cache[layer],
                cache.value_cache[layer],
                head_scores[layer],
            )
            cache.key_cache[layer] = keys
            cache.value_cache[layer] = values
        return cache

    layers = []
    for layer, (keys, values) in enumerate(cache):
        layers.append(trim_one(keys, values, head_scores[layer]))
    return tuple(layers)


def shared_keep_masks(
    scores: torch.Tensor,
    *,
    image_positions: torch.Tensor,
    prompt_len: int,
    n_keep: int,
) -> dict[int, torch.Tensor]:
    return intervention.make_keep_masks(
        scores,
        image_positions=image_positions,
        prompt_len=prompt_len,
        n_keep=n_keep,
    )


def decode_from_cache(
    *,
    model: Any,
    tokenizer: Any,
    cache: Any,
    first_token: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
) -> tuple[torch.Tensor, str]:
    eos = int(tokenizer.eos_token_id or model.config.eos_token_id or 2)
    ids = intervention.greedy_decode_with_kv(
        model,
        cache,
        first_token,
        prompt_len=prompt_len,
        eos_token_id=eos,
        max_new_tokens=max_new_tokens,
    )
    return ids, tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()


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
    total_keep_ratio: float,
    textvqa_ground_truth: dict[str, list[str]],
    scienceqa_ground_truth: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    record = torch.load(record_path, weights_only=False, map_location="cpu")
    with Image.open(agreement._resolve_image_path(record["image_path"])) as image:
        image = image.convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    input_ids = agreement._tokenize_prompt(
        record["prompt_text"], tokenizer
    ).unsqueeze(0).to(device)
    image_positions, prompt_len, placeholder = agreement.infer_image_positions(
        input_ids, image_feature_len
    )
    if not torch.equal(
        image_positions,
        record["image_token_indices"].to(dtype=torch.long),
    ):
        raise ValueError("Live image-token positions differ from the teacher record")
    user_positions = agreement.infer_user_question_positions(
        prompt=record["prompt_text"],
        question=record["question"],
        tokenizer=tokenizer,
        full_input_ids=input_ids,
        image_placeholder=placeholder,
        image_feature_len=image_feature_len,
    )
    student_positions = record["question_token_indices"].to(dtype=torch.long)
    image_idx = image_positions.to(device)
    user_idx = user_positions.to(device)
    student_idx = student_positions.to(device)

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
    if prefill.attentions is None or prefill.hidden_states is None:
        raise RuntimeError("Prefill did not return attentions/hidden states")
    first_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    base_cache = prefill.past_key_values

    h2o_heads = []
    question_scores = []
    qvik_scores = []
    for layer, attention in enumerate(prefill.attentions):
        # Exact H2O prefill criterion: every causal prefill query contributes.
        received = attention[0].sum(dim=1).float()
        h2o_heads.append(received.index_select(-1, image_idx).cpu())
        selected_question = attention[0, :, user_idx, :].index_select(
            -1, image_idx
        )
        question_scores.append(
            selected_question.float().mean(dim=(0, 1)).cpu()
        )
        qvik_scores.append(
            student.forward_layer(
                layer,
                # Artifact-specific provenance: the tradeoff trainer that made
                # this checkpoint used the post-layer stream H_{l+1}.
                prefill.hidden_states[layer + 1],
                image_idx,
                student_idx,
            )[0].float().cpu()
        )
    h2o_heads_tensor = torch.stack(h2o_heads)
    h2o_vector = normalize_rows(h2o_heads_tensor.mean(dim=1))
    question_vector = normalize_rows(torch.stack(question_scores))
    qvik_vector = torch.softmax(torch.stack(qvik_scores), dim=-1)
    del prefill

    max_new_tokens = int(record["max_new_tokens"])
    generated = model.generate(
        inputs=input_ids,
        images=image_tensor,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if generated.attentions is None or len(generated.attentions) == 0:
        raise RuntimeError("Full-cache generation did not return attentions")
    t_steps = len(generated.attentions)
    full_ids = generated.sequences[0, -t_steps:].detach().cpu()
    full_text = tokenizer.decode(
        full_ids.tolist(), skip_special_tokens=True
    ).strip()
    expected_answer_tokens = int(record["T"])
    expected_text = str(record["decoded"]).strip()
    if t_steps != expected_answer_tokens or full_text != expected_text:
        raise ValueError(
            "Exact teacher-generation replay differs from its stored trajectory: "
            f"tokens={t_steps}/{expected_answer_tokens}, "
            f"text={full_text!r}/{expected_text!r}"
        )
    if int(first_token.item()) != int(full_ids[0]):
        raise ValueError("Standalone prefill and generate disagree on the first token")

    future_by_layer: list[list[torch.Tensor]] = [
        [] for _ in range(len(generated.attentions[0]))
    ]
    for step_attentions in generated.attentions:
        for layer, attention in enumerate(step_attentions):
            selected = attention[0, :, -1, :].index_select(-1, image_idx)
            future_by_layer[layer].append(selected.float().cpu())
    future_heads_tensor = torch.stack(
        [
            torch.stack(layer_steps, dim=0).mean(dim=0)
            for layer_steps in future_by_layer
        ],
        dim=0,
    )
    captured_future_vector = normalize_rows(future_heads_tensor.mean(dim=1))
    saved_future_vector = normalize_rows(
        record["teacher_norm"].to(dtype=torch.float32)
    )
    teacher_max_abs_error = float(
        (captured_future_vector - saved_future_vector).abs().max()
    )
    teacher_mean_abs_error = float(
        (captured_future_vector - saved_future_vector).abs().mean()
    )
    if teacher_max_abs_error > 1e-3:
        raise ValueError(
            "Direct generation attention does not reproduce the stored teacher: "
            f"max_abs_error={teacher_max_abs_error:.6g}"
        )
    del generated

    matched_full_ids, matched_full_text = decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        cache=intervention.clone_cache(base_cache),
        first_token=first_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )

    n_visual = int(image_positions.numel())
    n_keep = exact_total_budget(
        n_visual=n_visual,
        prompt_len=prompt_len,
        total_keep_ratio=total_keep_ratio,
    )
    n_text = prompt_len - n_visual

    token_metrics, keep_masks = exact_head_token_metrics(
        h2o_heads=h2o_heads_tensor,
        question_scores=question_vector,
        qvik_scores=qvik_vector,
        future_heads=future_heads_tensor,
        k=n_keep,
    )
    vector_pairs = {
        "h2o_vs_future": agreement.compute_pair_metrics(
            h2o_vector, saved_future_vector
        ),
        "question_prefill_vs_future": agreement.compute_pair_metrics(
            question_vector, saved_future_vector
        ),
        "qvik_vs_future": agreement.compute_pair_metrics(
            qvik_vector, saved_future_vector
        ),
        "qvik_vs_h2o": agreement.compute_pair_metrics(
            qvik_vector, h2o_vector
        ),
        "captured_future_vs_saved_teacher": agreement.compute_pair_metrics(
            captured_future_vector, saved_future_vector
        ),
    }

    predictions = {
        "full_cache": full_text,
        "full_cache_manual": matched_full_text,
    }

    h2o_cache = trim_cache_per_head(
        intervention.clone_cache(base_cache),
        head_scores=h2o_heads_tensor,
        image_positions=image_positions,
        prompt_len=prompt_len,
        n_keep=n_keep,
    )
    _, predictions["h2o_prefill"] = decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        cache=h2o_cache,
        first_token=first_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    del h2o_cache

    shared_selectors = {
        "question_prefill": question_vector,
        "qvik": qvik_vector,
        "future_oracle": saved_future_vector,
    }
    for name, scores in shared_selectors.items():
        cache = intervention.clone_cache(base_cache)
        cache = intervention.trim_kv_cache_per_layer(
            cache,
            shared_keep_masks(
                scores,
                image_positions=image_positions,
                prompt_len=prompt_len,
                n_keep=n_keep,
            ),
        )
        _, predictions[name] = decode_from_cache(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            first_token=first_token,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        )
        del cache
    del base_cache

    answers, answer_index, choices = intervention.ground_truth_for(
        record, textvqa_ground_truth, scienceqa_ground_truth
    )
    scores = {
        name: intervention.score_prediction(
            str(record["dataset"]),
            prediction,
            answers=answers,
            answer_index=answer_index,
            choices=choices,
        )
        for name, prediction in predictions.items()
    }
    result = {
        "schema_version": 1,
        "dataset": str(record["dataset"]),
        "sample_id": str(record["sample_id"]),
        "source_record": str(record_path.resolve()),
        "baseline": "H2O-style all-prefill-query cumulative attention",
        "head_policy": "per-attention/KV-head Top-K (32 heads, no GQA)",
        "future_source": (
            "stored n600 teacher target from full-cache HF generate attention "
            "blocks: [last prompt, y_1, ..., y_(T-1)]"
        ),
        "head_resolved_future_source": (
            "direct per-head capture from the same full-cache generate blocks"
        ),
        "teacher_replay_max_abs_error": teacher_max_abs_error,
        "teacher_replay_mean_abs_error": teacher_mean_abs_error,
        "full_cache_manual_matches_generate": bool(
            torch.equal(matched_full_ids, full_ids)
        ),
        "requested_total_keep_ratio": total_keep_ratio,
        "actual_total_keep_ratio": (n_text + n_keep) / prompt_len,
        "actual_visual_keep_ratio": n_keep / n_visual,
        "prompt_len": prompt_len,
        "n_text": n_text,
        "n_visual": n_visual,
        "n_visual_kept_per_head": n_keep,
        "answer_tokens": int(full_ids.numel()),
        "vector_pairs": vector_pairs,
        "token_metrics": token_metrics,
        "predictions": predictions,
        "scores": scores,
    }
    return result, keep_masks


def save_packed_masks(
    path: Path,
    masks: dict[str, torch.Tensor],
    *,
    n_visual: int,
    n_keep: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            n_visual=np.int32(n_visual),
            n_keep=np.int32(n_keep),
            h2o_keep=np.packbits(masks["h2o_keep"].numpy(), axis=-1),
            question_keep=np.packbits(
                masks["question_keep"].numpy(), axis=-1
            ),
            qvik_keep=np.packbits(masks["qvik_keep"].numpy(), axis=-1),
            future_keep=np.packbits(masks["future_keep"].numpy(), axis=-1),
        )
    temporary.replace(path)


def bootstrap(
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    cursor = 0
    while cursor < repeats:
        chunk = min(512, repeats - cursor)
        indices = rng.integers(0, len(values), size=(chunk, len(values)))
        means[cursor : cursor + chunk] = values[indices].mean(axis=1)
        cursor += chunk
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> int:
    records = [
        json.loads(path.read_text())
        for path in sorted((args.output_dir / "samples").glob("*/*.json"))
    ]
    if not records:
        raise RuntimeError(f"No records under {args.output_dir / 'samples'}")
    expected = sum(
        1
        for dataset in agreement.DEFAULT_DATASETS
        for path in (args.teacher_root / dataset).glob("*.pt")
        if path.resolve() in agreement.recover_validation_paths(args.student_dir)
    )
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} held-out records, found {len(records)}")
    summary_dir = args.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Recover direct deployed-mask overlap from the packed [L,H,N] decisions.
    # This avoids conflating head-mean score similarity with actual eviction.
    for record in records:
        mask_path = (
            args.output_dir
            / "masks"
            / record["dataset"]
            / f"{Path(record['source_record']).stem}.npz"
        )
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing packed mask: {mask_path}")
        with np.load(mask_path) as packed:
            n_visual = int(packed["n_visual"])
            n_keep = int(packed["n_keep"])
            h2o_keep = np.unpackbits(
                packed["h2o_keep"], axis=-1
            )[..., :n_visual].astype(bool)
            question_keep = np.unpackbits(
                packed["question_keep"], axis=-1
            )[..., :n_visual].astype(bool)
            qvik_keep = np.unpackbits(
                packed["qvik_keep"], axis=-1
            )[..., :n_visual].astype(bool)
        if (
            h2o_keep.shape != question_keep.shape
            or h2o_keep.shape != qvik_keep.shape
            or n_keep <= 0
        ):
            raise ValueError(f"Invalid packed masks in {mask_path}")
        intersection = np.logical_and(h2o_keep, qvik_keep).sum(axis=-1)
        union = np.logical_or(h2o_keep, qvik_keep).sum(axis=-1)
        overlap_per_k = (intersection / n_keep).mean(axis=1)
        record["token_metrics"]["qvik_h2o_keep_overlap_per_k"] = (
            overlap_per_k.tolist()
        )
        record["token_metrics"]["qvik_h2o_keep_jaccard"] = (
            (intersection / np.maximum(union, 1)).mean(axis=1).tolist()
        )
        record["token_metrics"]["qvik_h2o_swap_fraction"] = (
            1.0 - overlap_per_k
        ).tolist()

    groups = [("all", records)] + [
        (dataset, [r for r in records if r["dataset"] == dataset])
        for dataset in agreement.DEFAULT_DATASETS
    ]

    vector_rows = []
    metric_names = agreement.METRIC_NAMES
    for group_index, (group, selected) in enumerate(groups):
        for pair_index, pair in enumerate(VECTOR_PAIRS):
            for metric_index, metric in enumerate(metric_names):
                values = np.asarray(
                    [
                        np.mean(record["vector_pairs"][pair][metric])
                        for record in selected
                    ]
                )
                mean, low, high = bootstrap(
                    values,
                    repeats=args.bootstrap_replicates,
                    seed=args.seed + group_index * 1000 + pair_index * 100 + metric_index,
                )
                vector_rows.append(
                    {
                        "group": group,
                        "pair": pair,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_samples": len(selected),
                    }
                )
        for metric_index, metric in enumerate(metric_names):
            values = np.asarray(
                [
                    np.mean(record["vector_pairs"]["qvik_vs_future"][metric])
                    - np.mean(record["vector_pairs"]["h2o_vs_future"][metric])
                    for record in selected
                ]
            )
            mean, low, high = bootstrap(
                values,
                repeats=args.bootstrap_replicates,
                seed=args.seed + 9_000 + group_index * 100 + metric_index,
            )
            vector_rows.append(
                {
                    "group": group,
                    "pair": "qvik_minus_h2o_vs_future",
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": len(selected),
                }
            )
    write_csv(summary_dir / "vector_agreement.csv", vector_rows)

    token_metric_names = tuple(records[0]["token_metrics"])
    token_rows = []
    for group_index, (group, selected) in enumerate(groups):
        for metric_index, metric in enumerate(token_metric_names):
            values = np.asarray(
                [
                    np.mean(record["token_metrics"][metric])
                    for record in selected
                ]
            )
            mean, low, high = bootstrap(
                values,
                repeats=args.bootstrap_replicates,
                seed=args.seed + 10_000 + group_index * 1000 + metric_index,
            )
            token_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": len(selected),
                }
            )
    write_csv(summary_dir / "token_keep_metrics.csv", token_rows)

    downstream_rows = []
    for group_index, (group, selected) in enumerate(groups):
        for selector_index, selector in enumerate(SELECTORS):
            values = np.asarray([record["scores"][selector] for record in selected])
            mean, low, high = bootstrap(
                values,
                repeats=args.bootstrap_replicates,
                seed=args.seed + 20_000 + group_index * 100 + selector_index,
            )
            downstream_rows.append(
                {
                    "group": group,
                    "selector": selector,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": len(selected),
                }
            )
        h2o = np.asarray([r["scores"]["h2o_prefill"] for r in selected])
        qvik = np.asarray([r["scores"]["qvik"] for r in selected])
        mean, low, high = bootstrap(
            qvik - h2o,
            repeats=args.bootstrap_replicates,
            seed=args.seed + 30_000 + group_index,
        )
        downstream_rows.append(
            {
                "group": group,
                "selector": "qvik_minus_h2o_prefill",
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_samples": len(selected),
            }
        )
    write_csv(summary_dir / "downstream.csv", downstream_rows)

    preservation_selectors = (
        "full_cache",
        "h2o_prefill",
        "question_prefill",
        "qvik",
        "future_oracle",
    )
    preservation_rows = []
    for group_index, (group, selected) in enumerate(groups):
        reference_text = [
            record["predictions"]["full_cache_manual"] for record in selected
        ]
        reference_score = np.asarray(
            [record["scores"]["full_cache_manual"] for record in selected]
        )
        for selector_index, selector in enumerate(preservation_selectors):
            matches = np.asarray(
                [
                    float(record["predictions"][selector] == expected)
                    for record, expected in zip(
                        selected, reference_text, strict=True
                    )
                ]
            )
            mean, low, high = bootstrap(
                matches,
                repeats=args.bootstrap_replicates,
                seed=args.seed + 35_000 + group_index * 100 + selector_index,
            )
            score_delta = (
                np.asarray([record["scores"][selector] for record in selected])
                - reference_score
            )
            preservation_rows.append(
                {
                    "group": group,
                    "selector": selector,
                    "reference": "full_cache_manual",
                    "exact_response_matches": int(matches.sum()),
                    "exact_response_match_rate": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "score_wins": int((score_delta > 0).sum()),
                    "score_ties": int((score_delta == 0).sum()),
                    "score_losses": int((score_delta < 0).sum()),
                    "n_samples": len(selected),
                }
            )
        matches = np.asarray(
            [
                float(
                    record["predictions"]["qvik"]
                    == record["predictions"]["h2o_prefill"]
                )
                for record in selected
            ]
        )
        mean, low, high = bootstrap(
            matches,
            repeats=args.bootstrap_replicates,
            seed=args.seed + 36_000 + group_index,
        )
        score_delta = np.asarray(
            [
                record["scores"]["qvik"] - record["scores"]["h2o_prefill"]
                for record in selected
            ]
        )
        preservation_rows.append(
            {
                "group": group,
                "selector": "qvik_vs_h2o_prefill",
                "reference": "h2o_prefill",
                "exact_response_matches": int(matches.sum()),
                "exact_response_match_rate": mean,
                "ci95_low": low,
                "ci95_high": high,
                "score_wins": int((score_delta > 0).sum()),
                "score_ties": int((score_delta == 0).sum()),
                "score_losses": int((score_delta < 0).sum()),
                "n_samples": len(selected),
            }
        )
    write_csv(summary_dir / "response_preservation.csv", preservation_rows)

    alignment_max = np.asarray(
        [record["teacher_replay_max_abs_error"] for record in records]
    )
    alignment_mean = np.asarray(
        [record["teacher_replay_mean_abs_error"] for record in records]
    )
    alignment_summary = {
        "n_samples": len(records),
        "sample_max_abs_error_mean": float(alignment_max.mean()),
        "sample_max_abs_error_worst": float(alignment_max.max()),
        "elementwise_mean_abs_error_mean": float(alignment_mean.mean()),
        "stored_teacher_trajectory_reproduced_all": True,
        "manual_decode_matches_teacher_generate_count": int(
            sum(record["full_cache_manual_matches_generate"] for record in records)
        ),
        "manual_decode_matches_teacher_generate_fraction": float(
            np.mean(
                [
                    record["full_cache_manual_matches_generate"]
                    for record in records
                ]
            )
        ),
        "manual_and_teacher_generate_scores_match_count": int(
            sum(
                record["scores"]["full_cache_manual"]
                == record["scores"]["full_cache"]
                for record in records
            )
        ),
    }
    (summary_dir / "teacher_alignment.json").write_text(
        json.dumps(alignment_summary, indent=2) + "\n"
    )

    layer_rows = []
    n_layers = len(records[0]["token_metrics"]["h2o_future_topk_recall"])
    for metric in (
        "h2o_future_topk_recall",
        "question_future_topk_recall",
        "qvik_future_topk_recall",
        "h2o_visual_future_mass_retained",
        "question_visual_future_mass_retained",
        "qvik_visual_future_mass_retained",
    ):
        for layer in range(n_layers):
            values = np.asarray(
                [record["token_metrics"][metric][layer] for record in records]
            )
            mean, low, high = bootstrap(
                values,
                repeats=min(2000, args.bootstrap_replicates),
                seed=args.seed + 40_000 + layer,
            )
            layer_rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(summary_dir / "layerwise_token_metrics.csv", layer_rows)

    vector_lookup = {
        (row["group"], row["pair"], row["metric"]): row for row in vector_rows
    }
    token_lookup = {
        (row["group"], row["metric"]): row for row in token_rows
    }
    downstream_lookup = {
        (row["group"], row["selector"]): row for row in downstream_rows
    }

    def ci(row: dict[str, Any]) -> str:
        return (
            f"{row['mean']:.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )

    lines = [
        "# Corrected Q2: H2O prefill vs. Full-Cache future utility",
        "",
        f"Held-out samples: {len(records)}. H2O accumulates attention from every "
        "causal prefill query and selects Top-K independently per attention/KV head.",
        "",
        "Teacher replay check: direct `generate()` attention capture matches the "
        f"stored training target with mean sample-max absolute error "
        f"{alignment_summary['sample_max_abs_error_mean']:.6g} "
        f"(worst {alignment_summary['sample_max_abs_error_worst']:.6g}).",
        "",
        "Downstream interventions use the matched manual-decode Full Cache as "
        "their primary control. It exactly matches the teacher/eager trajectory "
        f"for {alignment_summary['manual_decode_matches_teacher_generate_count']}/"
        f"{len(records)} responses, while benchmark scores match for "
        f"{alignment_summary['manual_and_teacher_generate_scores_match_count']}/"
        f"{len(records)}.",
        "",
        "## Vector-level agreement (head-mean diagnostic)",
        "",
        "| Signal | Spearman vs. Future | Cosine vs. Future | Jaccard@visual-20% (diagnostic) |",
        "|---|---:|---:|---:|",
    ]
    for label, pair in (
        ("H2O cumulative prefill", "h2o_vs_future"),
        ("Question-only prefill (secondary)", "question_prefill_vs_future"),
        ("Q-ViK", "qvik_vs_future"),
        ("Q-ViK − H2O (paired delta vs. Future)", "qvik_minus_h2o_vs_future"),
    ):
        lines.append(
            f"| {label} | "
            f"{ci(vector_lookup[('all', pair, 'spearman')])} | "
            f"{ci(vector_lookup[('all', pair, 'cosine')])} | "
            f"{ci(vector_lookup[('all', pair, 'jaccard_20')])} |"
        )
    lines.extend(
        [
            "",
            "Direct Q-ViK–H2O agreement: "
            f"Spearman {ci(vector_lookup[('all', 'qvik_vs_h2o', 'spearman')])}, "
            f"cosine {ci(vector_lookup[('all', 'qvik_vs_h2o', 'cosine')])}, "
            "and visual-Jaccard@20% "
            f"{ci(vector_lookup[('all', 'qvik_vs_h2o', 'jaccard_20')])}.",
            "",
            "Question-only attention is a secondary diagnostic rather than the "
            "H2O baseline. The stored Future target includes the first generation "
            "block (the last prompt query), which especially raises question-only "
            "agreement for short answers.",
        ]
    )
    lines.extend(
        [
            "",
            "## Exact-budget, head-resolved token selection",
            "",
            "| Method | Future Top-K recall | Jaccard vs. Future | Visual Future mass retained |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, method in (
        ("H2O", "h2o"),
        ("Question-only", "question"),
        ("Q-ViK", "qvik"),
    ):
        lines.append(
            f"| {label} | "
            f"{ci(token_lookup[('all', f'{method}_future_topk_recall')])} | "
            f"{ci(token_lookup[('all', f'{method}_jaccard_vs_future')])} | "
            f"{ci(token_lookup[('all', f'{method}_visual_future_mass_retained')])} |"
        )
    lines.extend(
        [
            "",
            "| Paired token diagnostic | Value |",
            "|---|---:|",
        ]
    )
    for label, metric in (
        ("Q-ViK − H2O Future recall", "qvik_minus_h2o_future_recall"),
        ("Q-ViK − H2O Jaccard", "qvik_minus_h2o_jaccard"),
        ("Q-ViK − H2O Future mass", "qvik_minus_h2o_future_mass"),
        ("Direct Q-ViK–H2O keep overlap / K", "qvik_h2o_keep_overlap_per_k"),
        ("Direct Q-ViK–H2O keep Jaccard", "qvik_h2o_keep_jaccard"),
        ("Direct Q-ViK–H2O swapped fraction", "qvik_h2o_swap_fraction"),
        ("Useful Q-ViK additions / K", "qvik_useful_additions_per_k"),
        ("Harmful removals: H2O-only Future hits / K", "qvik_harmful_removals_per_k"),
        ("Q-ViK head win fraction", "qvik_head_win_fraction"),
        ("Q-ViK head tie fraction", "qvik_head_tie_fraction"),
        ("Q-ViK head loss fraction", "qvik_head_loss_fraction"),
    ):
        lines.append(f"| {label} | {ci(token_lookup[('all', metric)])} |")
    lines.extend(
        [
            "",
            "## Held-out downstream score",
            "",
            "The matched manual-decode Full Cache is the primary control for "
            "H2O, Q-ViK, and Oracle interventions.",
            "",
            "| Selector | Overall | TextVQA | ScienceQA | GQA |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "full_cache_manual": "Full Cache (matched control; primary)",
        "full_cache": "Full Cache (teacher/eager trajectory)",
        "h2o_prefill": "H2O cumulative prefill",
        "question_prefill": "Question-only prefill",
        "qvik": "Q-ViK",
        "future_oracle": "Future Oracle",
    }
    for selector in SELECTORS:
        cells = [
            ci(downstream_lookup[(group, selector)])
            for group in ("all", *agreement.DEFAULT_DATASETS)
        ]
        lines.append(f"| {labels[selector]} | " + " | ".join(cells) + " |")
    delta = downstream_lookup[("all", "qvik_minus_h2o_prefill")]
    preservation_lookup = {
        (row["group"], row["selector"]): row for row in preservation_rows
    }
    lines.extend(
        [
            "",
            f"Paired downstream Q-ViK − H2O: {ci(delta)}.",
            "Its 95% paired CI includes zero, so downstream score superiority is "
            "not statistically resolved on this short-answer held-out set.",
            "",
            "## Response-level preservation",
            "",
            "| Selector | Exact response match | Score W/T/L vs. reference |",
            "|---|---:|---:|",
        ]
    )
    for label, selector in (
        ("Teacher/eager Full Cache vs. matched control", "full_cache"),
        ("H2O vs. matched Full Cache", "h2o_prefill"),
        ("Question-only vs. matched Full Cache", "question_prefill"),
        ("Q-ViK vs. matched Full Cache", "qvik"),
        ("Future Oracle vs. matched Full Cache", "future_oracle"),
        ("Q-ViK vs. H2O", "qvik_vs_h2o_prefill"),
    ):
        row = preservation_lookup[("all", selector)]
        lines.append(
            f"| {label} | {row['exact_response_matches']}/{row['n_samples']} "
            f"({row['exact_response_match_rate']:.3f} "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]) | "
            f"{row['score_wins']}/{row['score_ties']}/{row['score_losses']} |"
        )
    lines.extend(
        [
            "",
            "The vector table is secondary. The head-resolved token table and "
            "downstream intervention are the primary eviction results.",
        ]
    )
    (summary_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    specifications = (
        (
            "future_topk_recall",
            "Future Top-K recall",
        ),
        (
            "visual_future_mass_retained",
            "Visual Future mass retained",
        ),
    )
    for axis, (suffix, title) in zip(axes, specifications, strict=True):
        for method, label, color in (
            ("h2o", "H2O prefill", "#4C78A8"),
            ("question", "Question-only", "#72B7B2"),
            ("qvik", "Q-ViK", "#E45756"),
        ):
            metric = f"{method}_{suffix}"
            selected = sorted(
                [row for row in layer_rows if row["metric"] == metric],
                key=lambda row: row["layer"],
            )
            x = [row["layer"] for row in selected]
            y = [row["mean"] for row in selected]
            low = [row["ci95_low"] for row in selected]
            high = [row["ci95_high"] for row in selected]
            axis.plot(x, y, label=label, color=color, linewidth=2)
            axis.fill_between(x, low, high, color=color, alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Decoder layer")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Agreement / retained mass")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Corrected H2O Q2 token analysis (held-out n={len(records)})")
    fig.savefig(summary_dir / "layerwise_token_metrics.png", dpi=220)
    plt.close(fig)
    print(f"[summary] wrote {summary_dir / 'RESULTS.md'}", flush=True)
    return 0


def extract(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    validation = agreement.recover_validation_paths(args.student_dir)
    files = [
        path
        for dataset in args.datasets
        for path in sorted((args.teacher_root / dataset).glob("*.pt"))
        if path.resolve() in validation
    ]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("No requested held-out teacher records")
    print(
        f"[load] datasets={args.datasets} n={len(files)} device={args.device}",
        flush=True,
    )
    tokenizer, model, image_processor, student, image_feature_len = (
        load_exact_model_and_student(args)
    )
    textvqa_gt, scienceqa_gt = intervention.load_ground_truth()
    failures = []
    completed = 0
    skipped = 0
    started = time.time()
    for index, record_path in enumerate(files, start=1):
        output_path = (
            args.output_dir
            / "samples"
            / record_path.parent.name
            / f"{record_path.stem}.json"
        )
        mask_path = (
            args.output_dir
            / "masks"
            / record_path.parent.name
            / f"{record_path.stem}.npz"
        )
        if output_path.exists() and mask_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            result, masks = run_one(
                record_path=record_path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                student=student,
                image_feature_len=image_feature_len,
                device=torch.device(args.device),
                total_keep_ratio=args.total_keep_ratio,
                textvqa_ground_truth=textvqa_gt,
                scienceqa_ground_truth=scienceqa_gt,
            )
            agreement._dump_json(output_path, result)
            save_packed_masks(
                mask_path,
                masks,
                n_visual=result["n_visual"],
                n_keep=result["n_visual_kept_per_head"],
            )
            completed += 1
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "record": str(record_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[failed] {record_path}: {type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            torch.cuda.empty_cache()
        if index == 1 or index % 10 == 0 or index == len(files):
            print(
                f"[progress] {index}/{len(files)} completed={completed} "
                f"skipped={skipped} failures={len(failures)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    log_name = "extract_" + "_".join(args.datasets) + ".json"
    agreement._dump_json(
        args.output_dir / "logs" / log_name,
        {
            "datasets": args.datasets,
            "requested": len(files),
            "completed": completed,
            "skipped": skipped,
            "failures": failures,
            "elapsed_seconds": time.time() - started,
        },
    )
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    for name in ("teacher_root", "student_dir", "model_path", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.extract_only and args.summarize_only:
        raise ValueError("Choose at most one of --extract-only/--summarize-only")
    if not 0 < args.total_keep_ratio <= 1:
        raise ValueError("--total-keep-ratio must be in (0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        return summarize(args)
    status = extract(args)
    if status or args.extract_only:
        return status
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
