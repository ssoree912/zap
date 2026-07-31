#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled 7x200 comparison with one shared visual mask per layer.

All-prefill, semantic-question, Q-ViK, and raw Full-cache Future scores are
represented as ``[L, N_visual]``.  Attention-derived scores are averaged over
heads before Top-K.  The resulting one-dimensional prompt mask is applied
unchanged to every KV head for every actual intervention.

The Future reference retains the established causal generation trajectory:
``[last prompt query, y_1, ..., y_(T-1)]``.  It is captured from raw
Full-cache generation and is used only as an analysis reference.  This
controlled experiment does not perform an Oracle second-pass decode.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
WORKSPACE_ROOT = ZAP_ROOT.parent
QVIK_ROOT = WORKSPACE_ROOT / "Q-ViK"
for _path in (QVIK_ROOT, ZAP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BASE_EXP = ZAP_ROOT / "experiments" / "EXP-20260727-001-eval700-q2"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "eval1400_shared_control_base",
    BASE_EXP / "run_eval700_q2.py",
)
control = load_module(
    "eval1400_shared_control_utils",
    EXP_DIR / "shared_control_utils.py",
)
data = base.data
q2 = base.q2
process_images = base.process_images


EXPERIMENT_ID = "EXP-20260727-003-eval1400-shared-control"
SCHEMA_VERSION = 3
POLICY_VERSION = "shared-control-v1"
FUTURE_TRAJECTORY = "[last prompt query, y_1, ..., y_(T-1)]"
DEFAULT_OUTPUT = (
    ZAP_ROOT
    / "artifacts"
    / "rebuttal_eval1400_shared_control_llava15_total0p2"
)
DEFAULT_DATASETS = (
    "gqa",
    "textvqa",
    "docvqa",
    "chartqa",
    "coco_caption",
    "nocaps",
    "textcaps",
)
CAPTION_DATASETS = {"coco_caption", "nocaps", "textcaps"}
SCORE_TO_SELECTOR = {
    "all_prefill_shared": "all_prefill_shared",
    "question_shared": "question_shared",
    "qvik_shared": "qvik_shared",
}
SELECTORS = (
    "full_cache_generate",
    "full_cache_manual",
    "all_prefill_shared",
    "question_shared",
    "qvik_shared",
)
PRUNED_SELECTORS = SELECTORS[2:]
SELECTOR_LABELS = {
    "full_cache_generate": "Full Cache generate",
    "full_cache_manual": "Matched Full Cache",
    "all_prefill_shared": "All-prefill shared",
    "question_shared": "Semantic question shared",
    "qvik_shared": "Q-ViK shared",
}
SCORE_LABELS = {
    "all_prefill_shared": "All-prefill shared",
    "question_shared": "Semantic question shared",
    "qvik_shared": "Q-ViK shared",
    "future_shared": "Future shared (raw Full-cache reference)",
}
PAIRWISE_KEYS = tuple(
    control.pair_key(first, second)
    for first, second in control.PAIRWISE_SCORE_PAIRS
)
FUTURE_PAIRED_COMPARISONS = (
    (
        "qvik_shared_minus_all_prefill_shared",
        "qvik_shared",
        "all_prefill_shared",
    ),
    (
        "qvik_shared_minus_question_shared",
        "qvik_shared",
        "question_shared",
    ),
)
VECTOR_PAIRED_COMPARISONS = (
    (
        "qvik_shared_vs_future_shared_minus_"
        "all_prefill_shared_vs_future_shared",
        control.pair_key("qvik_shared", "future_shared"),
        control.pair_key("all_prefill_shared", "future_shared"),
    ),
    (
        "qvik_shared_vs_future_shared_minus_"
        "question_shared_vs_future_shared",
        control.pair_key("qvik_shared", "future_shared"),
        control.pair_key("question_shared", "future_shared"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=WORKSPACE_ROOT / "data/eval",
    )
    parser.add_argument(
        "--student-dir",
        type=Path,
        default=q2.agreement.DEFAULT_STUDENT_DIR,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=q2.agreement.DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--total-keep-ratio", type=float, default=0.2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--summary-seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--extract-only", action="store_true")
    action.add_argument("--summarize-only", action="store_true")
    action.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return cleaned[:160] or "sample"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_mask(mask: torch.Tensor) -> str:
    cpu = mask.detach().cpu().to(dtype=torch.bool).contiguous()
    digest = hashlib.sha256(
        f"shape={tuple(cpu.shape)};dtype=bool;".encode()
    )
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def validate_args(args: argparse.Namespace) -> int:
    invalid = sorted(set(args.datasets) - set(DEFAULT_DATASETS))
    if invalid:
        raise ValueError(
            f"Only the seven standard datasets are supported; invalid={invalid}"
        )
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets contains duplicates")
    if args.n_samples != 200:
        raise ValueError(
            f"{EXPERIMENT_ID} is fixed to 200 samples per dataset, got "
            f"{args.n_samples}"
        )
    if not math.isclose(
        float(args.total_keep_ratio),
        0.2,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{EXPERIMENT_ID} is fixed to total-cache keep ratio 0.2"
        )
    if args.sample_start < 0:
        raise ValueError("--sample-start must be nonnegative")
    sample_end = (
        args.sample_end if args.sample_end is not None else args.n_samples
    )
    if not args.sample_start <= sample_end <= args.n_samples:
        raise ValueError(
            f"Invalid sample slice [{args.sample_start}:{sample_end}] "
            f"for n={args.n_samples}"
        )
    return int(sample_end)


def prepare_samples(
    args: argparse.Namespace,
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    """Load deterministic rows and refuse changed manifests on resume."""

    selected: dict[str, list[Any]] = {}
    manifest_hashes: dict[str, str] = {}
    for dataset_name in args.datasets:
        samples = data.load_samples(
            dataset_name,
            args.eval_root,
            args.n_samples,
            args.sample_seed,
            load_images=not args.manifest_only,
        )
        if len(samples) != args.n_samples:
            raise RuntimeError(
                f"{dataset_name}: expected {args.n_samples}, got {len(samples)}"
            )
        payload = data.manifest_payload(samples)
        manifest_path = (
            args.output_dir / "manifests" / f"{dataset_name}.json"
        )
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing != payload:
                raise RuntimeError(
                    "Resume refused because the deterministic manifest changed: "
                    f"{manifest_path}. Use a fresh --output-dir."
                )
        else:
            data.write_manifest(samples, manifest_path)
        selected[dataset_name] = samples
        manifest_hashes[dataset_name] = data.manifest_sha256(samples)
    return selected, manifest_hashes


def run_config_path(args: argparse.Namespace) -> Path:
    identity = "\n".join(args.datasets)
    dataset_hash = hashlib.sha256(identity.encode()).hexdigest()[:12]
    worker = safe_name(args.worker_id or args.device)
    return (
        args.output_dir
        / "run_configs"
        / f"{worker}_{dataset_hash}.json"
    )


def write_or_validate_run_config(
    args: argparse.Namespace,
    *,
    sample_end: int,
    manifest_hashes: dict[str, str],
) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "datasets": list(args.datasets),
        "n_samples_per_dataset": args.n_samples,
        "sample_seed": args.sample_seed,
        "sample_slice": [args.sample_start, sample_end],
        "manifest_sha256": manifest_hashes,
        "model_path": str(args.model_path.resolve()),
        "student_dir": str(args.student_dir.resolve()),
        "device": args.device,
        "worker_id": args.worker_id,
        "budget_basis": "exact total prompt cache",
        "total_keep_ratio": args.total_keep_ratio,
        "all_text_kv_preserved": True,
        "score_shape": "[L,N_visual]",
        "head_policy": "head-average before Top-K",
        "selection_policy": "shared_layerwise_head_mean_topk",
        "mask_policy": "one shared layer-wise mask for every KV head",
        "future_reference_source": "raw Full-cache greedy generation attention",
        "future_trajectory": FUTURE_TRAJECTORY,
        "oracle_decode": "omitted",
        "image_preprocessing": (
            "qvik.llava15.mm_utils.process_images; same in-memory tensor for "
            "prefill and Future"
        ),
    }
    path = run_config_path(args)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError(
                f"Run-config fingerprint mismatch at {path}. "
                "Use a fresh --output-dir."
            )
        return
    atomic_json(path, payload)


def _n_kv_heads(cache: Any) -> int:
    if hasattr(cache, "key_cache"):
        keys = cache.key_cache[0]
    else:
        keys = cache[0][0]
    if keys.ndim != 4:
        raise ValueError(f"Expected cache keys [B,H,S,D], got {keys.shape}")
    return int(keys.shape[1])


def _decode_shared_selector(
    *,
    model: Any,
    tokenizer: Any,
    base_cache: Any,
    visual_keep: torch.Tensor,
    image_positions: torch.Tensor,
    prompt_len: int,
    first_token: torch.Tensor,
    max_new_tokens: int,
) -> str:
    prompt_masks = control.prompt_keep_masks(
        visual_keep,
        image_positions=image_positions,
        prompt_len=prompt_len,
    )
    cache = q2.intervention.clone_cache(base_cache)
    cache = q2.intervention.trim_kv_cache_per_layer(cache, prompt_masks)
    _, text = q2.decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        first_token=first_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    del cache
    return text


@torch.inference_mode()
def run_one(
    *,
    sample: Any,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    student: Any,
    image_feature_len: int,
    device: torch.device,
    total_keep_ratio: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    dataset_name = str(base.sample_value(sample, "dataset"))
    sample_id = str(base.sample_value(sample, "sample_id"))
    context = str(base.sample_value(sample, "context")).strip()
    prompt = base.build_prompt(context)
    image = base.sample_value(sample, "image").convert("RGB")

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    processed_image_sha256 = base.sha256_tensor(image_tensor)

    input_ids = q2.agreement._tokenize_prompt(
        prompt,
        tokenizer,
    ).unsqueeze(0).to(device)
    image_positions, prompt_len, placeholder = (
        q2.agreement.infer_image_positions(
            input_ids,
            image_feature_len,
        )
    )
    semantic_question = base.question_span_text(sample, prompt)
    question_positions = q2.agreement.infer_user_question_positions(
        prompt=prompt,
        question=semantic_question,
        tokenizer=tokenizer,
        full_input_ids=input_ids,
        image_placeholder=placeholder,
        image_feature_len=image_feature_len,
    )
    student_positions = base.student_question_positions(
        image_positions,
        prompt_len,
    )
    image_idx = image_positions.to(device)
    question_idx = question_positions.to(device)
    student_idx = student_positions.to(device)

    prefill = model(
        input_ids=input_ids,
        images=image_tensor,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    if prefill.attentions is None or prefill.hidden_states is None:
        raise RuntimeError(
            "Prefill did not return attentions and hidden states"
        )
    if int(prefill.logits.shape[1]) != prompt_len:
        raise ValueError(
            f"Expanded prompt mismatch: expected={prompt_len}, "
            f"actual={prefill.logits.shape[1]}"
        )
    first_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    base_cache = prefill.past_key_values
    n_kv_heads = _n_kv_heads(base_cache)

    all_prefill_heads: list[torch.Tensor] = []
    question_heads: list[torch.Tensor] = []
    qvik_logits: list[torch.Tensor] = []
    for layer, attention in enumerate(prefill.attentions):
        # [H,Q,K] -> [H,N_visual].  Heads are averaged only after all causal
        # prefill query rows have been accumulated.
        received = attention[0].sum(dim=1).float()
        all_prefill_heads.append(
            received.index_select(-1, image_idx).cpu()
        )

        # [H,|Q_semantic|,N_visual] -> [H,N_visual], then head-average below.
        question_received = attention[
            0,
            :,
            question_idx,
            :,
        ].index_select(-1, image_idx)
        question_heads.append(
            question_received.float().mean(dim=1).cpu()
        )

        qvik_logits.append(
            student.forward_layer(
                layer,
                prefill.hidden_states[layer + 1],
                image_idx,
                student_idx,
            )[0].float().cpu()
        )

    all_prefill_heads_tensor = torch.stack(all_prefill_heads)
    question_heads_tensor = torch.stack(question_heads)
    all_prefill_shared = control.head_average_scores(
        all_prefill_heads_tensor,
        name="all_prefill",
    )
    question_shared = control.head_average_scores(
        question_heads_tensor,
        name="semantic_question",
    )
    qvik_shared = torch.softmax(torch.stack(qvik_logits), dim=-1)
    n_attention_heads = int(all_prefill_heads_tensor.shape[1])
    del prefill

    max_new_tokens = int(base.sample_value(sample, "max_new_tokens"))
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
        raise RuntimeError(
            "Full-cache generation did not return attentions"
        )
    answer_steps = len(generated.attentions)
    full_ids = generated.sequences[0, -answer_steps:].detach().cpu()
    full_text = tokenizer.decode(
        full_ids.tolist(),
        skip_special_tokens=True,
    ).strip()
    if int(first_token.item()) != int(full_ids[0]):
        raise ValueError(
            "Standalone prefill and generate disagree on first answer token"
        )

    future_shared, future_heads_tensor = (
        control.future_scores_from_attention_blocks(
            generated.attentions,
            image_positions=image_idx,
        )
    )
    if future_heads_tensor.shape != all_prefill_heads_tensor.shape:
        raise ValueError(
            "Prefill/Future head-resolved score shapes differ: "
            f"{tuple(all_prefill_heads_tensor.shape)} vs "
            f"{tuple(future_heads_tensor.shape)}"
        )
    del generated

    matched_full_ids, matched_full_text = q2.decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        cache=q2.intervention.clone_cache(base_cache),
        first_token=first_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )

    n_visual = int(image_positions.numel())
    n_text = prompt_len - n_visual
    n_keep = q2.exact_total_budget(
        n_visual=n_visual,
        prompt_len=prompt_len,
        total_keep_ratio=total_keep_ratio,
    )
    if n_keep <= 0:
        raise ValueError(
            f"Total keep ratio {total_keep_ratio} leaves no visual tokens "
            f"(prompt={prompt_len}, text={n_text})"
        )

    shared_scores = {
        "all_prefill_shared": all_prefill_shared,
        "question_shared": question_shared,
        "qvik_shared": qvik_shared,
        "future_shared": future_shared,
    }
    token_metrics, keep_masks = control.shared_control_metrics(
        shared_scores,
        k=n_keep,
    )
    control.validate_visual_masks(
        keep_masks,
        layers=int(all_prefill_shared.shape[0]),
        n_visual=n_visual,
        n_keep=n_keep,
    )

    vector_pairs = {
        control.pair_key(first, second): (
            q2.agreement.compute_pair_metrics(
                shared_scores[first],
                shared_scores[second],
            )
        )
        for first, second in control.PAIRWISE_SCORE_PAIRS
    }

    predictions = {
        "full_cache_generate": full_text,
        "full_cache_manual": matched_full_text,
    }
    for score_name, selector_name in SCORE_TO_SELECTOR.items():
        predictions[selector_name] = _decode_shared_selector(
            model=model,
            tokenizer=tokenizer,
            base_cache=base_cache,
            visual_keep=keep_masks[f"{score_name}_keep"],
            image_positions=image_positions,
            prompt_len=prompt_len,
            first_token=first_token,
            max_new_tokens=max_new_tokens,
        )
    del base_cache

    scores = {
        selector: base.score_one_prediction(sample, prediction)
        for selector, prediction in predictions.items()
    }
    references = [
        str(value)
        for value in base.sample_value(sample, "references", ())
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "policy_version": POLICY_VERSION,
        "dataset": dataset_name,
        "sample_id": sample_id,
        "row_index": int(base.sample_value(sample, "row_index")),
        "semantic_question": semantic_question,
        "context": context,
        "references": references,
        "max_new_tokens": max_new_tokens,
        "answer_steps": answer_steps,
        "hit_max_new_tokens": bool(answer_steps >= max_new_tokens),
        "future_trajectory": FUTURE_TRAJECTORY,
        "future_reference_source": (
            "raw Full-cache greedy generation attention; head-averaged "
            "before shared Top-K"
        ),
        "oracle_decode": "omitted",
        "processed_image_sha256": processed_image_sha256,
        "same_processed_image_for_prefill_and_future": True,
        "question_token_count": int(question_positions.numel()),
        "student_prompt_tail_token_count": int(
            student_positions.numel()
        ),
        "prompt_len": prompt_len,
        "n_text": n_text,
        "n_visual": n_visual,
        "n_visual_kept_per_layer_and_kv_head": n_keep,
        "requested_total_keep_ratio": total_keep_ratio,
        "actual_total_keep_ratio": (n_text + n_keep) / prompt_len,
        "actual_visual_keep_ratio": n_keep / n_visual,
        "n_attention_heads": n_attention_heads,
        "n_kv_heads": n_kv_heads,
        "score_shape": "[L,N_visual]",
        "head_reduction": "arithmetic mean before Top-K",
        "selection_policy": "shared_layerwise_head_mean_topk",
        "mask_policy": (
            "one shared layer-wise visual mask broadcast unchanged to all "
            "KV heads"
        ),
        "all_compared_masks_shared_across_kv_heads": True,
        "applied_visual_mask_sha256": {
            score_name: sha256_mask(
                keep_masks[f"{score_name}_keep"]
            )
            for score_name in SCORE_TO_SELECTOR
        },
        "future_reference_mask_sha256": sha256_mask(
            keep_masks["future_shared_keep"]
        ),
        "all_prefill_definition": (
            "sum over every causal prefill query row per attention head, "
            "then mean over heads; shared Top-K"
        ),
        "question_definition": (
            "mean over semantic-question query rows and attention heads; "
            "shared Top-K"
        ),
        "future_definition": (
            "mean over raw Full-cache generation steps and attention heads; "
            f"trajectory={FUTURE_TRAJECTORY}; shared Top-K"
        ),
        "qvik_definition": "base checkpoint layer-wise logits; shared Top-K",
        "full_cache_manual_matches_generate": bool(
            torch.equal(matched_full_ids, full_ids)
        ),
        "vector_pairs": vector_pairs,
        "token_metrics": token_metrics,
        "predictions": predictions,
        "scores": scores,
    }
    del (
        image_tensor,
        input_ids,
        all_prefill_heads_tensor,
        question_heads_tensor,
        future_heads_tensor,
    )
    return result, keep_masks


def save_packed_shared_masks(
    path: Path,
    masks: dict[str, torch.Tensor],
    *,
    n_visual: int,
    n_keep: int,
    n_kv_heads: int,
) -> None:
    control.validate_visual_masks(
        masks,
        layers=int(next(iter(masks.values())).shape[0]),
        n_visual=n_visual,
        n_keep=n_keep,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays: dict[str, Any] = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "policy_version": np.str_(POLICY_VERSION),
        "mask_policy": np.str_("shared_across_all_kv_heads"),
        "future_reference": np.str_("raw_full_cache_generate_attention"),
        "future_trajectory": np.str_(FUTURE_TRAJECTORY),
        "n_visual": np.int32(n_visual),
        "n_keep": np.int32(n_keep),
        "n_kv_heads": np.int32(n_kv_heads),
    }
    arrays.update(
        {
            name: np.packbits(mask.numpy(), axis=-1)
            for name, mask in masks.items()
        }
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def sample_stem(sample: Any) -> str:
    return (
        f"{int(base.sample_value(sample, 'row_index')):06d}_"
        f"{safe_name(base.sample_value(sample, 'sample_id'))}"
    )


def cleanup_incomplete_outputs(
    result_path: Path,
    mask_path: Path,
) -> None:
    for path in (
        result_path,
        mask_path,
        result_path.with_suffix(result_path.suffix + ".tmp"),
        mask_path.with_suffix(mask_path.suffix + ".tmp"),
    ):
        path.unlink(missing_ok=True)


def extract(args: argparse.Namespace) -> int:
    sample_end = validate_args(args)
    selected, manifest_hashes = prepare_samples(args)
    write_or_validate_run_config(
        args,
        sample_end=sample_end,
        manifest_hashes=manifest_hashes,
    )
    if args.manifest_only:
        print(
            f"[manifest] datasets={len(args.datasets)} "
            f"samples={sum(map(len, selected.values()))}",
            flush=True,
        )
        return 0

    tokenizer, model, image_processor, student, image_feature_len = (
        q2.load_exact_model_and_student(args)
    )
    device = torch.device(args.device)
    work_selected = {
        name: values[args.sample_start:sample_end]
        for name, values in selected.items()
    }
    total = sum(len(values) for values in work_selected.values())
    completed = skipped = failures = 0
    started = time.time()

    for dataset_name in args.datasets:
        for sample in work_selected[dataset_name]:
            stem = sample_stem(sample)
            result_path = (
                args.output_dir
                / "samples"
                / dataset_name
                / f"{stem}.json"
            )
            mask_path = (
                args.output_dir
                / "masks"
                / dataset_name
                / f"{stem}.npz"
            )
            failure_path = (
                args.output_dir
                / "failures"
                / dataset_name
                / f"{stem}.txt"
            )
            complete_pair = result_path.exists() and mask_path.exists()
            if complete_pair and not args.overwrite:
                skipped += 1
                continue
            cleanup_incomplete_outputs(result_path, mask_path)
            try:
                result, masks = run_one(
                    sample=sample,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    student=student,
                    image_feature_len=image_feature_len,
                    device=device,
                    total_keep_ratio=args.total_keep_ratio,
                )
                result["mask_file"] = str(
                    mask_path.relative_to(args.output_dir)
                )
                result["qvik_checkpoint"] = str(
                    args.student_dir.resolve()
                )
                save_packed_shared_masks(
                    mask_path,
                    masks,
                    n_visual=int(result["n_visual"]),
                    n_keep=int(
                        result[
                            "n_visual_kept_per_layer_and_kv_head"
                        ]
                    ),
                    n_kv_heads=int(result["n_kv_heads"]),
                )
                result["mask_sha256"] = sha256_file(mask_path)
                atomic_json(result_path, result)
                failure_path.unlink(missing_ok=True)
                completed += 1
            except Exception as error:
                cleanup_incomplete_outputs(result_path, mask_path)
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(
                    f"{type(error).__name__}: {error}\n"
                )
                failures += 1
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"[failure] dataset={dataset_name} "
                    f"id={base.sample_value(sample, 'sample_id')}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            processed = completed + skipped + failures
            if (
                processed == 1
                or processed % 10 == 0
                or processed == total
            ):
                print(
                    f"[progress] {processed}/{total} "
                    f"completed={completed} skipped={skipped} "
                    f"failures={failures} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            torch.cuda.empty_cache()

    # Per-sample failures are declared common exclusions for every selector.
    # Unhandled worker/setup failures still propagate as a nonzero process
    # exit before reaching this return.
    if failures:
        print(
            f"[declared-common-exclusions] samples={failures}",
            flush=True,
        )
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(
    values: np.ndarray,
    args: argparse.Namespace,
    offset: int,
) -> tuple[float, float, float]:
    return q2.bootstrap(
        np.asarray(values, dtype=np.float64),
        repeats=args.bootstrap_replicates,
        seed=args.summary_seed + offset,
    )


def _unpack_mask(
    packed: Any,
    name: str,
    *,
    n_visual: int,
) -> torch.Tensor:
    unpacked = np.unpackbits(
        packed[name],
        axis=-1,
    )[..., :n_visual].astype(bool)
    return torch.from_numpy(unpacked)


def validate_record_and_mask(
    record: dict[str, Any],
    *,
    result_path: Path,
    mask_path: Path,
) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{result_path}: schema={record.get('schema_version')}"
        )
    if record.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(
            f"{result_path}: experiment_id={record.get('experiment_id')}"
        )
    if record.get("policy_version") != POLICY_VERSION:
        raise ValueError(
            f"{result_path}: policy_version={record.get('policy_version')}"
        )
    if record.get("future_trajectory") != FUTURE_TRAJECTORY:
        raise ValueError(
            f"{result_path}: Future trajectory changed"
        )
    if not record.get("all_compared_masks_shared_across_kv_heads"):
        raise ValueError(
            f"{result_path}: masks are not marked shared"
        )
    if record.get("mask_sha256") != sha256_file(mask_path):
        raise ValueError(f"{mask_path}: SHA-256 mismatch")

    n_visual = int(record["n_visual"])
    n_keep = int(
        record["n_visual_kept_per_layer_and_kv_head"]
    )
    with np.load(mask_path, allow_pickle=False) as packed:
        if int(packed["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"{mask_path}: packed schema mismatch")
        if str(packed["policy_version"]) != POLICY_VERSION:
            raise ValueError(f"{mask_path}: packed policy mismatch")
        if str(packed["mask_policy"]) != "shared_across_all_kv_heads":
            raise ValueError(f"{mask_path}: mask is not shared")
        if str(packed["future_trajectory"]) != FUTURE_TRAJECTORY:
            raise ValueError(f"{mask_path}: Future trajectory mismatch")
        if int(packed["n_visual"]) != n_visual:
            raise ValueError(f"{mask_path}: n_visual mismatch")
        if int(packed["n_keep"]) != n_keep:
            raise ValueError(f"{mask_path}: n_keep mismatch")
        if int(packed["n_kv_heads"]) != int(record["n_kv_heads"]):
            raise ValueError(f"{mask_path}: n_kv_heads mismatch")
        masks = {
            name: _unpack_mask(
                packed,
                name,
                n_visual=n_visual,
            )
            for name in (
                f"{score_name}_keep"
                for score_name in control.SCORE_NAMES
            )
        }

    layers = int(next(iter(masks.values())).shape[0])
    control.validate_visual_masks(
        masks,
        layers=layers,
        n_visual=n_visual,
        n_keep=n_keep,
    )
    applied_fingerprints = record.get(
        "applied_visual_mask_sha256",
        {},
    )
    for score_name in SCORE_TO_SELECTOR:
        expected_fingerprint = sha256_mask(
            masks[f"{score_name}_keep"]
        )
        if applied_fingerprints.get(score_name) != expected_fingerprint:
            raise ValueError(
                f"{mask_path}: stored {score_name} mask differs from the "
                "mask fingerprint used by the actual decode"
            )
    if record.get("future_reference_mask_sha256") != sha256_mask(
        masks["future_shared_keep"]
    ):
        raise ValueError(
            f"{mask_path}: Future reference mask fingerprint mismatch"
        )

    # Recompute every set-only metric from the persisted masks.  Future mass
    # needs the score vector and is checked at extraction time instead.
    future = masks["future_shared_keep"]
    for selector in control.DEPLOYABLE_SCORE_NAMES:
        keep = masks[f"{selector}_keep"]
        intersection = (keep & future).sum(dim=-1).float()
        union = (keep | future).sum(dim=-1).float()
        expected = record["token_metrics"]["future_agreement"][selector]
        np.testing.assert_allclose(
            expected["topk_recall"],
            (intersection / n_keep).numpy(),
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            expected["jaccard"],
            (intersection / union.clamp_min(1)).numpy(),
            rtol=0,
            atol=1e-7,
        )
    for first, second in control.PAIRWISE_SCORE_PAIRS:
        first_mask = masks[f"{first}_keep"]
        second_mask = masks[f"{second}_keep"]
        intersection = (first_mask & second_mask).sum(dim=-1).float()
        union = (first_mask | second_mask).sum(dim=-1).float()
        expected = record["token_metrics"][
            "pairwise_keep_agreement"
        ][control.pair_key(first, second)]
        np.testing.assert_allclose(
            expected["overlap_per_k"],
            (intersection / n_keep).numpy(),
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            expected["jaccard"],
            (intersection / union.clamp_min(1)).numpy(),
            rtol=0,
            atol=1e-7,
        )


def load_complete_records(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Consume only complete manifest-addressed JSON+NPZ pairs."""

    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        manifest_path = (
            args.output_dir / "manifests" / f"{dataset_name}.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("samples", [])
        if len(rows) != args.n_samples:
            raise RuntimeError(
                f"{dataset_name}: manifest has {len(rows)} rows, "
                f"expected {args.n_samples}"
            )
        for row in rows:
            stem = (
                f"{int(row['row_index']):06d}_"
                f"{safe_name(row['sample_id'])}"
            )
            result_path = (
                args.output_dir
                / "samples"
                / dataset_name
                / f"{stem}.json"
            )
            mask_path = (
                args.output_dir
                / "masks"
                / dataset_name
                / f"{stem}.npz"
            )
            failure_path = (
                args.output_dir
                / "failures"
                / dataset_name
                / f"{stem}.txt"
            )
            missing = [
                kind
                for kind, path in (
                    ("json", result_path),
                    ("npz", mask_path),
                )
                if not path.exists()
            ]
            if missing:
                if (
                    len(missing) == 2
                    and failure_path.exists()
                ):
                    reason = failure_path.read_text().strip()
                    if not reason:
                        raise RuntimeError(
                            f"Empty declared-failure marker: {failure_path}"
                        )
                    exclusions.append(
                        {
                            "dataset": dataset_name,
                            "sample_id": str(row["sample_id"]),
                            "row_index": int(row["row_index"]),
                            "reason": reason,
                            "failure_file": str(
                                failure_path.relative_to(
                                    args.output_dir
                                )
                            ),
                            "exclusion_scope": "all selectors",
                        }
                    )
                    continue
                marker = (
                    f"; failure marker={failure_path}"
                    if failure_path.exists()
                    else "; no failure marker"
                )
                raise RuntimeError(
                    "Undeclared or inconsistent incomplete sample pair: "
                    f"{dataset_name}/{row['sample_id']} missing={missing}"
                    f"{marker}"
                )
            if failure_path.exists():
                raise RuntimeError(
                    "Contradictory complete pair and failure marker: "
                    f"{dataset_name}/{row['sample_id']}"
                )
            record = json.loads(result_path.read_text())
            if (
                str(record["dataset"]) != dataset_name
                or str(record["sample_id"]) != str(row["sample_id"])
                or int(record["row_index"]) != int(row["row_index"])
            ):
                raise ValueError(
                    f"Manifest/result identity mismatch: {result_path}"
                )
            validate_record_and_mask(
                record,
                result_path=result_path,
                mask_path=mask_path,
            )
            records.append(record)

    if not records:
        raise RuntimeError("No complete JSON+NPZ sample pairs found")
    empty_datasets = [
        name
        for name in args.datasets
        if not any(record["dataset"] == name for record in records)
    ]
    if empty_datasets:
        raise RuntimeError(
            "Every requested dataset must retain at least one complete sample; "
            f"empty={empty_datasets}"
        )
    return records, exclusions


def _mean_metric(
    record: dict[str, Any],
    *keys: str,
) -> float:
    value: Any = record
    for key in keys:
        value = value[key]
    return float(np.mean(value))


def summarize(args: argparse.Namespace) -> int:
    validate_args(args)
    records, exclusions = load_complete_records(args)
    summary_dir = args.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(summary_dir / "exclusions.json", exclusions)

    dataset_order = [
        name
        for name in DEFAULT_DATASETS
        if any(record["dataset"] == name for record in records)
    ]
    groups = [("all", records)] + [
        (
            name,
            [
                record
                for record in records
                if record["dataset"] == name
            ],
        )
        for name in dataset_order
    ]

    vector_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        for pair_index, pair in enumerate(PAIRWISE_KEYS):
            for metric_index, metric in enumerate(
                q2.agreement.METRIC_NAMES
            ):
                values = np.asarray(
                    [
                        _mean_metric(
                            record,
                            "vector_pairs",
                            pair,
                            metric,
                        )
                        for record in selected
                    ]
                )
                mean, low, high = bootstrap_ci(
                    values,
                    args,
                    group_index * 10_000
                    + pair_index * 100
                    + metric_index,
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
    write_csv(summary_dir / "vector_agreement.csv", vector_rows)

    vector_paired_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        for comparison_index, (
            comparison,
            first_pair,
            second_pair,
        ) in enumerate(VECTOR_PAIRED_COMPARISONS):
            for metric_index, metric in enumerate(
                q2.agreement.METRIC_NAMES
            ):
                # Pair within each sample after reducing that sample's layer
                # trajectory. Bootstrap these per-sample deltas, never the
                # difference between independently bootstrapped means.
                values = np.asarray(
                    [
                        _mean_metric(
                            record,
                            "vector_pairs",
                            first_pair,
                            metric,
                        )
                        - _mean_metric(
                            record,
                            "vector_pairs",
                            second_pair,
                            metric,
                        )
                        for record in selected
                    ]
                )
                mean, low, high = bootstrap_ci(
                    values,
                    args,
                    50_000
                    + group_index * 1000
                    + comparison_index * 100
                    + metric_index,
                )
                vector_paired_rows.append(
                    {
                        "group": group,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_delta": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_samples": len(selected),
                    }
                )
    write_csv(
        summary_dir / "vector_agreement_paired.csv",
        vector_paired_rows,
    )

    future_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        for selector_index, selector in enumerate(
            control.DEPLOYABLE_SCORE_NAMES
        ):
            for metric_index, metric in enumerate(
                (
                    "topk_recall",
                    "jaccard",
                    "future_mass_retained",
                )
            ):
                values = np.asarray(
                    [
                        _mean_metric(
                            record,
                            "token_metrics",
                            "future_agreement",
                            selector,
                            metric,
                        )
                        for record in selected
                    ]
                )
                mean, low, high = bootstrap_ci(
                    values,
                    args,
                    100_000
                    + group_index * 10_000
                    + selector_index * 100
                    + metric_index,
                )
                future_rows.append(
                    {
                        "group": group,
                        "selector": selector,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_samples": len(selected),
                    }
                )
    write_csv(summary_dir / "future_agreement.csv", future_rows)

    future_paired_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        for comparison_index, (
            comparison,
            first_selector,
            second_selector,
        ) in enumerate(FUTURE_PAIRED_COMPARISONS):
            for metric_index, metric in enumerate(
                (
                    "topk_recall",
                    "jaccard",
                    "future_mass_retained",
                )
            ):
                # Each operand is the sample's layer mean. The bootstrap unit
                # is the resulting paired sample delta.
                values = np.asarray(
                    [
                        _mean_metric(
                            record,
                            "token_metrics",
                            "future_agreement",
                            first_selector,
                            metric,
                        )
                        - _mean_metric(
                            record,
                            "token_metrics",
                            "future_agreement",
                            second_selector,
                            metric,
                        )
                        for record in selected
                    ]
                )
                mean, low, high = bootstrap_ci(
                    values,
                    args,
                    150_000
                    + group_index * 1000
                    + comparison_index * 100
                    + metric_index,
                )
                future_paired_rows.append(
                    {
                        "group": group,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_delta": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_samples": len(selected),
                    }
                )
    write_csv(
        summary_dir / "future_agreement_paired.csv",
        future_paired_rows,
    )

    pairwise_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        for pair_index, pair in enumerate(PAIRWISE_KEYS):
            for metric_index, metric in enumerate(
                ("overlap_per_k", "jaccard")
            ):
                values = np.asarray(
                    [
                        _mean_metric(
                            record,
                            "token_metrics",
                            "pairwise_keep_agreement",
                            pair,
                            metric,
                        )
                        for record in selected
                    ]
                )
                mean, low, high = bootstrap_ci(
                    values,
                    args,
                    200_000
                    + group_index * 10_000
                    + pair_index * 100
                    + metric_index,
                )
                pairwise_rows.append(
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
    write_csv(
        summary_dir / "pairwise_keep_agreement.csv",
        pairwise_rows,
    )

    caption_corpus_scores: dict[str, dict[str, float]] = {}
    for dataset_name in dataset_order:
        if dataset_name not in CAPTION_DATASETS:
            continue
        selected = [
            record
            for record in records
            if record["dataset"] == dataset_name
        ]
        caption_corpus_scores[dataset_name] = {}
        for selector in SELECTORS:
            corpus_score = float(
                data.caption_results_rouge_l(
                    {
                        "answer": record["references"],
                        "pred": record["predictions"][selector],
                    }
                    for record in selected
                )
            )
            decomposed = float(
                np.mean(
                    [
                        record["scores"][selector]
                        for record in selected
                    ]
                )
            )
            if not np.isclose(
                corpus_score,
                decomposed,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{dataset_name}/{selector}: corpus ROUGE-L "
                    f"{corpus_score} != per-image mean {decomposed}"
                )
            caption_corpus_scores[dataset_name][
                selector
            ] = corpus_score

    downstream_rows: list[dict[str, Any]] = []
    downstream_deltas = (
        (
            "qvik_shared_minus_all_prefill_shared",
            "qvik_shared",
            "all_prefill_shared",
        ),
        (
            "question_shared_minus_all_prefill_shared",
            "question_shared",
            "all_prefill_shared",
        ),
    )
    for group_index, (group, selected) in enumerate(groups):
        for selector_index, selector in enumerate(SELECTORS):
            values = np.asarray(
                [record["scores"][selector] for record in selected]
            )
            mean, low, high = bootstrap_ci(
                values,
                args,
                300_000
                + group_index * 1000
                + selector_index,
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
        for delta_index, (label, first, second) in enumerate(
            downstream_deltas
        ):
            values = np.asarray(
                [
                    record["scores"][first]
                    - record["scores"][second]
                    for record in selected
                ]
            )
            mean, low, high = bootstrap_ci(
                values,
                args,
                350_000
                + group_index * 100
                + delta_index,
            )
            downstream_rows.append(
                {
                    "group": group,
                    "selector": label,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": len(selected),
                }
            )
    write_csv(summary_dir / "downstream.csv", downstream_rows)

    preservation_rows: list[dict[str, Any]] = []
    preservation_match_vectors: dict[
        tuple[str, str, str],
        np.ndarray,
    ] = {}
    for group_index, (group, selected) in enumerate(groups):
        for reference_index, reference in enumerate(
            ("full_cache_generate", "full_cache_manual")
        ):
            expected = [
                record["predictions"][reference]
                for record in selected
            ]
            for selector_index, selector in enumerate(PRUNED_SELECTORS):
                matches = np.asarray(
                    [
                        float(
                            record["predictions"][selector]
                            == reference_text
                        )
                        for record, reference_text in zip(
                            selected,
                            expected,
                            strict=True,
                        )
                    ]
                )
                preservation_match_vectors[
                    (group, reference, selector)
                ] = matches
                mean, low, high = bootstrap_ci(
                    matches,
                    args,
                    400_000
                    + group_index * 1000
                    + reference_index * 100
                    + selector_index,
                )
                preservation_rows.append(
                    {
                        "group": group,
                        "reference": reference,
                        "selector": selector,
                        "exact_matches": int(matches.sum()),
                        "match_rate": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_samples": len(selected),
                    }
                )
    write_csv(
        summary_dir / "response_preservation.csv",
        preservation_rows,
    )
    preservation_paired_rows: list[dict[str, Any]] = []
    preservation_comparisons = (
        (
            "qvik_shared_minus_all_prefill_shared",
            "qvik_shared",
            "all_prefill_shared",
        ),
        (
            "question_shared_minus_all_prefill_shared",
            "question_shared",
            "all_prefill_shared",
        ),
        (
            "qvik_shared_minus_question_shared",
            "qvik_shared",
            "question_shared",
        ),
    )
    for group_index, (group, selected) in enumerate(groups):
        for reference_index, reference in enumerate(
            ("full_cache_generate", "full_cache_manual")
        ):
            for comparison_index, (label, first, second) in enumerate(
                preservation_comparisons
            ):
                first_matches = preservation_match_vectors[
                    (group, reference, first)
                ]
                second_matches = preservation_match_vectors[
                    (group, reference, second)
                ]
                paired = first_matches - second_matches
                mean, low, high = bootstrap_ci(
                    paired,
                    args,
                    450_000
                    + group_index * 1000
                    + reference_index * 100
                    + comparison_index,
                )
                preservation_paired_rows.append(
                    {
                        "group": group,
                        "reference": reference,
                        "comparison": label,
                        "mean_delta": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "first_only_matches": int(
                            np.sum(
                                (first_matches == 1)
                                & (second_matches == 0)
                            )
                        ),
                        "second_only_matches": int(
                            np.sum(
                                (first_matches == 0)
                                & (second_matches == 1)
                            )
                        ),
                        "n_samples": len(selected),
                    }
                )
    write_csv(
        summary_dir / "response_preservation_paired.csv",
        preservation_paired_rows,
    )

    layer_rows: list[dict[str, Any]] = []
    n_layers = len(
        records[0]["token_metrics"]["future_agreement"][
            "all_prefill_shared"
        ]["topk_recall"]
    )
    for selector_index, selector in enumerate(
        control.DEPLOYABLE_SCORE_NAMES
    ):
        for metric_index, metric in enumerate(
            ("topk_recall", "jaccard", "future_mass_retained")
        ):
            for layer in range(n_layers):
                values = np.asarray(
                    [
                        record["token_metrics"]["future_agreement"][
                            selector
                        ][metric][layer]
                        for record in records
                    ]
                )
                mean, low, high = bootstrap_ci(
                    values,
                    args,
                    500_000
                    + selector_index * 10_000
                    + metric_index * 100
                    + layer,
                )
                layer_rows.append(
                    {
                        "layer": layer,
                        "selector": selector,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_samples": len(records),
                    }
                )
    write_csv(
        summary_dir / "layerwise_future_agreement.csv",
        layer_rows,
    )

    vector_lookup = {
        (row["group"], row["pair"], row["metric"]): row
        for row in vector_rows
    }
    vector_paired_lookup = {
        (
            row["group"],
            row["comparison"],
            row["metric"],
        ): row
        for row in vector_paired_rows
    }
    future_lookup = {
        (row["group"], row["selector"], row["metric"]): row
        for row in future_rows
    }
    future_paired_lookup = {
        (
            row["group"],
            row["comparison"],
            row["metric"],
        ): row
        for row in future_paired_rows
    }
    pairwise_lookup = {
        (row["group"], row["pair"], row["metric"]): row
        for row in pairwise_rows
    }
    downstream_lookup = {
        (row["group"], row["selector"]): row
        for row in downstream_rows
    }
    preservation_lookup = {
        (row["group"], row["reference"], row["selector"]): row
        for row in preservation_rows
    }
    preservation_paired_lookup = {
        (
            row["group"],
            row["reference"],
            row["comparison"],
        ): row
        for row in preservation_paired_rows
    }

    def formatted(row: dict[str, Any]) -> str:
        return (
            f"{row['mean']:.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )

    def formatted_delta(row: dict[str, Any]) -> str:
        return (
            f"{row['mean_delta']:+.4f} "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
        )

    keep_rates = np.asarray(
        [record["actual_visual_keep_ratio"] for record in records]
    )
    keep_counts = np.asarray(
        [
            record["n_visual_kept_per_layer_and_kv_head"]
            for record in records
        ]
    )
    exclusion_counts = {
        name: sum(
            row["dataset"] == name
            for row in exclusions
        )
        for name in DEFAULT_DATASETS
    }
    exclusion_reasons = {
        name: [
            {
                "sample_id": row["sample_id"],
                "row_index": row["row_index"],
                "reason": row["reason"],
            }
            for row in exclusions
            if row["dataset"] == name
        ]
        for name in DEFAULT_DATASETS
    }
    counts_text = ", ".join(
        f"{name}={sum(r['dataset'] == name for r in records)}"
        for name in dataset_order
    )
    lines = [
        "# Eval-1400 shared-mask controlled comparison",
        "",
        f"Complete samples: {len(records)} ({counts_text}). "
        f"MileBench is excluded. Declared common sample exclusions: "
        f"{len(exclusions)}.",
        "Declared common exclusions by dataset: "
        + ", ".join(
            f"{name}={exclusion_counts[name]}"
            for name in DEFAULT_DATASETS
        )
        + ".",
        "",
        "All compared selectors preserve every text KV and use exact "
        "total-prompt cache ratio 0.20. All-prefill, semantic question, "
        "Q-ViK, and Future are `[L,N_visual]` scores and select one shared "
        "layer-wise Top-K mask that is applied unchanged to every KV head.",
        f"The budget retains {keep_counts.mean():.1f} visual KVs on average "
        f"(range {keep_counts.min()}–{keep_counts.max()}), or "
        f"{100 * keep_rates.mean():.1f}% of 576 visual KVs.",
        "",
        "Future shared is derived from raw Full-cache greedy-generation "
        f"attention using `{FUTURE_TRAJECTORY}`. It is the reference for "
        "agreement only; no Oracle second-pass decode is run.",
        "",
        "## Exact-budget shared-mask pairwise agreement",
        "",
        "| Pair | Overlap/K | Jaccard |",
        "|---|---:|---:|",
    ]
    for first, second in control.PAIRWISE_SCORE_PAIRS:
        key = control.pair_key(first, second)
        lines.append(
            f"| {SCORE_LABELS[first]} ↔ {SCORE_LABELS[second]} | "
            f"{formatted(pairwise_lookup[('all', key, 'overlap_per_k')])} | "
            f"{formatted(pairwise_lookup[('all', key, 'jaccard')])} |"
        )

    lines.extend(
        [
            "",
            "## Agreement with raw Full-cache Future shared Top-K",
            "",
            "| Selector | Future Top-K recall | Jaccard | Future mass retained |",
            "|---|---:|---:|---:|",
        ]
    )
    for selector in control.DEPLOYABLE_SCORE_NAMES:
        lines.append(
            f"| {SCORE_LABELS[selector]} | "
            f"{formatted(future_lookup[('all', selector, 'topk_recall')])} | "
            f"{formatted(future_lookup[('all', selector, 'jaccard')])} | "
            f"{formatted(future_lookup[('all', selector, 'future_mass_retained')])} |"
        )

    lines.extend(
        [
            "",
            "## Paired deltas in Future agreement",
            "",
            "For each sample, each metric is averaged over layers first. "
            "The table reports the mean paired selector difference and a "
            "sample-bootstrap 95% CI.",
            "",
            "| Paired comparison | Δ Top-K recall | Δ Jaccard | "
            "Δ Future mass retained |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison, first, second in FUTURE_PAIRED_COMPARISONS:
        lines.append(
            f"| {SCORE_LABELS[first]} − {SCORE_LABELS[second]} | "
            f"{formatted_delta(future_paired_lookup[('all', comparison, 'topk_recall')])} | "
            f"{formatted_delta(future_paired_lookup[('all', comparison, 'jaccard')])} | "
            f"{formatted_delta(future_paired_lookup[('all', comparison, 'future_mass_retained')])} |"
        )

    lines.extend(
        [
            "",
            "## Head-averaged score-vector agreement",
            "",
            "| Pair | Spearman | Cosine |",
            "|---|---:|---:|",
        ]
    )
    for first, second in (
        ("all_prefill_shared", "question_shared"),
        ("all_prefill_shared", "future_shared"),
        ("question_shared", "future_shared"),
        ("qvik_shared", "future_shared"),
    ):
        key = control.pair_key(first, second)
        lines.append(
            f"| {SCORE_LABELS[first]} ↔ {SCORE_LABELS[second]} | "
            f"{formatted(vector_lookup[('all', key, 'spearman')])} | "
            f"{formatted(vector_lookup[('all', key, 'cosine')])} |"
        )

    vector_metric_labels = {
        "spearman": "Δ Spearman",
        "cosine": "Δ Cosine",
        "jaccard_10": "Δ Jaccard@10%",
        "jaccard_20": "Δ Jaccard@20%",
        "jaccard_50": "Δ Jaccard@50%",
    }
    vector_report_metrics = tuple(q2.agreement.METRIC_NAMES)
    lines.extend(
        [
            "",
            "## Paired deltas in score-vector agreement",
            "",
            "Each operand is the sample-level layer mean against Future "
            "shared. The difference is then sample-bootstrapped for a 95% CI. "
            "Jaccard columns are fixed visual-fraction vector diagnostics, "
            "not the exact total-budget mask metric above.",
            "",
            "| Paired comparison | "
            + " | ".join(
                vector_metric_labels.get(metric, f"Δ {metric}")
                for metric in vector_report_metrics
            )
            + " |",
            "|---|"
            + "".join("---:|" for _metric in vector_report_metrics),
        ]
    )
    vector_pair_labels = {
        VECTOR_PAIRED_COMPARISONS[0][0]: (
            "(Q-ViK shared ↔ Future shared) − "
            "(All-prefill shared ↔ Future shared)"
        ),
        VECTOR_PAIRED_COMPARISONS[1][0]: (
            "(Q-ViK shared ↔ Future shared) − "
            "(Semantic question shared ↔ Future shared)"
        ),
    }
    for comparison, _first_pair, _second_pair in (
        VECTOR_PAIRED_COMPARISONS
    ):
        values = [
            formatted_delta(
                vector_paired_lookup[
                    ("all", comparison, metric)
                ]
            )
            for metric in vector_report_metrics
        ]
        lines.append(
            f"| {vector_pair_labels[comparison]} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Post-eviction downstream score",
            "",
            "Scores are on a 0–100 scale. VQA uses each task metric; "
            "caption datasets use the local paper ROUGE-L setting.",
            "",
            "| Dataset | Full generate | Matched Full | All-prefill shared | "
            "Question shared | Q-ViK shared |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name in dataset_order:
        values = [
            100 * downstream_lookup[(dataset_name, selector)]["mean"]
            for selector in SELECTORS
        ]
        lines.append(
            f"| {dataset_name} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )
    macro_values = [
        100
        * float(
            np.mean(
                [
                    downstream_lookup[(name, selector)]["mean"]
                    for name in dataset_order
                ]
            )
        )
        for selector in SELECTORS
    ]
    lines.append(
        "| **7-benchmark macro** | "
        + " | ".join(
            f"**{value:.2f}**" for value in macro_values
        )
        + " |"
    )
    for label, names in (
        (
            "VQA macro",
            [
                name
                for name in dataset_order
                if name not in CAPTION_DATASETS
            ],
        ),
        (
            "Caption macro",
            [
                name
                for name in dataset_order
                if name in CAPTION_DATASETS
            ],
        ),
    ):
        if not names:
            continue
        values = [
            100
            * float(
                np.mean(
                    [
                        downstream_lookup[(name, selector)]["mean"]
                        for name in names
                    ]
                )
            )
            for selector in SELECTORS
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )

    for reference, heading in (
        (
            "full_cache_generate",
            "Response preservation against raw Full Cache generate",
        ),
        (
            "full_cache_manual",
            "Response preservation against matched manual Full Cache "
            "(diagnostic)",
        ),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Selector | Exact response matches | Match rate |",
                "|---|---:|---:|",
            ]
        )
        for selector in PRUNED_SELECTORS:
            row = preservation_lookup[("all", reference, selector)]
            lines.append(
                f"| {SELECTOR_LABELS[selector]} | "
                f"{row['exact_matches']}/{row['n_samples']} | "
                f"{row['match_rate']:.4f} "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
            )
        if reference == "full_cache_generate":
            lines.extend(
                [
                    "",
                    "Paired exact-match deltas against raw Full Cache "
                    "generate:",
                    "",
                ]
            )
            for label, _first, _second in preservation_comparisons:
                row = preservation_paired_lookup[
                    ("all", reference, label)
                ]
                lines.append(
                    f"- `{label}`: {row['mean_delta']:.4f} "
                    f"[{row['ci95_low']:.4f}, "
                    f"{row['ci95_high']:.4f}] "
                    f"({row['first_only_matches']} first-only vs. "
                    f"{row['second_only_matches']} second-only matches)."
                )

    manual_matches = sum(
        record["full_cache_manual_matches_generate"]
        for record in records
    )
    truncated = {
        name: sum(
            record["hit_max_new_tokens"]
            for record in records
            if record["dataset"] == name
        )
        for name in dataset_order
    }
    lines.extend(
        [
            "",
            f"Matched manual Full Cache reproduces raw `generate()` for "
            f"{manual_matches}/{len(records)} responses.",
            "",
            "Max-token truncation counts: "
            + ", ".join(
                f"{name}={count}/"
                f"{sum(r['dataset'] == name for r in records)}"
                for name, count in truncated.items()
            )
            + ".",
        ]
    )
    report_path = summary_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")

    manifest_hashes = {}
    for dataset_name in dataset_order:
        manifest = json.loads(
            (
                args.output_dir
                / "manifests"
                / f"{dataset_name}.json"
            ).read_text()
        )
        canonical = json.dumps(
            manifest,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        manifest_hashes[dataset_name] = hashlib.sha256(
            canonical
        ).hexdigest()
    atomic_json(
        summary_dir / "provenance.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "n_complete_samples": len(records),
            "n_declared_common_exclusions": len(exclusions),
            "declared_common_exclusion_counts": exclusion_counts,
            "declared_common_exclusion_reasons": exclusion_reasons,
            "datasets": dataset_order,
            "n_samples_per_dataset": args.n_samples,
            "sample_seed": args.sample_seed,
            "manifest_sha256": manifest_hashes,
            "total_keep_ratio": args.total_keep_ratio,
            "budget_basis": "exact total prompt cache",
            "head_reduction": "mean before Top-K",
            "selection_policy": "shared_layerwise_head_mean_topk",
            "mask_policy": (
                "one shared layer-wise mask for every KV head"
            ),
            "future_reference_source": (
                "raw Full-cache greedy generation attention"
            ),
            "future_trajectory": FUTURE_TRAJECTORY,
            "oracle_decode": "omitted",
            "paired_delta_protocol": (
                "mean layers within each sample, subtract selectors within "
                "that sample, then sample-bootstrap the paired deltas"
            ),
            "paired_delta_outputs": {
                "future_agreement": "future_agreement_paired.csv",
                "vector_agreement": "vector_agreement_paired.csv",
            },
            "manual_generate_exact_match_count": manual_matches,
            "truncation_counts": truncated,
            "caption_metric": (
                "local paper setting: tokenized COCO ROUGE-L"
            ),
            "caption_corpus_scores": caption_corpus_scores,
            "docvqa_metric": "official continuous max-ANLS",
        },
    )
    print(report_path.read_text())
    return 0


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        return summarize(args)
    return extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
