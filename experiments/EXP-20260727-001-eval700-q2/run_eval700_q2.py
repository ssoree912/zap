#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seven-benchmark Q2 analysis on 100 held-out examples per benchmark.

This runner keeps the comparison in visual-token space and applies every
deployable selector at the same *total prompt-cache* budget:

* H2O: cumulative attention received from every causal prefill query,
  retaining a separate Top-K set for every KV head.
* Question: attention received only from the semantic question-token rows,
  averaged over question rows and heads before a shared Top-K selection.
* Q-ViK: the base student's shared layer-wise visual-token prediction.
* Future Oracle: attention received during the full-cache answer trajectory.

The image tensor used for prefill scoring and Future extraction is the same
in-memory tensor.  No JPEG round-trip is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
Q2_EXP = ZAP_ROOT / "experiments" / "EXP-20260726-003-h2o-q2"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q2 = load_module("eval700_corrected_q2", Q2_EXP / "run_llava15_h2o_q2.py")
data = load_module("eval700_data", EXP_DIR / "eval700_data.py")

from qvik.llava15.conversation import conv_templates  # noqa: E402
from qvik.llava15.mm_utils import process_images  # noqa: E402


DEFAULT_OUTPUT = ZAP_ROOT / "artifacts" / "rebuttal_eval700_q2_total0p2"
DEFAULT_DATASETS = (
    "gqa",
    "textvqa",
    "docvqa",
    "chartqa",
    "coco_caption",
    "nocaps",
    "textcaps",
)
SELECTORS = (
    "full_cache_generate",
    "full_cache_manual",
    "h2o_prefill",
    "question_prefill",
    "qvik",
    "future_oracle",
)
VECTOR_PAIRS = (
    "h2o_vs_future",
    "question_vs_future",
    "qvik_vs_future",
    "qvik_vs_h2o",
    "qvik_vs_question",
)
CAPTION_DATASETS = {"coco_caption", "nocaps", "textcaps"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=ZAP_ROOT.parent / "data/eval")
    parser.add_argument("--student-dir", type=Path, default=q2.agreement.DEFAULT_STUDENT_DIR)
    parser.add_argument("--model-path", type=Path, default=q2.agreement.DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-keep-ratio", type=float, default=0.2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--summary-seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return value[:160] or "sample"


def sample_value(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def build_prompt(context: str) -> str:
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], f"<image>\n{context.strip()}")
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def question_span_text(sample: Any, prompt: str) -> str:
    context = str(sample_value(sample, "context", ""))
    semantic_context = context
    for suffix in (
        data.SINGLE_PHRASE_SUFFIX,
        data.CHARTQA_SUFFIX,
    ):
        if semantic_context.endswith(suffix):
            semantic_context = semantic_context[: -len(suffix)]
            break
    candidates = [
        sample_value(sample, "question_span"),
        semantic_context,
        sample_value(sample, "raw_question"),
        context,
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) and str(candidate) in prompt:
            return str(candidate)
    raw = str(sample_value(sample, "raw_question", ""))
    capitalized = raw.capitalize()
    if capitalized and capitalized in prompt:
        return capitalized
    raise ValueError(
        f"Could not locate semantic question span for {sample_value(sample, 'sample_id')}"
    )


def student_question_positions(
    image_positions: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Reproduce the base checkpoint's post-image prompt-tail convention."""
    first = int(image_positions[-1]) + 1
    if first >= prompt_len:
        raise ValueError("Student question/prompt-tail span is empty")
    return torch.arange(first, prompt_len, dtype=torch.long)


def sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(
        f"dtype={cpu.dtype};shape={tuple(cpu.shape)};".encode()
    )
    # NumPy cannot materialize torch.bfloat16 directly. Viewing the same
    # storage as bytes preserves an exact hash without any numeric conversion.
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def score_one_prediction(sample: Any, prediction: str) -> float:
    dataset_name = str(sample_value(sample, "dataset"))
    if dataset_name in CAPTION_DATASETS:
        # COCO ROUGE-L returns the arithmetic mean of its per-image scores.
        # A singleton call recovers the decomposable per-image component used
        # for sample bootstrap; summarize() also recomputes and verifies the
        # complete 100-image corpus aggregate for every selector.
        return float(
            data.caption_results_rouge_l(
                [
                    {
                        "answer": list(sample_value(sample, "references", ())),
                        "pred": prediction,
                    }
                ]
            )
        )
    return float(data.score_prediction(sample, prediction))


def add_mask_comparisons(
    metrics: dict[str, list[float]],
    masks: dict[str, torch.Tensor],
    *,
    n_keep: int,
) -> None:
    pairs = (
        ("qvik_h2o", masks["qvik_keep"], masks["h2o_keep"]),
        ("question_h2o", masks["question_keep"], masks["h2o_keep"]),
        ("qvik_question", masks["qvik_keep"], masks["question_keep"]),
    )
    for label, first, second in pairs:
        intersection = (first & second).sum(dim=-1).float()
        union = (first | second).sum(dim=-1).float()
        metrics[f"{label}_keep_overlap_per_k"] = (
            intersection.div(float(n_keep)).mean(dim=1).tolist()
        )
        metrics[f"{label}_keep_jaccard"] = (
            intersection.div(union.clamp_min(1)).mean(dim=1).tolist()
        )
        metrics[f"{label}_swap_fraction"] = (
            1.0 - intersection.div(float(n_keep)).mean(dim=1)
        ).tolist()


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
    dataset_name = str(sample_value(sample, "dataset"))
    sample_id = str(sample_value(sample, "sample_id"))
    context = str(sample_value(sample, "context")).strip()
    prompt = build_prompt(context)
    image = sample_value(sample, "image").convert("RGB")

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    processed_image_sha256 = sha256_tensor(image_tensor)

    input_ids = q2.agreement._tokenize_prompt(prompt, tokenizer).unsqueeze(0).to(device)
    image_positions, prompt_len, placeholder = q2.agreement.infer_image_positions(
        input_ids,
        image_feature_len,
    )
    semantic_question = question_span_text(sample, prompt)
    question_positions = q2.agreement.infer_user_question_positions(
        prompt=prompt,
        question=semantic_question,
        tokenizer=tokenizer,
        full_input_ids=input_ids,
        image_placeholder=placeholder,
        image_feature_len=image_feature_len,
    )
    student_positions = student_question_positions(image_positions, prompt_len)
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
        raise RuntimeError("Prefill did not return attentions and hidden states")
    if int(prefill.logits.shape[1]) != prompt_len:
        raise ValueError(
            f"Expanded prompt mismatch: expected={prompt_len}, "
            f"actual={prefill.logits.shape[1]}"
        )
    first_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    base_cache = prefill.past_key_values

    h2o_heads: list[torch.Tensor] = []
    question_heads: list[torch.Tensor] = []
    qvik_logits: list[torch.Tensor] = []
    for layer, attention in enumerate(prefill.attentions):
        # [H,Q,K] -> all-prefill accumulated attention received per KV.
        received = attention[0].sum(dim=1).float()
        h2o_heads.append(received.index_select(-1, image_idx).cpu())

        # [H,|Q_semantic|,N_visual] -> average only semantic question rows.
        question_received = attention[0, :, question_idx, :].index_select(
            -1,
            image_idx,
        )
        question_heads.append(question_received.float().mean(dim=1).cpu())

        qvik_logits.append(
            student.forward_layer(
                layer,
                # The base artifact was trained from post-layer H_{ell+1}.
                prefill.hidden_states[layer + 1],
                image_idx,
                student_idx,
            )[0].float().cpu()
        )

    h2o_heads_tensor = torch.stack(h2o_heads)
    question_heads_tensor = torch.stack(question_heads)
    h2o_vector = q2.normalize_rows(h2o_heads_tensor.mean(dim=1))
    question_vector = q2.normalize_rows(question_heads_tensor.mean(dim=1))
    qvik_vector = torch.softmax(torch.stack(qvik_logits), dim=-1)
    del prefill

    max_new_tokens = int(sample_value(sample, "max_new_tokens"))
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
    answer_steps = len(generated.attentions)
    full_ids = generated.sequences[0, -answer_steps:].detach().cpu()
    full_text = tokenizer.decode(
        full_ids.tolist(),
        skip_special_tokens=True,
    ).strip()
    if int(first_token.item()) != int(full_ids[0]):
        raise ValueError("Standalone prefill and generate disagree on first answer token")

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
    future_vector = q2.normalize_rows(future_heads_tensor.mean(dim=1))
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

    token_metrics, keep_masks = q2.exact_head_token_metrics(
        h2o_heads=h2o_heads_tensor,
        question_scores=question_vector,
        qvik_scores=qvik_vector,
        future_heads=future_heads_tensor,
        k=n_keep,
    )
    add_mask_comparisons(token_metrics, keep_masks, n_keep=n_keep)
    vector_pairs = {
        "h2o_vs_future": q2.agreement.compute_pair_metrics(
            h2o_vector,
            future_vector,
        ),
        "question_vs_future": q2.agreement.compute_pair_metrics(
            question_vector,
            future_vector,
        ),
        "qvik_vs_future": q2.agreement.compute_pair_metrics(
            qvik_vector,
            future_vector,
        ),
        "qvik_vs_h2o": q2.agreement.compute_pair_metrics(
            qvik_vector,
            h2o_vector,
        ),
        "qvik_vs_question": q2.agreement.compute_pair_metrics(
            qvik_vector,
            question_vector,
        ),
    }

    predictions = {
        "full_cache_generate": full_text,
        "full_cache_manual": matched_full_text,
    }
    h2o_cache = q2.trim_cache_per_head(
        q2.intervention.clone_cache(base_cache),
        head_scores=h2o_heads_tensor,
        image_positions=image_positions,
        prompt_len=prompt_len,
        n_keep=n_keep,
    )
    _, predictions["h2o_prefill"] = q2.decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        cache=h2o_cache,
        first_token=first_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    del h2o_cache

    for selector, scores in (
        ("question_prefill", question_vector),
        ("qvik", qvik_vector),
        ("future_oracle", future_vector),
    ):
        cache = q2.intervention.clone_cache(base_cache)
        cache = q2.intervention.trim_kv_cache_per_layer(
            cache,
            q2.shared_keep_masks(
                scores,
                image_positions=image_positions,
                prompt_len=prompt_len,
                n_keep=n_keep,
            ),
        )
        _, predictions[selector] = q2.decode_from_cache(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            first_token=first_token,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        )
        del cache
    del base_cache

    scores = {
        selector: score_one_prediction(sample, prediction)
        for selector, prediction in predictions.items()
    }
    references = [str(value) for value in sample_value(sample, "references", ())]
    result = {
        "schema_version": 2,
        "dataset": dataset_name,
        "sample_id": sample_id,
        "row_index": int(sample_value(sample, "row_index")),
        "semantic_question": semantic_question,
        "context": context,
        "references": references,
        "max_new_tokens": max_new_tokens,
        "answer_steps": answer_steps,
        "hit_max_new_tokens": bool(answer_steps >= max_new_tokens),
        "future_trajectory": "[last prompt query, y_1, ..., y_(T-1)]",
        "processed_image_sha256": processed_image_sha256,
        "same_processed_image_for_prefill_and_future": True,
        "question_token_count": int(question_positions.numel()),
        "student_prompt_tail_token_count": int(student_positions.numel()),
        "prompt_len": prompt_len,
        "n_text": n_text,
        "n_visual": n_visual,
        "n_visual_kept_per_head": n_keep,
        "requested_total_keep_ratio": total_keep_ratio,
        "actual_total_keep_ratio": (n_text + n_keep) / prompt_len,
        "actual_visual_keep_ratio": n_keep / n_visual,
        "h2o_definition": "sum over every causal prefill query; native per-head Top-K",
        "question_definition": (
            "mean over semantic-question query rows and heads; shared Top-K"
        ),
        "qvik_checkpoint": str(q2.agreement.DEFAULT_STUDENT_DIR),
        "full_cache_manual_matches_generate": bool(
            torch.equal(matched_full_ids, full_ids)
        ),
        "vector_pairs": vector_pairs,
        "token_metrics": token_metrics,
        "predictions": predictions,
        "scores": scores,
    }
    del image_tensor, input_ids
    return result, keep_masks


def write_manifest(samples: list[Any], path: Path) -> None:
    data.write_manifest(samples, path)


def extract(args: argparse.Namespace) -> int:
    invalid = sorted(set(args.datasets) - set(DEFAULT_DATASETS))
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")

    selected: dict[str, list[Any]] = {}
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
        selected[dataset_name] = samples
        write_manifest(
            samples,
            args.output_dir / "manifests" / f"{dataset_name}.json",
        )
    if args.manifest_only:
        print(
            f"[manifest] wrote {sum(map(len, selected.values()))} rows "
            f"under {args.output_dir / 'manifests'}"
        )
        return 0

    tokenizer, model, image_processor, student, image_feature_len = (
        q2.load_exact_model_and_student(args)
    )
    device = torch.device(args.device)
    run_config = {
        "datasets": list(args.datasets),
        "n_samples_per_dataset": args.n_samples,
        "sample_seed": args.sample_seed,
        "sample_slice": [args.sample_start, args.sample_end],
        "model_path": str(args.model_path.resolve()),
        "student_dir": str(args.student_dir.resolve()),
        "device": args.device,
        "total_keep_ratio": args.total_keep_ratio,
        "image_preprocessing": "qvik.llava15.mm_utils.process_images; one in-memory tensor",
    }
    atomic_json(
        args.output_dir
        / "run_configs"
        / f"{safe_name(args.device)}_{safe_name('_'.join(args.datasets))}.json",
        run_config,
    )

    if args.sample_start < 0:
        raise ValueError("sample-start must be nonnegative")
    sample_end = args.sample_end if args.sample_end is not None else args.n_samples
    if not args.sample_start <= sample_end <= args.n_samples:
        raise ValueError(
            f"Invalid sample slice [{args.sample_start}:{sample_end}] "
            f"for n={args.n_samples}"
        )
    work_selected = {
        name: values[args.sample_start:sample_end]
        for name, values in selected.items()
    }
    total = sum(len(values) for values in work_selected.values())
    completed = skipped = failures = 0
    started = time.time()
    for dataset_name in args.datasets:
        for sample in work_selected[dataset_name]:
            stem = f"{int(sample_value(sample, 'row_index')):06d}_{safe_name(sample_value(sample, 'sample_id'))}"
            result_path = args.output_dir / "samples" / dataset_name / f"{stem}.json"
            mask_path = args.output_dir / "masks" / dataset_name / f"{stem}.npz"
            failure_path = args.output_dir / "failures" / dataset_name / f"{stem}.txt"
            if result_path.exists() and mask_path.exists() and not args.overwrite:
                skipped += 1
                continue
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
                result["mask_file"] = str(mask_path.relative_to(args.output_dir))
                result["qvik_checkpoint"] = str(args.student_dir.resolve())
                atomic_json(result_path, result)
                q2.save_packed_masks(
                    mask_path,
                    masks,
                    n_visual=int(result["n_visual"]),
                    n_keep=int(result["n_visual_kept_per_head"]),
                )
                failure_path.unlink(missing_ok=True)
                completed += 1
            except Exception as error:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(f"{type(error).__name__}: {error}\n")
                failures += 1
                print(
                    f"[failure] dataset={dataset_name} "
                    f"id={sample_value(sample, 'sample_id')}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            processed = completed + skipped + failures
            if processed == 1 or processed % 10 == 0 or processed == total:
                print(
                    f"[progress] {processed}/{total} completed={completed} "
                    f"skipped={skipped} failures={failures} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            torch.cuda.empty_cache()
    return 1 if failures else 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(values: np.ndarray, args: argparse.Namespace, offset: int) -> tuple[float, float, float]:
    return q2.bootstrap(
        np.asarray(values, dtype=np.float64),
        repeats=args.bootstrap_replicates,
        seed=args.summary_seed + offset,
    )


def mean_layers(record: dict[str, Any], section: str, key: str, metric: str | None = None) -> float:
    value = record[section][key]
    if metric is not None:
        value = value[metric]
    return float(np.mean(value))


def write_layerwise_figure(
    summary_dir: Path,
    layer_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (row["metric"], int(row["layer"])): row for row in layer_rows
    }
    methods = (
        ("H2O all-prefill", "h2o", "#d55e00"),
        ("Semantic question", "question", "#009e73"),
        ("Q-ViK", "qvik", "#0072b2"),
    )
    panels = (
        ("future_topk_recall", "Future Top-K recall"),
        ("jaccard_vs_future", "Jaccard vs. Future"),
    )
    layers = sorted({int(row["layer"]) for row in layer_rows})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for axis, (suffix, ylabel) in zip(axes, panels, strict=True):
        for label, prefix, color in methods:
            rows = [
                lookup[(f"{prefix}_{suffix}", layer)] for layer in layers
            ]
            means = np.asarray([row["mean"] for row in rows])
            low = np.asarray([row["ci95_low"] for row in rows])
            high = np.asarray([row["ci95_high"] for row in rows])
            axis.plot(layers, means, label=label, color=color, linewidth=2)
            axis.fill_between(layers, low, high, color=color, alpha=0.14)
        axis.set_xlabel("Eviction layer")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.set_xlim(min(layers), max(layers))
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Exact total-cache 20% visual-token selection", fontsize=12)
    figure.tight_layout()
    figure.savefig(
        summary_dir / "layerwise_future_agreement.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        summary_dir / "layerwise_future_agreement.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def summarize(args: argparse.Namespace) -> int:
    records = [
        json.loads(path.read_text())
        for path in sorted((args.output_dir / "samples").glob("*/*.json"))
        if path.parent.name in args.datasets
    ]
    if not records:
        raise RuntimeError(f"No sample records under {args.output_dir / 'samples'}")
    expected = args.n_samples * len(args.datasets)
    if len(records) != expected and not args.allow_partial:
        counts: dict[str, int] = {}
        for record in records:
            counts[record["dataset"]] = counts.get(record["dataset"], 0) + 1
        raise RuntimeError(
            f"Expected {expected} sample records, found {len(records)}: {counts}"
        )

    summary_dir = args.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = [
        name for name in DEFAULT_DATASETS if any(r["dataset"] == name for r in records)
    ]
    groups = [("all", records)] + [
        (name, [record for record in records if record["dataset"] == name])
        for name in dataset_order
    ]

    vector_rows: list[dict[str, Any]] = []
    vector_metrics = q2.agreement.METRIC_NAMES
    for group_index, (group, selected) in enumerate(groups):
        for pair_index, pair in enumerate(VECTOR_PAIRS):
            for metric_index, metric in enumerate(vector_metrics):
                values = np.asarray(
                    [
                        mean_layers(record, "vector_pairs", pair, metric)
                        for record in selected
                    ]
                )
                mean, low, high = ci(
                    values,
                    args,
                    group_index * 10000 + pair_index * 100 + metric_index,
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
        for metric_index, metric in enumerate(vector_metrics):
            values = np.asarray(
                [
                    mean_layers(record, "vector_pairs", "qvik_vs_future", metric)
                    - mean_layers(record, "vector_pairs", "h2o_vs_future", metric)
                    for record in selected
                ]
            )
            mean, low, high = ci(
                values,
                args,
                100_000 + group_index * 100 + metric_index,
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

    token_rows: list[dict[str, Any]] = []
    token_keys = list(records[0]["token_metrics"])
    for group_index, (group, selected) in enumerate(groups):
        for metric_index, metric in enumerate(token_keys):
            values = np.asarray(
                [
                    mean_layers(record, "token_metrics", metric)
                    for record in selected
                ]
            )
            mean, low, high = ci(
                values,
                args,
                200_000 + group_index * 1000 + metric_index,
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

    caption_corpus_scores: dict[str, dict[str, float]] = {}
    for dataset_name in dataset_order:
        if dataset_name not in CAPTION_DATASETS:
            continue
        selected = [
            record for record in records if record["dataset"] == dataset_name
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
            decomposed_mean = float(
                np.mean([record["scores"][selector] for record in selected])
            )
            if not np.isclose(
                corpus_score,
                decomposed_mean,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{dataset_name}/{selector}: corpus ROUGE-L "
                    f"{corpus_score} != mean per-image score {decomposed_mean}"
                )
            caption_corpus_scores[dataset_name][selector] = corpus_score

    downstream_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        for selector_index, selector in enumerate(SELECTORS):
            values = np.asarray([record["scores"][selector] for record in selected])
            mean, low, high = ci(
                values,
                args,
                300_000 + group_index * 100 + selector_index,
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
        delta = np.asarray(
            [
                record["scores"]["qvik"] - record["scores"]["h2o_prefill"]
                for record in selected
            ]
        )
        mean, low, high = ci(delta, args, 400_000 + group_index)
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

    preservation_rows: list[dict[str, Any]] = []
    preservation_delta_rows: list[dict[str, Any]] = []
    for group_index, (group, selected) in enumerate(groups):
        reference = [record["predictions"]["full_cache_manual"] for record in selected]
        selector_matches: dict[str, np.ndarray] = {}
        for selector_index, selector in enumerate(SELECTORS):
            matches = np.asarray(
                [
                    float(record["predictions"][selector] == expected)
                    for record, expected in zip(selected, reference, strict=True)
                ]
            )
            selector_matches[selector] = matches
            mean, low, high = ci(
                matches,
                args,
                500_000 + group_index * 100 + selector_index,
            )
            preservation_rows.append(
                {
                    "group": group,
                    "selector": selector,
                    "exact_matches": int(matches.sum()),
                    "match_rate": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": len(selected),
                }
            )
        for delta_index, (label, first, second) in enumerate(
            (
                ("qvik_minus_h2o", "qvik", "h2o_prefill"),
                ("future_oracle_minus_h2o", "future_oracle", "h2o_prefill"),
            )
        ):
            paired = selector_matches[first] - selector_matches[second]
            mean, low, high = ci(
                paired,
                args,
                550_000 + group_index * 100 + delta_index,
            )
            preservation_delta_rows.append(
                {
                    "group": group,
                    "comparison": label,
                    "mean_delta": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "first_only_matches": int(
                        np.sum(
                            (selector_matches[first] == 1)
                            & (selector_matches[second] == 0)
                        )
                    ),
                    "second_only_matches": int(
                        np.sum(
                            (selector_matches[first] == 0)
                            & (selector_matches[second] == 1)
                        )
                    ),
                    "n_samples": len(selected),
                }
            )
    write_csv(summary_dir / "response_preservation.csv", preservation_rows)
    write_csv(
        summary_dir / "response_preservation_paired.csv",
        preservation_delta_rows,
    )

    layer_rows: list[dict[str, Any]] = []
    layer_metrics = (
        "h2o_future_topk_recall",
        "question_future_topk_recall",
        "qvik_future_topk_recall",
        "h2o_jaccard_vs_future",
        "question_jaccard_vs_future",
        "qvik_jaccard_vs_future",
        "h2o_visual_future_mass_retained",
        "question_visual_future_mass_retained",
        "qvik_visual_future_mass_retained",
    )
    n_layers = len(records[0]["token_metrics"][layer_metrics[0]])
    for metric_index, metric in enumerate(layer_metrics):
        for layer in range(n_layers):
            values = np.asarray(
                [record["token_metrics"][metric][layer] for record in records]
            )
            mean, low, high = ci(
                values,
                args,
                600_000 + metric_index * 100 + layer,
            )
            layer_rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": len(records),
                }
            )
    write_csv(summary_dir / "layerwise_token_metrics.csv", layer_rows)
    write_layerwise_figure(summary_dir, layer_rows)

    vector_lookup = {
        (row["group"], row["pair"], row["metric"]): row for row in vector_rows
    }
    token_lookup = {
        (row["group"], row["metric"]): row for row in token_rows
    }
    downstream_lookup = {
        (row["group"], row["selector"]): row for row in downstream_rows
    }
    preservation_lookup = {
        (row["group"], row["selector"]): row for row in preservation_rows
    }
    preservation_delta_lookup = {
        (row["group"], row["comparison"]): row
        for row in preservation_delta_rows
    }

    def formatted(row: dict[str, Any]) -> str:
        return (
            f"{row['mean']:.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )

    visual_keep_rates = np.asarray(
        [record["actual_visual_keep_ratio"] for record in records]
    )
    visual_keep_counts = np.asarray(
        [record["n_visual_kept_per_head"] for record in records]
    )
    lines = [
        "# Eval-700: H2O/question prefill vs. Future utility",
        "",
        f"Samples: {len(records)} ("
        + ", ".join(
            f"{name}={sum(r['dataset'] == name for r in records)}"
            for name in dataset_order
        )
        + "). "
        "MileBench is excluded.",
        "",
        "All selectors preserve every text KV and use the same total-prompt cache "
        f"ratio {args.total_keep_ratio:.2f}. H2O selects visual Top-K independently "
        "per KV head; Question, Q-ViK, and Future Oracle use shared layer-wise Top-K.",
        f"This total budget retains {visual_keep_counts.mean():.1f} visual KVs on "
        f"average (range {visual_keep_counts.min()}–{visual_keep_counts.max()}), "
        f"or {100 * visual_keep_rates.mean():.1f}% of the 576 visual KVs.",
        "",
        "Future is captured from full-cache greedy generation as "
        "`[last prompt query, y_1, ..., y_(T-1)]`, matching the student target.",
        "",
        "## Agreement with Future",
        "",
        "| Signal | Spearman | Cosine | visual Jaccard@20% |",
        "|---|---:|---:|---:|",
    ]
    for label, pair in (
        ("H2O all-prefill", "h2o_vs_future"),
        ("Semantic question only", "question_vs_future"),
        ("Q-ViK", "qvik_vs_future"),
        ("Q-ViK − H2O paired delta", "qvik_minus_h2o_vs_future"),
    ):
        lines.append(
            f"| {label} | "
            f"{formatted(vector_lookup[('all', pair, 'spearman')])} | "
            f"{formatted(vector_lookup[('all', pair, 'cosine')])} | "
            f"{formatted(vector_lookup[('all', pair, 'jaccard_20')])} |"
        )
    lines.extend(
        [
            "",
            "Direct Q-ViK–H2O vector agreement is "
            f"Spearman {formatted(vector_lookup[('all', 'qvik_vs_h2o', 'spearman')])}, "
            f"cosine {formatted(vector_lookup[('all', 'qvik_vs_h2o', 'cosine')])}, "
            "and visual Jaccard@20% "
            f"{formatted(vector_lookup[('all', 'qvik_vs_h2o', 'jaccard_20')])}.",
            "",
            "Q-ViK is substantially higher than H2O on Spearman and Top-K "
            "Jaccard, but not on cosine. The positive, diffuse attention "
            "distributions make cosine a poor proxy for the eviction mask; do "
            "not claim that every agreement metric increases.",
            "",
            "## Exact total-budget token selection",
            "",
            "| Selector | Future Top-K recall | Jaccard vs. Future | Future mass retained |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, prefix in (
        ("H2O all-prefill", "h2o"),
        ("Semantic question only", "question"),
        ("Q-ViK", "qvik"),
    ):
        lines.append(
            f"| {label} | "
            f"{formatted(token_lookup[('all', f'{prefix}_future_topk_recall')])} | "
            f"{formatted(token_lookup[('all', f'{prefix}_jaccard_vs_future')])} | "
            f"{formatted(token_lookup[('all', f'{prefix}_visual_future_mass_retained')])} |"
        )
    lines.extend(
        [
            "",
            "## Post-eviction downstream score",
            "",
            "Scores are reported on a 0–100 scale. VQA uses each benchmark's "
            "task metric; caption datasets use the paper's local ROUGE-L setting.",
            "",
            "| Dataset | Full generate | Matched Full | H2O | Question | Q-ViK | Future Oracle |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name in dataset_order:
        values = []
        for selector in (
            "full_cache_generate",
            "full_cache_manual",
            "h2o_prefill",
            "question_prefill",
            "qvik",
            "future_oracle",
        ):
            row = downstream_lookup[(dataset_name, selector)]
            values.append(f"{100 * row['mean']:.2f}")
        lines.append(f"| {dataset_name} | " + " | ".join(values) + " |")
    macro_values = []
    for selector in (
        "full_cache_generate",
        "full_cache_manual",
        "h2o_prefill",
        "question_prefill",
        "qvik",
        "future_oracle",
    ):
        macro_values.append(
            100
            * float(
                np.mean(
                    [
                        downstream_lookup[(name, selector)]["mean"]
                        for name in dataset_order
                    ]
                )
            )
        )
    lines.append(
        "| **7-benchmark macro** | "
        + " | ".join(f"**{value:.2f}**" for value in macro_values)
        + " |"
    )
    for label, names in (
        ("VQA macro", [name for name in dataset_order if name not in CAPTION_DATASETS]),
        ("Caption macro", [name for name in dataset_order if name in CAPTION_DATASETS]),
    ):
        subgroup_values = []
        for selector in (
            "full_cache_generate",
            "full_cache_manual",
            "h2o_prefill",
            "question_prefill",
            "qvik",
            "future_oracle",
        ):
            subgroup_values.append(
                100
                * float(
                    np.mean(
                        [
                            downstream_lookup[(name, selector)]["mean"]
                            for name in names
                        ]
                    )
                )
            )
        lines.append(
            f"| {label} | "
            + " | ".join(f"{value:.2f}" for value in subgroup_values)
            + " |"
        )

    response_delta = preservation_delta_lookup[("all", "qvik_minus_h2o")]
    score_delta = downstream_lookup[("all", "qvik_minus_h2o_prefill")]
    lines.extend(
        [
            "",
            "## Response preservation against matched Full Cache",
            "",
            "| Selector | Exact response matches | Match rate |",
            "|---|---:|---:|",
        ]
    )
    for label, selector in (
        ("H2O", "h2o_prefill"),
        ("Semantic question only", "question_prefill"),
        ("Q-ViK", "qvik"),
        ("Future Oracle", "future_oracle"),
    ):
        row = preservation_lookup[("all", selector)]
        lines.append(
            f"| {label} | {row['exact_matches']}/{row['n_samples']} | "
            f"{row['match_rate']:.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Paired Q-ViK − H2O response-preservation delta: "
            f"{response_delta['mean_delta']:.4f} "
            f"[{response_delta['ci95_low']:.4f}, "
            f"{response_delta['ci95_high']:.4f}] "
            f"({response_delta['first_only_matches']} Q-ViK-only matches vs. "
            f"{response_delta['second_only_matches']} H2O-only matches).",
            "",
            "The token-ranking claim is strongly supported, but aggregate task-score "
            "superiority is not resolved on this 100-example-per-task subset: "
            f"paired Q-ViK − H2O score delta is {100 * score_delta['mean']:.2f} "
            f"points [{100 * score_delta['ci95_low']:.2f}, "
            f"{100 * score_delta['ci95_high']:.2f}]. Caption ROUGE-L can improve "
            "when an eviction method changes wording even when it preserves the "
            "Full-Cache response less often, so task score and response fidelity "
            "must be reported separately.",
        ]
    )
    truncated = {
        name: sum(
            record["hit_max_new_tokens"]
            for record in records
            if record["dataset"] == name
        )
        for name in dataset_order
    }
    manual_matches = sum(
        record["full_cache_manual_matches_generate"] for record in records
    )
    lines.extend(
        [
            "",
            f"Matched manual Full Cache reproduces `generate()` for "
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
    (summary_dir / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        summary_dir / "provenance.json",
        {
            "n_samples": len(records),
            "datasets": dataset_order,
            "sample_seed": args.sample_seed,
            "n_samples_per_dataset": args.n_samples,
            "total_keep_ratio": args.total_keep_ratio,
            "manual_generate_exact_match_count": manual_matches,
            "truncation_counts": truncated,
            "caption_metric": "local paper setting: tokenized COCO ROUGE-L",
            "caption_corpus_scores": caption_corpus_scores,
            "docvqa_metric": "official continuous max-ANLS",
        },
    )
    print((summary_dir / "report.md").read_text())
    return 0


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        return summarize(args)
    return extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
