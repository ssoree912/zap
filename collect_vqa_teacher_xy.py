#!/usr/bin/env python3
"""Collect teacher XY shards for TextVQA and NLVR2.

Unified schema (mirrors ScienceQA structure):
  sample["question"]     – question / statement text
  sample["image_paths"]  – list of absolute image path strings
  sample["prompt_body"]  – pre-built prompt body (no template wrapper)

Prompt templates per dataset:
  textvqa:
    USER: <image>
    Question: {question}
    ASSISTANT:

  nlvr2:
    USER: <image>
    <image>
    Statement: {sentence}
    Is this statement True or False?
    Options:
    A. True
    B. False
    ASSISTANT:

Usage:
  conda run -n kv python collect_vqa_teacher_xy.py \\
    --dataset textvqa \\
    --data_dir /workspace/hd/data/textvqa/train \\
    --out_dir /workspace/hd/artifacts/sq_teacher/textvqa_xy/att_only_postvision \\
    --teacher_type att_only_postvision \\
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from kvzap.llava_extractor import _collect_postvision_minimal_sample, _parse_torch_dtype, configure_llava_processor

# re-use shard/writer utilities from ScienceQA script
from collect_scienceqa_teacher_xy import (
    XYShardWriter,
    _apply_target_transform,
    _parse_storage_dtype,
    _parse_target_dtype,
    _select_token_indices_top_random,
    _resolve_teacher_keys,
)


# ── Prompt templates ───────────────────────────────────────────────────────────

TEXTVQA_PROMPT_TEMPLATE = "USER: <image>\nQuestion: {question}\nASSISTANT:"

NLVR2_PROMPT_TEMPLATE = (
    "USER: <image>\n<image>\n"
    "Statement: {sentence}\n"
    "Is this statement True or False?\n"
    "Options:\nA. True\nB. False\n"
    "ASSISTANT:"
)


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_textvqa_samples(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load from {data_dir}/data.json produced by download_textvqa_nlvr2.py."""
    data_path = data_dir / "data.json"
    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)
    if limit is not None:
        records = records[:limit]

    samples = []
    for rec in records:
        img_path = str(data_dir.parent.parent / rec["image_path"])
        question = str(rec["question"]).strip()
        samples.append({
            "sample_id": str(rec["question_id"]),
            "question": question,
            "sentence": question,  # alias for template compatibility
            "image_paths": [img_path],
            "prompt_body": f"Question: {question}",
        })
    return samples


def _load_nlvr2_samples(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load from {data_dir}/data.json produced by download_textvqa_nlvr2.py."""
    data_path = data_dir / "data.json"
    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)
    if limit is not None:
        records = records[:limit]

    samples = []
    for rec in records:
        sentence = str(rec["sentence"]).strip()
        img_paths = [
            str(data_dir.parent.parent / p) for p in rec["image_paths"]
        ]
        if len(img_paths) != 2:
            continue  # skip malformed
        samples.append({
            "sample_id": str(rec["identifier"]),
            "question": sentence,
            "sentence": sentence,
            "image_paths": img_paths,
            "prompt_body": (
                f"Statement: {sentence}\n"
                "Is this statement True or False?\n"
                "Options:\nA. True\nB. False"
            ),
        })
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["textvqa", "nlvr2"],
                        help="Which dataset to collect XY for")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Split data dir, e.g. /workspace/hd/data/textvqa/train")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Output shard dir")
    parser.add_argument("--teacher_type", choices=["att_only_postvision", "splus_postvision"],
                        default="att_only_postvision")
    parser.add_argument("--implementation_model_name", type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--sample_image_tokens", choices=["all", "top_random"], default="all")
    parser.add_argument("--top_fraction", type=float, default=0.2)
    parser.add_argument("--random_fraction", type=float, default=0.2)
    parser.add_argument("--max_tokens_per_layer", type=int, default=None)
    parser.add_argument("--min_tokens_per_layer", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=50000)
    parser.add_argument("--target_transform", choices=["none", "log"], default="log")
    parser.add_argument("--target_eps", type=float, default=1e-8)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--storage_dtype", type=str, default="float16")
    parser.add_argument("--target_dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--teacher_head_reduction", choices=["none", "mean"], default="mean")
    parser.add_argument("--include_sample_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(out_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {out_dir}. Use --overwrite to continue.")

    # Select prompt template
    if args.dataset == "textvqa":
        prompt_template = TEXTVQA_PROMPT_TEMPLATE
        samples = _load_textvqa_samples(data_dir, limit=args.limit)
    else:
        prompt_template = NLVR2_PROMPT_TEMPLATE
        samples = _load_nlvr2_samples(data_dir, limit=args.limit)

    print(f"Dataset: {args.dataset}, split dir: {data_dir}")
    print(f"Samples loaded: {len(samples)}")
    print(f"Prompt template:\n{prompt_template}\n")

    teacher_att_key, teacher_splus_key, target_key = _resolve_teacher_keys(args.teacher_type)
    storage_dtype = _parse_storage_dtype(args.storage_dtype)
    target_dtype = _parse_target_dtype(args.target_dtype)

    model_kwargs: dict[str, Any] = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": _parse_torch_dtype(args.torch_dtype),
    }

    # Save run config
    run_config = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "prompt_template": prompt_template,
        "teacher_type": args.teacher_type,
        "n_samples": len(samples),
        **vars(args),
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Loading model: {args.implementation_model_name} …")
    processor = AutoProcessor.from_pretrained(args.implementation_model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.implementation_model_name, **model_kwargs
    )
    configure_llava_processor(processor, model.config)
    model = model.to(args.device)
    model.eval()

    writer = XYShardWriter(
        output_dir=out_dir,
        split_name=data_dir.name,
        max_rows_per_shard=args.shard_size,
        include_sample_ids=args.include_sample_ids,
    )
    rng = np.random.default_rng(args.seed)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "teacher_type": args.teacher_type,
        "n_samples_requested": len(samples),
        "n_samples_succeeded": 0,
        "n_samples_failed": 0,
        "n_rows_written": 0,
        "n_shards_written": 0,
        "failures": [],
    }

    for sample in tqdm(samples, desc=f"Collecting {args.dataset} XY shards"):
        try:
            record = _collect_postvision_minimal_sample(
                model=model,
                processor=processor,
                sample=sample,
                prompt_template=prompt_template,
                capture_vproj=True,
                save_hidden_image=True,
                save_teacher_scores=True,
                teacher_head_reduction=args.teacher_head_reduction,
                teacher_att_key=teacher_att_key,
                teacher_splus_key=teacher_splus_key,
                storage_dtype=storage_dtype,
            )

            hidden_image = record["hidden_image"]
            teacher = record[target_key]

            if not isinstance(hidden_image, torch.Tensor) or hidden_image.ndim != 3:
                raise ValueError(
                    f"hidden_image must be [L, I, D], got {type(hidden_image)} {getattr(hidden_image, 'shape', None)}"
                )
            if not isinstance(teacher, torch.Tensor) or teacher.ndim != 2:
                raise ValueError(
                    f"teacher must be [L, I], got {type(teacher)} {getattr(teacher, 'shape', None)}"
                )
            if hidden_image.shape[:2] != teacher.shape:
                raise ValueError(
                    f"Shape mismatch: hidden={tuple(hidden_image.shape)}, teacher={tuple(teacher.shape)}"
                )

            x_all: list[torch.Tensor] = []
            y_all: list[torch.Tensor] = []
            layer_all: list[torch.Tensor] = []
            sample_id_all: list[str] = []

            for layer_idx in range(int(hidden_image.shape[0])):
                layer_hidden = hidden_image[layer_idx].detach().cpu()
                layer_teacher = teacher[layer_idx].detach().cpu().float()

                if args.sample_image_tokens == "all":
                    selected = torch.arange(layer_hidden.shape[0], dtype=torch.long)
                else:
                    selected = _select_token_indices_top_random(
                        scores=layer_teacher,
                        rng=rng,
                        top_fraction=args.top_fraction,
                        random_fraction=args.random_fraction,
                        max_tokens_per_layer=args.max_tokens_per_layer,
                        min_tokens_per_layer=args.min_tokens_per_layer,
                    )

                if selected.numel() == 0:
                    continue

                x = layer_hidden.index_select(0, selected).contiguous()
                y = layer_teacher.index_select(0, selected).contiguous()
                y = _apply_target_transform(y, target_transform=args.target_transform,
                                            target_eps=args.target_eps)
                layer_ids = torch.full((selected.shape[0],), layer_idx, dtype=torch.long)

                x_all.append(x)
                y_all.append(y.to(dtype=target_dtype))
                layer_all.append(layer_ids)
                if args.include_sample_ids:
                    sample_id_all.extend([str(sample["sample_id"])] * int(selected.shape[0]))

            if x_all:
                writer.add(
                    x=torch.cat(x_all, dim=0),
                    y=torch.cat(y_all, dim=0),
                    layer_ids=torch.cat(layer_all, dim=0),
                    sample_ids=sample_id_all,
                )

            summary["n_samples_succeeded"] += 1

        except Exception as exc:
            summary["n_samples_failed"] += 1
            summary["failures"].append({"sample_id": sample.get("sample_id", "?"), "error": str(exc)})
            if not args.continue_on_error:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    writer.flush()
    writer.write_dataset_spec()
    summary["n_rows_written"] = writer.rows_written
    summary["n_shards_written"] = writer.shards_written

    with (out_dir / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Rows: {writer.rows_written}, Shards: {writer.shards_written}")
    print(f"Failed: {summary['n_samples_failed']} / {summary['n_samples_requested']}")


if __name__ == "__main__":
    main()
