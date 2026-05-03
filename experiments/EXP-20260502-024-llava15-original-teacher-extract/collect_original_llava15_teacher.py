# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect future-attention teacher labels from original LLaVA-1.5-7B.

This is intentionally experiment-local: the implementation-guide branch has a
HF `LlavaForConditionalGeneration` collector, but the requested checkpoint is
the original LLaVA repo format and must be loaded through the VFlowOpt LLaVA
package.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

VFLOWOPT_LLAVA_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    question: str
    image_path: Path
    prompt: str
    answer: str | None = None


def _sanitize_id(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:128] or "sample"


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _build_prompt(question: str, conv_template: str) -> str:
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _resolve_image(path_value: str, data_root: Path) -> Path | None:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [data_root / path, Path("/workspace/zap/data") / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_gqa(args: argparse.Namespace) -> list[Sample]:
    records = _read_json(Path(args.gqa_subset_json))
    samples: list[Sample] = []
    for rec in records:
        image_path = _resolve_image(str(rec["image_path"]), Path(args.data_root))
        if image_path is None:
            continue
        question = (
            f"{str(rec['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        samples.append(
            Sample(
                dataset="gqa",
                sample_id=str(rec.get("sample_id") or rec.get("question_id")),
                question=question,
                image_path=image_path,
                prompt=_build_prompt(question, args.conv_template),
                answer=str(rec.get("answer")) if rec.get("answer") is not None else None,
            )
        )
    return samples[: args.n_samples]


def _load_textvqa(args: argparse.Namespace) -> list[Sample]:
    ids_payload = _read_json(Path(args.textvqa_ids_json))
    selected_ids = [str(x) for x in ids_payload["sample_ids"]]
    id_to_rank = {sample_id: rank for rank, sample_id in enumerate(selected_ids)}

    records = _read_json(Path(args.textvqa_json))
    chosen: list[Sample | None] = [None] * len(selected_ids)
    for rec in records:
        qid = str(rec.get("question_id", rec.get("id", "")))
        if qid not in id_to_rank:
            continue
        image_path = _resolve_image(str(rec["image_path"]), Path(args.data_root))
        if image_path is None:
            continue
        question = (
            f"{str(rec['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        answers = rec.get("answers")
        answer = str(answers[0]) if isinstance(answers, list) and answers else None
        chosen[id_to_rank[qid]] = Sample(
            dataset="textvqa",
            sample_id=qid,
            question=question,
            image_path=image_path,
            prompt=_build_prompt(question, args.conv_template),
            answer=answer,
        )
    return [sample for sample in chosen if sample is not None][: args.n_samples]


def _load_scienceqa(args: argparse.Namespace) -> list[Sample]:
    ids_payload = _read_json(Path(args.scienceqa_ids_json))
    selected_ids = [str(x) for x in ids_payload["sample_ids"]]
    problems = _read_json(Path(args.scienceqa_problems_json))
    samples: list[Sample] = []
    for qid in selected_ids:
        prob = problems.get(qid)
        if not prob:
            continue
        split = qid.split("_", 1)[0]
        image_path = Path(args.scienceqa_images_root) / split / qid / str(prob.get("image", "image.png"))
        if not image_path.exists():
            continue
        choices = prob.get("choices") or []
        if args.scienceqa_prompt_style == "direct_letter":
            choice_lines = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
            body_parts = []
            hint = str(prob.get("hint", "")).strip()
            if hint:
                body_parts.append(f"Hint: {hint}")
            body_parts.append(f"Question: {str(prob.get('question', '')).strip()}")
            if choice_lines:
                body_parts.append(f"Choices:\n{choice_lines}")
            body_parts.append("Answer with the option's letter from the given choices directly.")
            question = "\n".join(body_parts)
        else:
            question = str(prob.get("question", "")).strip()
            if choices:
                lettered = " ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices))
                question = f"{question}\n{lettered}"
        answer_idx = prob.get("answer")
        answer = chr(65 + int(answer_idx)) if isinstance(answer_idx, int) else None
        samples.append(
            Sample(
                dataset="scienceqa",
                sample_id=qid,
                question=question,
                image_path=image_path.resolve(),
                prompt=_build_prompt(question, args.conv_template),
                answer=answer,
            )
        )
    return samples[: args.n_samples]


def load_samples(args: argparse.Namespace, dataset: str) -> list[Sample]:
    if dataset == "gqa":
        return _load_gqa(args)
    if dataset == "textvqa":
        return _load_textvqa(args)
    if dataset == "scienceqa":
        return _load_scienceqa(args)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _infer_image_positions(input_ids: torch.Tensor, image_feature_len: int) -> tuple[torch.Tensor, int]:
    raw_ids = input_ids[0].detach().cpu()
    placeholder_positions = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected exactly one image placeholder, found {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    image_positions = torch.arange(image_start, image_start + image_feature_len, dtype=torch.long)
    prompt_len_mm = int(raw_ids.numel() - 1 + image_feature_len)
    return image_positions, prompt_len_mm


def _infer_question_positions(prompt_len_mm: int, image_positions: torch.Tensor) -> torch.Tensor:
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


def _decode_generated(tokenizer: Any, sequences: torch.Tensor, t_steps: int) -> str:
    if sequences.numel() == 0 or t_steps <= 0:
        return ""
    answer_ids = sequences[0, -t_steps:].detach().cpu().tolist()
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def collect_one(
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    sample: Sample,
    max_new_tokens: int,
    device: torch.device,
    image_feature_len: int,
    eps: float,
) -> dict[str, Any]:
    with Image.open(sample.image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        sample.prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    image_positions, prompt_len_mm = _infer_image_positions(input_ids, image_feature_len)
    image_indices = image_positions.to(device=device)
    question_positions = _infer_question_positions(prompt_len_mm, image_positions)

    out = model.generate(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError("Generation did not return attentions")

    n_layers = len(out.attentions[0])
    n_img = int(image_positions.numel())
    teacher_raw = torch.zeros(n_layers, n_img, dtype=torch.float32)
    for step_attns in out.attentions:
        for layer_idx, attn in enumerate(step_attns):
            # First generate step can be [B,H,T,T]; later steps are [B,H,1,T].
            attn_to_img = attn[0, :, -1, :].index_select(dim=-1, index=image_indices)
            teacher_raw[layer_idx] += attn_to_img.float().mean(dim=0).detach().cpu()

    t_steps = len(out.attentions)
    teacher_raw /= float(t_steps)
    teacher_norm = teacher_raw / teacher_raw.sum(dim=-1, keepdim=True).clamp_min(eps)

    decoded = _decode_generated(tokenizer, out.sequences, t_steps)
    del out
    torch.cuda.empty_cache()

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "model": "llava-v1.5-7b-original",
        "prompt_text": sample.prompt,
        "question": sample.question,
        "answer": sample.answer,
        "decoded": decoded,
        "image_path": str(sample.image_path),
        "image_token_indices": image_positions.to(torch.long),
        "question_token_indices": question_positions.to(torch.long),
        "teacher_raw": teacher_raw.to(torch.float16),
        "teacher_norm": teacher_norm.to(torch.float16),
        "prompt_len_mm": int(prompt_len_mm),
        "T": int(t_steps),
        "n_img": int(n_img),
        "max_new_tokens": int(max_new_tokens),
    }


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, int]:
    print(
        f"[load] model={args.model_path} conv_template={args.conv_template} "
        f"device_map={args.device_map} attn=eager",
        flush=True,
    )
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="eager",
        multimodal=True,
    )
    model.eval()
    vision_tower = model.get_vision_tower()
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"image_feature_len={image_feature_len}",
        flush=True,
    )
    return tokenizer, model, image_processor, image_feature_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-v1.5-7b")
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["gqa", "textvqa", "scienceqa"])
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="/workspace/zap/artifacts/original_llava_teacher/future_decode_llava15_7b")
    parser.add_argument("--data-root", default="/workspace/zap/data")
    parser.add_argument("--gqa-subset-json", default="/workspace/zap/data/gqa/train/subset_600_seed42.json")
    parser.add_argument("--textvqa-json", default="/workspace/zap/data/textvqa/train/data.json")
    parser.add_argument("--textvqa-ids-json", default="/workspace/zap/artifacts/original_llava_teacher/subsets/textvqa_600_seed42_ids.json")
    parser.add_argument("--scienceqa-problems-json", default="/workspace/zap/data/scienceqa/problems.json")
    parser.add_argument("--scienceqa-images-root", default="/workspace/zap/data/scienceqa/images")
    parser.add_argument("--scienceqa-ids-json", default="/workspace/zap/artifacts/original_llava_teacher/subsets/scienceqa_600_seed42_ids.json")
    parser.add_argument("--scienceqa-prompt-style", choices=["choices_only", "direct_letter"], default="choices_only")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer, model, image_processor, image_feature_len = _load_model(args)
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, Any] = {}
    for dataset in args.datasets:
        samples = load_samples(args, dataset)
        if args.limit_per_dataset is not None:
            samples = samples[: args.limit_per_dataset]
        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] {dataset} samples={len(samples)} out={out_dir}", flush=True)

        saved = 0
        skipped: list[dict[str, str]] = []
        t_values: list[int] = []
        t0 = time.time()
        for idx, sample in enumerate(samples, start=1):
            out_path = out_dir / f"{_sanitize_id(sample.sample_id)}.pt"
            if out_path.exists() and not args.overwrite:
                try:
                    rec = torch.load(out_path, map_location="cpu")
                    t_values.append(int(rec.get("T", 0)))
                    saved += 1
                except Exception:
                    skipped.append({"sample_id": sample.sample_id, "error": "existing file unreadable"})
                continue
            try:
                rec = collect_one(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    image_feature_len=image_feature_len,
                    eps=1e-8,
                )
                torch.save(rec, out_path)
                saved += 1
                t_values.append(int(rec["T"]))
                if idx == 1:
                    print(
                        f"[sanity] {dataset} sid={sample.sample_id} "
                        f"L={tuple(rec['teacher_raw'].shape)} T={rec['T']} "
                        f"prompt_len_mm={rec['prompt_len_mm']} decoded={rec['decoded']!r}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"sample_id": sample.sample_id, "error": repr(exc)})
                print(f"[skip] {dataset} sid={sample.sample_id}: {exc}", flush=True)

            if idx % 25 == 0 or idx == len(samples):
                elapsed = time.time() - t0
                mean_t = float(np.mean(t_values)) if t_values else 0.0
                hit_max = sum(1 for value in t_values if value >= args.max_new_tokens)
                print(
                    f"[progress] {dataset} {idx}/{len(samples)} saved={saved} "
                    f"skipped={len(skipped)} rate={idx / max(elapsed, 1e-6):.3f}/s "
                    f"T_mean={mean_t:.3f} hit{args.max_new_tokens}={hit_max}/{len(t_values)}",
                    flush=True,
                )

        elapsed = time.time() - t0
        summary = {
            "dataset": dataset,
            "n_requested": len(samples),
            "n_saved": saved,
            "n_skipped": len(skipped),
            "skipped": skipped,
            "elapsed_seconds": elapsed,
            "max_new_tokens": args.max_new_tokens,
            "t_min": int(min(t_values)) if t_values else 0,
            "t_max": int(max(t_values)) if t_values else 0,
            "t_mean": float(np.mean(t_values)) if t_values else 0.0,
            f"hit{args.max_new_tokens}": int(sum(1 for value in t_values if value >= args.max_new_tokens)),
        }
        (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[done] {dataset} {summary}", flush=True)
        all_summaries[dataset] = summary

    root_summary_path = output_root / "_summary.json"
    if root_summary_path.exists():
        try:
            merged_summaries = json.loads(root_summary_path.read_text())
        except json.JSONDecodeError:
            merged_summaries = {}
    else:
        merged_summaries = {}
    merged_summaries.update(all_summaries)
    root_summary_path.write_text(json.dumps(merged_summaries, indent=2))
    print(f"[save] {output_root / '_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
