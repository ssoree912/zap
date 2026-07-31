# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect future-attention teacher labels from original LLaVA-1.5-7B.

This is intentionally experiment-local: the implementation-guide branch has a
Transformers LLaVA-wrapper collector, but the requested checkpoint is
the original LLaVA repo format and must be loaded through the VFlowOpt LLaVA
package.
"""

from __future__ import annotations

import argparse
import gc
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

from answer_correctness import is_correct_prediction

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
QVIK_ROOT = Path(os.environ.get("QVIK_ROOT", WORKSPACE_ROOT / "Q-ViK")).resolve()
for path in (QVIK_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from transformers import AutoTokenizer  # noqa: E402

from qvik.llava15.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from qvik.llava15.conversation import conv_templates  # noqa: E402
from qvik.llava15.mm_utils import tokenizer_image_token  # noqa: E402
from qvik.llava15.model.language_model.llava_llama import (  # noqa: E402
    LlavaLlamaForCausalLM,
)


def _patch_generation_config_nested_dicts() -> None:
    """Support older LLaVA configs with dict-valued nested model configs."""
    from types import SimpleNamespace

    from transformers.generation import configuration_utils

    original = configuration_utils.GenerationConfig.from_model_config.__func__

    @classmethod
    def patched(cls, model_config):
        for attr in ("decoder", "encoder", "text_config", "vision_config"):
            value = getattr(model_config, attr, None)
            if isinstance(value, dict):
                namespace = SimpleNamespace(**value)
                namespace.to_dict = lambda payload=value: payload
                setattr(model_config, attr, namespace)
        return original(cls, model_config)

    configuration_utils.GenerationConfig.from_model_config = patched


def _patch_llava_arch_for_dynamic_cache() -> None:
    """Make the vendored original-LLaVA code accept Transformers DynamicCache."""
    import qvik.llava15.model.llava_arch as llava_arch

    original = llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal

    def patched(self, input_ids, attention_mask, past_key_values, labels, images):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                if hasattr(past_key_values, "get_seq_length"):
                    past_len = past_key_values.get_seq_length()
                else:
                    past_len = max(cache[-1].shape[-2] for cache in past_key_values)
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_len + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels
        return original(self, input_ids, attention_mask, past_key_values, labels, images)

    llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = patched


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    question: str
    image_path: Path
    prompt: str
    answer: str | None = None
    answers: tuple[str, ...] = ()
    answer_index: int | None = None
    choices: tuple[str, ...] = ()


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
    candidates = (
        [path]
        if path.is_absolute()
        else [
            data_root / path,
            WORKSPACE_ROOT / "data/train" / path,
            Path("/workspace/zap/data") / path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_gqa(args: argparse.Namespace) -> list[Sample]:
    records = _read_json(Path(args.gqa_questions_json))
    samples: list[Sample] = []
    if isinstance(records, dict):
        rows = [(str(qid), rec) for qid, rec in records.items()]
    else:
        rows = [
            (str(rec.get("sample_id") or rec.get("question_id")), rec)
            for rec in records
        ]
    for sample_id, rec in rows:
        image_id = rec.get("imageId") or rec.get("image_id")
        image_value = rec.get("image_path")
        if image_value:
            image_path = _resolve_image(str(image_value), Path(args.data_root))
        elif image_id:
            image_path = Path(args.gqa_images_root) / f"{image_id}.jpg"
            image_path = image_path.resolve() if image_path.exists() else None
        else:
            image_path = None
        if image_path is None:
            continue
        answer = str(rec.get("answer", "")).strip()
        if not answer:
            continue
        question = (
            f"{str(rec['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        samples.append(
            Sample(
                dataset="gqa",
                sample_id=sample_id,
                question=question,
                image_path=image_path,
                prompt=_build_prompt(question, args.conv_template),
                answer=answer,
                answers=(answer,),
            )
        )
    random.Random(args.seed).shuffle(samples)
    return samples[: args.max_candidates] if args.max_candidates > 0 else samples


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
            answers=tuple(str(value) for value in answers) if isinstance(answers, list) else (),
        )
    samples = [sample for sample in chosen if sample is not None]
    return samples[: args.max_candidates] if args.max_candidates > 0 else samples


def _load_scienceqa(args: argparse.Namespace) -> list[Sample]:
    problems = _read_json(Path(args.scienceqa_problems_json))
    samples: list[Sample] = []
    for qid, prob in problems.items():
        split = str(prob.get("split") or qid.split("_", 1)[0])
        if split != args.scienceqa_split:
            continue
        image_path = Path(args.scienceqa_images_root) / split / qid / str(prob.get("image", "image.png"))
        if not image_path.exists():
            continue
        choices = tuple(str(choice) for choice in (prob.get("choices") or []))
        answer_idx = prob.get("answer")
        if not isinstance(answer_idx, int) or not 0 <= answer_idx < len(choices):
            continue
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
        answer = chr(65 + answer_idx)
        samples.append(
            Sample(
                dataset="scienceqa",
                sample_id=qid,
                question=question,
                image_path=image_path.resolve(),
                prompt=_build_prompt(question, args.conv_template),
                answer=answer,
                answers=(choices[answer_idx],),
                answer_index=answer_idx,
                choices=choices,
            )
        )
    random.Random(args.seed).shuffle(samples)
    return samples[: args.max_candidates] if args.max_candidates > 0 else samples


def _load_samples_from_teacher_manifest(
    manifest_root: Path,
    dataset: str,
    max_candidates: int,
) -> list[Sample]:
    """Reuse only sample metadata from an older cache; labels are recomputed."""
    samples: list[Sample] = []
    for path in sorted((manifest_root / dataset).glob("*.pt")):
        rec = torch.load(path, weights_only=False, map_location="cpu")
        image_path = Path(str(rec["image_path"]))
        if not image_path.exists():
            continue
        answers = rec.get("ground_truth_answers") or []
        answer = str(answers[0]) if answers else rec.get("answer")
        choices = tuple(str(value) for value in rec.get("ground_truth_choices", ()))
        answer_index = rec.get("ground_truth_answer_index")
        if dataset == "scienceqa" and not choices:
            continue
        if not answers and answer is None:
            continue
        prompt = str(rec["prompt_text"])
        samples.append(
            Sample(
                dataset=dataset,
                sample_id=str(rec.get("sample_id") or path.stem),
                question=str(rec.get("question") or prompt),
                image_path=image_path.resolve(),
                prompt=prompt,
                answer=str(answer) if answer is not None else None,
                answers=tuple(str(value) for value in answers) or (str(answer),),
                answer_index=int(answer_index) if answer_index is not None else None,
                choices=choices,
            )
        )
        if max_candidates > 0 and len(samples) >= max_candidates:
            break
    return samples


def load_samples(args: argparse.Namespace, dataset: str) -> list[Sample]:
    if args.sample_manifest_root:
        samples = _load_samples_from_teacher_manifest(
            Path(args.sample_manifest_root),
            dataset,
            args.max_candidates,
        )
        if samples:
            return samples
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


def _normalize_teacher(teacher: torch.Tensor, eps: float) -> torch.Tensor:
    return teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)


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
    question_weight: float,
    require_correct: bool,
    eps: float,
) -> tuple[dict[str, Any] | None, str, bool]:
    with Image.open(sample.image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)

    input_ids = tokenizer_image_token(
        sample.prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    image_positions, prompt_len_mm = _infer_image_positions(input_ids, image_feature_len)
    image_indices = image_positions.to(device=device)
    question_positions = _infer_question_positions(prompt_len_mm, image_positions)

    generated = model.generate(
        input_ids=input_ids,
        images=image_tensor,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer_ids = generated[:, input_ids.shape[1] :].detach()
    t_steps = int(answer_ids.shape[1])
    decoded = tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()
    del generated
    prediction_correct = is_correct_prediction(
        sample.dataset,
        decoded,
        sample.answers,
        answer_index=sample.answer_index,
        choices=sample.choices,
    )
    if require_correct and not prediction_correct:
        del answer_ids, input_ids, image_tensor
        gc.collect()
        torch.cuda.empty_cache()
        return None, decoded, False

    full_input_ids = torch.cat([input_ids, answer_ids], dim=1)
    full_out = model(
        input_ids=full_input_ids,
        images=image_tensor,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    if full_out.attentions is None or len(full_out.attentions) == 0:
        raise RuntimeError("Full question+answer pass did not return attentions")

    n_layers = len(full_out.attentions)
    n_img = int(image_positions.numel())
    teacher_question_raw = torch.zeros(n_layers, n_img, dtype=torch.float32)
    teacher_answer_raw = torch.zeros(n_layers, n_img, dtype=torch.float32)
    question_indices = question_positions.to(device=device)
    if question_indices.numel() == 0:
        question_indices = torch.tensor([prompt_len_mm - 1], device=device)
    if t_steps:
        answer_indices = torch.arange(
            prompt_len_mm,
            prompt_len_mm + t_steps,
            dtype=torch.long,
            device=device,
        )
    else:
        answer_indices = torch.tensor([prompt_len_mm - 1], device=device)

    for layer_idx, attn in enumerate(full_out.attentions):
        question_to_img = attn[0, :, question_indices, :].index_select(
            dim=-1,
            index=image_indices,
        )
        answer_to_img = attn[0, :, answer_indices, :].index_select(
            dim=-1,
            index=image_indices,
        )
        teacher_question_raw[layer_idx] = (
            question_to_img.float().mean(dim=(0, 1)).detach().cpu()
        )
        teacher_answer_raw[layer_idx] = (
            answer_to_img.float().mean(dim=(0, 1)).detach().cpu()
        )

    teacher_question_norm = _normalize_teacher(teacher_question_raw, eps)
    teacher_answer_norm = _normalize_teacher(teacher_answer_raw, eps)
    answer_weight = 1.0 - question_weight
    teacher_norm = _normalize_teacher(
        question_weight * teacher_question_norm
        + answer_weight * teacher_answer_norm,
        eps,
    )

    del full_out, full_input_ids, answer_ids
    gc.collect()
    torch.cuda.empty_cache()

    record = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "model": "llava-v1.5-7b-original",
        "prompt_text": sample.prompt,
        "question": sample.question,
        "answer": sample.answer,
        "decoded": decoded,
        "prediction": decoded,
        "prediction_correct": prediction_correct,
        "ground_truth_answers": list(sample.answers),
        "ground_truth_answer_index": sample.answer_index,
        "ground_truth_choices": list(sample.choices),
        "require_correct": require_correct,
        "image_path": str(sample.image_path),
        "image_token_indices": image_positions.to(torch.long),
        "question_token_indices": question_positions.to(torch.long),
        "teacher_raw": teacher_norm.to(torch.float16),
        "teacher_norm": teacher_norm.to(torch.float16),
        "teacher_question_raw": teacher_question_raw.to(torch.float16),
        "teacher_question_norm": teacher_question_norm.to(torch.float16),
        "teacher_answer_raw": teacher_answer_raw.to(torch.float16),
        "teacher_answer_norm": teacher_answer_norm.to(torch.float16),
        "teacher_question_weight": float(question_weight),
        "teacher_answer_weight": float(answer_weight),
        "teacher_signal": "question_answer_normalized_mix",
        "teacher_question_source": "causal_prefill_question_tokens",
        "teacher_answer_source": "generated_answer_tokens",
        "prompt_len_mm": int(prompt_len_mm),
        "T": int(t_steps),
        "n_img": int(n_img),
        "max_new_tokens": int(max_new_tokens),
    }
    return record, decoded, prediction_correct


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, int]:
    print(
        f"[load] model={args.model_path} conv_template={args.conv_template} "
        f"device={args.device} attn=eager",
        flush=True,
    )
    _patch_generation_config_nested_dicts()
    _patch_llava_arch_for_dynamic_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(torch.device(args.device)).eval()
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=torch.device(args.device), dtype=torch.float16)
    image_processor = vision_tower.image_processor
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"image_feature_len={image_feature_len}",
        flush=True,
    )
    return tokenizer, model, image_processor, image_feature_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(WORKSPACE_ROOT / "models/llava-v1.5-7b"))
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["gqa", "textvqa", "scienceqa"])
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Target number of teacher samples to save per dataset.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum shuffled candidates to inspect per dataset; 0 means all.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts/original_llava_teacher/qa50_llava15_300"))
    parser.add_argument("--data-root", default=str(WORKSPACE_ROOT / "data/train"))
    parser.add_argument(
        "--gqa-questions-json",
        default=str(WORKSPACE_ROOT / "data/train/gqa/val_balanced_questions.json"),
    )
    parser.add_argument(
        "--gqa-images-root",
        default=str(WORKSPACE_ROOT / "data/train/gqa/images"),
    )
    parser.add_argument("--textvqa-json", default=str(WORKSPACE_ROOT / "data/train/textvqa/train/data.json"))
    parser.add_argument("--textvqa-ids-json", default=str(REPO_ROOT / "artifacts/original_llava_teacher/subsets/textvqa_300_seed0_ids.json"))
    parser.add_argument("--scienceqa-problems-json", default=str(WORKSPACE_ROOT / "data/train/scienceqa/problems.json"))
    parser.add_argument("--scienceqa-images-root", default=str(WORKSPACE_ROOT / "data/train/scienceqa/images"))
    parser.add_argument("--scienceqa-ids-json", default=str(REPO_ROOT / "artifacts/original_llava_teacher/subsets/scienceqa_300_seed0_ids.json"))
    parser.add_argument("--scienceqa-split", default="train")
    parser.add_argument("--scienceqa-prompt-style", choices=["choices_only", "direct_letter"], default="choices_only")
    parser.add_argument(
        "--sample-manifest-root",
        default=str(WORKSPACE_ROOT / "data/train/teacher/llava15_qa50_all"),
        help="Optional older cache used only to select sample IDs/prompts/images.",
    )
    parser.add_argument("--question-weight", type=float, default=0.5)
    parser.add_argument(
        "--require-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save teacher records only when the base-model prediction is correct.",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.question_weight <= 1.0:
        raise ValueError("--question-weight must be in [0, 1]")
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

        existing = 0
        existing_correct = 0
        valid_existing_paths: set[Path] = set()
        for path in sorted(out_dir.glob("*.pt")):
            try:
                rec = torch.load(path, weights_only=False, map_location="cpu")
            except Exception:
                continue
            existing += 1
            prediction_correct = bool(rec.get("prediction_correct", False))
            existing_correct += int(prediction_correct)
            if not args.require_correct or prediction_correct:
                valid_existing_paths.add(path)

        saved = len(valid_existing_paths)
        newly_saved = 0
        evaluated = 0
        rejected_incorrect = 0
        incorrect_examples: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        t_values: list[int] = []
        t0 = time.time()
        for idx, sample in enumerate(samples, start=1):
            if saved >= args.n_samples:
                break
            out_path = out_dir / f"{_sanitize_id(sample.sample_id)}.pt"
            if out_path in valid_existing_paths and not args.overwrite:
                try:
                    rec = torch.load(out_path, map_location="cpu")
                    t_values.append(int(rec.get("T", 0)))
                except Exception:
                    skipped.append({"sample_id": sample.sample_id, "error": "existing file unreadable"})
                continue
            evaluated += 1
            try:
                rec, decoded, prediction_correct = collect_one(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    image_feature_len=image_feature_len,
                    question_weight=args.question_weight,
                    require_correct=args.require_correct,
                    eps=1e-8,
                )
                if rec is None:
                    rejected_incorrect += 1
                    if len(incorrect_examples) < 100:
                        incorrect_examples.append(
                            {
                                "sample_id": sample.sample_id,
                                "prediction": decoded,
                                "answers": list(sample.answers),
                                "answer_index": sample.answer_index,
                            }
                        )
                    if idx % 25 == 0:
                        print(
                            f"[progress] {dataset} evaluated={evaluated}/{len(samples)} "
                            f"saved={saved}/{args.n_samples} "
                            f"rejected={rejected_incorrect} skipped={len(skipped)}",
                            flush=True,
                        )
                    continue
                torch.save(rec, out_path)
                saved += 1
                newly_saved += 1
                t_values.append(int(rec["T"]))
                if newly_saved == 1:
                    print(
                        f"[sanity] {dataset} sid={sample.sample_id} "
                        f"L={tuple(rec['teacher_raw'].shape)} T={rec['T']} "
                        f"prompt_len_mm={rec['prompt_len_mm']} decoded={rec['decoded']!r} "
                        f"correct={prediction_correct}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"sample_id": sample.sample_id, "error": repr(exc)})
                print(f"[skip] {dataset} sid={sample.sample_id}: {exc}", flush=True)

            if idx % 25 == 0 or saved == args.n_samples:
                elapsed = time.time() - t0
                mean_t = float(np.mean(t_values)) if t_values else 0.0
                hit_max = sum(1 for value in t_values if value >= args.max_new_tokens)
                print(
                    f"[progress] {dataset} evaluated={evaluated}/{len(samples)} "
                    f"saved={saved}/{args.n_samples} rejected={rejected_incorrect} "
                    f"skipped={len(skipped)} rate={evaluated / max(elapsed, 1e-6):.3f}/s "
                    f"T_mean={mean_t:.3f} hit{args.max_new_tokens}={hit_max}/{len(t_values)}",
                    flush=True,
                )

        elapsed = time.time() - t0
        summary = {
            "dataset": dataset,
            "n_requested": args.n_samples,
            "n_saved": saved,
            "n_existing": existing,
            "n_existing_correct": existing_correct,
            "n_newly_saved": newly_saved,
            "n_candidates": len(samples),
            "n_evaluated": evaluated,
            "n_rejected_incorrect": rejected_incorrect,
            "incorrect_examples": incorrect_examples,
            "n_skipped": len(skipped),
            "skipped": skipped,
            "elapsed_seconds": elapsed,
            "max_new_tokens": args.max_new_tokens,
            "question_weight": args.question_weight,
            "answer_weight": 1.0 - args.question_weight,
            "require_correct": args.require_correct,
            "t_min": int(min(t_values)) if t_values else 0,
            "t_max": int(max(t_values)) if t_values else 0,
            "t_mean": float(np.mean(t_values)) if t_values else 0.0,
            f"hit{args.max_new_tokens}": int(sum(1 for value in t_values if value >= args.max_new_tokens)),
        }
        (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[done] {dataset} {summary}", flush=True)
        all_summaries[dataset] = summary
        if saved != args.n_samples:
            print(
                f"[error] {dataset}: exhausted candidates before reaching target "
                f"{saved}/{args.n_samples}",
                flush=True,
            )
            return 2

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
