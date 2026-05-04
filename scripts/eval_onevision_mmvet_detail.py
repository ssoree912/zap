#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-cache LLaVA-OneVision inference for MMVet and detail_1k.

This script is intentionally separate from the older LLaVA-1.5 PrefixKV
evaluation scripts. It uses the original LLaVA-OneVision repo loader because
the local checkpoint is a LlavaQwenForCausalLM checkpoint, not a HF
LlavaOnevisionForConditionalGeneration checkpoint.

ROUGE protocol:
  * If --rouge-ref-path is provided, score generations against that full-cache
    reference file.
  * Otherwise, the current full-cache generation is used as the ROUGE reference
    and written to fullcache_rouge_ref.json for later pruned/student runs.

PPL protocol:
  * Teacher-force the original dataset answer, not the full-cache generation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DynamicCache
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if LLAVA_ONEVISION_ROOT.exists() and str(LLAVA_ONEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision


def patch_siglip_loader_to_local_init() -> None:
    """Avoid a hard-coded external SigLIP path in this LLaVA-OneVision checkout.

    The local checkpoint already contains `model.vision_tower.*` weights. The
    upstream file tries to initialize SigLIP from a lab-specific absolute path
    before those weights are loaded. Initializing the module from config lets
    `from_pretrained()` fill it from the local checkpoint shards.
    """
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):  # noqa: ANN001, ARG001
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model

DEFAULT_MODEL_PATH = "/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"
DATASET_DEFAULTS = {
    "mmvet": {
        "data_path": "/workspace/data/mm-vet/mm-vet.json",
        "image_path": "/workspace/data/mm-vet",
        "eval_samples": 218,
        "max_new_tokens": 128,
    },
    "detail_1k": {
        "data_path": "/workspace/data/detail_1k.json",
        "image_path": "/workspace/data",
        "eval_samples": 1000,
        "max_new_tokens": 512,
    },
}


def load_samples(data_path: str, image_path: str, eval_samples: int | None) -> list[dict[str, Any]]:
    """Normalize detail_1k and MMVet JSON into id/image/question/answer records."""
    with open(data_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items = [{"id": k, **v} for k, v in raw.items()]
    else:
        items = list(raw)

    samples: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if "conversations" in item:
            convs = item["conversations"]
            if len(convs) < 2:
                raise ValueError(f"Expected >=2 conversation turns in sample {idx}")
            question = convs[0]["value"]
            answer = convs[1]["value"]
            image_rel = item["image"]
        else:
            question = item["question"]
            answer = item["answer"]
            image_rel = item.get("image") or item.get("imagename")
            if image_rel and "mm-vet" in data_path and not image_rel.startswith("images/"):
                image_rel = f"images/{image_rel}"

        question = question.replace("<image>", "").replace("\n\n", "\n").strip()
        sample_id = item.get("id", item.get("sample_id", str(idx)))
        samples.append({
            "id": sample_id,
            "image": image_rel,
            "image_file": str(Path(image_path) / image_rel),
            "question": question,
            "answer": answer,
        })

    if eval_samples is not None:
        samples = samples[:eval_samples]
    return samples


def build_prompt(question: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    content = f"{DEFAULT_IMAGE_TOKEN}\n{question.strip()}"
    conv.append_message(conv.roles[0], content)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def encode_prompt(tokenizer: Any, question: str, conv_template: str, device: torch.device) -> tuple[torch.Tensor, str]:
    prompt_text, stop_str = build_prompt(question, conv_template)
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    return input_ids, stop_str


def prepare_image_tensors(
    *,
    image: Image.Image,
    image_processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, list[tuple[int, int]]]:
    image_tensors = process_images([image], image_processor, model.config)
    if isinstance(image_tensors, torch.Tensor):
        image_tensors = image_tensors.to(device=device, dtype=dtype)
    else:
        image_tensors = [tensor.to(device=device, dtype=dtype) for tensor in image_tensors]
    return image_tensors, [image.size]


def _cache_seq_len(past_key_values: Any) -> int:
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        return int(past_key_values.key_cache[0].shape[-2])
    return int(past_key_values[0][0].shape[-2])


def _infer_image_positions(input_ids: torch.Tensor, prompt_len_mm: int) -> tuple[torch.Tensor, int]:
    raw_ids = input_ids[0].detach().cpu()
    placeholder_positions = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected exactly one image placeholder, found {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    image_feature_len = int(prompt_len_mm - raw_ids.numel() + 1)
    if image_feature_len <= 0:
        raise ValueError(
            f"Invalid image feature length={image_feature_len} prompt_len={prompt_len_mm} raw_len={raw_ids.numel()}"
        )
    image_positions = torch.arange(image_start, image_start + image_feature_len, dtype=torch.long)
    return image_positions, image_feature_len


def _infer_question_positions(prompt_len_mm: int, image_positions: torch.Tensor) -> torch.Tensor:
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


def _resolve_eos_token_id(tokenizer: Any, model_config: Any) -> int:
    eid = tokenizer.eos_token_id
    if eid is None:
        cfg = getattr(model_config, "eos_token_id", None)
        eid = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    return int(eid if eid is not None else 151645)


def _trim_kv_cache_per_layer(past_kv: Any, keep_masks: dict[int, torch.Tensor]) -> Any:
    if hasattr(past_kv, "key_cache"):
        for li in range(len(past_kv.key_cache)):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(past_kv.key_cache[li].device)
            past_kv.key_cache[li] = past_kv.key_cache[li][:, :, mask, :].contiguous()
            past_kv.value_cache[li] = past_kv.value_cache[li][:, :, mask, :].contiguous()
        return past_kv

    if hasattr(past_kv, "layers"):
        for li, layer in enumerate(past_kv.layers):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(layer.keys.device)
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return past_kv

    new_cache = DynamicCache()
    for li, (k, v) in enumerate(past_kv):
        if li in keep_masks:
            mask = keep_masks[li].to(k.device)
            k = k[:, :, mask, :].contiguous()
            v = v[:, :, mask, :].contiguous()
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    return new_cache


@torch.no_grad()
def _greedy_decode_with_kv(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    past_kv: Any,
    first_next_token: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
) -> str:
    eos_token_id = _resolve_eos_token_id(tokenizer, model.config)
    out_tokens: list[int] = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id:
        return ""

    next_token = first_next_token
    pos = int(prompt_len)
    device = next_token.device
    cache_pos = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(max_new_tokens - 1):
        cache_pos[0] = pos
        out = model(
            input_ids=next_token,
            past_key_values=past_kv,
            cache_position=cache_pos,
            position_ids=cache_pos.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(next_token.item())
        out_tokens.append(tok)
        pos += 1
        if tok == eos_token_id:
            break
    return tokenizer.decode(out_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def _student_keep_masks(
    *,
    student: VisualUtilityStudentOneVision,
    H_all: tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    prompt_len: int,
    keep_ratio: float,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    image_positions, image_feature_len = _infer_image_positions(input_ids, prompt_len)
    question_positions = _infer_question_positions(prompt_len, image_positions)
    n_img = int(image_positions.numel())
    n_text = int(prompt_len) - n_img
    n_keep = max(1, int(round(n_img - (1.0 - keep_ratio) * int(prompt_len))))
    n_keep = min(n_keep, n_img)

    image_idx_dev = image_positions.to(device=device)
    q_idx_dev = question_positions.to(device=device)

    keep_masks: dict[int, torch.Tensor] = {}
    for li in student.layer_indices:
        if li + 1 >= len(H_all):
            continue
        H_l = H_all[li + 1]
        scores = student.forward_layer(li, H_l, image_idx_dev, q_idx_dev).squeeze(0)
        if n_keep >= n_img:
            continue
        top = torch.topk(scores, k=n_keep, largest=True).indices
        mask = torch.ones(int(prompt_len), dtype=torch.bool)
        image_keep = torch.zeros(n_img, dtype=torch.bool)
        image_keep[top.detach().cpu()] = True
        mask[image_positions] = image_keep
        keep_masks[li] = mask

    stats = {
        "prompt_len": int(prompt_len),
        "n_image_original": n_img,
        "n_image_kept": n_keep,
        "n_text": n_text,
        "image_feature_len": image_feature_len,
        "total_keep_ratio": (n_text + n_keep) / max(1, int(prompt_len)),
        "image_keep_ratio": n_keep / max(1, n_img),
    }
    return keep_masks, stats


@torch.no_grad()
def generate_answer(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    conv_template: str,
    max_new_tokens: int,
) -> str:
    input_ids, stop_str = encode_prompt(tokenizer, sample["question"], conv_template, device)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    image_tensors, image_sizes = prepare_image_tensors(
        image=image,
        image_processor=image_processor,
        model=model,
        device=device,
        dtype=dtype,
    )

    outputs = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_tensors,
        image_sizes=image_sizes,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


@torch.no_grad()
def generate_answer_with_student(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudentOneVision,
    keep_ratio: float,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    conv_template: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    input_ids, _stop_str = encode_prompt(tokenizer, sample["question"], conv_template, device)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    image_tensors, image_sizes = prepare_image_tensors(
        image=image,
        image_processor=image_processor,
        model=model,
        device=device,
        dtype=dtype,
    )
    prefill = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_tensors,
        image_sizes=image_sizes,
        modalities=["image"],
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill.past_key_values
    prompt_len = _cache_seq_len(past_kv)
    keep_masks, keep_stats = _student_keep_masks(
        student=student,
        H_all=prefill.hidden_states,
        input_ids=input_ids,
        prompt_len=prompt_len,
        keep_ratio=keep_ratio,
        device=device,
    )
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill
    past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)
    text = _greedy_decode_with_kv(
        model=model,
        tokenizer=tokenizer,
        past_kv=past_kv,
        first_next_token=next_token,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.empty_cache()
    return text, keep_stats


@torch.no_grad()
def compute_answer_nll(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    conv_template: str,
    max_answer_tokens: int | None,
) -> tuple[float, int]:
    prompt_ids, _ = encode_prompt(tokenizer, sample["question"], conv_template, device)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    image_tensors, image_sizes = prepare_image_tensors(
        image=image,
        image_processor=image_processor,
        model=model,
        device=device,
        dtype=dtype,
    )

    answer_ids = tokenizer.encode(
        sample["answer"],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    if max_answer_tokens is not None:
        answer_ids = answer_ids[:, :max_answer_tokens]
    n_answer = int(answer_ids.shape[1])
    if n_answer == 0:
        return 0.0, 0

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : int(prompt_ids.shape[1])] = IGNORE_INDEX
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        images=image_tensors,
        image_sizes=image_sizes,
        use_cache=False,
        return_dict=True,
    )
    return float(outputs.loss.float().item()) * n_answer, n_answer


@torch.no_grad()
def compute_answer_nll_with_student(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudentOneVision,
    keep_ratio: float,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    conv_template: str,
    max_answer_tokens: int | None,
) -> tuple[float, int]:
    prompt_ids, _ = encode_prompt(tokenizer, sample["question"], conv_template, device)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    image_tensors, image_sizes = prepare_image_tensors(
        image=image,
        image_processor=image_processor,
        model=model,
        device=device,
        dtype=dtype,
    )

    answer_ids = tokenizer.encode(
        sample["answer"],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    if max_answer_tokens is not None:
        answer_ids = answer_ids[:, :max_answer_tokens]
    n_answer = int(answer_ids.shape[1])
    if n_answer == 0:
        return 0.0, 0

    prefill = model(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        images=image_tensors,
        image_sizes=image_sizes,
        modalities=["image"],
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill.past_key_values
    prompt_len = _cache_seq_len(past_kv)
    keep_masks, _keep_stats = _student_keep_masks(
        student=student,
        H_all=prefill.hidden_states,
        input_ids=prompt_ids,
        prompt_len=prompt_len,
        keep_ratio=keep_ratio,
        device=device,
    )
    past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

    sum_nll = F.cross_entropy(
        prefill.logits[:, -1, :].float(),
        answer_ids[:, 0],
        reduction="sum",
    ).float()
    del prefill

    if n_answer > 1:
        cache_pos = torch.zeros(1, dtype=torch.long, device=device)
        for idx in range(n_answer - 1):
            cache_pos[0] = int(prompt_len + idx)
            out = model(
                input_ids=answer_ids[:, idx : idx + 1],
                past_key_values=past_kv,
                cache_position=cache_pos,
                position_ids=cache_pos.unsqueeze(0),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_kv = out.past_key_values
            sum_nll = sum_nll + F.cross_entropy(
                out.logits[:, -1, :].float(),
                answer_ids[:, idx + 1],
                reduction="sum",
            ).float()

    torch.cuda.empty_cache()
    return float(sum_nll.item()), n_answer


def load_rouge_refs(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "per_sample" in raw:
        records = raw["per_sample"]
    elif isinstance(raw, dict):
        records = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "answer": v}
                   for k, v in raw.items()]
    else:
        records = list(raw)

    refs: dict[str, str] = {}
    for rec in records:
        sid = str(rec.get("id", rec.get("sample_id")))
        ref = (
            rec.get("fullcache_answer")
            or rec.get("rouge_reference")
            or rec.get("pred")
            or rec.get("prediction")
            or rec.get("answer")
        )
        if sid and ref is not None:
            refs[sid] = str(ref)
    return refs


def resolve_rouge_ref_path(path: str | None, dataset: str) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        candidate = p / dataset / "fullcache_rouge_ref.json"
        if candidate.exists():
            return str(candidate)
    return str(p)


def _rouge_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    return tokens or ["<empty>"]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for tok_long in long:
        cur = [0]
        for j, tok_short in enumerate(short, start=1):
            if tok_long == tok_short:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, reference: str) -> float:
    pred_tokens = _rouge_tokens(pred)
    ref_tokens = _rouge_tokens(reference)
    lcs = _lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_score(pred: str, reference: str) -> float:
    pred_text = pred.strip() or "<empty>"
    ref_text = reference.strip() or "<empty>"
    alternatives = [a.strip() for a in re.split(r"<OR>|<AND>", ref_text) if a.strip()]
    if not alternatives:
        alternatives = ["<empty>"]
    return max(_rouge_l_f1(pred_text, alt) for alt in alternatives)


def evaluate_dataset(
    *,
    dataset: str,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudentOneVision | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    defaults = DATASET_DEFAULTS[dataset]
    data_path = args.data_path or defaults["data_path"]
    image_path = args.image_path or defaults["image_path"]
    eval_samples = args.eval_samples if args.eval_samples is not None else defaults["eval_samples"]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else defaults["max_new_tokens"]

    output_dir = Path(args.output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    predictions_path = output_dir / "predictions.jsonl"
    fullcache_ref_path = output_dir / "fullcache_rouge_ref.json"

    if result_path.exists() and not args.overwrite:
        print(f"[skip] {dataset}: {result_path} exists")
        return
    if predictions_path.exists():
        predictions_path.unlink()

    samples = load_samples(data_path, image_path, eval_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {data_path}")

    rouge_ref_path = resolve_rouge_ref_path(args.rouge_ref_path, dataset)
    rouge_refs = load_rouge_refs(rouge_ref_path)
    use_self_ref = not bool(rouge_refs)
    per_sample: list[dict[str, Any]] = []
    ref_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scores: list[float] = []
    total_nll = 0.0
    total_tokens = 0
    t_start = time.time()

    desc = f"{dataset} onevision fullcache"
    iterator = tqdm(samples, desc=desc)
    for sample in iterator:
        try:
            keep_stats = None
            if student is not None and args.keep_ratio < 1.0:
                pred, keep_stats = generate_answer_with_student(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    student=student,
                    keep_ratio=args.keep_ratio,
                    sample=sample,
                    device=device,
                    dtype=dtype,
                    conv_template=args.conv_template,
                    max_new_tokens=max_new_tokens,
                )
                sum_nll, n_tokens = compute_answer_nll_with_student(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    student=student,
                    keep_ratio=args.keep_ratio,
                    sample=sample,
                    device=device,
                    dtype=dtype,
                    conv_template=args.conv_template,
                    max_answer_tokens=args.max_answer_tokens,
                )
            else:
                pred = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    sample=sample,
                    device=device,
                    dtype=dtype,
                    conv_template=args.conv_template,
                    max_new_tokens=max_new_tokens,
                )
                sum_nll, n_tokens = compute_answer_nll(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    sample=sample,
                    device=device,
                    dtype=dtype,
                    conv_template=args.conv_template,
                    max_answer_tokens=args.max_answer_tokens,
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sample["id"], "error": repr(exc)})
            gc.collect()
            torch.cuda.empty_cache()
            continue

        sid = str(sample["id"])
        rouge_ref = pred if use_self_ref else rouge_refs.get(sid, "")
        rouge_l = rouge_l_score(pred, rouge_ref)
        total_nll += sum_nll
        total_tokens += n_tokens
        scores.append(rouge_l)

        record = {
            "id": sid,
            "image": sample["image"],
            "question": sample["question"],
            "pred": pred,
            "rouge_reference": rouge_ref,
            "gt_answer": sample["answer"],
            "rouge_l_f": rouge_l,
            "n_answer_tokens": n_tokens,
            "sum_nll": sum_nll,
            "keep_stats": keep_stats,
        }
        per_sample.append(record)
        ref_records.append({
            "id": sid,
            "image": sample["image"],
            "question": sample["question"],
            "answer": pred,
            "fullcache_answer": pred,
            "gt_answer": sample["answer"],
        })

        with predictions_path.open("a" if predictions_path.exists() else "w") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if total_tokens:
            iterator.set_postfix(
                rouge=f"{sum(scores) / len(scores):.4f}",
                ppl=f"{math.exp(total_nll / total_tokens):.3f}",
            )

    if not per_sample:
        raise RuntimeError(f"{dataset}: all samples failed")
    if total_tokens == 0:
        raise RuntimeError(f"{dataset}: no answer tokens scored")

    elapsed = time.time() - t_start
    result = {
        "dataset": dataset,
        "model_path": args.model_path,
        "student_path": args.student_path,
        "keep_ratio": args.keep_ratio,
        "rouge_reference": "self_fullcache" if use_self_ref else rouge_ref_path,
        "rouge_l_f_mean": sum(scores) / len(scores),
        "ppl": math.exp(total_nll / total_tokens),
        "n_samples": len(per_sample),
        "n_failures": len(failures),
        "n_total_answer_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "sec_per_sample": elapsed / max(len(per_sample), 1),
        "args": vars(args),
        "per_sample": per_sample,
        "failures": failures,
    }

    with result_path.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    with fullcache_ref_path.open("w") as f:
        json.dump(ref_records, f, ensure_ascii=False, indent=2)

    print(
        f"[DONE] {dataset} rouge_l={result['rouge_l_f_mean']:.4f} "
        f"ppl={result['ppl']:.4f} n={len(per_sample)} fail={len(failures)} "
        f"-> {result_path}"
    )
    print(f"[REF] {dataset} full-cache ROUGE reference -> {fullcache_ref_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_DEFAULTS), default=["mmvet", "detail_1k"])
    parser.add_argument("--data-path", default=None,
                        help="Override data path. Only valid when one dataset is selected.")
    parser.add_argument("--image-path", default=None,
                        help="Override image root. Only valid when one dataset is selected.")
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--rouge-ref-path", default=None,
                        help="Optional full-cache reference JSON or output root containing <dataset>/fullcache_rouge_ref.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-path", default=None)
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.data_path or args.image_path) and len(args.datasets) != 1:
        raise ValueError("--data-path/--image-path overrides require selecting exactly one dataset")

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device(args.device)
    patch_siglip_loader_to_local_init()
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    if args.device_map != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    if dtype != torch.float16:
        model = model.to(dtype=dtype)

    student = None
    if args.student_path:
        student = VisualUtilityStudentOneVision.from_pretrained(args.student_path, map_location="cpu")
        student = student.to(device=device, dtype=dtype).eval()
        print(
            f"[student] path={args.student_path} keep_ratio={args.keep_ratio} "
            f"layers={student.layer_indices}",
            flush=True,
        )

    for dataset in args.datasets:
        evaluate_dataset(
            dataset=dataset,
            args=args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            student=student,
            device=device,
            dtype=dtype,
        )


if __name__ == "__main__":
    main()
