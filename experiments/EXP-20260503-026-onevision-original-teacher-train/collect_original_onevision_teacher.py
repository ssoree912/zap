#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect future-attention teacher labels from original LLaVA-OneVision.

This collector targets `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov`,
which is a LLaVA-OneVision `LlavaQwenForCausalLM` checkpoint, not a HF
`LlavaOnevisionForConditionalGeneration` checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
import traceback
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache

REPO_ROOT = Path("/workspace/zap")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(LLAVA_ONEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
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


def patch_siglip_loader_to_local_init() -> None:
    """Avoid the hard-coded SigLIP path in this LLaVA-OneVision checkout."""
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


def _sanitize_id(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:128] or "sample"


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _build_prompt(question: str, conv_template: str) -> str:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _resolve_image(path_value: str, data_root: Path) -> Path | None:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [data_root / path, REPO_ROOT / "data" / path]
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


def _to_image_inputs(image_tensor: Any, device: torch.device) -> Any:
    if isinstance(image_tensor, torch.Tensor):
        return image_tensor.to(device=device, dtype=torch.float16)
    return [tensor.to(device=device, dtype=torch.float16) for tensor in image_tensor]


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
        if isinstance(cfg, (list, tuple)):
            eid = cfg[0]
        else:
            eid = cfg
    return int(eid if eid is not None else 151645)


def _get_decoder_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    raise AttributeError("Could not locate decoder layers for attention capture")


def _decode_generated(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def _generate_token_ids(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    image_tensor: Any,
    image_size: tuple[int, int],
    max_new_tokens: int,
) -> list[int]:
    out = model.generate(
        inputs=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ids = out.sequences[0].tolist() if hasattr(out, "sequences") else out[0].tolist()
    if len(ids) > max_new_tokens:
        ids = ids[-max_new_tokens:]
    eos_token_id = _resolve_eos_token_id(tokenizer, model.config)
    if eos_token_id in ids:
        ids = ids[: ids.index(eos_token_id)]
    return [int(tok) for tok in ids]


@torch.no_grad()
def collect_one(
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    sample: Sample,
    max_new_tokens: int,
    device: torch.device,
    eps: float,
) -> dict[str, Any]:
    with Image.open(sample.image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    image_tensor = _to_image_inputs(image_tensor, device)

    input_ids = tokenizer_image_token(
        sample.prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    generated_ids = _generate_token_ids(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        image_tensor=image_tensor,
        image_size=image_size,
        max_new_tokens=max_new_tokens,
    )
    if not generated_ids:
        raise RuntimeError("model.generate produced zero answer tokens")

    prefill = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill.past_key_values
    if isinstance(past_kv, tuple):
        past_kv = DynamicCache.from_legacy_cache(past_kv)
    prompt_len_mm = _cache_seq_len(past_kv)
    image_positions, image_feature_len = _infer_image_positions(input_ids, prompt_len_mm)
    question_positions = _infer_question_positions(prompt_len_mm, image_positions)
    image_indices = image_positions.to(device=device)

    attn_mask = torch.ones((1, prompt_len_mm), dtype=torch.long, device=device)
    generated: list[int] = []
    teacher_raw: torch.Tensor | None = None
    t_steps = 0
    attn_layers = _get_decoder_layers(model)
    original_forwards: dict[int, Any] = {}
    captured_attn: dict[int, torch.Tensor] = {}

    def make_patched(layer_idx: int, original_forward: Any):
        def patched(*args, **kwargs):
            kwargs["output_attentions"] = True
            out = original_forward(*args, **kwargs)
            attn_w = out[1]
            if attn_w is not None:
                captured_attn[layer_idx] = (
                    attn_w[0, :, -1, :]
                    .index_select(dim=-1, index=image_indices)
                    .float()
                    .mean(dim=0)
                    .detach()
                    .cpu()
                )
            return (out[0], None) + out[2:]

        return patched

    for li, layer in enumerate(attn_layers):
        original_forwards[li] = layer.self_attn.forward
        layer.self_attn.forward = make_patched(li, original_forwards[li])

    try:
        for tok in generated_ids[:max_new_tokens]:
            generated.append(tok)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                dim=1,
            )
            captured_attn.clear()
            cache_position = torch.tensor([prompt_len_mm + t_steps], dtype=torch.long, device=device)
            next_token = torch.tensor([[tok]], dtype=torch.long, device=device)
            step_out = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_kv = step_out.past_key_values
            if isinstance(past_kv, tuple):
                past_kv = DynamicCache.from_legacy_cache(past_kv)
            if teacher_raw is None:
                teacher_raw = torch.zeros(
                    len(attn_layers),
                    int(image_positions.numel()),
                    dtype=torch.float32,
                )
            if len(captured_attn) != len(attn_layers):
                raise RuntimeError(
                    f"Captured attentions for {len(captured_attn)}/{len(attn_layers)} layers"
                )
            for layer_idx in range(len(attn_layers)):
                teacher_raw[layer_idx] += captured_attn[layer_idx]
            t_steps += 1
    finally:
        for li, layer in enumerate(attn_layers):
            layer.self_attn.forward = original_forwards[li]

    if t_steps == 0 or teacher_raw is None:
        raise RuntimeError("Decode loop produced zero teacher steps")

    teacher_raw /= float(t_steps)
    teacher_raw = torch.nan_to_num(teacher_raw, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = teacher_raw.sum(dim=-1, keepdim=True)
    bad_rows = (~torch.isfinite(row_sum)) | (row_sum <= eps)
    teacher_norm = teacher_raw / row_sum.clamp_min(eps)
    if bad_rows.any():
        teacher_norm[bad_rows.squeeze(-1)] = 1.0 / max(1, int(image_positions.numel()))
    if not torch.isfinite(teacher_norm).all():
        raise RuntimeError("teacher_norm contains non-finite values after cleanup")

    decoded = _decode_generated(tokenizer, generated)
    del prefill, past_kv
    torch.cuda.empty_cache()

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "model": "llava-onevision-qwen2-7b-original",
        "prompt_text": sample.prompt,
        "question": sample.question,
        "question_text": sample.question,
        "answer": sample.answer,
        "decoded": decoded,
        "image_path": str(sample.image_path),
        "image_token_indices": image_positions.to(torch.long),
        "question_token_indices": question_positions.to(torch.long),
        "teacher_raw": teacher_raw.to(torch.float16),
        "teacher_norm": teacher_norm.to(torch.float16),
        "prompt_len_mm": int(prompt_len_mm),
        "T": int(t_steps),
        "n_img": int(image_positions.numel()),
        "image_feature_len": int(image_feature_len),
        "max_new_tokens": int(max_new_tokens),
        "bad_teacher_rows": int(bad_rows.sum().item()),
    }


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    patch_siglip_loader_to_local_init()
    print(
        f"[load] model={args.model_path} model_name={args.model_name} "
        f"conv_template={args.conv_template} device_map={args.device_map} attn=sdpa",
        flush=True,
    )
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="sdpa",
        multimodal=True,
    )
    model.eval()
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"hidden={model.config.hidden_size}",
        flush=True,
    )
    return tokenizer, model, image_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "gqa", "scienceqa"])
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="/workspace/zap/artifacts/original_onevision_teacher/future_decode_qwen2_7b")
    parser.add_argument("--data-root", default="/workspace/zap/data")
    parser.add_argument("--gqa-subset-json", default="/workspace/zap/data/gqa/train/subset_600_seed42.json")
    parser.add_argument("--textvqa-json", default="/workspace/zap/data/textvqa/train/data.json")
    parser.add_argument("--textvqa-ids-json", default="/workspace/zap/artifacts/original_llava_teacher/subsets/textvqa_600_seed42_ids.json")
    parser.add_argument("--scienceqa-problems-json", default="/workspace/zap/data/scienceqa/problems.json")
    parser.add_argument("--scienceqa-images-root", default="/workspace/zap/data/scienceqa/images")
    parser.add_argument("--scienceqa-ids-json", default="/workspace/zap/artifacts/original_llava_teacher/subsets/scienceqa_600_seed42_ids.json")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer, model, image_processor = _load_model(args)
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
        n_img_values: list[int] = []
        t0 = time.time()
        for idx, sample in enumerate(samples, start=1):
            out_path = out_dir / f"{_sanitize_id(sample.sample_id)}.pt"
            if out_path.exists() and not args.overwrite:
                try:
                    rec = torch.load(out_path, map_location="cpu")
                    t_values.append(int(rec.get("T", 0)))
                    n_img_values.append(int(rec.get("n_img", 0)))
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
                    eps=1e-8,
                )
                torch.save(rec, out_path)
                saved += 1
                t_values.append(int(rec["T"]))
                n_img_values.append(int(rec["n_img"]))
                if idx == 1:
                    print(
                        f"[sanity] {dataset} sid={sample.sample_id} "
                        f"shape={tuple(rec['teacher_raw'].shape)} T={rec['T']} "
                        f"prompt_len_mm={rec['prompt_len_mm']} n_img={rec['n_img']} "
                        f"decoded={rec['decoded']!r}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"sample_id": sample.sample_id, "error": repr(exc)})
                print(
                    f"[skip] {dataset} sid={sample.sample_id}: {exc}\n{traceback.format_exc()}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()

            if idx % 25 == 0 or idx == len(samples):
                elapsed = time.time() - t0
                mean_t = float(np.mean(t_values)) if t_values else 0.0
                hit_max = sum(1 for value in t_values if value >= args.max_new_tokens)
                mean_img = float(np.mean(n_img_values)) if n_img_values else 0.0
                print(
                    f"[progress] {dataset} {idx}/{len(samples)} saved={saved} "
                    f"skipped={len(skipped)} rate={idx / max(elapsed, 1e-6):.3f}/s "
                    f"T_mean={mean_t:.3f} hit{args.max_new_tokens}={hit_max}/{len(t_values)} "
                    f"n_img_mean={mean_img:.1f}",
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
            "n_img_min": int(min(n_img_values)) if n_img_values else 0,
            "n_img_max": int(max(n_img_values)) if n_img_values else 0,
            "n_img_mean": float(np.mean(n_img_values)) if n_img_values else 0.0,
        }
        (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[done] {dataset} {summary}", flush=True)
        all_summaries[dataset] = summary

    root_summary_path = output_root / "_summary.json"
    root_summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"[save] {root_summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
