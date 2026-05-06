#!/usr/bin/env python3
"""Collect future-attention teacher labels from HF eval arrow datasets.

Datasets supported (default location: /mnt/srv/home/dlpc.3842/zap/data/eval/):
    - textvqa_val      (HF arrow,  fields: image, question, answers, ...)
    - docvqa_val       (HF arrow,  fields: image, question, answers, ...)
    - gqa              (instructions + images sub-datasets, joined on imageId)

Each .pt file mirrors the schema produced by `collect_original_llava15_teacher.py`,
with one difference: instead of a filesystem `image_path`, the PIL image is stored
in-place under `image_bytes` (bytes of a re-saved JPEG). This keeps the dump
self-contained and removes path-remapping headaches.

Usage:
    python collect_eval_teacher.py \
        --eval-root /mnt/srv/home/dlpc.3842/zap/data/eval \
        --datasets textvqa_val docvqa_val gqa \
        --n-samples 100 \
        --output-root /mnt/srv/home/dlpc.3842/zap/artifacts/eval_teacher_llava15_7b
"""
from __future__ import annotations

import argparse
import io
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

# Reuse the original collector's LLaVA infrastructure
VFLOWOPT_LLAVA_ROOT = Path(os.environ.get(
    "VFLOWOPT_LLAVA_ROOT",
    "/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision",
))
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from datasets import load_from_disk  # noqa: E402


@dataclass
class Sample:
    dataset: str
    sample_id: str
    question: str
    image: Any            # PIL.Image
    prompt: str
    answer: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_id(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:128] or "sample"


def _build_prompt(question: str, conv_template: str) -> str:
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _ensure_rgb(im):
    return im.convert("RGB") if im.mode != "RGB" else im


def _select_random_indices(total: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = min(n, total)
    return rng.choice(total, size=n, replace=False).tolist()


def _load_textvqa_val(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    ds = load_from_disk(str(root / "textvqa_val"))
    idxs = _select_random_indices(len(ds), n, seed)
    out: list[Sample] = []
    for ex in ds.select(idxs):
        q = (
            f"{str(ex['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        ans = ex["answers"][0] if isinstance(ex.get("answers"), list) and ex["answers"] else None
        out.append(Sample(
            dataset="textvqa_val",
            sample_id=str(ex["question_id"]),
            question=q,
            image=_ensure_rgb(ex["image"]),
            prompt=_build_prompt(q, conv_template),
            answer=str(ans) if ans is not None else None,
        ))
    return out


def _load_docvqa_val(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    ds = load_from_disk(str(root / "docvqa_val"))
    idxs = _select_random_indices(len(ds), n, seed)
    out: list[Sample] = []
    for ex in ds.select(idxs):
        q = (
            f"{str(ex['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        ans = ex["answers"][0] if isinstance(ex.get("answers"), list) and ex["answers"] else None
        out.append(Sample(
            dataset="docvqa_val",
            sample_id=str(ex["questionId"]),
            question=q,
            image=_ensure_rgb(ex["image"]),
            prompt=_build_prompt(q, conv_template),
            answer=str(ans) if ans is not None else None,
        ))
    return out


def _load_gqa(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    """GQA: join instructions (question/answer) with images dataset by imageId."""
    inst = load_from_disk(str(root / "gqa" / "instructions"))
    imgs = load_from_disk(str(root / "gqa" / "images"))
    img_lookup = {ex["id"]: ex["image"] for ex in imgs}

    # Filter to instructions whose image is available, then random-sample
    eligible_idxs = [i for i, ex in enumerate(inst) if ex["imageId"] in img_lookup]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(eligible_idxs), size=min(n, len(eligible_idxs)), replace=False)
    pick = [eligible_idxs[i] for i in chosen]

    out: list[Sample] = []
    for ex in inst.select(pick):
        q = (
            f"{str(ex['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        ans = ex.get("answer")
        out.append(Sample(
            dataset="gqa",
            sample_id=str(ex["id"]),
            question=q,
            image=_ensure_rgb(img_lookup[ex["imageId"]]),
            prompt=_build_prompt(q, conv_template),
            answer=str(ans) if ans is not None else None,
        ))
    return out


def _load_chartqa(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    ds = load_from_disk(str(root / "chartqa"))
    idxs = _select_random_indices(len(ds), n, seed)
    out: list[Sample] = []
    for i, ex in zip(idxs, ds.select(idxs)):
        q = (
            f"{str(ex['question']).strip()}\n"
            "Answer the question using a single word or phrase."
        )
        out.append(Sample(
            dataset="chartqa",
            sample_id=f"chartqa_{i}",
            question=q,
            image=_ensure_rgb(ex["image"]),
            prompt=_build_prompt(q, conv_template),
            answer=str(ex.get("answer", "")) if ex.get("answer") is not None else None,
        ))
    return out


def _caption_prompt(extra: str = "") -> str:
    return "Provide a one-sentence caption for the provided image." + (f" {extra}" if extra else "")


def _load_coco(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    ds = load_from_disk(str(root / "coco2017_cap_val"))
    idxs = _select_random_indices(len(ds), n, seed)
    out: list[Sample] = []
    for ex in ds.select(idxs):
        q = str(ex.get("question") or _caption_prompt()).strip()
        ans = ex.get("answer")
        first_ans = ans[0] if isinstance(ans, list) and ans else (str(ans) if ans else None)
        out.append(Sample(
            dataset="coco_cap",
            sample_id=str(ex["question_id"]),
            question=q,
            image=_ensure_rgb(ex["image"]),
            prompt=_build_prompt(q, conv_template),
            answer=str(first_ans) if first_ans is not None else None,
        ))
    return out


def _load_nocaps(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    ds = load_from_disk(str(root / "nocaps_val"))
    idxs = _select_random_indices(len(ds), n, seed)
    out: list[Sample] = []
    for ex in ds.select(idxs):
        q = _caption_prompt()
        caps = ex.get("annotations_captions") or []
        first_ans = caps[0] if caps else None
        out.append(Sample(
            dataset="nocaps",
            sample_id=str(ex["image_id"]),
            question=q,
            image=_ensure_rgb(ex["image"]),
            prompt=_build_prompt(q, conv_template),
            answer=str(first_ans) if first_ans is not None else None,
        ))
    return out


def _load_textcaps(root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    ds = load_from_disk(str(root / "textcaps_val"))
    idxs = _select_random_indices(len(ds), n, seed)
    out: list[Sample] = []
    for ex in ds.select(idxs):
        q = str(ex.get("question") or _caption_prompt()).strip()
        refs = ex.get("reference_strs") or []
        first_ans = refs[0] if refs else None
        out.append(Sample(
            dataset="textcaps",
            sample_id=str(ex["question_id"]),
            question=q,
            image=_ensure_rgb(ex["image"]),
            prompt=_build_prompt(q, conv_template),
            answer=str(first_ans) if first_ans is not None else None,
        ))
    return out


def load_samples(name: str, root: Path, conv_template: str, n: int, seed: int) -> list[Sample]:
    fns = {
        "textvqa_val": _load_textvqa_val,
        "docvqa_val": _load_docvqa_val,
        "gqa": _load_gqa,
        "chartqa": _load_chartqa,
        "coco2017_cap_val": _load_coco,
        "nocaps_val": _load_nocaps,
        "textcaps_val": _load_textcaps,
    }
    if name not in fns:
        raise ValueError(f"Unsupported dataset: {name}")
    return fns[name](root, conv_template, n, seed)


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample collection (mirrors collect_original_llava15_teacher.collect_one,
# but accepts a PIL image directly instead of a path)
# ──────────────────────────────────────────────────────────────────────────────

def _infer_image_positions(input_ids: torch.Tensor, image_feature_len: int) -> tuple[torch.Tensor, int]:
    raw_ids = input_ids[0].detach().cpu()
    placeholder_positions = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholder_positions) != 1:
        raise ValueError(f"Expected one image placeholder, found {len(placeholder_positions)}")
    image_start = int(placeholder_positions[0])
    image_positions = torch.arange(image_start, image_start + image_feature_len, dtype=torch.long)
    prompt_len_mm = int(raw_ids.numel() - 1 + image_feature_len)
    return image_positions, prompt_len_mm


def _infer_question_positions(prompt_len_mm: int, image_positions: torch.Tensor) -> torch.Tensor:
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


def _decode_generated(tokenizer, sequences: torch.Tensor, t_steps: int) -> str:
    if sequences.numel() == 0 or t_steps <= 0:
        return ""
    answer_ids = sequences[0, -t_steps:].detach().cpu().tolist()
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def _image_to_jpeg_bytes(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def collect_one(
    *,
    model,
    tokenizer,
    image_processor,
    sample: Sample,
    max_new_tokens: int,
    device: torch.device,
    image_feature_len: int,
    eps: float,
) -> dict[str, Any]:
    image = sample.image
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        sample.prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)

    image_positions, prompt_len_mm = _infer_image_positions(input_ids, image_feature_len)
    image_indices = image_positions.to(device=device)
    question_positions = _infer_question_positions(prompt_len_mm, image_positions)

    out = model.generate(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        modalities=["image"],
        do_sample=False, num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError("No attentions returned")

    n_layers = len(out.attentions[0])
    n_img = int(image_positions.numel())
    teacher_raw = torch.zeros(n_layers, n_img, dtype=torch.float32)
    for step_attns in out.attentions:
        for layer_idx, attn in enumerate(step_attns):
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
        "image_bytes": _image_to_jpeg_bytes(image),    # self-contained
        "image_path": "",                              # legacy field (empty)
        "image_token_indices": image_positions.to(torch.long),
        "question_token_indices": question_positions.to(torch.long),
        "teacher_raw": teacher_raw.to(torch.float16),
        "teacher_norm": teacher_norm.to(torch.float16),
        "prompt_len_mm": int(prompt_len_mm),
        "T": int(t_steps),
        "n_img": int(n_img),
        "max_new_tokens": int(max_new_tokens),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", default="/mnt/srv/home/dlpc.3842/zap/data/eval")
    p.add_argument("--datasets", nargs="+", default=[
        "textvqa_val", "docvqa_val", "gqa", "chartqa",
        "coco2017_cap_val", "nocaps_val", "textcaps_val",
    ])
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--output-root", required=True,
                   help="Directory; one subdir per dataset will be created.")
    p.add_argument("--model-path", default="/mnt/srv/home/dlpc.3842/zap/ckpts/llava-v1.5-7b")
    p.add_argument("--model-name", default="llava-v1.5-7b")
    p.add_argument("--conv-template", default="vicuna_v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_model(args):
    print(f"[load] {args.model_path} attn=eager", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="eager",
        multimodal=True,
    )
    model.eval()
    vt = model.get_vision_tower()
    image_feature_len = int(getattr(vt, "num_patches", 576))
    print(f"[load-ok] image_feature_len={image_feature_len}", flush=True)
    return tokenizer, model, image_processor, image_feature_len


def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_root)
    device = torch.device(args.device)

    tokenizer, model, image_processor, image_feature_len = load_model(args)

    for dataset in args.datasets:
        try:
            samples = load_samples(dataset, eval_root, args.conv_template, args.n_samples, args.seed)
        except Exception as exc:
            print(f"[skip-dataset] {dataset}: {exc}", flush=True)
            continue
        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] {dataset} samples={len(samples)} → {out_dir}", flush=True)

        t0 = time.time()
        saved = 0
        for idx, sample in enumerate(samples, 1):
            out_path = out_dir / f"{_sanitize_id(sample.sample_id)}.pt"
            if out_path.exists() and not args.overwrite:
                saved += 1
                continue
            try:
                rec = collect_one(
                    model=model, tokenizer=tokenizer, image_processor=image_processor,
                    sample=sample, max_new_tokens=args.max_new_tokens, device=device,
                    image_feature_len=image_feature_len, eps=1e-8,
                )
                torch.save(rec, out_path)
                saved += 1
                if idx == 1:
                    print(f"[sanity] {dataset} sid={sample.sample_id} "
                          f"L={tuple(rec['teacher_raw'].shape)} T={rec['T']} "
                          f"prompt_len_mm={rec['prompt_len_mm']} decoded={rec['decoded']!r}",
                          flush=True)
            except Exception as exc:
                print(f"[skip] {dataset}/{sample.sample_id}: {exc}", flush=True)

            if idx % 20 == 0 or idx == len(samples):
                rate = idx / max(time.time() - t0, 1e-6)
                print(f"[progress] {dataset} {idx}/{len(samples)} saved={saved} {rate:.2f}/s",
                      flush=True)

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
