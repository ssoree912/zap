#!/usr/bin/env python3
"""Teacher 데이터 추출: llava-onevision-qwen2-7b-ov (LLaVA format)
textvqa / gqa / scienceqa 각 n_samples개 추출.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache

DATA_ROOT   = Path("/workspace/zap/data")
REPO_ROOT   = Path("/workspace/zap")
LLAVA_ROOT  = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
TF_ROOT     = Path("/workspace/VFlowOpt/src/transformers-4.46.0/src")

for p in [REPO_ROOT, LLAVA_ROOT, TF_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


# ── patch SigLIP loader ───────────────────────────────────────────────────────
def patch_siglip():
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


# ── Sample dataclass ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Sample:
    dataset:    str
    sample_id:  str
    question:   str
    image_path: Path
    prompt:     str
    answer:     str | None = None


def _build_prompt(question: str, conv_template: str) -> str:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_textvqa(n: int, seed: int, conv_template: str) -> list[Sample]:
    data = json.loads((DATA_ROOT / "textvqa/train/data.json").read_text())
    rng  = random.Random(seed)
    rng.shuffle(data)
    samples: list[Sample] = []
    for rec in data:
        img_path = DATA_ROOT / str(rec["image_path"])
        if not img_path.exists():
            continue
        q  = f"{rec['question'].strip()}\nAnswer the question using a single word or phrase."
        ans = rec["answers"][0] if rec.get("answers") else None
        samples.append(Sample(
            dataset="textvqa",
            sample_id=str(rec["question_id"]),
            question=q,
            image_path=img_path.resolve(),
            prompt=_build_prompt(q, conv_template),
            answer=ans,
        ))
        if len(samples) >= n:
            break
    return samples


def load_gqa(n: int, seed: int, conv_template: str) -> list[Sample]:
    data   = json.loads((DATA_ROOT / "gqa/train_balanced_questions.json").read_text())
    qids   = list(data.keys())
    img_dir = DATA_ROOT / "gqa/images"
    rng    = random.Random(seed)
    rng.shuffle(qids)
    samples: list[Sample] = []
    for qid in qids:
        rec      = data[qid]
        img_path = img_dir / f"{rec['imageId']}.jpg"
        if not img_path.exists():
            continue
        q = f"{rec['question'].strip()}\nAnswer the question using a single word or phrase."
        samples.append(Sample(
            dataset="gqa",
            sample_id=qid,
            question=q,
            image_path=img_path.resolve(),
            prompt=_build_prompt(q, conv_template),
            answer=str(rec.get("answer", "")),
        ))
        if len(samples) >= n:
            break
    return samples


def load_scienceqa(n: int, seed: int, conv_template: str) -> list[Sample]:
    problems  = json.loads((DATA_ROOT / "scienceqa/problems.json").read_text())
    img_root  = DATA_ROOT / "scienceqa/images"
    # only train split with image
    qids = [
        qid for qid, rec in problems.items()
        if qid.startswith("train_") and rec.get("image")
        and (img_root / "train" / qid / rec["image"]).exists()
    ]
    rng = random.Random(seed)
    rng.shuffle(qids)
    samples: list[Sample] = []
    for qid in qids:
        rec     = problems[qid]
        img_path = img_root / "train" / qid / rec["image"]
        choices  = rec.get("choices") or []
        q = rec["question"].strip()
        if choices:
            q += "\n" + " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        ans_idx = rec.get("answer")
        ans = chr(65 + int(ans_idx)) if isinstance(ans_idx, int) else None
        samples.append(Sample(
            dataset="scienceqa",
            sample_id=qid,
            question=q,
            image_path=img_path.resolve(),
            prompt=_build_prompt(q, conv_template),
            answer=ans,
        ))
        if len(samples) >= n:
            break
    return samples


# ── Model helpers ─────────────────────────────────────────────────────────────
def _cache_seq_len(past_kv: Any) -> int:
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    if hasattr(past_kv, "key_cache") and past_kv.key_cache:
        return int(past_kv.key_cache[0].shape[-2])
    return int(past_kv[0][0].shape[-2])


def _infer_image_positions(input_ids: torch.Tensor, prompt_len_mm: int):
    raw = input_ids[0].detach().cpu()
    pos = torch.where(raw == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(pos) != 1:
        raise ValueError(f"Expected 1 image placeholder, found {len(pos)}")
    start = int(pos[0])
    feat_len = int(prompt_len_mm - raw.numel() + 1)
    if feat_len <= 0:
        raise ValueError(f"Bad image feature len={feat_len}")
    return torch.arange(start, start + feat_len, dtype=torch.long), feat_len


def _resolve_eos(tokenizer: Any, config: Any) -> int:
    eid = tokenizer.eos_token_id
    if eid is None:
        cfg = getattr(config, "eos_token_id", None)
        eid = cfg[0] if isinstance(cfg, (list, tuple)) else cfg
    return int(eid if eid is not None else 151645)


def _get_decoder_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    raise AttributeError("Cannot find decoder layers")


# ── Single-sample collection ──────────────────────────────────────────────────
@torch.no_grad()
def collect_one(*, model, tokenizer, image_processor, sample: Sample,
                max_new_tokens: int, device: torch.device, eps: float = 1e-8) -> dict:
    with Image.open(sample.image_path) as img:
        img      = img.convert("RGB")
        img_size = img.size
        img_tensor = process_images([img], image_processor, model.config)
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.to(device=device, dtype=torch.float16)
    else:
        img_tensor = [t.to(device=device, dtype=torch.float16) for t in img_tensor]

    input_ids = tokenizer_image_token(
        sample.prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # generate
    out = model.generate(
        inputs=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=img_tensor,
        image_sizes=[img_size],
        modalities=["image"],
        do_sample=False, num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    eos_id = _resolve_eos(tokenizer, model.config)
    gen_ids = out.sequences[0].tolist() if hasattr(out, "sequences") else out[0].tolist()
    gen_ids = gen_ids[-max_new_tokens:] if len(gen_ids) > max_new_tokens else gen_ids
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    if not gen_ids:
        raise RuntimeError("generate produced zero tokens")

    # prefill
    prefill = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=img_tensor,
        image_sizes=[img_size],
        modalities=["image"],
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill.past_key_values
    if isinstance(past_kv, tuple):
        past_kv = DynamicCache.from_legacy_cache(past_kv)
    prompt_len_mm  = _cache_seq_len(past_kv)
    image_positions, feat_len = _infer_image_positions(input_ids, prompt_len_mm)
    last_img = int(image_positions.max().item())
    q_positions = (
        torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)
        if last_img + 1 < prompt_len_mm else torch.empty(0, dtype=torch.long)
    )
    image_idx = image_positions.to(device)

    # decode loop: capture attention per layer
    attn_layers   = _get_decoder_layers(model)
    orig_forwards: dict[int, Any] = {}
    captured: dict[int, torch.Tensor] = {}

    def make_patched(li: int, orig):
        def patched(*args, **kwargs):
            kwargs["output_attentions"] = True
            out = orig(*args, **kwargs)
            w = out[1]
            if w is not None:
                captured[li] = (
                    w[0, :, -1, :].index_select(-1, image_idx)
                    .float().mean(0).detach().cpu()
                )
            return (out[0], None) + out[2:]
        return patched

    for li, layer in enumerate(attn_layers):
        orig_forwards[li] = layer.self_attn.forward
        layer.self_attn.forward = make_patched(li, orig_forwards[li])

    attn_mask   = torch.ones((1, prompt_len_mm), dtype=torch.long, device=device)
    teacher_raw = None
    t_steps     = 0
    try:
        for tok in gen_ids[:max_new_tokens]:
            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)], dim=1
            )
            captured.clear()
            cache_pos = torch.tensor([prompt_len_mm + t_steps], dtype=torch.long, device=device)
            step_out  = model(
                input_ids=torch.tensor([[tok]], dtype=torch.long, device=device),
                attention_mask=attn_mask,
                past_key_values=past_kv,
                cache_position=cache_pos,
                position_ids=cache_pos.unsqueeze(0),
                use_cache=True, output_attentions=False, return_dict=True,
            )
            past_kv = step_out.past_key_values
            if isinstance(past_kv, tuple):
                past_kv = DynamicCache.from_legacy_cache(past_kv)
            if teacher_raw is None:
                teacher_raw = torch.zeros(len(attn_layers), int(image_positions.numel()))
            if len(captured) != len(attn_layers):
                raise RuntimeError(f"Captured {len(captured)}/{len(attn_layers)} layers")
            for li in range(len(attn_layers)):
                teacher_raw[li] += captured[li]
            t_steps += 1
    finally:
        for li, layer in enumerate(attn_layers):
            layer.self_attn.forward = orig_forwards[li]

    if t_steps == 0 or teacher_raw is None:
        raise RuntimeError("Zero decode steps")

    teacher_raw /= float(t_steps)
    teacher_raw  = torch.nan_to_num(teacher_raw, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum      = teacher_raw.sum(-1, keepdim=True)
    bad          = (~torch.isfinite(row_sum)) | (row_sum <= eps)
    teacher_norm = teacher_raw / row_sum.clamp_min(eps)
    if bad.any():
        teacher_norm[bad.squeeze(-1)] = 1.0 / max(1, int(image_positions.numel()))
    if not torch.isfinite(teacher_norm).all():
        raise RuntimeError("teacher_norm has non-finite values")

    decoded = tokenizer.decode([int(t) for t in gen_ids], skip_special_tokens=True).strip()
    del prefill, past_kv
    torch.cuda.empty_cache()

    return {
        "sample_id":            sample.sample_id,
        "dataset":              sample.dataset,
        "model":                "llava-onevision-qwen2-7b-ov",
        "prompt_text":          sample.prompt,
        "question_text":        sample.question,
        "answer":               sample.answer,
        "decoded":              decoded,
        "image_path":           str(sample.image_path),
        "image_token_indices":  image_positions.to(torch.long),
        "question_token_indices": q_positions.to(torch.long),
        "teacher_raw":          teacher_raw.to(torch.float16),
        "teacher_norm":         teacher_norm.to(torch.float16),
        "prompt_len_mm":        int(prompt_len_mm),
        "T":                    int(t_steps),
        "n_img":                int(image_positions.numel()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",    default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    p.add_argument("--model-name",    default="llava_qwen")
    p.add_argument("--conv-template", default="qwen_1_5")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--device-map",    default="auto")
    p.add_argument("--datasets",      nargs="+", default=["textvqa", "gqa", "scienceqa"])
    p.add_argument("--n-samples",     type=int, default=300)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--output-root",   default="/workspace/zap/data/train/teacher_onevision_ov")
    p.add_argument("--overwrite",     action="store_true")
    return p.parse_args()


def _sanitize_id(v) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(v).strip())
    return s[:128] or "sample"


LOADERS = {"textvqa": load_textvqa, "gqa": load_gqa, "scienceqa": load_scienceqa}


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    patch_siglip()
    print(f"[load] {args.model_path}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="sdpa",
        multimodal=True,
    )
    model.eval()
    print(f"[load-ok] {model.__class__.__name__} layers={model.config.num_hidden_layers}", flush=True)

    device     = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        loader  = LOADERS[dataset]
        samples = loader(args.n_samples, args.seed, args.conv_template)
        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] {dataset} n={len(samples)}", flush=True)

        saved = 0; skipped = []; t_vals = []; t0 = time.time()
        for idx, sample in enumerate(samples, 1):
            out_path = out_dir / f"{_sanitize_id(sample.sample_id)}.pt"
            if out_path.exists() and not args.overwrite:
                rec = torch.load(out_path, map_location="cpu", weights_only=False)
                t_vals.append(int(rec.get("T", 0))); saved += 1; continue
            try:
                rec = collect_one(
                    model=model, tokenizer=tokenizer, image_processor=image_processor,
                    sample=sample, max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                torch.save(rec, out_path)
                saved += 1; t_vals.append(int(rec["T"]))
                if idx == 1:
                    print(f"[sanity] {dataset} sid={sample.sample_id} "
                          f"n_img={rec['n_img']} T={rec['T']} decoded={rec['decoded']!r}", flush=True)
            except Exception as exc:
                skipped.append({"sample_id": sample.sample_id, "error": repr(exc)})
                print(f"[skip] {dataset} sid={sample.sample_id}: {exc}\n{traceback.format_exc()}", flush=True)
                gc.collect(); torch.cuda.empty_cache()

            if idx % 25 == 0 or idx == len(samples):
                elapsed = time.time() - t0
                print(
                    f"[progress] {dataset} {idx}/{len(samples)} saved={saved} "
                    f"skipped={len(skipped)} rate={idx/max(elapsed,1e-6):.3f}/s "
                    f"T_mean={float(np.mean(t_vals)) if t_vals else 0:.2f}",
                    flush=True,
                )

        summary = {
            "dataset": dataset, "n_saved": saved, "n_skipped": len(skipped),
            "skipped": skipped, "elapsed": time.time() - t0,
            "t_mean": float(np.mean(t_vals)) if t_vals else 0.0,
        }
        (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[done] {dataset} {summary}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
