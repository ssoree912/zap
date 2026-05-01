#!/usr/bin/env python3
"""Stage 1 teacher cache for LLaVA-1.5-7B.

Runs greedy generate with output_attentions=True, averages attention across
heads / decode steps at each image position, normalizes to a per-layer
distribution, and saves the [L, N_I] teacher tensor.

LLaVA-1.5 prompts are short (~600-700 tokens with 576 image tokens) so the
standard model.generate(..., output_attentions=True) approach is memory-safe
on 1 GPU — decode-step attentions are [B, H, 1, T_kv] with KV cache.

Output layout: ``<output_root>/<dataset>/<sample_id>.pt`` with the same
schema as the OneVision collector (teacher_raw, teacher_norm,
image_token_indices, question_token_indices, prompt_len_mm, T, n_img,
trajectory_m).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, "/workspace/zap")
from ..llava_15b_extractor import (
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)

_LLAVA15_PROMPT = "USER: <image>\n{question}\nASSISTANT:"


def _fmt(question: str) -> str:
    return _LLAVA15_PROMPT.format(question=question)


# ── dataset loaders ─────────────────────────────────────────────────────────
# All return list of (sample_id, prompt_str, image_path_str).


def _resolve_image_path(img_path: str) -> str | None:
    """Resolve image path, trying train/ subdirectory as fallback."""
    p = Path(img_path)
    if p.exists():
        return str(p)
    # samples.json may have been created before moving data under train/
    alt = Path(str(img_path).replace("/workspace/zap/data/", "/workspace/zap/data/train/", 1))
    if alt.exists():
        return str(alt)
    return None


def load_samples_from_json(
    samples_json: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    """Generic loader for pre-sampled datasets saved as samples.json.

    JSON format: list of {sample_id, image_path, question, [answer]}.
    """
    with samples_json.open() as f:
        records = json.load(f)
    candidates: list[tuple[str, str, str]] = []
    for rec in records:
        sid = str(rec["sample_id"])
        question = str(rec["question"]).strip()
        resolved = _resolve_image_path(rec["image_path"])
        if resolved is None:
            continue
        candidates.append((sid, _fmt(question), resolved))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_scienceqa_samples(
    problems_json: Path, images_root: Path, split: str, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    with problems_json.open() as f:
        problems = json.load(f)
    candidates: list[tuple[str, str, str]] = []
    for qid, prob in problems.items():
        if not qid.startswith(f"{split}_"):
            continue
        img_path = images_root / split / qid / "image.png"
        if not img_path.exists():
            continue
        question = prob.get("question", "").strip()
        choices = prob.get("choices", [])
        if choices:
            lettered = " ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
            question = f"{question}\n{lettered}"
        candidates.append((qid, _fmt(question), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_gqa_samples(
    questions_json: Path, images_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    with questions_json.open() as f:
        questions = json.load(f)
    candidates: list[tuple[str, str, str]] = []
    for qid, rec in questions.items():
        image_id = rec.get("imageId") or rec.get("image_id")
        if not image_id:
            continue
        img_path = images_root / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        candidates.append((str(qid), _fmt(f"Question: {question}\nAnswer the question briefly."), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_st_vqa_samples(
    data_json: Path, images_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    with data_json.open() as f:
        payload = json.load(f)
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    candidates: list[tuple[str, str, str]] = []
    for i, rec in enumerate(records):
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        rel = rec.get("file_path") or rec.get("file_name")
        if not rel:
            continue
        img_path = images_root / rel
        if not img_path.exists():
            alt = images_root / Path(rel).name
            if alt.exists():
                img_path = alt
            else:
                continue
        qid = str(rec.get("question_id", rec.get("id", f"st_vqa_{i:06d}")))
        candidates.append((qid, _fmt(f"Question: {question}\nAnswer the question briefly."), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_textvqa_samples(
    data_json: Path, data_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    with data_json.open() as f:
        payload = json.load(f)
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    candidates: list[tuple[str, str, str]] = []
    for rec in records:
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        rel = rec.get("image_path", "")
        img_path = data_root / rel if rel and not Path(rel).is_absolute() else Path(rel)
        if not img_path.exists():
            continue
        qid = str(rec.get("question_id", rec.get("id", f"tvqa_{len(candidates):06d}")))
        candidates.append((qid, _fmt(f"Question: {question}\nAnswer the question briefly."), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


# ── question-position helper ─────────────────────────────────────────────────


def infer_question_positions(
    prompt_len_mm: int,
    image_positions: torch.Tensor,
) -> torch.Tensor:
    """Return question positions: every prompt token after the last image token."""
    if image_positions.numel() == 0:
        return torch.arange(prompt_len_mm, dtype=torch.long)
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


# ── teacher collection ────────────────────────────────────────────────────────


def _collect_single_trajectory(
    model,
    inputs: dict,
    image_indices: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Run one generate pass and return (teacher [L, N_I] fp32 cpu, T)."""
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=0,
        num_beams=1,
        output_attentions=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    L = len(out.attentions[0])
    n_img = int(image_indices.numel())
    teacher = torch.zeros(L, n_img, dtype=torch.float32)
    T = len(out.attentions)

    for step_attns in out.attentions:
        for l in range(L):
            a = step_attns[l][0, :, -1, :].index_select(dim=-1, index=image_indices)
            teacher[l] += a.float().mean(dim=0).cpu()
    teacher /= float(T)

    del out
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return teacher, T


@torch.no_grad()
def collect_one(
    model,
    processor,
    image,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    trajectory_m: int = 1,
    trajectory_temperature: float = 0.7,
    trajectory_top_p: float = 0.9,
    eps: float = 1e-8,
) -> dict:
    n_images = len(image) if isinstance(image, (list, tuple)) else 1
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
        prompt_inputs=inputs,
        model_config=model.config,
        num_images=n_images,
    )
    image_indices = image_positions.to(device)
    n_img = int(image_indices.numel())
    question_positions = infer_question_positions(prompt_len_mm, image_positions)

    M = max(1, trajectory_m)
    use_sampling = M > 1

    traj_scores: list[torch.Tensor] = []
    t_lengths: list[int] = []
    for _ in range(M):
        score, T = _collect_single_trajectory(
            model=model,
            inputs=inputs,
            image_indices=image_indices,
            max_new_tokens=max_new_tokens,
            do_sample=use_sampling,
            temperature=trajectory_temperature,
            top_p=trajectory_top_p,
            eps=eps,
        )
        traj_scores.append(score)
        t_lengths.append(T)

    stacked = torch.stack(traj_scores, dim=0)
    teacher = stacked.mean(dim=0)
    teacher_norm = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)

    result = dict(
        teacher_raw=teacher.to(torch.float16),
        teacher_norm=teacher_norm.to(torch.float16),
        image_token_indices=image_positions.to(torch.long),
        question_token_indices=question_positions.to(torch.long),
        prompt_len_mm=int(prompt_len_mm),
        T=int(np.mean(t_lengths)),
        n_img=int(n_img),
        trajectory_m=M,
    )
    if M > 1:
        result["teacher_var"] = stacked.var(dim=0).to(torch.float16)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["scienceqa", "gqa", "textvqa", "st_vqa", "chartqa", "docvqa", "infovqa", "llava_instruct", "mmvet"],
    )
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default="/workspace/zap/data/train/teacher_llava15")
    # scienceqa (raw)
    p.add_argument("--problems-json", default="/workspace/zap/data/scienceqa/problems.json")
    p.add_argument("--images-root", default="/workspace/zap/data/scienceqa/images")
    p.add_argument("--split", default="train")
    # gqa (raw)
    p.add_argument("--gqa-questions-json", default="/workspace/zap/data/gqa/val_balanced_questions.json")
    p.add_argument("--gqa-images-root", default="/workspace/zap/data/gqa/images")
    # st_vqa (raw)
    p.add_argument("--st-vqa-json", default="/workspace/zap/data/st_vqa/train_task_3.json")
    p.add_argument("--st-vqa-images-root", default="/workspace/zap/data/st_vqa")
    # pre-built samples.json paths (matching onevision layout)
    p.add_argument("--chartqa-samples-json", default="/workspace/zap/data/chartqa_train_sample/samples.json")
    p.add_argument("--docvqa-samples-json", default="/workspace/zap/data/docvqa_train_sample/samples.json")
    p.add_argument("--infovqa-samples-json", default="/workspace/zap/data/infovqa_train_sample/samples.json")
    p.add_argument("--llava-instruct-samples-json", default="/workspace/zap/data/train/llava_instruct_sample/samples.json")
    p.add_argument("--mmvet-samples-json", default="/workspace/zap/data/train/mmvet_sample/samples.json")
    p.add_argument("--textvqa-json", default="/workspace/zap/data/textvqa/train/data.json")
    p.add_argument("--textvqa-data-root", default="/workspace/zap/data")
    # trajectory options
    p.add_argument("--trajectory-m", type=int, default=1)
    p.add_argument("--trajectory-temperature", type=float, default=0.7)
    p.add_argument("--trajectory-top-p", type=float, default=0.9)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[load] {args.model} dtype=bf16 attn=eager device={device}", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model)
    processor = configure_llava_processor(processor, model.config)
    print(f"[info] num_hidden_layers={model.config.text_config.num_hidden_layers}", flush=True)

    if args.dataset == "scienceqa":
        samples = load_scienceqa_samples(
            problems_json=Path(args.problems_json),
            images_root=Path(args.images_root),
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "gqa":
        samples = load_gqa_samples(
            questions_json=Path(args.gqa_questions_json),
            images_root=Path(args.gqa_images_root),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "textvqa":
        samples = load_textvqa_samples(
            data_json=Path(args.textvqa_json),
            data_root=Path(args.textvqa_data_root),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "st_vqa":
        samples = load_st_vqa_samples(
            data_json=Path(args.st_vqa_json),
            images_root=Path(args.st_vqa_images_root),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "chartqa":
        samples = load_samples_from_json(Path(args.chartqa_samples_json), args.n_samples, args.seed)
    elif args.dataset == "docvqa":
        samples = load_samples_from_json(Path(args.docvqa_samples_json), args.n_samples, args.seed)
    elif args.dataset == "infovqa":
        samples = load_samples_from_json(Path(args.infovqa_samples_json), args.n_samples, args.seed)
    elif args.dataset == "llava_instruct":
        samples = load_samples_from_json(Path(args.llava_instruct_samples_json), args.n_samples, args.seed)
    else:  # mmvet
        samples = load_samples_from_json(Path(args.mmvet_samples_json), args.n_samples, args.seed)
    print(f"[info] dataset={args.dataset} loaded {len(samples)} samples", flush=True)

    saved = 0
    skipped: list[tuple[str, str]] = []
    t_list: list[int] = []
    t0 = time.time()

    for idx, (sid, prompt, img_path) in enumerate(samples):
        safe_sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid))[:128]
        out_path = out_dir / f"{safe_sid}.pt"
        if out_path.exists():
            saved += 1
            continue
        try:
            if isinstance(img_path, (list, tuple)):
                image = [Image.open(p).convert("RGB") for p in img_path]
                stored_path = list(img_path)
            else:
                image = Image.open(img_path).convert("RGB")
                stored_path = img_path
            rec = collect_one(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                trajectory_m=args.trajectory_m,
                trajectory_temperature=args.trajectory_temperature,
                trajectory_top_p=args.trajectory_top_p,
            )
            rec.update(
                sample_id=sid,
                dataset=args.dataset,
                model="llava-1.5-7b-hf",
                prompt_text=prompt,
                image_path=stored_path,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            torch.save(rec, out_path)
            saved += 1
            t_list.append(rec["T"])
            if idx == 0:
                print(
                    f"[sanity] sid={sid} L={rec['teacher_raw'].shape[0]} "
                    f"N_I={rec['teacher_raw'].shape[1]} T={rec['T']} "
                    f"trajectory_m={rec['trajectory_m']} "
                    f"prompt_len_mm={rec['prompt_len_mm']} "
                    f"|img|={rec['image_token_indices'].numel()} "
                    f"|q|={rec['question_token_indices'].numel()}",
                    flush=True,
                )
        except (RuntimeError, ValueError, IOError) as e:
            skipped.append((sid, repr(e)))
            print(f"[skip] {sid}: {e}", flush=True)
            continue

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"[progress] {idx+1}/{len(samples)} | rate={(idx+1)/max(elapsed,1e-6):.2f}/s "
                f"| elapsed={elapsed:.1f}s | T_mean={np.mean(t_list):.2f} "
                f"| saved={saved} skipped={len(skipped)}",
                flush=True,
            )
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"[done] dataset={args.dataset} saved={saved}/{len(samples)} "
        f"skipped={len(skipped)} elapsed={elapsed:.1f}s "
        f"T_mean={np.mean(t_list) if t_list else 0:.2f}",
        flush=True,
    )

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(
            dict(
                dataset=args.dataset,
                n_requested=args.n_samples,
                n_saved=saved,
                n_skipped=len(skipped),
                skipped=skipped,
                t_mean=float(np.mean(t_list)) if t_list else 0.0,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                elapsed_seconds=elapsed,
                trajectory_m=args.trajectory_m,
                trajectory_temperature=args.trajectory_temperature,
                trajectory_top_p=args.trajectory_top_p,
            ),
            indent=2,
        )
    )
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
