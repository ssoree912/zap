#!/usr/bin/env python3
"""Collect per-sample prefill + future decode attention for mismatch analysis.

For each sample:
  Pass 1 (cheap): model forward on prompt only, output_attentions=True
      → prefill[l] = mean over heads of A[h, question_idx, image_idx]   [L, N_I]
  Pass 2 (expensive): model.generate with output_attentions=True
      → future[l]  = mean over heads & decode steps of A_decode[h, t, image_idx] [L, N_I]

Output: artifacts/EXP-20260426-001-figure/attn_dump/{dataset}/{sample_id}.pt
  {
    'prefill':          [L, N_I]  float16   (question→image, prefill)
    'future':           [L, N_I]  float16   (decode→image,  time-averaged)
    'image_indices':    [N_I]     long
    'question_indices': [N_Q]     long
    'image_path':       str
    'question':         str
    'answer':           str
    'sample_id':        str
  }

Usage examples:
  # mm-vet, GPU 0, 50 single-image samples
  python collect_attn_dump.py --dataset mmvet --n-samples 50 --device cuda:0

  # MileBench DocVQA, GPU 1, 20 samples
  python collect_attn_dump.py --dataset DocVQA --n-samples 20 --device cuda:1
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, "/workspace/zap")
from foresight.llava_extractor import infer_llava_image_positions_no_forward

MILEBENCH_SINGLE_IMAGE_DATASETS = [
    "DocVQA", "OCR-VQA", "SlideVQA", "MultiModalQA",
    "TQA", "WikiVQA", "WebQA", "GPR1200",
    "TextNeedleInAHaystack", "ImageNeedleInAHaystack", "MMCoQA",
]
MMVET_PRIORITY_CAPS = {"ocr", "math", "spat"}

MODEL_PATH = "/workspace/zap/ckpts/llava-1.5-7b-hf"
MILEBENCH_ROOT = "/workspace/zap/data/MileBench"
MMVET_JSON = "/workspace/data/mm-vet/mm-vet.json"
MMVET_IMG_ROOT = "/workspace/data/mm-vet"
OUT_ROOT = "/workspace/zap/artifacts/EXP-20260426-001-figure/attn_dump"


# ── dataset loaders ────────────────────────────────────────────────────────────

def load_milebench_samples(dataset_name: str, n: int) -> list[dict]:
    """Return list of {sample_id, question, image_path, answer}."""
    data_root = Path(MILEBENCH_ROOT) / dataset_name
    json_path = data_root / f"{dataset_name}.json"
    img_root = data_root / "images"
    raw = json.loads(json_path.read_text())
    records = raw["data"]

    samples = []
    for rec in records:
        task = rec.get("task_instance", {})
        # prefer combined_1_images (key image), fall back to single-image samples
        combined = task.get("combined_1_images", [])
        imgs = task.get("images_path", [])
        img_file = combined[0] if combined else (imgs[0] if len(imgs) == 1 else None)
        if img_file is None:
            continue
        img_path = img_root / img_file
        if not img_path.exists():
            continue
        context = task.get("context", "")
        question = re.sub(r"\{image#\d+\}", "", context).strip()
        answer = str(rec.get("response", ""))
        samples.append({
            "sample_id": str(rec["sample_id"]),
            "question": question,
            "image_path": str(img_path),
            "answer": answer,
        })
        if len(samples) >= n:
            break
    return samples


def load_mmvet_samples(n: int) -> list[dict]:
    """Return list of {sample_id, question, image_path, answer}, priority cap tags first."""
    records = json.loads(Path(MMVET_JSON).read_text())
    img_root = Path(MMVET_IMG_ROOT)

    priority, rest = [], []
    for rec in records:
        caps = set(rec.get("capability", []))
        img_path = img_root / rec["image"]
        if not img_path.exists():
            continue
        entry = {
            "sample_id": rec["id"],
            "question": rec["question"],
            "image_path": str(img_path),
            "answer": rec.get("answer", ""),
        }
        if caps & MMVET_PRIORITY_CAPS:
            priority.append(entry)
        else:
            rest.append(entry)

    combined = priority + rest
    return combined[:n]


# ── attention collection ───────────────────────────────────────────────────────

def _build_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


@torch.no_grad()
def collect_prefill_attn(
    model: LlavaForConditionalGeneration,
    inputs: dict,
    image_indices: torch.Tensor,
    question_indices: torch.Tensor,
) -> torch.Tensor:
    """Single forward pass → question→image attention [L, N_I] float32 cpu."""
    out = model(**inputs, output_attentions=True, use_cache=False)
    # out.attentions: tuple of L tensors, each [1, H, N, N]
    L = len(out.attentions)
    N_I = image_indices.numel()
    prefill = torch.zeros(L, N_I, dtype=torch.float32)
    img_idx = image_indices.cpu().long()
    # all non-image positions = role + instruction + question tokens
    N = out.attentions[0].shape[-1]
    all_pos = torch.arange(N)
    img_set = set(img_idx.tolist())
    text_idx = all_pos[[i for i in range(N) if i not in img_set]]

    for l, attn_l in enumerate(out.attentions):
        # [1, H, N, N] — text-to-image: all text positions → image positions
        a = attn_l[0, :, :, :].float().cpu()  # [H, N, N]
        if text_idx.numel() > 0:
            a_ti = a[:, text_idx, :][:, :, img_idx]  # [H, N_T, N_I]
            prefill[l] = a_ti.mean(dim=(0, 1))
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return prefill


@torch.no_grad()
def collect_future_attn(
    model: LlavaForConditionalGeneration,
    processor,
    inputs: dict,
    image_indices: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, str]:
    """generate (greedy) with output_attentions → (decode→image [L,N_I], generated_text)."""
    img_idx = image_indices.to(device=next(model.parameters()).device, dtype=torch.long)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        output_attentions=True,
        return_dict_in_generate=True,
        use_cache=True,
    )
    # decode generated tokens (strip prompt)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0, prompt_len:]
    generated_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    L = len(out.attentions[0])
    N_I = image_indices.numel()
    T = len(out.attentions)
    future = torch.zeros(L, N_I, dtype=torch.float32)
    for step_attns in out.attentions:
        for l in range(L):
            a = step_attns[l][0, :, -1, :].index_select(-1, img_idx)  # [H, N_I]
            future[l] += a.float().mean(dim=0).cpu()
    future /= float(T)
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return future, generated_text


@torch.no_grad()
def collect_one(
    model: LlavaForConditionalGeneration,
    processor,
    sample: dict,
    device: torch.device,
    max_new_tokens: int,
) -> dict | None:
    try:
        image = Image.open(sample["image_path"]).convert("RGB")
        prompt = _build_prompt(sample["question"])
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
            prompt_inputs=inputs,
            model_config=model.config,
            num_images=1,
        )
        image_indices = image_positions.to(device)
        # question = everything after last image token
        last_img = int(image_positions.max().item())
        if last_img + 1 < prompt_len_mm:
            question_indices = torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long).to(device)
        else:
            question_indices = torch.empty(0, dtype=torch.long, device=device)

        prefill = collect_prefill_attn(model, inputs, image_indices, question_indices)
        future, generated_answer = collect_future_attn(
            model, processor, inputs, image_indices, max_new_tokens
        )

        return {
            "prefill": prefill.to(torch.float16),
            "future": future.to(torch.float16),
            "image_indices": image_positions.cpu().to(torch.long),
            "question_indices": question_indices.cpu().to(torch.long),
            "image_path": sample["image_path"],
            "question": sample["question"],
            "gt_answer": sample["answer"],
            "generated_answer": generated_answer,  # full-cache model output
            "sample_id": sample["sample_id"],
        }
    except Exception as e:
        print(f"[WARN] sample {sample['sample_id']} failed: {e}")
        return None


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help=f"'mmvet' or one of {MILEBENCH_SINGLE_IMAGE_DATASETS}")
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--out-root", default=OUT_ROOT)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] model {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=str(device),
        attn_implementation="eager",  # sdpa does not return attention weights
    )
    model.eval()

    if args.dataset.lower() == "mmvet":
        samples = load_mmvet_samples(args.n_samples)
    elif args.dataset in MILEBENCH_SINGLE_IMAGE_DATASETS:
        samples = load_milebench_samples(args.dataset, args.n_samples)
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: mmvet, {MILEBENCH_SINGLE_IMAGE_DATASETS}")

    print(f"[collect] {len(samples)} samples from '{args.dataset}' → {out_dir}")
    ok, skip = 0, 0
    for sample in tqdm(samples):
        out_path = out_dir / f"{sample['sample_id']}.pt"
        if out_path.exists():
            skip += 1
            continue
        result = collect_one(model, processor, sample, device, args.max_new_tokens)
        if result is not None:
            torch.save(result, out_path)
            ok += 1
        else:
            skip += 1

    print(f"[done] saved={ok}  skipped/failed={skip}  dir={out_dir}")


if __name__ == "__main__":
    main()
