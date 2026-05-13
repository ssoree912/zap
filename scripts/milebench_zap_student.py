#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZAP foresight student eviction on LLaVA-1.5 + MileBench (image-only).

Image keep ratio basis. Only image tokens are evicted; text tokens are
preserved verbatim. Pipeline:
  1. Prefill the LLaVA-1.5 model (look-m fork, kv_mode=origin) with hidden
     states captured.
  2. Score image tokens layer by layer with the trained `VisualUtilityStudent`.
  3. Build per-layer keep masks; keep budget on the image side is
     `ceil(keep_ratio * n_image_tokens)`. Text positions are always kept.
  4. Trim the per-layer KV cache and greedy-decode from the trimmed cache.

Prompt + image processing are byte-identical to `milebench_unified.py`
fullcache so the keep-ratio sweep is comparable to the fullcache baseline.

Usage:
    python milebench_zap_student.py \\
        --keep_ratio 0.5 --dataset all \\
        --output_dir <dir> --device cuda:0 \\
        --student_path /mnt/srv/home/dlpc.3842/zap/artifacts/student_llava15_original_future_1800_lr1e4_15ep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


DATA_ROOT = "/mnt/srv/home/dlpc.3842/zap/data/MileBench"
LLAVA15_CKPT = "/mnt/srv/home/dlpc.3842/zap/ckpts/llava-v1.5-7b"
DEFAULT_STUDENT = "/mnt/srv/home/dlpc.3842/zap/artifacts/student_llava15_original_future_1800_lr1e4_15ep"

LLAVA_FORK = "/mnt/srv/home/dlpc.3842/look-m/LLaVA-mix_merge_v1"
LOOKM_ROOT = "/mnt/srv/home/dlpc.3842/look-m"
ZAP_ROOT = "/mnt/srv/home/dlpc.3842/zap"

DATASETS = ["ALFRED", "CLEVR-Change", "IEdit", "Spot-the-Diff"]
ALL_MILEBENCH = [
    "ALFRED", "ActionLocalization", "ActionPrediction", "ActionSequence",
    "CLEVR-Change", "CharacterOrder", "CounterfactualInference", "DocVQA",
    "EgocentricNavigation", "GPR1200", "IEdit", "ImageNeedleInAHaystack",
    "MMCoQA", "MovingAttribute", "MovingDirection", "MultiModalQA",
    "OCR-VQA", "ObjectExistence", "ObjectInteraction", "ObjectShuffle",
    "SceneTransition", "SlideVQA", "Spot-the-Diff", "StateChange",
    "TQA", "TextNeedleInAHaystack", "WebQA", "WikiVQA",
]
N_TOKENS_PER_IMAGE = 576
MAX_CONTEXT_LEN = 4096
CONV_NAME = "vicuna_v1"
MAX_NEW_TOKENS = 512  # default; overridable via --max_new_tokens


def setup_paths() -> None:
    for p in (LLAVA_FORK, LOOKM_ROOT, ZAP_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def to_int(v: Any) -> Any:
    try:
        return int(v)
    except Exception:
        return str(v)


def _patch_rotary_for_kv_pruning(model) -> None:
    """Patch LlamaRotaryEmbedding so cos/sin cache extends to position_ids range.

    The vendored v433 LlamaAttention computes kv_seq_len = past_kv_len + new_tok,
    which after KV trimming is shorter than the original prompt. Returning only
    that many cos/sin entries is fine for normal generation but breaks KV
    pruning where position_ids reference *original* positions (past prompt_len).
    Override forward to always return the full cached cos/sin so RoPE indexing
    by position_ids works regardless of the trimmed cache length.
    """
    from llava.model.kv_token_merge.v433_modeling_llama import LlamaRotaryEmbedding

    def patched_forward(self, x, seq_len=None):
        target = max(seq_len or 0, self.max_position_embeddings)
        if target > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=target, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )

    LlamaRotaryEmbedding.forward = patched_forward


def load_model(device: str):
    setup_paths()
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=LLAVA15_CKPT,
        model_base=None,
        model_name=LLAVA15_CKPT,
        device_map="cuda",
        kv_mode="origin",
        hh_ratio=0.0,
        recent_ratio=0.0,
    )
    _patch_rotary_for_kv_pruning(model)
    model.eval()
    return tokenizer, model, image_processor


def load_student(student_path: str, device: str):
    setup_paths()
    from kvpress.presses.visual_utility_student import VisualUtilityStudent

    student = VisualUtilityStudent.from_pretrained(student_path)
    student = student.to(device=device, dtype=torch.float16).eval()
    return student


def build_input(question: str, tokenizer, model):
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import tokenizer_image_token

    mm_use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))
    single_img_tok = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if mm_use_im_start_end
        else DEFAULT_IMAGE_TOKEN
    )

    input_prompt = question.replace("<ImageHere>", single_img_tok)
    input_prompt = input_prompt.replace(
        single_img_tok + single_img_tok, single_img_tok + "\n" + single_img_tok
    )

    conv = conv_templates[CONV_NAME].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    return input_ids, stop_str or ""


def get_image_positions(input_ids: torch.Tensor, n_tokens_per_image: int) -> torch.Tensor:
    """Compute image token positions in the merged (multimodal) sequence.

    LLaVA-1.5 expands each IMAGE_TOKEN_INDEX placeholder into n_tokens_per_image
    in the embedding sequence. Returns 1-D LongTensor.
    """
    from llava.constants import IMAGE_TOKEN_INDEX

    ids = input_ids[0]
    placeholders = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
    positions: list[int] = []
    offset = 0
    for ph in placeholders:
        merged_start = ph + offset
        positions.extend(range(merged_start, merged_start + n_tokens_per_image))
        offset += n_tokens_per_image - 1
    return torch.tensor(positions, dtype=torch.long)


def trim_kv(past_kv, keep_masks: dict[int, torch.Tensor]):
    """Trim per-layer KV cache, preserving the input container type.

    The look-m fork uses transformers 4.33's vendored LlamaModel which returns
    past_key_values as tuple-of-tuples; converting to DynamicCache would break
    the next forward pass.
    """
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

    # legacy tuple-of-tuples — keep the same format
    new_layers = []
    for li, (k, v) in enumerate(past_kv):
        if li in keep_masks:
            mask = keep_masks[li].to(k.device)
            k = k[:, :, mask, :].contiguous()
            v = v[:, :, mask, :].contiguous()
        new_layers.append((k, v))
    return tuple(new_layers)


@torch.inference_mode()
def greedy_decode(model, past_kv, first_tok, eos_id, max_new_tokens, prompt_len: int):
    """Greedy decode with explicit position_ids.

    After KV trimming the cache is shorter than the original prompt, so the
    LlamaModel can't infer the right RoPE position from past_key_values length
    alone. We pass position_ids = [prompt_len, prompt_len+1, ...] for each new
    token so RoPE matches what prefill saw.
    """
    out = [int(first_tok.item())]
    if out[0] == eos_id:
        return torch.tensor(out, dtype=torch.long)
    cur = first_tok
    pos = prompt_len  # next-token position in the original (pre-trim) sequence
    device = cur.device
    for _ in range(max_new_tokens - 1):
        position_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
        # Build attention_mask covering trimmed cache + new token.
        if isinstance(past_kv, tuple):
            kv_len = past_kv[0][0].shape[-2]
        else:
            kv_len = past_kv.key_cache[0].shape[-2]
        attention_mask = torch.ones(1, kv_len + 1, dtype=torch.long, device=device)
        result = model(
            input_ids=cur,
            past_key_values=past_kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past_kv = result.past_key_values
        cur = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(cur.item())
        out.append(tok)
        pos += 1
        if tok == eos_id:
            break
    return torch.tensor(out, dtype=torch.long)


@torch.inference_mode()
def generate_with_student(
    *,
    tokenizer,
    model,
    image_processor,
    student,
    question: str,
    image_paths: list[str],
    keep_ratio: float,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, dict[str, Any]]:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, KeywordsStoppingCriteria

    input_ids, stop_str = build_input(question, tokenizer, model)
    input_ids = input_ids.to(device)
    if image_paths:
        pil = [Image.open(p).convert("RGB") for p in image_paths]
        image_tensor = process_images(pil, image_processor, model.config).to(device, torch.float16)
    else:
        image_tensor = None

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Fast path: full cache (keep_ratio >= 1.0) or no images.
    if keep_ratio >= 1.0 or image_tensor is None:
        out = model.generate(
            input_ids, images=image_tensor,
            stopping_criteria=[KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)],
            pad_token_id=pad_id,
            max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=False,
            temperature=0.0, use_cache=True,
        )
        ans = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if stop_str and stop_str in ans:
            ans = ans.split(stop_str)[0].strip()
        return ans, {}

    # Prefill with hidden states.
    prefill = model(
        input_ids, images=image_tensor,
        use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True,
    )
    H_all = prefill.hidden_states  # tuple of [1, L_mm, D], len=n_layers+1
    past_kv = prefill.past_key_values
    next_tok = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prompt_len = int(prefill.logits.shape[1])
    eos_id = tokenizer.eos_token_id

    image_positions = get_image_positions(input_ids, N_TOKENS_PER_IMAGE).to(device)
    n_img = image_positions.numel()
    n_text = prompt_len - n_img

    # keep_ratio is image-token based: keep ceil(keep_ratio * n_img) image
    # tokens, while text tokens are always kept unconditionally.
    n_keep_img = min(
        n_img,
        max(1, int(torch.ceil(torch.tensor(keep_ratio * n_img)).item())),
    )

    stats = {
        "keep_ratio_basis": "image",
        "prompt_len": prompt_len,
        "n_text": n_text,
        "n_image_original": n_img,
        "n_image_kept": n_keep_img,
        "image_token_ratio": n_img / max(1, prompt_len),
        "text_token_ratio": n_text / max(1, prompt_len),
        "image_keep_ratio": n_keep_img / max(1, n_img),
        "total_keep_ratio": (n_text + n_keep_img) / max(1, prompt_len),
    }

    if n_keep_img >= n_img:
        ans_ids = greedy_decode(model, past_kv, next_tok, eos_id, max_new_tokens, prompt_len)
        ans = tokenizer.decode(ans_ids.tolist(), skip_special_tokens=True).strip()
        if stop_str and stop_str in ans:
            ans = ans.split(stop_str)[0].strip()
        return ans, stats

    last_img = int(image_positions.max().item())
    q_positions = (
        torch.arange(last_img + 1, prompt_len, dtype=torch.long, device=device)
        if last_img + 1 < prompt_len
        else torch.empty(0, dtype=torch.long, device=device)
    )

    keep_masks: dict[int, torch.Tensor] = {}
    for li in student.layer_indices:
        H_l = H_all[li + 1]  # hidden_states[0] is embedding output; layer i → index i+1
        scores = student.forward_layer(li, H_l, image_positions, q_positions).squeeze(0)  # [n_img]
        top = torch.topk(scores, k=n_keep_img, largest=True).indices
        image_keep = torch.zeros(n_img, dtype=torch.bool, device=device)
        image_keep[top] = True
        mask = torch.ones(prompt_len, dtype=torch.bool, device=device)
        mask[image_positions] = image_keep
        keep_masks[li] = mask.cpu()

    del H_all
    past_kv = trim_kv(past_kv, keep_masks)

    ans_ids = greedy_decode(model, past_kv, next_tok, eos_id, max_new_tokens, prompt_len)
    torch.cuda.empty_cache()
    ans = tokenizer.decode(ans_ids.tolist(), skip_special_tokens=True).strip()
    if stop_str and stop_str in ans:
        ans = ans.split(stop_str)[0].strip()
    return ans, stats


def run_dataset(
    *,
    dataset: str,
    tokenizer,
    model,
    image_processor,
    student,
    keep_ratio: float,
    output_dir: Path,
    device: str,
    overwrite: bool,
    limit: int | None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    combine_image: int | None = None,
) -> None:
    setup_paths()
    from utils import MileBenchDataset

    task_out = output_dir / dataset
    task_out.mkdir(parents=True, exist_ok=True)
    pred_path = task_out / "pred.json"
    if pred_path.exists() and not overwrite:
        print(f"[skip] {dataset}: pred.json exists")
        return

    data_path = Path(DATA_ROOT) / dataset / f"{dataset}.json"
    img_dir = str(Path(DATA_ROOT) / dataset / "images")
    core = json.loads(data_path.read_text())
    samples_raw = core["data"]
    if limit is not None:
        samples_raw = samples_raw[:limit]

    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in samples_raw:
        if combine_image:
            by_n[combine_image].append(s)
        else:
            by_n[len(s["task_instance"]["images_path"])].append(s)

    predictions: list[dict[str, Any]] = []
    keep_stats: list[dict[str, Any]] = []
    for n_img in sorted(by_n):
        ds = MileBenchDataset(
            annotation=by_n[n_img],
            task_instructions=core["meta_data"]["task_instruction"],
            img_dir=img_dir,
            max_context_len=MAX_CONTEXT_LEN,
            n_tokens_per_image=N_TOKENS_PER_IMAGE,
            tokenizer=tokenizer,
            dataset_name=dataset,
            combine_image=combine_image,
        )
        for idx in tqdm(range(len(ds)), desc=f"{dataset}(n={n_img})"):
            item = ds[idx]
            try:
                ans, stats = generate_with_student(
                    tokenizer=tokenizer, model=model, image_processor=image_processor,
                    student=student, question=item["context"],
                    image_paths=item["raw_img_list"], keep_ratio=keep_ratio, device=device,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc(file=sys.stderr)
                print(f"[warn] {dataset} sample {item['sample_id']} failed: {exc}", file=sys.stderr)
                ans, stats = "", {}
            if stats:
                keep_stats.append(stats)
            predictions.append(
                {
                    "sample_id": to_int(item["sample_id"]),
                    "image": item["raw_img_list"],
                    "question": item["context"],
                    "gt_response": str(item["response"]),
                    "gen_model_id": (
                        f"llava15_zap_student_keep{keep_ratio:g}"
                        + (f"_combine{combine_image}" if combine_image else "")
                    ),
                    "pred_response": ans,
                    "gen_kwargs": {
                        "max_new_tokens": max_new_tokens, "do_sample": False,
                        "temperature": 0.0, "use_cache": True,
                        "method": "zap_student", "keep_ratio": keep_ratio,
                        "combine_image": combine_image,
                    },
                }
            )

    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"[{dataset}] saved {len(predictions)} -> {pred_path}")
    if keep_stats:
        n = len(keep_stats)
        summary = {
            "dataset": dataset, "keep_ratio": keep_ratio, "n_samples": n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in keep_stats) / n,
            "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in keep_stats) / n,
        }
        (task_out / "keep_ratio_stats.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[{dataset}] avg image_keep={summary['avg_image_keep_ratio']:.4f} "
            f"avg total_keep={summary['avg_total_keep_ratio']:.4f}"
        )


def score(output_dir: Path, datasets: list[str]) -> None:
    setup_paths()
    from qvik.eval.score_milebench_predictions import score_dataset

    summary: dict[str, Any] = {}
    for ds in datasets:
        try:
            summary[ds] = score_dataset(
                data_root=DATA_ROOT, result_dir=str(output_dir), dataset=ds,
                allow_partial=False, overwrite=True,
            )
        except Exception as exc:  # noqa: BLE001
            summary[ds] = {"error": repr(exc)}
            print(f"[score-err] {ds}: {exc}", file=sys.stderr)
    (output_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_ratio", type=float, required=True)
    parser.add_argument("--dataset", default="all",
                        help="dataset name, 'all' (4-dataset subset), or 'full' (all 28 MileBench)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student_path", default=DEFAULT_STUDENT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Generation cap per sample (default 512).")
    parser.add_argument("--combine_image", type=int, default=None,
                        help="Use MileBench combined_N_images (single stitched grid). "
                             "Pass 1 for combined_1_images. Default: None (multi-image).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "all":
        datasets = DATASETS
    elif args.dataset == "full":
        datasets = ALL_MILEBENCH
    elif "," in args.dataset:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    else:
        datasets = [args.dataset]
    print(
        f"[init] keep_ratio={args.keep_ratio} keep_ratio_basis=image "
        f"max_new_tokens={args.max_new_tokens} "
        f"student={args.student_path} datasets={datasets}"
    )

    tokenizer, model, image_processor = load_model(args.device)
    student = load_student(args.student_path, args.device)
    print(f"[init] student layers={len(student.layer_indices)}")

    for ds in datasets:
        run_dataset(
            dataset=ds, tokenizer=tokenizer, model=model, image_processor=image_processor,
            student=student, keep_ratio=args.keep_ratio, output_dir=output_dir,
            device=args.device, overwrite=args.overwrite, limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            combine_image=args.combine_image,
        )

    if not args.no_score:
        score(output_dir, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
