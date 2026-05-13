#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZAP foresight student eviction on LLaVA-OneVision + MileBench (image-only).

OneVision counterpart of `milebench_zap_student.py`. Uses the LLaVA-OneVision
repo's LlavaQwenForCausalLM (LMMs-Lab format checkpoint, NOT HF format).
Pipeline:
  1. Run multimodal preprocessing (`prepare_inputs_labels_for_multimodal`) to
     get inputs_embeds and the merged sequence length plus image-token spans.
  2. Prefill with hidden states; capture per-layer hidden_states.
  3. Score image tokens with `VisualUtilityStudentOneVision`; build per-layer
     keep masks (image-only). The keep budget is
     `ceil(keep_ratio * n_image_tokens)`, and text positions are always kept.
  4. Trim per-layer KV cache and greedy-decode from the trimmed cache, passing
     position_ids based on the original (pre-trim) sequence length.

Prompt + image processing match `milebench_unified.py` fullcache for OneVision
(multi-image goes through `video` modality with square-padded frames).

Usage:
    python milebench_zap_student_onevision.py \\
        --keep_ratio 0.5 --dataset all \\
        --output_dir <dir> --device cuda:0 \\
        --student_path /workspace/zap/ckpts/student_onevision_A_ep20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


DATA_ROOT = "/workspace/zap/data/MileBench"
ONEVISION_CKPT = "/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"
DEFAULT_STUDENT = "/workspace/zap/ckpts/student_onevision_A_ep20"

ONEVISION_REPO = "/workspace/VFlowOpt/src/LLaVA-OneVision"
LOOKM_ROOT = "/workspace/look-m"
ZAP_ROOT = "/workspace/zap"

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
N_TOKENS_PER_IMAGE = 200  # OneVision pooled tokens (matches fullcache)
MAX_CONTEXT_LEN = 4096
CONV_NAME = "qwen_1_5"
MAX_NEW_TOKENS = 64


def setup_paths() -> None:
    for p in (ONEVISION_REPO, LOOKM_ROOT, ZAP_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def to_int(v: Any) -> Any:
    try:
        return int(v)
    except Exception:
        return str(v)


def load_model(device: str):
    setup_paths()
    from llava.model.builder import load_pretrained_model

    overwrite_config = {
        "image_aspect_ratio": "anyres_max_9",
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
    }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=ONEVISION_CKPT,
        model_base=None,
        model_name="llava_qwen",
        device_map=device,
        attn_implementation="sdpa",
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()
    return tokenizer, model, image_processor


def load_student(student_path: str, device: str):
    setup_paths()
    from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision

    student = VisualUtilityStudentOneVision.from_pretrained(student_path)
    student = student.to(device=device, dtype=torch.float16).eval()
    return student


def collapse_image_placeholders(question: str, n_images: int) -> str:
    placeholder = "<ImageHere>"
    count = question.count(placeholder)
    if count <= 1 or n_images <= 1:
        return question
    out = question
    for _ in range(count - 1):
        out = out.replace(placeholder, "", 1)
    return out


def build_input(question: str, tokenizer, model):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import tokenizer_image_token

    input_prompt = question.replace("<ImageHere>", DEFAULT_IMAGE_TOKEN)

    conv = copy.deepcopy(conv_templates[CONV_NAME])
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    return input_ids, stop_str or ""


def process_visuals(image_paths, image_processor, model, device):
    if not image_paths:
        return None, None, None
    from llava.mm_utils import process_images

    pil = [Image.open(p).convert("RGB") for p in image_paths]
    if len(pil) > 1:
        bg = tuple(int(x * 255) for x in image_processor.image_mean)
        squared = []
        for im in pil:
            w, h = im.size
            if w != h:
                side = max(w, h)
                square = Image.new(im.mode, (side, side), bg)
                square.paste(im, ((side - w) // 2, (side - h) // 2))
                im = square
            t = image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
            squared.append(t)
        return [torch.stack(squared, dim=0).to(device=device, dtype=torch.float16)], [pil[0].size], ["video"]

    image_tensor = process_images(pil, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)
    return image_tensor, [im.size for im in pil], ["image"] * len(pil)


def prepare_multimodal_inputs(*, model, tokenizer, image_processor, question, image_paths, device):
    """Run OneVision's multimodal merging to get inputs_embeds + image span info."""
    from llava.constants import IMAGE_TOKEN_INDEX

    n = len(image_paths)
    question_for_prompt = collapse_image_placeholders(question, n) if n > 1 else question
    input_ids, stop_str = build_input(question_for_prompt, tokenizer, model)
    input_ids = input_ids.to(device)

    image_tensor, image_sizes, modalities = process_visuals(image_paths, image_processor, model, device)

    if image_tensor is None:
        inputs_embeds = model.get_model().embed_tokens(input_ids)
        prompt_len = int(inputs_embeds.shape[1])
        image_indices = torch.empty(0, dtype=torch.long, device=device)
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        attention_mask = None
    else:
        # OneVision: prepare_inputs_labels_for_multimodal merges image features into the embed sequence.
        with torch.no_grad():
            prepared = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor,
                modalities=modalities, image_sizes=image_sizes,
            )
        # Returns (input_ids, position_ids, attention_mask, past_kv, inputs_embeds, labels)
        # In some forks it returns 8 elements with extras; last entries are ignored here.
        if len(prepared) == 8:
            _, position_ids, attention_mask, _, inputs_embeds, _, _, _ = prepared
        else:
            _, position_ids, attention_mask, _, inputs_embeds, _ = prepared
        prompt_len = int(inputs_embeds.shape[1])
        if position_ids is None:
            position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

        # Image positions: indices in the merged sequence that are image tokens.
        # The non-image input_ids count tells us how many text positions there are; the
        # remaining slots (in order, where placeholders sat) are images.
        n_text = int((input_ids != IMAGE_TOKEN_INDEX).sum().item())
        n_img_total = max(0, prompt_len - n_text)
        # Find placeholder positions in input_ids and compute merged image spans.
        ph_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
        n_placeholders = max(1, len(ph_positions))
        per_ph = n_img_total // n_placeholders if n_placeholders > 0 else 0
        image_indices_list: list[int] = []
        offset = 0
        for ph in ph_positions:
            merged_start = ph + offset
            image_indices_list.extend(range(merged_start, merged_start + per_ph))
            offset += per_ph - 1  # placeholder was 1 token → now per_ph
        image_indices = torch.tensor(image_indices_list, dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "prompt_len": prompt_len,
        "image_indices": image_indices,
        "stop_str": stop_str,
    }


def trim_kv(past_kv, keep_masks):
    """Trim per-layer KV cache, preserving the input container type."""
    if hasattr(past_kv, "key_cache"):
        for li in range(len(past_kv.key_cache)):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(past_kv.key_cache[li].device)
            past_kv.key_cache[li] = past_kv.key_cache[li][:, :, mask, :].contiguous()
            past_kv.value_cache[li] = past_kv.value_cache[li][:, :, mask, :].contiguous()
        # Normalize all layers to min length so the next forward's causal_mask matches.
        if past_kv.key_cache:
            min_len = min(k.shape[-2] for k in past_kv.key_cache)
            for li in range(len(past_kv.key_cache)):
                if past_kv.key_cache[li].shape[-2] > min_len:
                    past_kv.key_cache[li] = past_kv.key_cache[li][:, :, :min_len, :].contiguous()
                    past_kv.value_cache[li] = past_kv.value_cache[li][:, :, :min_len, :].contiguous()
        return past_kv

    new_layers = []
    for li, (k, v) in enumerate(past_kv):
        if li in keep_masks:
            mask = keep_masks[li].to(k.device)
            k = k[:, :, mask, :].contiguous()
            v = v[:, :, mask, :].contiguous()
        new_layers.append((k, v))
    return tuple(new_layers)


@torch.inference_mode()
def greedy_decode(model, past_kv, first_tok, eos_ids: set[int], max_new_tokens: int, prompt_len: int):
    out = [int(first_tok.item())]
    if out[0] in eos_ids:
        return torch.tensor(out, dtype=torch.long)
    cur = first_tok
    pos = prompt_len
    device = cur.device
    for _ in range(max_new_tokens - 1):
        position_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
        result = model(
            input_ids=cur,
            past_key_values=past_kv,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=True,
            return_dict=True,
        )
        past_kv = result.past_key_values
        cur = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(cur.item())
        out.append(tok)
        pos += 1
        if tok in eos_ids:
            break
    return torch.tensor(out, dtype=torch.long)


def resolve_eos_ids(tokenizer, model_config) -> set[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    cfg_eos = getattr(model_config, "eos_token_id", None)
    if isinstance(cfg_eos, (list, tuple)):
        ids.extend(int(x) for x in cfg_eos if x is not None)
    elif cfg_eos is not None:
        ids.append(int(cfg_eos))
    if not ids:
        ids = [151645]  # Qwen2 <|im_end|>
    return set(ids)


@torch.inference_mode()
def generate_with_student(
    *, tokenizer, model, image_processor, student,
    question, image_paths, keep_ratio, device,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    prepared = prepare_multimodal_inputs(
        model=model, tokenizer=tokenizer, image_processor=image_processor,
        question=question, image_paths=image_paths, device=device,
    )
    inputs_embeds = prepared["inputs_embeds"]
    position_ids = prepared["position_ids"]
    attention_mask = prepared["attention_mask"]
    prompt_len = prepared["prompt_len"]
    image_indices = prepared["image_indices"]
    stop_str = prepared["stop_str"]

    eos_ids = resolve_eos_ids(tokenizer, model.config)

    # Fast path: full cache (or no images).
    if keep_ratio >= 1.0 or image_indices.numel() == 0:
        out = model.generate(
            prepared["input_ids"], images=None,  # already merged via prepare_inputs_labels
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=False,
            temperature=0.0, use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        ans = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        if stop_str and stop_str in ans:
            ans = ans.split(stop_str)[0].strip()
        return ans, {}

    # Prefill with hidden states using inputs_embeds (post-merge).
    prefill = model.model(  # underlying Qwen2Model to skip the image-merge wrapper
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    H_all = prefill.hidden_states
    past_kv = prefill.past_key_values
    last_logits = model.lm_head(prefill.last_hidden_state[:, -1:, :])
    next_tok = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)

    n_img = image_indices.numel()
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
        ans_ids = greedy_decode(model, past_kv, next_tok, eos_ids, max_new_tokens, prompt_len)
        ans = tokenizer.decode(ans_ids.tolist(), skip_special_tokens=True).strip()
        if stop_str and stop_str in ans:
            ans = ans.split(stop_str)[0].strip()
        return ans, stats

    last_img = int(image_indices.max().item())
    q_positions = (
        torch.arange(last_img + 1, prompt_len, dtype=torch.long, device=device)
        if last_img + 1 < prompt_len
        else torch.empty(0, dtype=torch.long, device=device)
    )

    keep_masks: dict[int, torch.Tensor] = {}
    for li in student.layer_indices:
        H_l = H_all[li + 1]  # hidden_states[0] is the embedding output
        scores = student.forward_layer(li, H_l, image_indices, q_positions).squeeze(0)
        top = torch.topk(scores, k=n_keep_img, largest=True).indices
        image_keep = torch.zeros(n_img, dtype=torch.bool, device=device)
        image_keep[top] = True
        mask = torch.ones(prompt_len, dtype=torch.bool, device=device)
        mask[image_indices] = image_keep
        keep_masks[li] = mask.cpu()

    del H_all
    past_kv = trim_kv(past_kv, keep_masks)

    ans_ids = greedy_decode(model, past_kv, next_tok, eos_ids, max_new_tokens, prompt_len)
    torch.cuda.empty_cache()
    ans = tokenizer.decode(ans_ids.tolist(), skip_special_tokens=True).strip()
    if stop_str and stop_str in ans:
        ans = ans.split(stop_str)[0].strip()
    return ans, stats


def run_dataset(
    *, dataset, tokenizer, model, image_processor, student,
    keep_ratio, output_dir, device, overwrite, limit,
    max_new_tokens: int = MAX_NEW_TOKENS,
    combine_image: int | None = None,
):
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
                        f"onevision_zap_student_keep{keep_ratio:g}"
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
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student_path", default=DEFAULT_STUDENT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Generation cap per sample (default 64).")
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
