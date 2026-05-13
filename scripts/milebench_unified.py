#!/usr/bin/env python3
"""Unified MileBench full-cache eval for LLaVA-1.5 and LLaVA-OneVision.

This file is the canonical evaluator. Identical copies live in:
  - zap/scripts/milebench_unified.py
  - look-m/scripts/milebench_unified.py
  - VFlowOpt_llava1.5/scripts/milebench_unified.py

For full-cache mode, the same code path is exercised regardless of which folder
launches the script, so pred.json files must be byte-identical across folders.
KV-pruning experiments stay in their respective folders and import / mirror the
same prompt construction so their predictions remain comparable to the
full-cache baseline.

Inference backend: VFlowOpt_llava1.5/src/LLaVA-OneVision (handles both LLaVA-1.5
via LlavaLlamaForCausalLM and LLaVA-OneVision via LlavaQwenForCausalLM).

Prompt truncation: look-m/utils.py MileBenchDataset (per-image budget; left
truncate the middle, keep instruction + question).

Scoring: zap/qvik/eval/look_milebench_metrics.py LookMileBenchEvaluator
(matches look-m/evaluate.py output and look-m/score.py aggregation).

Usage:
    python milebench_unified.py --model llava15 --output_dir <dir> --device cuda:0
    python milebench_unified.py --model onevision --output_dir <dir> --device cuda:0
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


# --- Fixed paths ----------------------------------------------------------
DATA_ROOT = "/mnt/srv/home/dlpc.3842/zap/data/MileBench"
LLAVA15_CKPT = "/mnt/srv/home/dlpc.3842/zap/ckpts/llava-v1.5-7b"
ONEVISION_CKPT = "/mnt/srv/home/dlpc.3842/zap/ckpts/llava-onevision-qwen2-7b-ov"

ONEVISION_REPO = "/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision"
LOOKM_ROOT = "/mnt/srv/home/dlpc.3842/look-m"
ZAP_ROOT = "/mnt/srv/home/dlpc.3842/zap"

# --- Datasets -------------------------------------------------------------
DATASETS = ["ALFRED", "CLEVR-Change", "IEdit", "Spot-the-Diff"]

# --- Generation kwargs (per-model max_new_tokens) -------------------------
# All other knobs are shared so full-cache outputs reproduce across folders.
_GEN_BASE: dict[str, Any] = {
    "min_new_tokens": 1,
    "do_sample": False,
    "temperature": 0.0,
    "use_cache": True,
}
LLAVA15_MAX_NEW_TOKENS = 512
ONEVISION_MAX_NEW_TOKENS = 64

# --- Per-model prompt-truncation budget -----------------------------------
# The MileBench dataset truncator computes how many tokens an image will expand
# to in order to fit the prompt under max_context_len. These numbers describe
# the visual token budget per image for each model; they only affect prompt
# truncation, not the actual visual encoding.
LLAVA15_N_TOKENS_PER_IMAGE = 576       # 24*24 patches
ONEVISION_N_TOKENS_PER_IMAGE = 200     # OneVision pooled tokens (look-m default)
MAX_CONTEXT_LEN = 4096

# --- Per-model conv template ----------------------------------------------
LLAVA15_CONV = "vicuna_v1"
ONEVISION_CONV = "qwen_1_5"


def gen_kwargs_for(model_kind: str) -> dict[str, Any]:
    out = dict(_GEN_BASE)
    out["max_new_tokens"] = (
        LLAVA15_MAX_NEW_TOKENS if model_kind == "llava15" else ONEVISION_MAX_NEW_TOKENS
    )
    return out


def setup_paths() -> None:
    for p in (ONEVISION_REPO, LOOKM_ROOT, ZAP_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------
# Model loaders
# --------------------------------------------------------------------------
def load_model(model_kind: str, device: str):
    """Load a multimodal model via the LLaVA-OneVision repo's builder.

    Returns (tokenizer, model, image_processor, kind_meta).
    """
    setup_paths()
    from llava.model.builder import load_pretrained_model

    if model_kind == "llava15":
        # Triggers the v1.5 branch in the OneVision builder (uses LlavaLlamaForCausalLM).
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=LLAVA15_CKPT,
            model_base=None,
            model_name="llava-v1.5-7b",
            device_map=device,
            attn_implementation="sdpa",
            multimodal=True,
        )
        kind_meta = {
            "n_tokens_per_image": LLAVA15_N_TOKENS_PER_IMAGE,
            "conv": LLAVA15_CONV,
        }
    elif model_kind == "onevision":
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
        kind_meta = {
            "n_tokens_per_image": ONEVISION_N_TOKENS_PER_IMAGE,
            "conv": ONEVISION_CONV,
        }
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    model.eval()
    return tokenizer, model, image_processor, kind_meta


# --------------------------------------------------------------------------
# Prompt construction (input-side, identical across folders)
# --------------------------------------------------------------------------
def build_input_ids(
    *,
    question: str,
    tokenizer,
    model,
    conv_name: str,
) -> tuple[torch.Tensor, str]:
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import tokenizer_image_token

    mm_use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))
    if mm_use_im_start_end:
        single_img_tok = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        single_img_tok = DEFAULT_IMAGE_TOKEN

    # MileBenchDataset uses '<ImageHere>' as its placeholder; substitute the
    # correct LLaVA token. The CLEVR-Change inputs sometimes contain two
    # adjacent placeholders — separate them with a newline so the conv
    # template tokenizer treats them as distinct images.
    input_prompt = question.replace("<ImageHere>", single_img_tok)
    input_prompt = input_prompt.replace(
        single_img_tok + single_img_tok,
        single_img_tok + "\n" + single_img_tok,
    )

    conv = copy.deepcopy(conv_templates[conv_name])
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)
    return input_ids, stop_str or ""


def process_images_for_model(
    *,
    model_kind: str,
    image_paths: list[str],
    image_processor,
    model,
    device: str,
) -> tuple[Any, list[tuple[int, int]] | None, list[str] | None]:
    """Return (image_tensor_or_list, image_sizes, modalities)."""
    if not image_paths:
        return None, None, None

    from llava.mm_utils import process_images

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    if model_kind == "onevision" and len(pil_images) > 1:
        # For OneVision, multi-image inputs are handled as video frames with
        # square padding so spatial pooling produces a fixed token count.
        bg = tuple(int(x * 255) for x in image_processor.image_mean)
        squared = []
        for im in pil_images:
            w, h = im.size
            if w != h:
                side = max(w, h)
                square = Image.new(im.mode, (side, side), bg)
                square.paste(im, ((side - w) // 2, (side - h) // 2))
                im = square
            t = image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
            squared.append(t)
        video_tensor = torch.stack(squared, dim=0).to(device=device, dtype=torch.float16)
        return [video_tensor], [pil_images[0].size], ["video"]

    image_tensor = process_images(pil_images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)
    image_sizes = [im.size for im in pil_images]
    modalities = ["image"] * len(pil_images)
    return image_tensor, image_sizes, modalities


def collapse_image_placeholders_for_onevision(question: str, n_images: int) -> str:
    """OneVision video modality expects a single image placeholder.

    When the prompt has multiple <ImageHere> placeholders and we're routing the
    images through video modality, collapse them down to one placeholder so the
    remaining markers sit verbatim in the prompt as natural text.
    """
    placeholder = "<ImageHere>"
    count = question.count(placeholder)
    if count <= 1 or n_images <= 1:
        return question
    # Replace the first n_images-1 occurrences with empty so only one remains.
    out = question
    for _ in range(count - 1):
        out = out.replace(placeholder, "", 1)
    return out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
@torch.inference_mode()
def generate_answer(
    *,
    model_kind: str,
    tokenizer,
    model,
    image_processor,
    question: str,
    image_paths: list[str],
    conv_name: str,
    device: str,
) -> str:
    n_images = len(image_paths)
    if model_kind == "onevision":
        question_for_prompt = collapse_image_placeholders_for_onevision(question, n_images)
    else:
        question_for_prompt = question

    input_ids, stop_str = build_input_ids(
        question=question_for_prompt,
        tokenizer=tokenizer,
        model=model,
        conv_name=conv_name,
    )
    input_ids = input_ids.to(device)

    image_tensor, image_sizes, modalities = process_images_for_model(
        model_kind=model_kind,
        image_paths=image_paths,
        image_processor=image_processor,
        model=model,
        device=device,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    gen_kwargs = gen_kwargs_for(model_kind)
    gen_kwargs["pad_token_id"] = pad_token_id

    try:
        if model_kind == "onevision":
            out = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                modalities=modalities,
                **gen_kwargs,
            )
        else:
            out = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **gen_kwargs,
            )
    except TypeError:
        # Older signature: no image_sizes / modalities.
        out = model.generate(input_ids, images=image_tensor, **gen_kwargs)

    # The OneVision builder's generate path strips the prompt, returning only
    # newly generated tokens. Decode and strip the stop string.
    answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    if stop_str and stop_str in answer:
        answer = answer.split(stop_str)[0].strip()
    return answer


# --------------------------------------------------------------------------
# Dataset iteration (using look-m's MileBenchDataset for prompt truncation)
# --------------------------------------------------------------------------
def _to_int_if_possible(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return str(value)


def run_dataset(
    *,
    model_kind: str,
    dataset: str,
    tokenizer,
    model,
    image_processor,
    n_tokens_per_image: int,
    conv_name: str,
    output_dir: Path,
    device: str,
    overwrite: bool,
    limit: int | None,
) -> Path | None:
    setup_paths()
    from utils import MileBenchDataset

    task_out = output_dir / dataset
    task_out.mkdir(parents=True, exist_ok=True)
    pred_path = task_out / "pred.json"
    if pred_path.exists() and not overwrite:
        print(f"[skip] {dataset}: {pred_path} exists (use --overwrite)")
        return pred_path

    data_path = Path(DATA_ROOT) / dataset / f"{dataset}.json"
    img_dir = str(Path(DATA_ROOT) / dataset / "images")
    core = json.loads(data_path.read_text())
    samples_raw = core["data"]
    if limit is not None:
        samples_raw = samples_raw[:limit]

    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in samples_raw:
        by_n[len(s["task_instance"]["images_path"])].append(s)

    predictions: list[dict[str, Any]] = []
    for n_img in sorted(by_n):
        sub = by_n[n_img]
        ds = MileBenchDataset(
            annotation=sub,
            task_instructions=core["meta_data"]["task_instruction"],
            img_dir=img_dir,
            max_context_len=MAX_CONTEXT_LEN,
            n_tokens_per_image=n_tokens_per_image,
            tokenizer=tokenizer,
            dataset_name=dataset,
            combine_image=None,
        )
        for idx in tqdm(range(len(ds)), desc=f"{dataset}(n={n_img})"):
            item = ds[idx]
            sample_id = item["sample_id"]
            try:
                answer = generate_answer(
                    model_kind=model_kind,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    question=item["context"],
                    image_paths=item["raw_img_list"],
                    conv_name=conv_name,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {dataset} sample {sample_id} failed: {exc}", file=sys.stderr)
                answer = ""
            predictions.append(
                {
                    "sample_id": _to_int_if_possible(sample_id),
                    "image": item["raw_img_list"],
                    "question": item["context"],
                    "gt_response": str(item["response"]),
                    "gen_model_id": f"{model_kind}_fullcache",
                    "pred_response": answer,
                    "gen_kwargs": gen_kwargs_for(model_kind),
                }
            )

    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"[{dataset}] saved {len(predictions)} predictions -> {pred_path}")
    return pred_path


# --------------------------------------------------------------------------
# Scoring (LookMileBenchEvaluator)
# --------------------------------------------------------------------------
def score_results(*, output_dir: Path, datasets: list[str]) -> dict[str, Any]:
    setup_paths()
    from qvik.eval.score_milebench_predictions import score_dataset

    results: dict[str, Any] = {}
    for ds in datasets:
        try:
            metrics = score_dataset(
                data_root=DATA_ROOT,
                result_dir=str(output_dir),
                dataset=ds,
                allow_partial=False,
                overwrite=True,
            )
            results[ds] = metrics
        except Exception as exc:  # noqa: BLE001
            print(f"[score-err] {ds}: {exc}", file=sys.stderr)
            results[ds] = {"error": repr(exc)}
    summary_path = output_dir / "_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[summary] {summary_path}")
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava15", "onevision"], required=True)
    parser.add_argument("--dataset", default="all", help="dataset name or 'all'")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of samples per dataset (debug only).")
    parser.add_argument("--no-score", action="store_true",
                        help="Generate predictions but skip scoring.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    print(f"[init] model={args.model} datasets={datasets} output={output_dir}")

    tokenizer, model, image_processor, kind_meta = load_model(args.model, args.device)
    print(
        f"[init] model loaded: n_tokens_per_image={kind_meta['n_tokens_per_image']} "
        f"conv={kind_meta['conv']}"
    )

    for ds in datasets:
        run_dataset(
            model_kind=args.model,
            dataset=ds,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            n_tokens_per_image=kind_meta["n_tokens_per_image"],
            conv_name=kind_meta["conv"],
            output_dir=output_dir,
            device=args.device,
            overwrite=args.overwrite,
            limit=args.limit,
        )

    if not args.no_score:
        score_results(output_dir=output_dir, datasets=datasets)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
