#!/usr/bin/env python3
"""Collect future-supervised teacher labels for image token importance prediction.

For each sample this script runs a full LLaVA forward pass (prefill + greedy decode)
and extracts:
  - hidden_image  : [n_layers, N_image, D]  prefill hidden states at image positions
  - future_splus  : [n_layers, N_image]     decode→image attention (teacher label)

future_splus[l, i] = mean over heads × mean over decode steps of
                     A^(l)[decode_q, image_key_i]

The output records are saved to {out_dir}/records/{sample_id}.pt and are compatible
with train_future_supervised_probe.py as both extractor and teacher.

Usage:
  python collect_future_supervised_labels.py \\
    --dataset textvqa \\
    --data_dir /workspace/zap/data/textvqa/train \\
    --out_dir /workspace/zap/artifacts/future_teacher/textvqa \\
    --device cuda:0 \\
    --max_new_tokens 64 \\
    --selected_layers -4 -3 -2 -1
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from kvzap.llava_extractor import (
    LlavaAnalysisDataCollector,
    _parse_torch_dtype,
    configure_llava_processor,
)


def _resolve_image_indices(record: dict[str, Any]) -> torch.Tensor:
    if "image_indices_mm" in record:
        return record["image_indices_mm"].long().flatten()
    if "is_image_pos_mm" in record:
        return record["is_image_pos_mm"].nonzero(as_tuple=False).flatten().long()
    raise KeyError("Record missing image_indices_mm / is_image_pos_mm")


def compute_future_labels(
    record: dict[str, Any],
    selected_layers: list[int],
    storage_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Extract hidden_image and future_splus from a collect_sample record.

    Args:
        record: output of LlavaAnalysisDataCollector.collect_sample()
        selected_layers: layer indices (negative OK, e.g. [-4, -3, -2, -1])
        storage_dtype: dtype for saved tensors

    Returns:
        dict with:
          hidden_image   [n_layers, N_image, D]
          future_splus   [n_layers, N_image]
          image_indices_mm  [N_image]
          n_decode_steps int
    """
    # attn_answer_to_prompt: list[n_transformer_layers] of [H, T, N_prompt]
    # hidden_prompt:         list[n_transformer_layers + 1] of [N_prompt, D]
    #
    # Decode-step validity: LlavaAnalysisDataCollector.collect_sample() builds
    # analysis_input_ids from _trim_after_eos(generated_ids), so full_len_mm already
    # excludes post-EOS tokens.  T = full_len_mm - prompt_len_mm contains only valid
    # decode steps — no padding masking is needed here.
    attn_list = record["attn_answer_to_prompt"]   # list[L] of [H, T, N_prompt]
    hidden_list = record["hidden_prompt"]          # list[L+1] of [N_prompt, D]
    n_layers_total = len(attn_list)

    assert n_layers_total > 0, "attn_answer_to_prompt is empty"
    assert len(hidden_list) in (n_layers_total, n_layers_total + 1), (
        f"hidden_prompt length {len(hidden_list)} unexpected for {n_layers_total} attn layers"
    )

    image_idx = _resolve_image_indices(record)    # [N_image]
    N_image = image_idx.numel()
    if N_image == 0:
        raise ValueError("No image tokens found in record")

    resolved = [l % n_layers_total for l in selected_layers]
    # hidden_list[0] is embedding; transformer layer l output is hidden_list[l+1]
    hidden_layer_offset = 1 if len(hidden_list) == n_layers_total + 1 else 0

    hidden_image_layers: list[torch.Tensor] = []
    future_splus_layers: list[torch.Tensor] = []

    for l in resolved:
        # ── prefill hidden at image positions ────────────────────────────
        h = hidden_list[l + hidden_layer_offset].float()  # [N_prompt, D]
        h_img = h[image_idx]                              # [N_image, D]
        hidden_image_layers.append(h_img.to(storage_dtype))

        # ── decode → image attention ─────────────────────────────────────
        attn = attn_list[l].float()       # [H, T, N_prompt]
        attn_img = attn[:, :, image_idx]  # [H, T, N_image]
        # aggregate: mean over heads (dim 0) and decode steps (dim 1)
        label = attn_img.mean(dim=0).mean(dim=0)  # [N_image]
        future_splus_layers.append(label.to(storage_dtype))

    T = int(attn_list[0].shape[1])

    return {
        "hidden_image": torch.stack(hidden_image_layers, dim=0),  # [n_sel, N_image, D]
        "future_splus": torch.stack(future_splus_layers, dim=0),  # [n_sel, N_image]
        "image_indices_mm": image_idx.cpu(),
        "n_decode_steps": T,
    }


def collect_dataset(
    *,
    model: LlavaForConditionalGeneration,
    processor: Any,
    samples: list[dict[str, Any]],
    prompt_template: str,
    out_dir: Path,
    selected_layers: list[int],
    max_new_tokens: int,
    storage_dtype: torch.dtype,
    overwrite: bool,
) -> dict[str, int]:
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    collector = LlavaAnalysisDataCollector(model, processor)
    stats = {"saved": 0, "skipped": 0, "failed": 0}

    for sample in tqdm(samples, desc="Collecting future labels"):
        sid = sample["sample_id"]
        out_path = records_dir / f"{sid}.pt"

        if out_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        try:
            record = collector.collect_sample(
                sample=sample,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                capture_vproj=False,
                save_image_hidden_states=False,
                save_full_attentions=False,
                save_wo=False,
            )

            labels = compute_future_labels(record, selected_layers, storage_dtype)

            save_record = {
                "sample_id": sid,
                "prompt_len_mm": record["prompt_len_mm"],
                "full_len_mm": record["full_len_mm"],
                "is_image_pos_mm": record["is_image_pos_mm"].cpu(),
                "image_indices_mm": labels["image_indices_mm"],
                "hidden_image": labels["hidden_image"],   # [n_sel, N_image, D]
                "future_splus": labels["future_splus"],   # [n_sel, N_image]
                "n_decode_steps": labels["n_decode_steps"],
            }
            torch.save(save_record, out_path)
            stats["saved"] += 1

        except Exception as exc:
            print(f"[WARN] sample {sid} failed: {exc}")
            traceback.print_exc()
            stats["failed"] += 1

    return stats


# ── dataset loaders ─────────────────────────────────────────────────────────

TEXTVQA_TEMPLATE = "USER: <image>\nQuestion: {question}\nASSISTANT:"
SCIENCEQA_TEMPLATE = "USER: <image>\n{prompt_body}\nASSISTANT:"
NLVR2_TEMPLATE = (
    "USER: <image>\n<image>\n"
    "Statement: {sentence}\n"
    "Is this statement True or False?\n"
    "Options:\nA. True\nB. False\n"
    "ASSISTANT:"
)

DATASET_TEMPLATES = {
    "textvqa": TEXTVQA_TEMPLATE,
    "nlvr2": NLVR2_TEMPLATE,
    "scienceqa": SCIENCEQA_TEMPLATE,
}


def _load_json_samples(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    data_path = data_dir / "data.json"
    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)
    if limit is not None:
        records = records[:limit]

    # image_paths in data.json may be relative to /workspace/zap/data/ (e.g. nlvr2),
    # or absent with fallback to image_path (e.g. textvqa). Resolve to absolute.
    img_root = data_dir.parent.parent  # e.g. /workspace/zap/data

    samples = []
    for rec in records:
        sid = str(rec.get("question_id", rec.get("sample_id", rec.get("identifier", len(samples)))))
        question = str(rec.get("question", rec.get("sentence", ""))).strip()
        raw_paths = rec.get("image_paths") or [rec["image_path"]]
        resolved = []
        for p in raw_paths:
            pp = Path(p)
            resolved.append(str(pp if pp.is_absolute() else (img_root / pp)))
        samples.append({
            "sample_id": sid,
            "question": question,
            "sentence": question,
            "prompt_body": question,
            "image_paths": resolved,
        })
    return samples


def _load_scienceqa_samples(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    """ScienceQA uses problems.json + pid_splits.json, not data.json."""
    from build_scienceqa_manifest import iter_scienceqa_samples

    OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]
    samples = []
    for raw in iter_scienceqa_samples(base_dir=data_dir, split="train", limit=limit):
        hint = raw.get("hint", "")
        question = raw.get("question", "")
        choices = raw.get("choices", [])
        lines = []
        if hint:
            lines.append(f"Context: {hint}")
        lines.append(f"Question: {question}")
        lines.append("Options:")
        for idx, choice in enumerate(choices):
            lines.append(f"{OPTION_LETTERS[idx]}. {choice}")
        lines.append("Select the best answer based on the image and text.")
        prompt_body = "\n".join(lines)
        samples.append({
            "sample_id": raw["sample_id"],
            "question": question,
            "sentence": question,
            "prompt_body": prompt_body,
            "image_paths": [raw["image_path"]],
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASET_TEMPLATES), required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--selected_layers", type=int, nargs="+", default=[-4, -3, -2, -1])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _parse_torch_dtype(args.torch_dtype)
    storage_dtype = torch.float16

    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
    )
    configure_llava_processor(processor, model.config)
    model.eval()

    data_dir = Path(args.data_dir).resolve()
    if args.dataset == "scienceqa":
        samples = _load_scienceqa_samples(data_dir, args.limit)
    else:
        samples = _load_json_samples(data_dir, args.limit)
    prompt_template = DATASET_TEMPLATES[args.dataset]

    print(f"Samples: {len(samples)}, layers: {args.selected_layers}, T_max: {args.max_new_tokens}")

    stats = collect_dataset(
        model=model,
        processor=processor,
        samples=samples,
        prompt_template=prompt_template,
        out_dir=out_dir,
        selected_layers=args.selected_layers,
        max_new_tokens=args.max_new_tokens,
        storage_dtype=storage_dtype,
        overwrite=args.overwrite,
    )

    config = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "selected_layers": args.selected_layers,
        "n_samples": len(samples),
        **stats,
    }
    with (out_dir / "collect_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
