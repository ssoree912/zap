#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone MileBench evaluation using LLaVA-OneVision + PrefixKV KV pruning.

PrefixKV prunes KV cache tokens based on cumulative attention scores (image + text,
no distinction), using per-layer ratios from a pre-computed conf file.

Outputs pred.json compatible with /workspace/look-m/evaluate.py and score.py.

Usage:
    # 1. Profile first (if conf doesn't exist yet):
    python qvik/eval/milebench_onevision_prefixkv.py \
        --dataset ActionLocalization \
        --keep_ratio 0.5 \
        --output_dir /workspace/zap/experiments/.../outputs/keep050 \
        --profile --profile_samples 50

    # 2. Evaluate:
    python qvik/eval/milebench_onevision_prefixkv.py \
        --dataset ActionLocalization \
        --keep_ratio 0.5 \
        --output_dir /workspace/zap/experiments/.../outputs/keep050
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/workspace/zap")
sys.path.insert(0, "/workspace/PrefixKV")

from prefixkv import obtain_cdf_num


# ---------------------------------------------------------------------------
# KV cache helpers (inlined to avoid kvpress import chain)
# ---------------------------------------------------------------------------

def _resolve_eos_token_id(processor, model_config) -> int:
    eid = processor.tokenizer.eos_token_id
    if eid is None:
        cfg = getattr(model_config, "eos_token_id", None)
        if isinstance(cfg, (list, tuple)) and cfg:
            eid = cfg[0]
        else:
            eid = cfg
    if eid is None:
        eid = 151645  # Qwen2 default <|im_end|>
    return int(eid)


def _trim_kv_cache_per_layer(past_kv, keep_masks: dict):
    """Trim per-layer KV cache to positions where keep_masks[l] is True.

    After trimming, each layer may have a different KV length. DynamicCache.get_seq_length()
    only queries layer 0, which may be shorter than other layers, causing causal_mask to be
    too narrow. Override get_seq_length to return the max across layers.
    """
    if hasattr(past_kv, "key_cache"):
        for li in range(len(past_kv.key_cache)):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(past_kv.key_cache[li].device)
            past_kv.key_cache[li] = past_kv.key_cache[li][:, :, mask, :].contiguous()
            past_kv.value_cache[li] = past_kv.value_cache[li][:, :, mask, :].contiguous()
        # Normalize all layers to the same (minimum) KV length to avoid causal_mask mismatch.
        # PrefixKV trims layers to different lengths; DynamicCache.get_seq_length() only
        # checks layer 0, so the causal_mask would be too narrow for layers with more tokens.
        if past_kv.key_cache:
            min_len = min(k.shape[-2] for k in past_kv.key_cache if hasattr(k, "shape"))
            for li in range(len(past_kv.key_cache)):
                if not hasattr(past_kv.key_cache[li], "shape"):
                    continue
                if past_kv.key_cache[li].shape[-2] > min_len:
                    past_kv.key_cache[li] = past_kv.key_cache[li][:, :, :min_len, :].contiguous()
                    past_kv.value_cache[li] = past_kv.value_cache[li][:, :, :min_len, :].contiguous()
        return past_kv

    if hasattr(past_kv, "layers"):
        for li, layer in enumerate(past_kv.layers):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(layer.keys.device)
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return past_kv

    from transformers import DynamicCache
    new_cache = DynamicCache()
    for li, (k, v) in enumerate(past_kv):
        if li in keep_masks:
            mask = keep_masks[li].to(k.device)
            k = k[:, :, mask, :].contiguous()
            v = v[:, :, mask, :].contiguous()
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    # Normalize all layers to min length to avoid causal_mask mismatch
    if new_cache.key_cache:
        min_len = min(k.shape[-2] for k in new_cache.key_cache)
        for li in range(len(new_cache.key_cache)):
            if new_cache.key_cache[li].shape[-2] > min_len:
                new_cache.key_cache[li] = new_cache.key_cache[li][:, :, :min_len, :].contiguous()
                new_cache.value_cache[li] = new_cache.value_cache[li][:, :, :min_len, :].contiguous()
    return new_cache


@torch.no_grad()
def _greedy_decode_with_kv(model, past_kv, first_next_token, prompt_len, eos_token_id, max_new_tokens):
    out_tokens = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id:
        return torch.tensor(out_tokens, dtype=torch.long)

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
    return torch.tensor(out_tokens, dtype=torch.long)

DATA_ROOT = "/workspace/zap/data/MileBench"
CONF_DIR = Path("/workspace/PrefixKV/confs")
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_NEW_TOKENS = 32
START_SIZE = 4
PROTECT_SIZE = 1
MAX_LEN = 4096       # attention matrix O(seq^2); 4096 → ~512MB/layer, fits in ~5GB free
# Single tile only: 2 tiles total (1 grid + 1 thumbnail) × 729 ≈ 1458 image tokens → fits easily in 4096
MAX_PINPOINTS = [[384, 384]]


def build_prompt(sample: dict, meta: dict) -> str:
    ann = sample["task_instance"]
    task_instruction = meta["task_instruction"][sample["task_instruction_id"]]

    context = ann["context"]
    n_img = len(ann["images_path"])
    for i in range(1, n_img + 1):
        context = context.replace(f"{{image#{i}}}", f"<Image {i}> ")
        context = context.replace(f"{{table#{i}}}", f"<Image {i}> ")

    if ann.get("choice_list"):
        choice_str = "\nChoice List:\n"
        choice_str += "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ann["choice_list"]))
        choice_str += "\nYour answer is: "
        context += choice_str

    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}"


def conf_path(model_name: str, ratio: float) -> Path:
    return CONF_DIR / f"prefixkv_{model_name}_{ratio}.json"


def load_conf(model_name: str, ratio: float) -> np.ndarray:
    p = conf_path(model_name, ratio)
    if not p.exists():
        raise FileNotFoundError(
            f"PrefixKV conf not found: {p}\n"
            f"Run with --profile --profile_samples 50 first."
        )
    with open(p) as f:
        return np.array(json.load(f))


class LlavaOnevisionPrefixKV:
    """LLaVA-OneVision (original repo) with PrefixKV attention-score KV pruning."""

    def __init__(
        self,
        pretrained: str = "/workspace/zap/model/llava-onevision-qwen2-7b-ov",
        keep_ratio: float = 0.5,
        device: str = "cuda:0",
        conv_template: str = "qwen_1_5",
    ) -> None:
        _LLAVA_SRC = "/workspace/VFlowOpt/src/LLaVA-OneVision"
        _TF_SRC = "/workspace/VFlowOpt/src/transformers-4.46.0/src"
        for p in [_LLAVA_SRC, _TF_SRC]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        self._device = torch.device(device)
        model_name = get_model_name_from_path(pretrained)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, model_name,
            device_map=device,
            attn_implementation="eager",  # needed for output_attentions=True
            multimodal=True,
        )
        self._model = model.eval()
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._conv_template = conv_template
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._config = self._model.config

        self.model_name = Path(pretrained).name
        self.keep_ratio = float(keep_ratio)
        self.ratio = round(1.0 - keep_ratio, 10)
        self.layer_num = getattr(self._config, "num_hidden_layers", 28)

    def _prepare_inputs(self, prompt_text: str, image: Image.Image):
        """Convert prompt + image to inputs_embeds dict and return (inputs, prompt_len)."""
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        conv = conv_templates[self._conv_template].copy()
        question = prompt_text
        if DEFAULT_IMAGE_TOKEN not in question:
            question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)
        attention_mask = input_ids.ne(
            self._tokenizer.pad_token_id if self._tokenizer.pad_token_id else self._tokenizer.eos_token_id
        ).to(self._device)

        image_tensor = self._process_images([image], self._image_processor, self._config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self._device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self._device, dtype=torch.float16)

        _, _, attention_mask, _, inputs_embeds, _ = self._model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, None,
            image_tensor, ["image"], [image.size],
        )
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}, int(inputs_embeds.shape[1])

    @torch.no_grad()
    def generate(self, inputs: dict, prompt_len: int, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Full-cache baseline generation (keep_ratio=1.0)."""
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_with_prefixkv(
        self,
        inputs: dict,
        prompt_len: int,
        layer_ratios: np.ndarray,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        """Prefill → PrefixKV attention-score pruning → greedy decode."""
        score_sums: dict[int, torch.Tensor] = {}
        attn_layers = self._model.model.layers
        original_forwards: dict[int, object] = {}

        def make_patched(li, orig_fwd):
            def patched(*args, **kwargs):
                kwargs["output_attentions"] = True
                out = orig_fwd(*args, **kwargs)
                attn_w = out[1]
                if attn_w is not None:
                    score_sums[li] = attn_w.float().mean(dim=1).sum(dim=-2)[0].detach().cpu()
                return (out[0], None) + out[2:]
            return patched

        for li, layer in enumerate(attn_layers):
            original_forwards[li] = layer.self_attn.forward
            layer.self_attn.forward = make_patched(li, original_forwards[li])

        try:
            prefill = self._model(
                **inputs,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as e:
            for li, layer in enumerate(attn_layers):
                layer.self_attn.forward = original_forwards[li]
            print(f"[warn] prefill failed ({e}), falling back to full cache.", file=sys.stderr)
            return self.generate(inputs, max_new_tokens)
        finally:
            for li, layer in enumerate(attn_layers):
                layer.self_attn.forward = original_forwards[li]

        past_kv = prefill.past_key_values
        eos_token_id = self._tokenizer.eos_token_id or 151645
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del prefill

        forget_nums = (layer_ratios * prompt_len).round().astype(np.int32)

        # Build per-layer boolean keep masks from accumulated scores
        keep_masks: dict[int, torch.Tensor] = {}
        for li in range(self.layer_num):
            forget_num = int(forget_nums[li])
            if forget_num <= 0 or li not in score_sums:
                continue
            mid_start = START_SIZE
            mid_end = prompt_len - PROTECT_SIZE
            if mid_end <= mid_start:
                continue

            scores_mid = score_sums[li][mid_start:mid_end]
            keep_in_mid = scores_mid.argsort(descending=False)[forget_num:] + mid_start
            keep_in_mid = keep_in_mid.sort().values

            pre = torch.arange(START_SIZE)
            post = torch.tensor([mid_end])
            keep_idx = torch.cat([pre, keep_in_mid, post])

            mask = torch.zeros(prompt_len, dtype=torch.bool)
            mask[keep_idx] = True
            keep_masks[li] = mask

        torch.cuda.empty_cache()

        past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

        answer_ids = _greedy_decode_with_kv(
            self._model, past_kv, next_token,
            prompt_len=prompt_len,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.empty_cache()
        return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

    @torch.no_grad()
    def profile_sample(self, inputs: dict, prompt_len: int) -> list[float] | None:
        """Compute PrefixKV per-layer forget ratios for one sample."""
        try:
            score_sums: dict[int, torch.Tensor] = {}

            attn_layers = self._model.model.layers
            original_forwards: dict[int, object] = {}

            def make_patched(li, orig_fwd):
                def patched(*args, **kwargs):
                    kwargs["output_attentions"] = True
                    out = orig_fwd(*args, **kwargs)  # orig_fwd is already bound
                    attn_w = out[1]
                    if attn_w is not None:
                        # [B, H, q, k] → mean H, sum q → [k]
                        score_sums[li] = attn_w.float().mean(dim=1).sum(dim=-2)[0].detach().cpu()
                    # Return None for attn weights to free GPU memory immediately
                    return (out[0], None) + out[2:]
                return patched

            for li, layer in enumerate(attn_layers):
                original_forwards[li] = layer.self_attn.forward
                layer.self_attn.forward = make_patched(li, original_forwards[li])

            try:
                self._model(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
            finally:
                for li, layer in enumerate(attn_layers):
                    layer.self_attn.forward = original_forwards[li]

            if not score_sums:
                return None

            score_stack = torch.stack([score_sums[li] for li in range(len(attn_layers))])  # [L, seq]

            target_num = prompt_len * self.ratio
            forget_nums = obtain_cdf_num(
                score_stack[:, :prompt_len],
                target_num,
                START_SIZE + PROTECT_SIZE,
            )
            if forget_nums.sum() != 0:
                forget_nums = forget_nums * target_num / forget_nums.sum()
            forget_nums = forget_nums.astype(np.int32)

            torch.cuda.empty_cache()
            return (forget_nums / max(1, prompt_len)).tolist()
        except Exception as e:
            import traceback
            print(f"[profile_sample] error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            torch.cuda.empty_cache()
            return None


def run_profile(model: LlavaOnevisionPrefixKV, samples: list, meta: dict, combined_img_root: str, n: int) -> np.ndarray:
    """Profile on up to n samples and return averaged per-layer ratios."""
    all_ratios: list[list[float]] = []
    for sample in tqdm(samples[:n], desc="Profiling"):
        try:
            ann = sample["task_instance"]
            img_file = ann["combined_1_images"][0]
            img_path = os.path.join(combined_img_root, img_file)
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        try:
            prompt_text = build_prompt(sample, meta)
            inputs, prompt_len = model._prepare_inputs(prompt_text, image)
        except Exception as e:
            print(f"[profile] input prep failed: {e}", file=sys.stderr)
            continue

        ratios = model.profile_sample(inputs, prompt_len)
        if ratios is not None:
            all_ratios.append(ratios)

    if not all_ratios:
        raise RuntimeError(
            f"Profiling failed: 0 valid samples out of {min(n, len(samples))} attempted. "
            "Check that combined_1_images exist and output_attentions works."
        )

    avg_ratios = np.mean(all_ratios, axis=0)
    print(f"[profile] {len(all_ratios)} samples → avg per-layer ratios: {avg_ratios.round(3).tolist()}")
    return avg_ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrained", default="/workspace/zap/model/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Profile mode: compute per-layer ratios and save conf, then evaluate.")
    parser.add_argument("--profile_samples", type=int, default=50,
                        help="Number of calibration samples for profiling.")
    args = parser.parse_args()

    task_out = os.path.join(args.output_dir, args.dataset)
    pred_path = os.path.join(task_out, "pred.json")
    os.makedirs(task_out, exist_ok=True)

    if os.path.exists(pred_path) and not args.overwrite:
        print(f"[skip] {args.dataset}: {pred_path} exists")
        return

    # Load data
    data_path = os.path.join(DATA_ROOT, args.dataset, f"{args.dataset}.json")
    data = json.load(open(data_path))
    meta = data["meta_data"]
    samples = data["data"]
    combined_img_root = os.path.join(DATA_ROOT, args.dataset, "combined_1_images")

    ratio = round(1.0 - args.keep_ratio, 10)
    print(f"[{args.dataset}] {len(samples)} samples | keep_ratio={args.keep_ratio} ratio={ratio}")

    model = LlavaOnevisionPrefixKV(
        pretrained=args.pretrained,
        keep_ratio=args.keep_ratio,
        device=args.device,
    )

    # Profiling step
    if args.profile or not conf_path(model.model_name, ratio).exists():
        if not args.profile and not conf_path(model.model_name, ratio).exists():
            print(f"[info] Conf not found. Running profile on {args.profile_samples} samples first.")
        avg_ratios = run_profile(model, samples, meta, combined_img_root, args.profile_samples)
        CONF_DIR.mkdir(parents=True, exist_ok=True)
        p = conf_path(model.model_name, ratio)
        with open(p, "w") as f:
            json.dump(avg_ratios.tolist(), f)
        print(f"[profile] conf saved → {p}")

    layer_ratios = load_conf(model.model_name, ratio)

    # Evaluation loop
    predictions = []
    for sample in tqdm(samples, desc=args.dataset):
        ann = sample["task_instance"]
        prompt_text = build_prompt(sample, meta)

        img_file = ann["combined_1_images"][0]
        img_path = os.path.join(combined_img_root, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {img_path}: {e}", file=sys.stderr)
            predictions.append({
                "sample_id": sample["sample_id"],
                "pred_response": "",
                "gt_response": sample["response"],
            })
            continue

        try:
            inputs, prompt_len = model._prepare_inputs(prompt_text, image)
        except Exception as e:
            print(f"[warn] input prep failed for sample {sample['sample_id']}: {e}", file=sys.stderr)
            predictions.append({"sample_id": sample["sample_id"], "pred_response": "", "gt_response": sample["response"]})
            continue

        if args.keep_ratio >= 1.0:
            answer = model.generate(inputs, prompt_len, MAX_NEW_TOKENS)
        else:
            answer = model.generate_with_prefixkv(inputs, prompt_len, layer_ratios, MAX_NEW_TOKENS)

        predictions.append({
            "sample_id": sample["sample_id"],
            "pred_response": answer,
            "gt_response": sample["response"],
        })

    json.dump(predictions, open(pred_path, "w"), ensure_ascii=False, indent=2)
    print(f"[{args.dataset}] saved → {pred_path}")


if __name__ == "__main__":
    main()
