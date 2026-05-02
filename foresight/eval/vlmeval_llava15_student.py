# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLMEvalKit-compatible LLaVA-1.5 wrapper with KV pruning.

Subclasses VLMEvalKit's `LLaVA` and replaces the standard `model.generate()`
call with a manual prefill → student-scored image-token KV pruning → decode
loop. Use this to evaluate the trained `VisualUtilityStudent` probe under
`vlmeval`.

Registered class name: `LLaVA_v1_5_HF_Student` — the launcher script
monkey-patches this class onto `vlmeval.vlm` before invoking `run.py`.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, "/workspace/zap")
from kvpress.presses.visual_utility_student import VisualUtilityStudent
from ..llava_15b_extractor import (
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)


def _trim_kv_cache_per_layer(past_kv, keep_masks: dict[int, torch.Tensor]):
    """Trim per-layer KV cache to positions where keep_masks[l] is True."""
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

    from transformers import DynamicCache
    new_cache = DynamicCache()
    for li, (k, v) in enumerate(past_kv):
        if li in keep_masks:
            mask = keep_masks[li].to(k.device)
            k = k[:, :, mask, :].contiguous()
            v = v[:, :, mask, :].contiguous()
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    return new_cache


def _resolve_eos_token_id(processor, model_config) -> int:
    eid = processor.tokenizer.eos_token_id
    if eid is None:
        cfg = getattr(model_config, "eos_token_id", None)
        if isinstance(cfg, (list, tuple)) and cfg:
            eid = cfg[0]
        else:
            eid = cfg
    if eid is None:
        eid = 2  # LLaMA default </s>
    return int(eid)


@torch.no_grad()
def _greedy_decode_with_kv(
    model,
    past_kv,
    first_next_token: torch.Tensor,
    prompt_len: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """Emit `first_next_token` then iterate with explicit cache_position/position_ids.

    When KV cache has been per-layer trimmed, `past_kv.get_seq_length()` no
    longer equals the true absolute position. We pass `cache_position` and
    `position_ids` set to `prompt_len + step` so RoPE aligns correctly.
    """
    out_tokens: list[int] = [int(first_next_token.item())]
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


def _import_base_class():
    from vlmeval.vlm.llava.llava import LLaVA
    return LLaVA


class LLaVA_v1_5_HF_Student:
    """Behaves like VLMEvalKit's `LLaVA` but uses student-driven KV pruning.

    We do NOT subclass at import time to avoid triggering heavy llava module
    loads during VLMEvalKit's discovery phase.
    """

    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = False
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(
        self,
        model_path: str = "/workspace/zap/ckpts/llava-1.5-7b-hf",
        student_path: str = "/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4",
        keep_ratio: float = 0.5,
        max_new_tokens: int = 32,
        per_layer_keep_ratios: dict[int, float] | None = None,
        entropy_budget: bool = False,
        grid_h: int = 24,
        grid_w: int = 24,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        configure_llava_processor(self.processor, self.model.config)

        self.student = VisualUtilityStudent.from_pretrained(student_path)
        self.student = self.student.to(device="cuda", dtype=torch.float16).eval()
        self.keep_ratio = float(keep_ratio)
        self.per_layer_keep_ratios = per_layer_keep_ratios
        self.entropy_budget = entropy_budget
        self.max_new_tokens = int(max_new_tokens)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.model_path = model_path
        self.student_path = student_path
        self._reported_keep_budget = False

        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1

    # — VLMEvalKit BaseModel interface ————————————————————————————————————————

    @property
    def is_api(self) -> bool:
        return False

    def use_custom_prompt(self, dataset: str) -> bool:
        return False

    def build_prompt(self, line, dataset: str | None = None):
        BaseClass = _import_base_class()
        if not hasattr(self, "_proxy_for_prompt"):
            self._proxy_for_prompt = object.__new__(BaseClass)
            self._proxy_for_prompt.processor = self.processor
        return BaseClass.build_prompt(self._proxy_for_prompt, line, dataset)

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset=None):
        return self.dump_image_func(line)

    def message_to_promptimg(self, *args, **kwargs):
        BaseClass = _import_base_class()
        return BaseClass.message_to_promptimg(self, *args, **kwargs)

    # — generation ————————————————————————————————————————————————————————————

    @torch.no_grad()
    def generate(self, message, dataset=None):
        return self.generate_inner(message, dataset)

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_image(message, dataset)

    def generate_inner_image(self, message, dataset=None):
        content = ""
        images: list[Image.Image] = []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {"role": "user", "content": [{"type": "text", "text": content}]},
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt",
        ).to("cuda", torch.float16)

        # No image: fall back to raw generate.
        if not images:
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return self.processor.decode(
                output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

        # Prefill with hidden states.
        with torch.no_grad():
            prefill = self.model(
                **inputs,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
        H_all = prefill.hidden_states
        past_kv = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        eos_token_id = _resolve_eos_token_id(self.processor, self.model.config)

        # Locate image tokens; bypass pruning if expansion failed.
        try:
            image_positions, prompt_len = infer_llava_image_positions_no_forward(
                prompt_inputs={
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                model_config=self.model.config,
                num_images=len(images),
            )
        except ValueError:
            answer_ids = _greedy_decode_with_kv(
                self.model,
                past_kv,
                next_token,
                prompt_len=int(inputs["input_ids"].shape[1]),
                eos_token_id=eos_token_id,
                max_new_tokens=self.max_new_tokens,
            )
            return self.processor.decode(answer_ids.tolist(), skip_special_tokens=True)

        n_img = int(image_positions.numel())
        n_text = int(prompt_len) - n_img
        n_keep = max(1, int(round(n_img - (1.0 - self.keep_ratio) * int(prompt_len))))
        if not self._reported_keep_budget:
            print(
                "[llava15-student] keep_ratio_basis=total "
                f"keep_ratio={self.keep_ratio} entropy_budget={self.entropy_budget} "
                f"prompt_len={int(prompt_len)} "
                f"image_tokens={n_img} text_tokens={n_text} image_tokens_kept={n_keep}",
                file=sys.stderr,
                flush=True,
            )
            self._reported_keep_budget = True

        last_img = int(image_positions.max().item())
        question_positions = (
            torch.arange(last_img + 1, prompt_len, dtype=torch.long)
            if last_img + 1 < prompt_len
            else torch.empty(0, dtype=torch.long)
        )

        image_idx_dev = image_positions.to("cuda")
        q_idx_dev = question_positions.to("cuda")

        do_prune = self.keep_ratio < 1.0 and n_keep < n_img
        if self.per_layer_keep_ratios:
            do_prune = True
        if self.entropy_budget:
            do_prune = True

        if do_prune:
            keep_masks: dict[int, torch.Tensor] = {}
            n_layers = len(self.student.layer_indices)

            # Collect student scores for all layers.
            all_scores: dict[int, torch.Tensor] = {}
            for li in self.student.layer_indices:
                H_l = H_all[li + 1]
                scores = self.student.layers[str(li)](
                    H_l, image_idx_dev, q_idx_dev, self.grid_h, self.grid_w
                ).squeeze(0)  # [n_img]
                all_scores[li] = scores

            # Compute per-layer n_keep.
            if self.entropy_budget:
                total_budget = n_keep * n_layers
                entropies = {}
                for li, scores in all_scores.items():
                    p = torch.softmax(scores.float(), dim=0)
                    h = -(p * p.clamp(min=1e-12).log()).sum().item()
                    entropies[li] = h
                h_sum = sum(entropies.values()) + 1e-12
                raw_alloc = {li: total_budget * (h / h_sum) for li, h in entropies.items()}
                floor_alloc = {li: max(1, int(v)) for li, v in raw_alloc.items()}
                remainder = total_budget - sum(floor_alloc.values())
                for li in sorted(entropies, key=lambda x: entropies[x], reverse=True):
                    if remainder <= 0:
                        break
                    floor_alloc[li] += 1
                    remainder -= 1
                layer_n_keep = {li: min(n_img, floor_alloc[li]) for li in self.student.layer_indices}
            else:
                layer_n_keep = {}
                for li in self.student.layer_indices:
                    if self.per_layer_keep_ratios is not None and li in self.per_layer_keep_ratios:
                        li_ratio = self.per_layer_keep_ratios[li]
                    else:
                        li_ratio = self.keep_ratio
                    layer_n_keep[li] = max(1, int(round(n_img - (1.0 - li_ratio) * int(prompt_len))))

            # Build keep masks.
            for li in self.student.layer_indices:
                li_n_keep = layer_n_keep[li]
                if li_n_keep >= n_img:
                    continue
                scores = all_scores[li]
                top = torch.topk(scores, k=li_n_keep, largest=True).indices
                mask = torch.ones(prompt_len, dtype=torch.bool)
                image_keep = torch.zeros(n_img, dtype=torch.bool)
                image_keep[top.cpu()] = True
                mask[image_positions.cpu()] = image_keep
                keep_masks[li] = mask

            past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

        answer_ids = _greedy_decode_with_kv(
            self.model,
            past_kv,
            next_token,
            prompt_len=int(prompt_len),
            eos_token_id=eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )

        text = self.processor.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
        torch.cuda.empty_cache()
        return text
