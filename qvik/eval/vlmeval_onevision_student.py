# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLMEvalKit-compatible LLaVA-OneVision wrapper with KV pruning.

Subclasses VLMEvalKit's `LLaVA_OneVision_HF` and replaces the standard
`model.generate()` call with a manual prefill → student-scored image-token
KV pruning → decode loop. Use this to evaluate the trained
`VisualUtilityStudentOneVision` probe under `vlmeval`.

Registered class name: `LLaVA_OneVision_HF_Student` — VLMEvalKit's
`build_model_from_config` finds custom classes by attribute lookup on
`vlmeval.vlm`, so the launcher script monkey-patches this class onto that
module before invoking `run.py`.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, "/workspace/zap")
from kvpress.presses.visual_utility_student_onevision import (
    VisualUtilityStudentOneVision,
)
from ..llava_onevision_extractor import (
    configure_onevision_processor,
    infer_onevision_image_positions_no_forward,
)


def _trim_kv_cache_per_layer(past_kv, keep_masks: dict[int, torch.Tensor]):
    """Trim per-layer KV cache to positions where keep_masks[l] is True.

    Supports DynamicCache (HF >=4.36, key_cache/value_cache lists),
    legacy .layers API, and tuple-of-tuples. Always returns the same type
    as the input so the model's _update_causal_mask doesn't see a type change.
    """
    # HF >=4.36 DynamicCache: key_cache[li] shape [B, H, S, D]
    if hasattr(past_kv, "key_cache"):
        for li in range(len(past_kv.key_cache)):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(past_kv.key_cache[li].device)
            past_kv.key_cache[li] = past_kv.key_cache[li][:, :, mask, :].contiguous()
            past_kv.value_cache[li] = past_kv.value_cache[li][:, :, mask, :].contiguous()
        return past_kv

    # Legacy cache with .layers attribute
    if hasattr(past_kv, "layers"):
        for li, layer in enumerate(past_kv.layers):
            if li not in keep_masks:
                continue
            mask = keep_masks[li].to(layer.keys.device)
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return past_kv

    # tuple-of-tuples → trim and convert back to DynamicCache so model doesn't choke
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
    """Qwen2 sometimes carries `eos_token_id` as a list — guard against that."""
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


@torch.no_grad()
def _greedy_decode_with_kv(
    model,
    past_kv,
    first_next_token: torch.Tensor,
    prompt_len: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """`first_next_token` is the prefill argmax — emit it then iterate.

    Critical: when KV cache has been per-layer trimmed, `past_kv.get_seq_length()`
    no longer equals the *true* absolute position the next query should rotate
    at. We must explicitly pass `cache_position` and `position_ids` set to the
    true absolute position (`prompt_len + step`) so RoPE for new query/key
    aligns with the un-trimmed K vectors that retain their original rotations.
    """
    out_tokens: list[int] = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id:
        return torch.tensor(out_tokens, dtype=torch.long)

    next_token = first_next_token
    pos = int(prompt_len)  # absolute position of the about-to-be-decoded token
    device = next_token.device
    cache_pos = torch.zeros(1, dtype=torch.long, device=device)  # reused in-place
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
    # Imported lazily so that pure environment probes (e.g. `import` checks
    # in run.py's discovery phase) do not trigger heavy llava module loads.
    from vlmeval.vlm.llava.llava import LLaVA_OneVision_HF
    return LLaVA_OneVision_HF


class LLaVA_OneVision_HF_Student:
    """Behaves like `LLaVA_OneVision_HF` but uses student-driven KV pruning.

    We do NOT subclass at import time — we wrap delegation so that VLMEvalKit's
    discovery doesn't try to instantiate the parent constructor before we
    inject the trained student.
    """

    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(
        self,
        model_path: str = "/workspace/zap/model/llava-onevision-qwen2-7b-ov-hf",
        student_path: str = "/workspace/zap/ckpts/student_onevision_A_lr1e4_20ep",
        keep_ratio: float = 0.5,
        max_new_tokens: int = 32,
        per_layer_keep_ratios: dict[int, float] | None = None,
        entropy_budget: bool = False,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        configure_onevision_processor(self.processor, self.model.config)

        self.student = VisualUtilityStudentOneVision.from_pretrained(student_path)
        self.student = self.student.to(device="cuda", dtype=torch.float16).eval()
        self.keep_ratio = float(keep_ratio)
        self.per_layer_keep_ratios = per_layer_keep_ratios  # dict[layer_idx -> ratio], overrides keep_ratio per layer
        self.entropy_budget = entropy_budget  # if True, redistribute total budget across layers proportional to entropy
        self.max_new_tokens = int(max_new_tokens)
        self.model_path = model_path
        self.student_path = student_path
        self._reported_keep_budget = False

        # Required attributes mirroring `LLaVA_OneVision_HF` so VLMEvalKit
        # treats us as a regular VLM (it inspects these for routing logic).
        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1

    # — VLMEvalKit BaseModel interface delegation ——————————————————————————
    @property
    def is_api(self) -> bool:
        return False

    def use_custom_prompt(self, dataset: str) -> bool:
        return False

    def build_prompt(self, line, dataset: str | None = None):
        # Delegate to the parent class's prompt builder (text + image messages).
        BaseClass = _import_base_class()
        if not hasattr(self, "_proxy_for_prompt"):
            # Build a lightweight stand-in just for prompt construction.
            self._proxy_for_prompt = object.__new__(BaseClass)
            self._proxy_for_prompt.processor = self.processor
        return BaseClass.build_prompt(self._proxy_for_prompt, line, dataset)

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset=None):
        # VLMEvalKit calls this internally when reading TSV-stored images.
        return self.dump_image_func(line)

    def message_to_promptimg(self, *args, **kwargs):
        BaseClass = _import_base_class()
        return BaseClass.message_to_promptimg(self, *args, **kwargs)

    # — generation —————————————————————————————————————————————————————————

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

        # No image case: fall back to raw generate.
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

        # Locate image tokens; if expansion failed (unlikely), bypass pruning.
        try:
            image_positions, prompt_len = infer_onevision_image_positions_no_forward(
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
            return self.processor.decode(
                answer_ids.tolist(), skip_special_tokens=True
            )

        n_img = int(image_positions.numel())
        n_text = int(prompt_len) - n_img
        n_keep = max(1, int(round(n_img - (1.0 - self.keep_ratio) * int(prompt_len))))
        if not self._reported_keep_budget:
            print(
                "[onevision-student] keep_ratio_basis=total "
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

        # Skip pruning entirely when keep_ratio>=1 (lets us A/B against the
        # exact same code path as student@<1.0; no decode-loop drift).
        do_prune = self.keep_ratio < 1.0 and n_keep < n_img
        if self.per_layer_keep_ratios:
            do_prune = True  # any per-layer override activates pruning
        if self.entropy_budget:
            do_prune = True  # entropy redistribution requires pruning even at keep_ratio<1 cases
        if do_prune:
            keep_masks: dict[int, torch.Tensor] = {}
            n_layers = len(self.student.layer_indices)

            # --- Step 1: collect all student scores (needed for entropy budget) ---
            all_scores: dict[int, torch.Tensor] = {}
            for li in self.student.layer_indices:
                H_l = H_all[li + 1]
                scores = self.student.layers[str(li)](H_l, image_idx_dev, q_idx_dev).squeeze(0)  # [n_img]
                all_scores[li] = scores

            # --- Step 2: compute per-layer n_keep ---
            if self.entropy_budget:
                # Entropy of softmax(student scores) per layer, used as proxy for
                # teacher utility entropy (Sub-exp A showed teacher entropy varies 0.77–0.93
                # across layers). High-entropy layers get more budget. NOTE: student scores
                # are raw logits; softmax here approximates the utility distribution. Alignment
                # between student and teacher is imperfect (Spearman 0.48–0.77, Sub-exp B),
                # so this is an approximation — not the oracle teacher-entropy allocation.
                total_budget = n_keep * n_layers  # same total tokens as uniform
                entropies = {}
                for li, scores in all_scores.items():
                    p = torch.softmax(scores.float(), dim=0)
                    h = -(p * p.clamp(min=1e-12).log()).sum().item()
                    entropies[li] = h
                h_sum = sum(entropies.values()) + 1e-12
                # Allocate proportional to entropy: high entropy → more budget
                raw_alloc = {li: total_budget * (h / h_sum) for li, h in entropies.items()}
                # Floor allocation, then distribute remainder to highest-entropy layers
                floor_alloc = {li: max(1, int(v)) for li, v in raw_alloc.items()}
                remainder = total_budget - sum(floor_alloc.values())
                for li in sorted(entropies, key=lambda x: entropies[x], reverse=True):
                    if remainder <= 0:
                        break
                    floor_alloc[li] += 1
                    remainder -= 1
                layer_n_keep = {li: min(n_img, floor_alloc[li]) for li in self.student.layer_indices}
            else:
                # Uniform or per_layer_keep_ratios override
                layer_n_keep = {}
                for li in self.student.layer_indices:
                    if self.per_layer_keep_ratios is not None and li in self.per_layer_keep_ratios:
                        li_ratio = self.per_layer_keep_ratios[li]
                    else:
                        li_ratio = self.keep_ratio
                    layer_n_keep[li] = max(1, int(round(n_img - (1.0 - li_ratio) * int(prompt_len))))

            # --- Step 3: build keep masks ---
            for li in self.student.layer_indices:
                li_n_keep = layer_n_keep[li]
                if li_n_keep >= n_img:
                    continue  # keep all tokens for this layer
                scores = all_scores[li]
                top = torch.topk(scores, k=li_n_keep, largest=True).indices
                mask = torch.ones(prompt_len, dtype=torch.bool)
                image_keep = torch.zeros(n_img, dtype=torch.bool)
                image_keep[top.cpu()] = True
                mask[image_positions.cpu()] = image_keep
                keep_masks[li] = mask

            past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

        # Match the original kvpress/direct-generate semantics: the first answer
        # token comes from prefill logits; the pruned KV affects subsequent decode.
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
