# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lmms-eval model wrapper: LLaVA-1.5-7B + student-driven KV pruning.

Register as `llava15_student` via @register_model. The launcher (eval.py)
monkey-patches this module into the lmms-eval models package before running.

Example CLI (via eval.py):
    CUDA_VISIBLE_DEVICES=2 python eval.py --model llava15 --framework lmms -- \\
        --model llava15_student \\
        --model_args pretrained=/workspace/zap/ckpts/llava-1.5-7b-hf,student_path=/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4,keep_ratio=0.3 \\
        --tasks chartqa_local \\
        --batch_size 1 \\
        --output_path results/
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, "/workspace/zap")

from kvpress.presses.visual_utility_student import VisualUtilityStudent
from ..llava_15b_extractor import (
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)
from .vlmeval_llava15_student import (
    _greedy_decode_with_kv,
    _resolve_eos_token_id,
    _trim_kv_cache_per_layer,
)

try:
    from lmms_eval import utils
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
except ImportError as e:
    raise ImportError(
        "lmms_eval not found. Add VFlowOpt/src/lmms_eval-0.2.4 to PYTHONPATH."
    ) from e

DEFAULT_IMAGE_TOKEN = "<image>"


@register_model("llava15_student")
class Llava15Student(lmms):
    """LLaVA-1.5-7B with student-scored image-token KV pruning for lmms-eval."""

    def __init__(
        self,
        pretrained: str = "/workspace/zap/ckpts/llava-1.5-7b-hf",
        student_path: str = "/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4",
        keep_ratio: float = 0.5,
        device: str = "cuda:0",
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        grid_h: int = 24,
        grid_w: int = 24,
        stats_output_dir: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "Only batch_size=1 is supported."

        self._model = LlavaForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(self._device).eval()

        self._processor = AutoProcessor.from_pretrained(pretrained)
        configure_llava_processor(self._processor, self._model.config)
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

        self.student = VisualUtilityStudent.from_pretrained(student_path)
        self.student = self.student.to(device=self._device, dtype=torch.float16).eval()
        self.keep_ratio = float(keep_ratio)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.stats_output_dir = stats_output_dir
        self._reported_keep_budget = False
        self._img_keep_sum = 0
        self._img_total_sum = 0
        self._img_sample_count = 0
        self._keep_stats: list[dict] = []
        self._rank = 0
        self._world_size = 1

    # --- lmms interface properties ---

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._config, "max_position_embeddings", 4096)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood not implemented for Llava15Student")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation not implemented for Llava15Student")

    # --- main generation ---

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        task_name = None

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (len(requests) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        self._keep_stats.clear()
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            if task_name is None:
                task_name = task
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = [img for sublist in visuals for img in sublist]

            gen_kwargs = all_gen_kwargs[0]
            gen_kwargs.pop("until", None)
            max_new_tokens = gen_kwargs.get("max_new_tokens", 32)

            context = contexts[0]
            # Strip any <image> tokens embedded by lmms-eval — LLaVA 1.5 uses
            # a fixed template that places <image> before the question text.
            clean_context = context.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if visuals:
                text = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                    f"USER: {DEFAULT_IMAGE_TOKEN}\n{clean_context} ASSISTANT:"
                )
            else:
                text = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                    f"USER: {clean_context} ASSISTANT:"
                )

            inputs = self._processor(
                images=visuals if visuals else None,
                text=text,
                return_tensors="pt",
            ).to(self._device, torch.float16)

            output = self._generate_with_student(inputs, visuals, max_new_tokens)
            res.append(output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        self._save_keep_stats(task_name)
        return res

    def _save_keep_stats(self, task_name: Optional[str]) -> None:
        if not self._keep_stats:
            return
        n = len(self._keep_stats)
        summary = {
            "task": task_name,
            "keep_ratio": self.keep_ratio,
            "n_samples": n,
            "avg_image_token_ratio": sum(s["image_token_ratio"] for s in self._keep_stats) / n,
            "avg_text_token_ratio": sum(s["text_token_ratio"] for s in self._keep_stats) / n,
            "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in self._keep_stats) / n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in self._keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in self._keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in self._keep_stats) / n,
            "samples": self._keep_stats,
        }
        out_dir = self.stats_output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{task_name or 'unknown'}_keep_ratio_stats.json"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[keep-ratio] stats saved → {fpath} "
            f"(avg_total={summary['avg_total_keep_ratio']:.4f} avg_image={summary['avg_image_keep_ratio']:.4f})",
            file=sys.stderr, flush=True,
        )

    @torch.no_grad()
    def _generate_with_student(self, inputs, visuals, max_new_tokens: int) -> str:
        def _safe_generate():
            try:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                return self._tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            except ValueError as e:
                print(f"[lmms-llava15-student] WARNING: generate() fallback failed ({e}), skipping sample.",
                      file=sys.stderr, flush=True)
                return ""

        if not visuals or self.keep_ratio >= 1.0:
            return _safe_generate()

        try:
            prefill = self._model(
                **inputs,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
        except ValueError:
            return _safe_generate()

        H_all = prefill.hidden_states
        past_kv = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        eos_token_id = _resolve_eos_token_id(self._processor, self._model.config)

        try:
            image_positions, prompt_len = infer_llava_image_positions_no_forward(
                prompt_inputs={"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                model_config=self._model.config,
                num_images=len(visuals),
            )
        except ValueError:
            return self._tokenizer.decode(
                _greedy_decode_with_kv(
                    self._model, past_kv, next_token,
                    prompt_len=int(inputs["input_ids"].shape[1]),
                    eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
                ).tolist(), skip_special_tokens=True
            ).strip()

        n_img = int(image_positions.numel())
        n_text = int(prompt_len) - n_img
        n_keep = max(1, int(round(n_img - (1.0 - self.keep_ratio) * int(prompt_len))))

        self._img_keep_sum += n_keep
        self._img_total_sum += n_img
        self._img_sample_count += 1
        self._keep_stats.append({
            "n_image_original": n_img,
            "n_image_kept": n_keep,
            "n_text": n_text,
            "prompt_len": int(prompt_len),
            "image_token_ratio": n_img / max(1, int(prompt_len)),
            "text_token_ratio": n_text / max(1, int(prompt_len)),
            "total_keep_ratio": (n_text + n_keep) / max(1, int(prompt_len)),
            "image_keep_ratio": n_keep / max(1, n_img),
        })
        if not self._reported_keep_budget:
            print(
                f"[lmms-llava15-student] keep_ratio_basis=total keep_ratio={self.keep_ratio} "
                f"prompt_len={int(prompt_len)} image_tokens={n_img} text_tokens={n_text} "
                f"image_tokens_kept={n_keep}",
                file=sys.stderr, flush=True,
            )
            self._reported_keep_budget = True
        if self._img_sample_count % 200 == 0:
            avg_img_ratio = self._img_keep_sum / max(1, self._img_total_sum)
            print(
                f"[lmms-llava15-student] img_keep_ratio_avg={avg_img_ratio:.4f} "
                f"({self._img_keep_sum}/{self._img_total_sum}) over {self._img_sample_count} samples",
                file=sys.stderr, flush=True,
            )

        last_img = int(image_positions.max().item())
        q_positions = (
            torch.arange(last_img + 1, prompt_len, dtype=torch.long)
            if last_img + 1 < prompt_len else torch.empty(0, dtype=torch.long)
        )
        image_idx_dev = image_positions.to(self._device)
        q_idx_dev = q_positions.to(self._device)

        keep_masks: dict[int, torch.Tensor] = {}
        for li in self.student.layer_indices:
            H_l = H_all[li + 1]
            scores = self.student.layers[str(li)](
                H_l, image_idx_dev, q_idx_dev, self.grid_h, self.grid_w
            ).squeeze(0)
            if n_keep >= n_img:
                continue
            top = torch.topk(scores, k=n_keep, largest=True).indices
            mask = torch.ones(prompt_len, dtype=torch.bool)
            image_keep = torch.zeros(n_img, dtype=torch.bool)
            image_keep[top.cpu()] = True
            mask[image_positions.cpu()] = image_keep
            keep_masks[li] = mask

        del H_all

        past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)
        answer_ids = _greedy_decode_with_kv(
            self._model, past_kv, next_token,
            prompt_len=int(prompt_len),
            eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
        )
        torch.cuda.empty_cache()
        return self._processor.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
