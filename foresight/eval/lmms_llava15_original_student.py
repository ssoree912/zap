# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lmms-eval wrapper for original-repo LLaVA-1.5-7B + student KV pruning.

This intentionally does not use the Transformers LLaVA wrapper.
It loads the original LLaVA checkpoint layout through `llava.model.builder`,
matching the teacher extraction/training path used for the original labels.
"""

from __future__ import annotations

import copy
import importlib.metadata as importlib_metadata
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

ZAP_ROOT = Path(os.environ.get("ZAP_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
QVIK_ROOT = Path(os.environ.get("QVIK_ROOT", ZAP_ROOT.parent / "Q-ViK")).resolve()
for _path in (str(QVIK_ROOT), str(ZAP_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _patch_vflowopt_transformers_version_checks() -> None:
    original_version = importlib_metadata.version

    def version(package_name: str) -> str:
        if package_name == "tokenizers":
            return "0.20.3"
        if package_name == "huggingface-hub":
            return "0.26.5"
        return original_version(package_name)

    importlib_metadata.version = version


def _patch_torch_load_legacy_bin_mmap() -> None:
    original_torch_load = torch.load

    def load(*args, **kwargs):
        retry_kwargs = dict(kwargs)
        while True:
            try:
                return original_torch_load(*args, **retry_kwargs)
            except RuntimeError as exc:
                if retry_kwargs.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry_kwargs.pop("mmap", None)
                    continue
                raise
            except pickle.UnpicklingError:
                if retry_kwargs.get("weights_only") is True:
                    retry_kwargs["weights_only"] = False
                    retry_kwargs.pop("mmap", None)
                    continue
                raise

    torch.load = load


_patch_vflowopt_transformers_version_checks()
_patch_torch_load_legacy_bin_mmap()

from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402

from .kv_decode_utils import (  # noqa: E402
    greedy_decode_with_kv,
    trim_kv_cache_per_layer,
)

from qvik.llava15.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from qvik.llava15.conversation import conv_templates
from qvik.llava15.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from qvik.llava15.model.builder import load_pretrained_model

try:
    from lmms_eval import utils
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover - import error is environment-specific.
    raise ImportError(
        "lmms_eval not found. Add /workspace/VFlowOpt/src/lmms_eval-0.2.4 to PYTHONPATH."
    ) from exc


def _flatten(items):
    return [child for item in items for child in item]


def _resolve_eos_token_id(tokenizer, model_config) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        cfg = getattr(model_config, "eos_token_id", None)
        eos = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    return int(2 if eos is None else eos)


def _infer_original_image_positions(
    input_ids: torch.Tensor,
    image_feature_len: int,
) -> tuple[torch.Tensor, int]:
    """Map raw IMAGE_TOKEN_INDEX placeholders to multimodal prompt positions."""
    raw_ids = input_ids[0].detach().cpu().tolist()
    image_positions: list[int] = []
    cursor = 0
    for token_id in raw_ids:
        if int(token_id) == IMAGE_TOKEN_INDEX:
            image_positions.extend(range(cursor, cursor + image_feature_len))
            cursor += image_feature_len
        else:
            cursor += 1
    if not image_positions:
        raise ValueError("No image placeholders found in input_ids.")
    return torch.tensor(image_positions, dtype=torch.long), cursor


def _decode_generated(tokenizer, sequences: torch.Tensor, input_len: int, max_new_tokens: int) -> str:
    ids = sequences[0].detach().cpu()
    # Original LLaVA generate() with inputs_embeds commonly returns only new
    # tokens; HF-style generate() may return prompt+new. Handle both.
    if ids.numel() > max_new_tokens + 1 and ids.numel() > input_len:
        ids = ids[input_len:]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()


@register_model("llava15_original_student")
class Llava15OriginalStudent(lmms):
    """Original LLaVA-1.5-7B with student-scored image-token KV pruning."""

    def __init__(
        self,
        pretrained: str = "/workspace/zap/ckpts/llava-v1.5-7b",
        student_path: str = (
            "/workspace/zap/artifacts/original_llava_teacher/"
            "student_llava15_original_future_1800_lr1e4_15ep"
        ),
        vision_tower_path: str = "",
        keep_ratio: float = 0.5,
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        model_name: Optional[str] = None,
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 32,
        image_feature_len: int = 576,
        grid_h: int = 24,
        grid_w: int = 24,
        stats_output_dir: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model_args for llava15_original_student: {kwargs}")

        self._device = torch.device(device)
        self.device_map = device_map
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("Original LLaVA student wrapper only supports batch_size=1.")

        resolved_model_name = model_name or get_model_name_from_path(pretrained)
        (
            self._tokenizer,
            self._model,
            self._image_processor,
            self._max_length,
        ) = load_pretrained_model(
            pretrained,
            None,
            resolved_model_name,
            device_map=device_map,
        )

        self._model.eval()
        if device_map != "auto":
            self._model.to(self._device)
        try:
            self._model.tie_weights()
        except Exception:
            pass
        self._config = self._model.config

        self._model_dtype = next(self._model.parameters()).dtype
        self.student = VisualUtilityStudent.from_pretrained(student_path)
        self.student = self.student.to(device=self._device, dtype=self._model_dtype).eval()
        self.student_path = student_path
        self.keep_ratio = float(keep_ratio)
        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        self.image_feature_len = int(image_feature_len)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.stats_output_dir = stats_output_dir
        self._rank = 0
        self._world_size = 1
        self._reported_keep_budget = False
        self._img_keep_sum = 0
        self._img_total_sum = 0
        self._img_sample_count = 0
        self._keep_stats: list[dict] = []

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
        return self._max_length

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
        try:
            return self._tokenizer.decode(tokens)
        except TypeError:
            return self._tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for Llava15OriginalStudent.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation is not implemented.")

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

            batched_visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            flattened_visuals = _flatten(batched_visuals)

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs.pop("until", None)
            max_new_tokens = int(gen_kwargs.pop("max_new_tokens", self.max_new_tokens))

            prompt = self._build_prompt(contexts[0], batched_visuals[0])
            image_tensor, image_sizes = self._prepare_images(flattened_visuals)
            input_ids = tokenizer_image_token(
                prompt,
                self._tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self._device)

            output = self._generate_with_student(
                input_ids=input_ids,
                image_tensor=image_tensor,
                image_sizes=image_sizes,
                num_images=len(flattened_visuals),
                max_new_tokens=max_new_tokens,
            )
            res.append(output)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        self._save_keep_stats(task_name)
        return res

    def _build_prompt(self, context: str, visuals: list) -> str:
        question = context
        if visuals and DEFAULT_IMAGE_TOKEN not in question:
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
            question = f"{image_tokens}\n{question}"

        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _prepare_images(self, visuals: list):
        if not visuals:
            return None, []
        image_sizes = [image.size for image in visuals]
        image_tensor = process_images(visuals, self._image_processor, self._config)
        if isinstance(image_tensor, list):
            image_tensor = [
                tensor.to(dtype=self._model_dtype, device=self._device) for tensor in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(dtype=self._model_dtype, device=self._device)
        return image_tensor, image_sizes

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
            f"[llava15-original-student] stats saved -> {fpath} "
            f"(avg_total={summary['avg_total_keep_ratio']:.4f} "
            f"avg_image={summary['avg_image_keep_ratio']:.4f})",
            file=sys.stderr,
            flush=True,
        )

    @torch.no_grad()
    def _generate_with_student(
        self,
        *,
        input_ids: torch.Tensor,
        image_tensor,
        image_sizes: list,
        num_images: int,
        max_new_tokens: int,
    ) -> str:
        eos_token_id = _resolve_eos_token_id(self._tokenizer, self._model.config)
        pad_token_id = self._tokenizer.pad_token_id or eos_token_id
        modalities = ["image"] * max(1, num_images)

        def _safe_generate() -> str:
            out = self._model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long),
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            sequences = out.sequences if hasattr(out, "sequences") else out
            return _decode_generated(
                self._tokenizer,
                sequences,
                input_len=int(input_ids.shape[1]),
                max_new_tokens=max_new_tokens,
            )

        if image_tensor is None or num_images == 0 or self.keep_ratio >= 1.0:
            return _safe_generate()

        try:
            prefill = self._model(
                input_ids=input_ids,
                images=image_tensor,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as exc:
            print(
                f"[llava15-original-student] WARNING: prefill failed ({exc}); falling back.",
                file=sys.stderr,
                flush=True,
            )
            return _safe_generate()

        H_all = prefill.hidden_states
        past_kv = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        try:
            image_positions, prompt_len = _infer_original_image_positions(
                input_ids,
                self.image_feature_len,
            )
        except ValueError:
            answer_ids = greedy_decode_with_kv(
                self._model,
                past_kv,
                next_token,
                prompt_len=int(H_all[-1].shape[1]),
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
            )
            return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

        actual_prompt_len = int(H_all[-1].shape[1])
        if actual_prompt_len != int(prompt_len):
            print(
                f"[llava15-original-student] WARNING: prompt_len mismatch "
                f"inferred={prompt_len} actual={actual_prompt_len}; using actual for decode.",
                file=sys.stderr,
                flush=True,
            )
            prompt_len = actual_prompt_len

        n_img = int(image_positions.numel())
        n_text = int(prompt_len) - n_img
        n_keep = max(1, int(round(n_img - (1.0 - self.keep_ratio) * int(prompt_len))))
        n_keep = min(n_keep, n_img)

        self._img_keep_sum += n_keep
        self._img_total_sum += n_img
        self._img_sample_count += 1
        self._keep_stats.append(
            {
                "n_image_original": n_img,
                "n_image_kept": n_keep,
                "n_text": n_text,
                "prompt_len": int(prompt_len),
                "image_token_ratio": n_img / max(1, int(prompt_len)),
                "text_token_ratio": n_text / max(1, int(prompt_len)),
                "total_keep_ratio": (n_text + n_keep) / max(1, int(prompt_len)),
                "image_keep_ratio": n_keep / max(1, n_img),
            }
        )
        if not self._reported_keep_budget:
            print(
                f"[llava15-original-student] keep_ratio_basis=total keep_ratio={self.keep_ratio} "
                f"prompt_len={int(prompt_len)} image_tokens={n_img} text_tokens={n_text} "
                f"image_tokens_kept={n_keep}",
                file=sys.stderr,
                flush=True,
            )
            self._reported_keep_budget = True
        if self._img_sample_count % 200 == 0:
            avg_img_ratio = self._img_keep_sum / max(1, self._img_total_sum)
            print(
                f"[llava15-original-student] img_keep_ratio_avg={avg_img_ratio:.4f} "
                f"({self._img_keep_sum}/{self._img_total_sum}) "
                f"over {self._img_sample_count} samples",
                file=sys.stderr,
                flush=True,
            )

        last_img = int(image_positions.max().item())
        q_positions = (
            torch.arange(last_img + 1, int(prompt_len), dtype=torch.long)
            if last_img + 1 < int(prompt_len)
            else torch.empty(0, dtype=torch.long)
        )
        image_idx_dev = image_positions.to(self._device)
        q_idx_dev = q_positions.to(self._device)

        keep_masks: dict[int, torch.Tensor] = {}
        for layer_idx in self.student.layer_indices:
            H_l = H_all[layer_idx + 1]
            scores = self.student.layers[str(layer_idx)](
                H_l,
                image_idx_dev,
                q_idx_dev,
                self.grid_h,
                self.grid_w,
            ).squeeze(0)
            if n_keep >= n_img:
                continue
            top = torch.topk(scores, k=n_keep, largest=True).indices
            mask = torch.ones(int(prompt_len), dtype=torch.bool)
            image_keep = torch.zeros(n_img, dtype=torch.bool)
            image_keep[top.detach().cpu()] = True
            mask[image_positions] = image_keep
            keep_masks[layer_idx] = mask

        del H_all
        past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
        answer_ids = greedy_decode_with_kv(
            self._model,
            past_kv,
            next_token,
            prompt_len=int(prompt_len),
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.empty_cache()
        return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
