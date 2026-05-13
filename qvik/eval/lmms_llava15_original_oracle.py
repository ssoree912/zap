# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lmms-eval wrapper for original-repo LLaVA-1.5-7B + future-oracle KV pruning.

The "Future Oracle" is a privileged upper bound:
  1. Run full-cache prefill + decode and collect answer-to-image attention
     per-layer (mean over heads, mean over decode steps). This produces a
     teacher score for every image KV position.
  2. Throw away the full-cache answer. Re-prefill the same prompt to obtain a
     fresh KV cache.
  3. Prune image KV per layer using top-k teacher scores under the same
     `n_keep` budget that the student wrapper uses. Text KV is preserved.
  4. Greedy-decode from the pruned cache and return THAT newly generated
     answer for evaluation.

Step (4) is the metric target: we never evaluate the full-cache answer.
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
VFLOWOPT_LLAVA_ROOT = Path(
    os.environ.get(
        "VFLOWOPT_LLAVA_ROOT",
        ZAP_ROOT.parent / "VFlowOpt_llava1.5/src/LLaVA-OneVision",
    )
).resolve()
VFLOWOPT_TRANSFORMERS_ROOT = Path(
    os.environ.get(
        "VFLOWOPT_TRANSFORMERS_ROOT",
        ZAP_ROOT.parent / "VFlowOpt_llava1.5/src/transformers-4.46.0/src",
    )
).resolve()
for _path in (str(ZAP_ROOT), str(VFLOWOPT_TRANSFORMERS_ROOT), str(VFLOWOPT_LLAVA_ROOT)):
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

from .kv_decode_utils import (  # noqa: E402
    greedy_decode_with_kv,
    trim_kv_cache_per_layer,
)

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except Exception as exc:  # pragma: no cover - import error is environment-specific.
    raise ImportError(
        "Original LLaVA package is required. Expected it under "
        "/workspace/VFlowOpt/src/LLaVA-OneVision."
    ) from exc

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


@register_model("llava15_original_oracle")
class Llava15OriginalOracle(lmms):
    """Original LLaVA-1.5-7B with future-oracle (teacher-score) KV pruning."""

    def __init__(
        self,
        pretrained: str = "/workspace/zap/model/llava-1.5-7b-hf",
        vision_tower_path: str = "",
        keep_ratio: float = 0.5,
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        model_name: Optional[str] = None,
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        attn_implementation: str = "eager",
        max_new_tokens: int = 32,
        teacher_max_new_tokens: int = 0,
        image_feature_len: int = 576,
        stats_output_dir: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model_args for llava15_original_oracle: {kwargs}")

        self._device = torch.device(device)
        self.device_map = device_map
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("Original LLaVA oracle wrapper only supports batch_size=1.")
        if attn_implementation != "eager":
            raise ValueError(
                "Future Oracle requires attn_implementation='eager' so that "
                "output_attentions=True is supported during the teacher pass."
            )

        llava_model_args = {"multimodal": True, "attn_implementation": attn_implementation}
        if vision_tower_path:
            llava_model_args["overwrite_config"] = {"mm_vision_tower": vision_tower_path}
        resolved_model_name = model_name or get_model_name_from_path(pretrained)
        try:
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
                **llava_model_args,
            )
        except TypeError:
            llava_model_args.pop("multimodal", None)
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
                **llava_model_args,
            )

        self._model.eval()
        if device_map != "auto":
            self._model.to(self._device)
        try:
            self._model.tie_weights()
        except Exception:
            pass
        self._config = self._model.config

        self.keep_ratio = float(keep_ratio)
        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        # Teacher pass max_new_tokens defaults to the eval pass length so the
        # privileged signal exactly mirrors the answer being evaluated.
        self.teacher_max_new_tokens = int(teacher_max_new_tokens) or int(max_new_tokens)
        self.image_feature_len = int(image_feature_len)
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
        raise NotImplementedError("loglikelihood is not implemented for Llava15OriginalOracle.")

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

            output = self._generate_with_oracle(
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
                tensor.to(dtype=torch.float16, device=self._device) for tensor in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(dtype=torch.float16, device=self._device)
        return image_tensor, image_sizes

    def _save_keep_stats(self, task_name: Optional[str]) -> None:
        if not self._keep_stats:
            return
        n = len(self._keep_stats)
        summary = {
            "task": task_name,
            "keep_ratio": self.keep_ratio,
            "method": "future_oracle",
            "n_samples": n,
            "avg_image_token_ratio": sum(s["image_token_ratio"] for s in self._keep_stats) / n,
            "avg_text_token_ratio": sum(s["text_token_ratio"] for s in self._keep_stats) / n,
            "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in self._keep_stats) / n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in self._keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in self._keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in self._keep_stats) / n,
            "avg_teacher_steps": sum(s["teacher_steps"] for s in self._keep_stats) / n,
            "samples": self._keep_stats,
        }
        out_dir = self.stats_output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{task_name or 'unknown'}_keep_ratio_stats.json"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[llava15-original-oracle] stats saved -> {fpath} "
            f"(avg_total={summary['avg_total_keep_ratio']:.4f} "
            f"avg_image={summary['avg_image_keep_ratio']:.4f} "
            f"teacher_T_mean={summary['avg_teacher_steps']:.2f})",
            file=sys.stderr,
            flush=True,
        )

    @torch.no_grad()
    def _collect_teacher_scores(
        self,
        *,
        input_ids: torch.Tensor,
        image_tensor,
        image_sizes: list,
        modalities: list,
        image_positions: torch.Tensor,
        eos_token_id: int,
        teacher_max_new_tokens: int,
    ) -> tuple[torch.Tensor, int]:
        """Run full-cache prefill + per-step decode with output_attentions=True.

        Returns (teacher [L, n_img] fp32 cpu, T) where T is decode steps used.
        Aggregation: mean over heads, mean over decode steps.
        """
        prefill = self._model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=modalities,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        past_kv = prefill.past_key_values
        prompt_len_actual = int(prefill.logits.shape[1])
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del prefill

        n_img = int(image_positions.numel())
        image_idx_dev = image_positions.to(self._device)

        teacher: torch.Tensor | None = None
        L_layers: int | None = None
        T = 0
        pos = int(prompt_len_actual)
        cache_pos = torch.zeros(1, dtype=torch.long, device=next_token.device)

        for _ in range(int(teacher_max_new_tokens)):
            cache_pos[0] = pos
            step = self._model(
                input_ids=next_token,
                past_key_values=past_kv,
                cache_position=cache_pos,
                position_ids=cache_pos.unsqueeze(0),
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_kv = step.past_key_values
            attentions = step.attentions
            if L_layers is None:
                L_layers = len(attentions)
                teacher = torch.zeros(L_layers, n_img, dtype=torch.float32)
            for l_idx in range(L_layers):
                # attentions[l]: [1, H, 1, T_now] — gather image positions, mean heads
                a = attentions[l_idx][0, :, -1, :].index_select(dim=-1, index=image_idx_dev)
                teacher[l_idx] += a.float().mean(dim=0).cpu()
            T += 1

            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del step
            pos += 1
            if int(next_token.item()) == eos_token_id:
                break

        if T == 0 or teacher is None:
            raise RuntimeError("Teacher decode loop produced zero steps.")
        teacher /= float(T)

        del past_kv
        torch.cuda.empty_cache()
        return teacher, T

    @torch.no_grad()
    def _generate_with_oracle(
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
                images=image_tensor,
                image_sizes=image_sizes,
                modalities=modalities,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            sequences = out.sequences if hasattr(out, "sequences") else out
            ids = sequences[0].detach().cpu()
            in_len = int(input_ids.shape[1])
            if ids.numel() > max_new_tokens + 1 and ids.numel() > in_len:
                ids = ids[in_len:]
            return self._tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()

        if image_tensor is None or num_images == 0 or self.keep_ratio >= 1.0:
            return _safe_generate()

        try:
            image_positions, _inferred_prompt_len = _infer_original_image_positions(
                input_ids,
                self.image_feature_len,
            )
        except ValueError:
            return _safe_generate()

        # ── Pass 1: collect teacher scores via full-cache decode ──────────
        try:
            teacher, T_teacher = self._collect_teacher_scores(
                input_ids=input_ids,
                image_tensor=image_tensor,
                image_sizes=image_sizes,
                modalities=modalities,
                image_positions=image_positions,
                eos_token_id=eos_token_id,
                teacher_max_new_tokens=self.teacher_max_new_tokens,
            )
        except Exception as exc:
            print(
                f"[llava15-original-oracle] WARNING: teacher pass failed ({exc}); "
                f"falling back to full-cache generate.",
                file=sys.stderr,
                flush=True,
            )
            return _safe_generate()

        # ── Pass 2: fresh prefill for the eviction decode ─────────────────
        try:
            prefill = self._model(
                input_ids=input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                modalities=modalities,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as exc:
            print(
                f"[llava15-original-oracle] WARNING: re-prefill failed ({exc}); "
                f"falling back to full-cache generate.",
                file=sys.stderr,
                flush=True,
            )
            return _safe_generate()

        past_kv = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prompt_len = int(prefill.logits.shape[1])
        del prefill

        n_img = int(image_positions.numel())
        n_text = prompt_len - n_img
        n_keep = max(1, int(round(n_img - (1.0 - self.keep_ratio) * prompt_len)))
        n_keep = min(n_keep, n_img)

        self._img_keep_sum += n_keep
        self._img_total_sum += n_img
        self._img_sample_count += 1
        self._keep_stats.append(
            {
                "n_image_original": n_img,
                "n_image_kept": n_keep,
                "n_text": n_text,
                "prompt_len": prompt_len,
                "teacher_steps": T_teacher,
                "image_token_ratio": n_img / max(1, prompt_len),
                "text_token_ratio": n_text / max(1, prompt_len),
                "total_keep_ratio": (n_text + n_keep) / max(1, prompt_len),
                "image_keep_ratio": n_keep / max(1, n_img),
            }
        )
        if not self._reported_keep_budget:
            print(
                f"[llava15-original-oracle] keep_ratio_basis=total keep_ratio={self.keep_ratio} "
                f"prompt_len={prompt_len} image_tokens={n_img} text_tokens={n_text} "
                f"image_tokens_kept={n_keep} teacher_T={T_teacher}",
                file=sys.stderr,
                flush=True,
            )
            self._reported_keep_budget = True

        # ── Build per-layer keep masks from teacher scores ────────────────
        L_layers = teacher.shape[0]
        keep_masks: dict[int, torch.Tensor] = {}
        if n_keep < n_img:
            for layer_idx in range(L_layers):
                scores = teacher[layer_idx]  # [n_img] cpu fp32
                top = torch.topk(scores, k=n_keep, largest=True).indices
                mask = torch.ones(prompt_len, dtype=torch.bool)
                image_keep = torch.zeros(n_img, dtype=torch.bool)
                image_keep[top] = True
                mask[image_positions] = image_keep
                keep_masks[layer_idx] = mask

        past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
        answer_ids = greedy_decode_with_kv(
            self._model,
            past_kv,
            next_token,
            prompt_len=prompt_len,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.empty_cache()
        return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
