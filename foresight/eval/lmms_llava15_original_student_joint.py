# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint text+image KV-eviction baseline for original-repo LLaVA-1.5-7B.

This is an ablation variant of `Llava15OriginalStudent`. The main method keeps
all text tokens and only prunes image tokens; this variant ranks BOTH text and
image prompt tokens under a shared budget.

Per-modality scoring:
- image tokens: student-predicted scores ŝ^(l)
- text tokens: average prefill attention received from question (post-image)
  tokens, with causal masking handled by counting only valid (q, p) pairs

Both modalities are rank-normalized within layer, concatenated into a unified
prompt-token score, and the top-(keep_ratio · prompt_len) entries are kept.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

ZAP_ROOT = Path("/workspace/zap")
VFLOWOPT_LLAVA_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
for _path in (str(ZAP_ROOT), str(VFLOWOPT_LLAVA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from .kv_decode_utils import greedy_decode_with_kv, trim_kv_cache_per_layer  # noqa: E402
from .lmms_llava15_original_student import (  # noqa: E402
    Llava15OriginalStudent,
    _infer_original_image_positions,
    _resolve_eos_token_id,
)

try:
    from lmms_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover - import error is environment-specific.
    raise ImportError(
        "lmms_eval not found. Add /workspace/VFlowOpt/src/lmms_eval-0.2.4 to PYTHONPATH."
    ) from exc


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Map values to ranks in [0, 1]. Higher input -> higher rank."""
    if x.numel() <= 1:
        return torch.zeros_like(x, dtype=torch.float32)
    ranks = x.float().argsort(stable=True).argsort(stable=True).float()
    return ranks / float(x.numel() - 1)


def _text_scores_from_attention(
    attn_weights_l: torch.Tensor,
    q_positions: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Average prefill attention received by each prompt position from query tokens.

    attn_weights_l: [B=1, N_h, T, T] for layer l.
    q_positions:    [N_Q] long indices of question tokens (after last image).
    Returns:        [prompt_len] mean attention received per prompt position.

    Causal masking is handled by dividing each column-sum by the number of
    query rows whose absolute position >= the column position.
    """
    if q_positions.numel() == 0:
        return torch.zeros(prompt_len, dtype=torch.float32, device=attn_weights_l.device)

    attn = attn_weights_l[0].mean(dim=0)  # [T, T] (head mean)
    rows = attn[q_positions]  # [N_Q, T]
    rows = rows[:, :prompt_len]  # restrict to prompt range
    score_sum = rows.sum(dim=0)  # [prompt_len]

    cols = torch.arange(prompt_len, device=q_positions.device)
    valid = q_positions.unsqueeze(1) >= cols.unsqueeze(0)  # [N_Q, prompt_len]
    valid_count = valid.sum(dim=0).clamp(min=1).float()
    return score_sum / valid_count


@register_model("llava15_original_student_joint")
class Llava15OriginalStudentJoint(Llava15OriginalStudent):
    """Joint text+image KV-pruning ablation.

    Keeps ``round(keep_ratio * prompt_len)`` total prompt tokens at every
    pruned layer, ranking text via prefill attention and image via student.
    """

    def __init__(
        self,
        attn_implementation: str = "eager",
        score_normalize: str = "rank",
        **kwargs,
    ) -> None:
        # eager attention is required so prefill returns full attention weights.
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        if score_normalize not in {"rank", "minmax", "none"}:
            raise ValueError(f"unsupported score_normalize={score_normalize!r}")
        self.score_normalize = score_normalize

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
            from .lmms_llava15_original_student import _decode_generated  # local import

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
                image_sizes=image_sizes,
                modalities=modalities,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
        except Exception as exc:
            print(
                f"[llava15-joint] WARNING: prefill failed ({exc}); falling back.",
                file=sys.stderr,
                flush=True,
            )
            return _safe_generate()

        H_all = prefill.hidden_states
        attns = prefill.attentions
        past_kv = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if attns is None:
            print(
                "[llava15-joint] WARNING: attentions are None; cannot score text. "
                "Falling back to safe generate.",
                file=sys.stderr,
                flush=True,
            )
            return _safe_generate()

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
                f"[llava15-joint] WARNING: prompt_len mismatch "
                f"inferred={prompt_len} actual={actual_prompt_len}; using actual.",
                file=sys.stderr,
                flush=True,
            )
            prompt_len = actual_prompt_len

        n_img = int(image_positions.numel())
        n_text = int(prompt_len) - n_img
        n_keep_total = max(1, int(round(self.keep_ratio * int(prompt_len))))
        n_keep_total = min(n_keep_total, int(prompt_len))

        # text/image position bookkeeping
        all_idx = torch.arange(int(prompt_len), dtype=torch.long)
        image_set = torch.zeros(int(prompt_len), dtype=torch.bool)
        image_set[image_positions] = True
        text_positions = all_idx[~image_set]
        n_text_actual = int(text_positions.numel())
        assert n_text_actual == n_text, (n_text_actual, n_text)

        last_img = int(image_positions.max().item())
        q_positions = (
            torch.arange(last_img + 1, int(prompt_len), dtype=torch.long)
            if last_img + 1 < int(prompt_len)
            else torch.empty(0, dtype=torch.long)
        )
        image_idx_dev = image_positions.to(self._device)
        q_idx_dev = q_positions.to(self._device)
        text_idx_dev = text_positions.to(self._device)

        self._img_keep_sum += 0  # placeholder; updated below per layer
        self._img_total_sum += n_img
        self._img_sample_count += 1

        if not self._reported_keep_budget:
            print(
                f"[llava15-joint] keep_ratio_basis=total keep_ratio={self.keep_ratio} "
                f"prompt_len={int(prompt_len)} image_tokens={n_img} text_tokens={n_text} "
                f"keep_total={n_keep_total} score_normalize={self.score_normalize}",
                file=sys.stderr,
                flush=True,
            )
            self._reported_keep_budget = True

        # Per-layer joint top-k
        keep_masks: dict[int, torch.Tensor] = {}
        per_sample_text_kept = 0
        per_sample_image_kept = 0
        for layer_idx in self.student.layer_indices:
            H_l = H_all[layer_idx + 1]
            img_scores = self.student.layers[str(layer_idx)](
                H_l,
                image_idx_dev,
                q_idx_dev,
                self.grid_h,
                self.grid_w,
            ).squeeze(0)  # [N_I]

            attn_l = attns[layer_idx]  # [B, N_h, T, T]
            text_scores_full = _text_scores_from_attention(
                attn_l, q_idx_dev, int(prompt_len)
            )  # [prompt_len]
            text_scores = text_scores_full[text_idx_dev]  # [N_text]

            if self.score_normalize == "rank":
                img_norm = _rank_normalize(img_scores)
                text_norm = _rank_normalize(text_scores)
            elif self.score_normalize == "minmax":
                img_norm = _minmax(img_scores)
                text_norm = _minmax(text_scores)
            else:  # "none"
                img_norm = img_scores.float()
                text_norm = text_scores.float()

            unified = torch.full(
                (int(prompt_len),),
                float("-inf"),
                dtype=torch.float32,
                device=img_norm.device,
            )
            unified[image_idx_dev] = img_norm
            unified[text_idx_dev] = text_norm

            top = torch.topk(unified, k=n_keep_total, largest=True).indices
            mask = torch.zeros(int(prompt_len), dtype=torch.bool)
            mask[top.detach().cpu()] = True
            keep_masks[layer_idx] = mask

            kept_in_layer = int(mask.sum().item())
            kept_image_in_layer = int(mask[image_positions].sum().item())
            per_sample_text_kept += kept_in_layer - kept_image_in_layer
            per_sample_image_kept += kept_image_in_layer

        n_layers = max(1, len(self.student.layer_indices))
        avg_text_kept = per_sample_text_kept / n_layers
        avg_image_kept = per_sample_image_kept / n_layers
        self._img_keep_sum += avg_image_kept
        self._keep_stats.append(
            {
                "n_image_original": n_img,
                "n_image_kept": avg_image_kept,
                "n_text": n_text,
                "n_text_kept": avg_text_kept,
                "prompt_len": int(prompt_len),
                "image_token_ratio": n_img / max(1, int(prompt_len)),
                "text_token_ratio": n_text / max(1, int(prompt_len)),
                "total_keep_ratio": (avg_text_kept + avg_image_kept) / max(1, int(prompt_len)),
                "image_keep_ratio": avg_image_kept / max(1, n_img),
                "text_keep_ratio": avg_text_kept / max(1, n_text),
                "n_keep_total": n_keep_total,
            }
        )

        del H_all
        del attns
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


def _minmax(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    lo = x.min()
    hi = x.max()
    if (hi - lo).abs() < 1e-12:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)
