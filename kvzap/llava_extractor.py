# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLaVA analysis data extraction utilities.

This module implements a two-pass extraction pipeline for LLaVA models. For each
image/question sample, it can:
1. Generate an answer in text space
2. Run a second analysis forward pass with hidden states and attentions enabled
3. Extract prompt/answer splits in multimodal space
4. Save per-layer attention blocks, hidden states, W_O, and optional ||W_O v_i||
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.integrations.finegrained_fp8 import FP8Linear
from transformers.models.llama.modeling_llama import repeat_kv


def _parse_torch_dtype(dtype: str | None) -> torch.dtype | str:
    if dtype is None or dtype == "auto":
        return "auto"
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    parsed = getattr(torch, dtype)
    if not isinstance(parsed, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return parsed


def _sanitize_sample_id(value: Any, idx: int) -> str:
    text = str(value) if value is not None else f"sample-{idx:06d}"
    text = text.strip() or f"sample-{idx:06d}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:128]


def _load_json_records(dataset_path: Path) -> list[dict[str, Any]]:
    if dataset_path.suffix == ".jsonl":
        records = []
        with dataset_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if dataset_path.suffix == ".json":
        with dataset_path.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("JSON dataset must be either a list of records or a dict with a `data` list")

    raise ValueError("Unsupported dataset format. Expected `.jsonl` or `.json`.")


def load_llava_samples(
    dataset_path: str,
    image_root: str | None = None,
    question_column: str = "question",
    image_column: str = "image_path",
    id_column: str = "sample_id",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset_file = Path(dataset_path).resolve()
    image_base = Path(image_root).resolve() if image_root is not None else dataset_file.parent
    raw_records = _load_json_records(dataset_file)

    samples = []
    for idx, record in enumerate(raw_records):
        if question_column not in record:
            raise KeyError(f"Missing `{question_column}` in sample #{idx}")

        image_value = record.get(image_column)
        if image_value is None and image_column != "image_path":
            image_value = record.get("image_path")
        if image_value is None and image_column != "image":
            image_value = record.get("image")
        if image_value is None:
            raise KeyError(f"Missing image path field `{image_column}` in sample #{idx}")

        image_path = Path(str(image_value))
        if not image_path.is_absolute():
            image_path = image_base / image_path

        sample = {
            "sample_id": _sanitize_sample_id(record.get(id_column), idx),
            "question": str(record[question_column]),
            "image_path": str(image_path.resolve()),
            "raw": record,
        }
        samples.append(sample)

        if limit is not None and len(samples) >= limit:
            break

    return samples


def _get_model_device(model: LlavaForConditionalGeneration) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_model_float_dtype(model: LlavaForConditionalGeneration) -> torch.dtype | None:
    for parameter in model.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype
    return None


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device, float_dtype: torch.dtype | None = None
) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if float_dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _trim_after_eos(sequence: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    sequence = sequence.detach()
    if eos_token_id is None:
        return sequence.clone()
    eos_positions = (sequence == eos_token_id).nonzero(as_tuple=False)
    if eos_positions.numel() == 0:
        return sequence.clone()
    return sequence[: eos_positions[0].item() + 1].clone()


def _get_image_token_id(config: Any) -> int:
    for attribute in ("image_token_index", "image_token_id"):
        if hasattr(config, attribute):
            return int(getattr(config, attribute))
    raise AttributeError("Could not find LLaVA image token id on the model config")


def _get_image_seq_length(config: Any) -> int | None:
    if hasattr(config, "image_seq_length") and getattr(config, "image_seq_length") is not None:
        return int(getattr(config, "image_seq_length"))
    return None


def _infer_merged_length_from_merge_output(merged: Any) -> int | None:
    if torch.is_tensor(merged) and merged.ndim >= 2:
        return int(merged.shape[1])

    if isinstance(merged, (tuple, list)):
        for value in merged:
            if torch.is_tensor(value) and value.ndim >= 3:
                return int(value.shape[1])
        for value in merged:
            if torch.is_tensor(value) and value.ndim >= 2:
                return int(value.shape[1])

    return None


def reconstruct_image_mask(
    input_ids: torch.Tensor,
    merged_len: int,
    image_token_id: int,
    image_seq_length: int | None = None,
) -> torch.Tensor:
    """
    Reconstruct prompt-space image positions in multimodal merged space.

    This supports both common LLaVA variants:
    - processors that already expand `<image>` into one token per image feature
    - processors/models that keep a single placeholder token and expand in-model
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported for image mask reconstruction")

    text_ids = input_ids[0].detach().cpu()
    placeholder_mask = text_ids == image_token_id
    n_placeholders = int(placeholder_mask.sum().item())
    mask = torch.zeros(merged_len, dtype=torch.bool)

    if n_placeholders == 0:
        return mask

    text_len = int(text_ids.shape[0])
    if merged_len < text_len:
        raise ValueError(f"Merged length {merged_len} cannot be smaller than text length {text_len}")

    if merged_len == text_len:
        mask.copy_(placeholder_mask)
        return mask

    expansion = (merged_len - text_len + n_placeholders) // n_placeholders
    expected_len = text_len - n_placeholders + n_placeholders * expansion
    if expected_len != merged_len:
        if image_seq_length is None:
            raise ValueError(
                "Could not infer image placeholder expansion from merged length. "
                f"text_len={text_len}, merged_len={merged_len}, n_placeholders={n_placeholders}"
            )
        expansion = image_seq_length
        expected_len = text_len - n_placeholders + n_placeholders * expansion
        if expected_len != merged_len:
            raise ValueError(
                "Image mask reconstruction still failed after falling back to `image_seq_length`. "
                f"Expected {expected_len}, got {merged_len}."
            )

    cursor = 0
    for token_id in text_ids.tolist():
        if token_id == image_token_id:
            mask[cursor : cursor + expansion] = True
            cursor += expansion
        else:
            cursor += 1

    return mask


def enable_merge_trace(model: LlavaForConditionalGeneration) -> dict[str, Any]:
    """
    Wrap the LLaVA multimodal placeholder/merge path so the extractor can record
    which prompt positions correspond to image features in merged space.
    """

    ctx: dict[str, Any] = {"is_image_pos_mm": None, "owner": None, "attr": None, "original": None}
    merge_owner = getattr(model, "model", model)

    if hasattr(merge_owner, "get_placeholder_mask"):
        original = merge_owner.get_placeholder_mask

        def wrapped(self, *args, **kwargs):
            mask = original(*args, **kwargs)
            reduced = mask.all(dim=-1) if mask.ndim == 3 else mask
            ctx["is_image_pos_mm"] = reduced[0].detach().cpu().to(torch.bool)
            return mask

        merge_owner.get_placeholder_mask = types.MethodType(wrapped, merge_owner)
        ctx.update({"owner": merge_owner, "attr": "get_placeholder_mask", "original": original})
        return ctx

    for owner in (model, merge_owner):
        if owner is None or not hasattr(owner, "_merge_input_ids_with_image_features"):
            continue

        original = owner._merge_input_ids_with_image_features

        def wrapped(self, *args, **kwargs):
            merged = original(*args, **kwargs)
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) >= 3:
                input_ids = args[2]

            merged_len = _infer_merged_length_from_merge_output(merged)
            if input_ids is not None and merged_len is not None:
                ctx["is_image_pos_mm"] = reconstruct_image_mask(
                    input_ids=input_ids.detach().cpu(),
                    merged_len=merged_len,
                    image_token_id=_get_image_token_id(self.config),
                    image_seq_length=_get_image_seq_length(self.config),
                )
            return merged

        owner._merge_input_ids_with_image_features = types.MethodType(wrapped, owner)
        ctx.update({"owner": owner, "attr": "_merge_input_ids_with_image_features", "original": original})
        return ctx

    return ctx


def disable_merge_trace(ctx: dict[str, Any]) -> None:
    if ctx.get("owner") is not None:
        setattr(ctx["owner"], ctx["attr"], ctx["original"])


def _resolve_prompt_image_mask(
    model: LlavaForConditionalGeneration,
    prompt_input_ids: torch.Tensor,
    prompt_len_mm: int,
    trace_ctx: dict[str, Any],
) -> torch.Tensor:
    mask = trace_ctx.get("is_image_pos_mm")
    if mask is not None:
        mask = mask.flatten().to(torch.bool).cpu()
        if mask.shape[0] == prompt_len_mm:
            return mask

    return reconstruct_image_mask(
        input_ids=prompt_input_ids.detach().cpu(),
        merged_len=prompt_len_mm,
        image_token_id=_get_image_token_id(model.config),
        image_seq_length=_get_image_seq_length(model.config),
    )


def get_decoder_module(model: LlavaForConditionalGeneration) -> torch.nn.Module:
    candidates = [
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "model", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layers") and len(candidate.layers) > 0 and hasattr(candidate.layers[0], "self_attn"):
            return candidate

    raise AttributeError("Could not locate the decoder layers inside the LLaVA language model")


def get_num_decoder_layers(model: LlavaForConditionalGeneration) -> int:
    return len(get_decoder_module(model).layers)


def _get_effective_linear_weight(linear: torch.nn.Module, dtype: torch.dtype | None = None) -> torch.Tensor:
    weight = linear.weight.detach()
    if isinstance(linear, FP8Linear):
        out_dtype = dtype if dtype is not None else torch.float32
        scale = linear.weight_scale_inv.to(out_dtype)
        scale = scale.repeat_interleave(linear.block_size[0], dim=0)
        scale = scale.repeat_interleave(linear.block_size[1], dim=1)
        return weight.to(out_dtype) * scale
    if dtype is not None:
        return weight.to(dtype)
    return weight


def get_attention_output_projection_weight(model: LlavaForConditionalGeneration, layer_idx: int) -> torch.Tensor:
    layer = get_decoder_module(model).layers[layer_idx]
    return _get_effective_linear_weight(layer.self_attn.o_proj).cpu()


def register_analysis_hooks(
    model: LlavaForConditionalGeneration, capture_vproj: bool = True
) -> dict[str, Any]:
    ctx: dict[str, Any] = {"hooks": [], "vproj_outputs": {}}
    if not capture_vproj:
        return ctx

    decoder = get_decoder_module(model)
    for layer_idx, layer in enumerate(decoder.layers):
        if not hasattr(layer.self_attn, "v_proj"):
            raise AttributeError(
                f"Layer {layer_idx} does not expose `v_proj`. "
                "This extractor currently expects a LLaMA-like LLaVA attention module."
            )

        def hook_fn(module, inputs, outputs, layer_idx: int = layer_idx):
            tensor = outputs[0] if isinstance(outputs, tuple) else outputs
            ctx["vproj_outputs"][layer_idx] = tensor.detach().cpu()

        handle = layer.self_attn.v_proj.register_forward_hook(hook_fn)
        ctx["hooks"].append(handle)

    return ctx


def remove_analysis_hooks(hook_ctx: dict[str, Any]) -> None:
    for handle in hook_ctx.get("hooks", []):
        handle.remove()


def compute_wov_norm_from_hooks(
    model: LlavaForConditionalGeneration, hook_ctx: dict[str, Any], prompt_len_mm: int
) -> list[torch.Tensor]:
    norms = []
    decoder = get_decoder_module(model)

    for layer_idx, layer in enumerate(decoder.layers):
        if layer_idx not in hook_ctx["vproj_outputs"]:
            raise RuntimeError(f"Missing v_proj activations for layer {layer_idx}")

        vcat = hook_ctx["vproj_outputs"][layer_idx]
        if vcat.ndim != 3:
            raise RuntimeError(f"Unexpected v_proj activation shape for layer {layer_idx}: {tuple(vcat.shape)}")

        num_heads = int(getattr(layer.self_attn, "num_heads", layer.self_attn.config.num_attention_heads))
        num_kv_heads = int(
            getattr(layer.self_attn, "num_key_value_heads", getattr(layer.self_attn.config, "num_key_value_heads", num_heads))
        )
        num_kv_groups = int(getattr(layer.self_attn, "num_key_value_groups", num_heads // num_kv_heads))
        head_dim = int(getattr(layer.self_attn, "head_dim", vcat.shape[-1] // num_kv_heads))

        values = vcat.view(vcat.shape[0], vcat.shape[1], num_kv_heads, head_dim).permute(0, 2, 1, 3)
        values = repeat_kv(values, num_kv_groups)

        wo = _get_effective_linear_weight(layer.self_attn.o_proj, dtype=values.dtype).transpose(0, 1).cpu()
        wo = wo.view(num_heads, head_dim, wo.shape[-1])
        projected = torch.einsum("h d m, b h s d -> b h s m", wo, values)
        norms.append(projected.norm(dim=-1)[0, :, :prompt_len_mm].detach().cpu())

    return norms


def _to_cpu_nested(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return [_to_cpu_nested(v) for v in value]
    if isinstance(value, list):
        return [_to_cpu_nested(v) for v in value]
    return value


def _decode_answer(processor: Any, answer_ids: torch.Tensor) -> str:
    return processor.tokenizer.decode(
        answer_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _build_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.format(question=question.strip())


def _open_image(image_path: str):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


def _get_visual_inputs(prompt_inputs: dict[str, Any]) -> dict[str, Any]:
    visual_inputs = {}
    for key in ("pixel_values", "image_sizes"):
        if key in prompt_inputs:
            visual_inputs[key] = prompt_inputs[key]
    return visual_inputs


def validate_extraction_record(record: dict[str, Any]) -> dict[str, Any]:
    errors = []

    prompt_len_text = int(record["prompt_len_text"])
    answer_len_text = int(record["answer_len_text"])
    prompt_len_mm = int(record["prompt_len_mm"])
    full_len_mm = int(record["full_len_mm"])

    if int(record["prompt_ids_text"].shape[0]) != prompt_len_text:
        errors.append("prompt_ids_text length does not match prompt_len_text")
    if int(record["full_ids_text"].shape[0]) != prompt_len_text + answer_len_text:
        errors.append("full_ids_text length does not equal prompt_len_text + answer_len_text")
    if full_len_mm != prompt_len_mm + answer_len_text:
        errors.append("full_len_mm does not equal prompt_len_mm + answer_len_text")

    n_layers = len(record["W_O"])
    if len(record["hidden_prompt"]) != n_layers + 1:
        errors.append("hidden_prompt should contain embeddings plus one tensor per layer")
    if len(record["hidden_answer"]) != n_layers + 1:
        errors.append("hidden_answer should contain embeddings plus one tensor per layer")
    if len(record["attn_answer_to_prompt"]) != n_layers:
        errors.append("attn_answer_to_prompt should contain one tensor per decoder layer")
    if record["wov_norm_prompt"] is not None and len(record["wov_norm_prompt"]) != n_layers:
        errors.append("wov_norm_prompt should contain one tensor per decoder layer")

    for idx, hidden in enumerate(record["hidden_prompt"]):
        if hidden.ndim != 2 or hidden.shape[0] != prompt_len_mm:
            errors.append(f"hidden_prompt[{idx}] has invalid shape {tuple(hidden.shape)}")

    for idx, hidden in enumerate(record["hidden_answer"]):
        if hidden.ndim != 2 or hidden.shape[0] != answer_len_text:
            errors.append(f"hidden_answer[{idx}] has invalid shape {tuple(hidden.shape)}")

    for idx, attn in enumerate(record["attn_answer_to_prompt"]):
        if attn.ndim != 3 or attn.shape[1] != answer_len_text or attn.shape[2] != prompt_len_mm:
            errors.append(f"attn_answer_to_prompt[{idx}] has invalid shape {tuple(attn.shape)}")
            continue
        row_sums = attn.sum(dim=-1)
        if torch.any(row_sums > 1 + 1e-4):
            errors.append(f"attn_answer_to_prompt[{idx}] contains row sums greater than 1")

    for idx, weight in enumerate(record["W_O"]):
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            errors.append(f"W_O[{idx}] has invalid shape {tuple(weight.shape)}")

    is_image_pos_mm = record["is_image_pos_mm"]
    is_text_prompt_pos_mm = record["is_text_prompt_pos_mm"]
    if is_image_pos_mm.ndim != 1 or is_image_pos_mm.shape[0] != prompt_len_mm:
        errors.append("is_image_pos_mm has invalid shape")
    if is_text_prompt_pos_mm.ndim != 1 or is_text_prompt_pos_mm.shape[0] != prompt_len_mm:
        errors.append("is_text_prompt_pos_mm has invalid shape")
    if not torch.equal(~is_image_pos_mm, is_text_prompt_pos_mm):
        errors.append("is_text_prompt_pos_mm must be the boolean complement of is_image_pos_mm")

    return {"ok": len(errors) == 0, "errors": errors}


class LlavaAnalysisDataCollector:
    """
    Collects per-sample LLaVA analysis records using a two-pass extraction scheme.
    """

    def __init__(self, model: LlavaForConditionalGeneration, processor: Any):
        self.model = model
        self.processor = processor
        self.device = _get_model_device(model)
        self.float_dtype = _get_model_float_dtype(model)
        self.eos_token_id = processor.tokenizer.eos_token_id

    def collect_sample(
        self,
        sample: dict[str, Any],
        prompt_template: str,
        max_new_tokens: int = 32,
        capture_vproj: bool = True,
        save_image_hidden_states: bool = False,
        save_full_attentions: bool = False,
    ) -> dict[str, Any]:
        image = _open_image(sample["image_path"])
        prompt_text = _build_prompt(sample["question"], prompt_template)

        prompt_inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")
        prompt_inputs = _move_batch_to_device(prompt_inputs, self.device, self.float_dtype)
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_len_text = int(prompt_input_ids.shape[1])

        trace_ctx = enable_merge_trace(self.model)
        try:
            with torch.no_grad():
                prompt_out = self.model(
                    **prompt_inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            disable_merge_trace(trace_ctx)

        prompt_len_mm = int(prompt_out.logits.shape[1])
        is_image_pos_mm = _resolve_prompt_image_mask(self.model, prompt_input_ids, prompt_len_mm, trace_ctx)
        is_text_prompt_pos_mm = ~is_image_pos_mm

        with torch.no_grad():
            generated_ids = self.model.generate(
                **prompt_inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        full_ids_text = _trim_after_eos(generated_ids[0], self.eos_token_id)
        answer_ids_text = full_ids_text[prompt_len_text:]
        answer_len_text = int(answer_ids_text.shape[0])

        analysis_input_ids = full_ids_text.unsqueeze(0).to(self.device)
        analysis_inputs = {
            "input_ids": analysis_input_ids,
            "attention_mask": torch.ones_like(analysis_input_ids),
            **_get_visual_inputs(prompt_inputs),
        }

        hook_ctx = register_analysis_hooks(self.model, capture_vproj=capture_vproj)
        try:
            with torch.no_grad():
                full_out = self.model(
                    **analysis_inputs,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            remove_analysis_hooks(hook_ctx)

        if full_out.hidden_states is None or full_out.attentions is None:
            raise RuntimeError(
                "LLaVA analysis forward pass did not return hidden states or attentions. "
                "Make sure the model uses an attention implementation that supports dense attentions."
            )

        full_len_mm = int(full_out.hidden_states[0].shape[1])
        prompt_pos_mm = torch.arange(prompt_len_mm, dtype=torch.long)
        answer_pos_mm = torch.arange(prompt_len_mm, full_len_mm, dtype=torch.long)

        hidden_prompt = [hidden[0, :prompt_len_mm, :].detach().cpu() for hidden in full_out.hidden_states]
        hidden_answer = [hidden[0, prompt_len_mm:full_len_mm, :].detach().cpu() for hidden in full_out.hidden_states]
        attn_answer_to_prompt = [
            attn[0, :, prompt_len_mm:full_len_mm, :prompt_len_mm].detach().cpu() for attn in full_out.attentions
        ]

        weights = [
            get_attention_output_projection_weight(self.model, layer_idx) for layer_idx in range(get_num_decoder_layers(self.model))
        ]

        wov_norm_prompt = None
        if capture_vproj:
            wov_norm_prompt = compute_wov_norm_from_hooks(self.model, hook_ctx, prompt_len_mm)

        record = {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "image_path": sample["image_path"],
            "prompt_text": prompt_text,
            "prompt_len_text": prompt_len_text,
            "answer_len_text": answer_len_text,
            "prompt_len_mm": prompt_len_mm,
            "full_len_mm": full_len_mm,
            "prompt_ids_text": prompt_input_ids[0].detach().cpu(),
            "full_ids_text": full_ids_text.detach().cpu(),
            "answer_ids_text": answer_ids_text.detach().cpu(),
            "prompt_pos_mm": prompt_pos_mm,
            "answer_pos_mm": answer_pos_mm,
            "is_image_pos_mm": is_image_pos_mm.detach().cpu(),
            "is_text_prompt_pos_mm": is_text_prompt_pos_mm.detach().cpu(),
            "hidden_prompt": hidden_prompt,
            "hidden_answer": hidden_answer,
            "attn_answer_to_prompt": attn_answer_to_prompt,
            "W_O": weights,
            "wov_norm_prompt": wov_norm_prompt,
            "answer_text": _decode_answer(self.processor, answer_ids_text.detach().cpu()),
        }

        if save_image_hidden_states:
            record["image_hidden_states"] = _to_cpu_nested(full_out.image_hidden_states)

        if save_full_attentions:
            record["full_attentions"] = [attn[0].detach().cpu() for attn in full_out.attentions]

        return record


def extract_llava_analysis_data(
    dataset_path: str,
    output_dir: str,
    implementation_model_name: str = "llava-hf/llava-1.5-7b-hf",
    report_model_name: str = "liuhaotian/llava-v1.5-7b",
    image_root: str | None = None,
    question_column: str = "question",
    image_column: str = "image_path",
    id_column: str = "sample_id",
    prompt_template: str = "USER: <image>\n{question}\nASSISTANT:",
    max_new_tokens: int = 32,
    torch_dtype: str = "auto",
    device_map: str | None = "auto",
    attn_implementation: str = "eager",
    limit: int | None = None,
    capture_vproj: bool = True,
    save_image_hidden_states: bool = False,
    save_full_attentions: bool = False,
    max_full_attention_samples: int = 0,
    overwrite: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """
    Extract per-sample LLaVA analysis records and save them to disk.

    The dataset must be a `.jsonl` or `.json` file containing at least:
    - `question`
    - `image_path` (or the column selected by `image_column`)
    - `sample_id` is optional
    """

    output_path = Path(output_dir).resolve()
    records_path = output_path / "records"
    metadata_path = output_path / "metadata.jsonl"
    validation_path = output_path / "validation_report.json"
    config_path = output_path / "run_config.json"

    output_path.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_path.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_path}")
    records_path.mkdir(parents=True, exist_ok=True)

    samples = load_llava_samples(
        dataset_path=dataset_path,
        image_root=image_root,
        question_column=question_column,
        image_column=image_column,
        id_column=id_column,
        limit=limit,
    )

    model_kwargs = {
        "attn_implementation": attn_implementation,
        "device_map": device_map,
        "torch_dtype": _parse_torch_dtype(torch_dtype),
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

    print(f"Loading LLaVA processor and model: {implementation_model_name}")
    processor = AutoProcessor.from_pretrained(implementation_model_name)
    model = LlavaForConditionalGeneration.from_pretrained(implementation_model_name, **model_kwargs)
    model.eval()

    collector = LlavaAnalysisDataCollector(model, processor)

    summary = {
        "report_model_name": report_model_name,
        "implementation_model_name": implementation_model_name,
        "dataset_path": str(Path(dataset_path).resolve()),
        "n_samples_requested": len(samples),
        "n_samples_succeeded": 0,
        "n_samples_failed": 0,
        "n_validation_failures": 0,
        "failures": [],
    }

    with config_path.open("w") as f:
        json.dump(
            {
                "report_model_name": report_model_name,
                "implementation_model_name": implementation_model_name,
                "dataset_path": str(Path(dataset_path).resolve()),
                "image_root": str(Path(image_root).resolve()) if image_root is not None else None,
                "question_column": question_column,
                "image_column": image_column,
                "id_column": id_column,
                "prompt_template": prompt_template,
                "max_new_tokens": max_new_tokens,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "attn_implementation": attn_implementation,
                "limit": limit,
                "capture_vproj": capture_vproj,
                "save_image_hidden_states": save_image_hidden_states,
                "save_full_attentions": save_full_attentions,
                "max_full_attention_samples": max_full_attention_samples,
            },
            f,
            indent=2,
        )

    with metadata_path.open("w") as metadata_file:
        for sample_idx, sample in enumerate(tqdm(samples, desc="Extracting LLaVA analysis records")):
            try:
                record = collector.collect_sample(
                    sample=sample,
                    prompt_template=prompt_template,
                    max_new_tokens=max_new_tokens,
                    capture_vproj=capture_vproj,
                    save_image_hidden_states=save_image_hidden_states,
                    save_full_attentions=save_full_attentions and sample_idx < max_full_attention_samples,
                )
                validation = validate_extraction_record(record)

                record_file = records_path / f"{sample['sample_id']}.pt"
                torch.save(record, record_file)

                metadata = {
                    "sample_id": sample["sample_id"],
                    "question": sample["question"],
                    "image_path": sample["image_path"],
                    "record_path": str(record_file),
                    "answer_text": record["answer_text"],
                    "prompt_len_text": record["prompt_len_text"],
                    "answer_len_text": record["answer_len_text"],
                    "prompt_len_mm": record["prompt_len_mm"],
                    "full_len_mm": record["full_len_mm"],
                    "validation_ok": validation["ok"],
                    "validation_errors": validation["errors"],
                }
                metadata_file.write(json.dumps(metadata) + "\n")

                summary["n_samples_succeeded"] += 1
                if not validation["ok"]:
                    summary["n_validation_failures"] += 1
                    summary["failures"].append(
                        {
                            "sample_id": sample["sample_id"],
                            "type": "validation",
                            "errors": validation["errors"],
                        }
                    )
            except Exception as exc:
                summary["n_samples_failed"] += 1
                failure = {"sample_id": sample["sample_id"], "type": "exception", "error": str(exc)}
                summary["failures"].append(failure)
                metadata_file.write(json.dumps(failure) + "\n")
                if not continue_on_error:
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    with validation_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary
