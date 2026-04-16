#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from kvzap.image_teacher_utils import load_pt_record
from kvzap.llava_extractor import (
    _get_model_device,
    _get_model_float_dtype,
    _move_batch_to_device,
    _parse_torch_dtype,
    _trim_after_eos,
    compute_wov_norm_from_hooks,
    configure_llava_processor,
    register_analysis_hooks,
    remove_analysis_hooks,
)


EPS = 1e-8


def _open_images(image_paths: list[str]):
    from PIL import Image

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images[0] if len(images) == 1 else images



def _get_visual_inputs(prompt_inputs: dict[str, Any]) -> dict[str, Any]:
    visual_inputs = {}
    for key in ("pixel_values", "image_sizes"):
        if key in prompt_inputs:
            visual_inputs[key] = prompt_inputs[key]
    return visual_inputs



def _to_storage_cpu(tensor: torch.Tensor, storage_dtype: torch.dtype | None) -> torch.Tensor:
    out = tensor.detach()
    if storage_dtype is not None and torch.is_floating_point(out):
        out = out.to(storage_dtype)
    return out.cpu()



def _decode_answer(processor: Any, answer_ids: torch.Tensor) -> str:
    return processor.tokenizer.decode(
        answer_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()



def _resolve_image_indices(base_record: dict[str, Any]) -> torch.Tensor:
    if "image_indices_mm" in base_record:
        return base_record["image_indices_mm"].long().flatten().cpu()
    if "image_pos_mm" in base_record:
        return base_record["image_pos_mm"].long().flatten().cpu()
    if "is_image_pos_mm" in base_record:
        return base_record["is_image_pos_mm"].nonzero(as_tuple=False).flatten().long().cpu()
    raise KeyError("Base record is missing image position info")



def _sorted_record_paths(records_dir: Path) -> list[Path]:
    paths = [p for p in records_dir.glob("*.pt") if p.is_file()]

    def key_fn(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(paths, key=key_fn)



def build_teacher_record(
    model: LlavaForConditionalGeneration,
    processor: Any,
    base_record: dict[str, Any],
    max_new_tokens: int,
    storage_dtype: torch.dtype | None,
) -> dict[str, Any]:
    device = _get_model_device(model)
    float_dtype = _get_model_float_dtype(model)

    image_paths = list(base_record["image_paths"])
    images = _open_images(image_paths)
    prompt_text = str(base_record["prompt_text"])

    prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, float_dtype)
    prompt_len_text = int(prompt_inputs["input_ids"].shape[1])

    with torch.no_grad():
        generated_ids = model.generate(
            **prompt_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    full_ids_text = _trim_after_eos(generated_ids[0], processor.tokenizer.eos_token_id)
    answer_ids_text = full_ids_text[prompt_len_text:]

    analysis_input_ids = full_ids_text.unsqueeze(0).to(device)
    analysis_inputs = {
        "input_ids": analysis_input_ids,
        "attention_mask": torch.ones_like(analysis_input_ids),
        **_get_visual_inputs(prompt_inputs),
    }

    hook_ctx = register_analysis_hooks(model, capture_vproj=True, store_on_cpu=False)
    try:
        with torch.no_grad():
            full_out = model(
                **analysis_inputs,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        remove_analysis_hooks(hook_ctx)

    if full_out.hidden_states is None or full_out.attentions is None:
        raise RuntimeError("Analysis forward pass did not return hidden states/attentions")

    prompt_len_mm = int(base_record["prompt_len_mm"])
    full_len_mm = int(full_out.hidden_states[0].shape[1])
    answer_pos_mm = torch.arange(prompt_len_mm, full_len_mm, device=device, dtype=torch.long)
    image_indices_cpu = _resolve_image_indices(base_record)
    image_indices = image_indices_cpu.to(device)

    n_layers = len(full_out.attentions)
    hidden_all = full_out.hidden_states
    if len(hidden_all) == n_layers + 1:
        hidden_layers = hidden_all[1:]
    elif len(hidden_all) == n_layers:
        hidden_layers = hidden_all
    else:
        raise ValueError(f"Unexpected hidden_states length {len(hidden_all)} for {n_layers} layers")

    wov_norm_prompt = compute_wov_norm_from_hooks(
        model,
        hook_ctx,
        prompt_len_mm,
        to_cpu=False,
    )

    att_only_answer_layers: list[torch.Tensor] = []
    splus_answer_layers: list[torch.Tensor] = []

    for layer_idx in range(n_layers):
        attn = full_out.attentions[layer_idx][0].detach()  # [H, S, S]
        if attn.dim() != 3:
            raise ValueError(f"Unexpected attention shape at layer {layer_idx}: {tuple(attn.shape)}")

        answer_block = attn.index_select(1, answer_pos_mm).index_select(2, image_indices)  # [H, A, I]
        wnorm_image = wov_norm_prompt[layer_idx].index_select(1, image_indices)  # [H, I]
        hidden_layer = hidden_layers[layer_idx][0].detach()
        answer_hidden = hidden_layer.index_select(0, answer_pos_mm)  # [A, D]

        if answer_hidden.shape[0] == 0:
            att_only = torch.zeros(
                (answer_block.shape[0], answer_block.shape[2]),
                device=answer_block.device,
                dtype=answer_block.dtype,
            )
            splus = att_only
        else:
            h_norm = torch.norm(answer_hidden, dim=-1).clamp_min(EPS)
            att_only = answer_block.max(dim=1).values
            splus = (
                answer_block
                * h_norm.reciprocal().view(1, -1, 1)
                * wnorm_image.unsqueeze(1)
            ).max(dim=1).values

        att_only_answer_layers.append(_to_storage_cpu(att_only, storage_dtype))
        splus_answer_layers.append(_to_storage_cpu(splus, storage_dtype))

    out = {
        "sample_id": base_record["sample_id"],
        "question": base_record.get("question", ""),
        "image_path": base_record.get("image_path", image_paths[0] if image_paths else ""),
        "image_paths": image_paths,
        "prompt_text": prompt_text,
        "prompt_len_text": int(base_record.get("prompt_len_text", prompt_len_text)),
        "prompt_len_mm": prompt_len_mm,
        "image_pos_mm": image_indices_cpu,
        "image_indices_mm": image_indices_cpu,
        "answer_text": _decode_answer(processor, answer_ids_text.detach().cpu()),
        "att_only_answer": torch.stack(att_only_answer_layers, dim=0),
        "splus_answer": torch.stack(splus_answer_layers, dim=0),
        "att_only_postvision": _to_storage_cpu(base_record["att_only_postvision"], storage_dtype),
        "splus_postvision": _to_storage_cpu(base_record["splus_postvision"], storage_dtype),
    }

    return out



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_teacher_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--implementation_model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--storage_dtype", type=str, default="float16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_teacher_dir).resolve()
    base_records_dir = base_dir / "records"
    if not base_records_dir.is_dir():
        raise ValueError(f"Base teacher records directory not found: {base_records_dir}")

    output_dir = Path(args.output_dir).resolve()
    records_out = output_dir / "records"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    records_out.mkdir(parents=True, exist_ok=True)

    record_paths = _sorted_record_paths(base_records_dir)
    if args.limit is not None:
        record_paths = record_paths[: args.limit]

    model_kwargs: dict[str, Any] = {
        "device_map": None if args.device_map in ("", "none", "None") else args.device_map,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": _parse_torch_dtype(args.torch_dtype),
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    storage_dtype: torch.dtype | None = None
    if args.storage_dtype not in ("", "none", "None"):
        parsed = _parse_torch_dtype(args.storage_dtype)
        if isinstance(parsed, str):
            raise ValueError(f"Invalid storage dtype: {args.storage_dtype}")
        storage_dtype = parsed

    print(f"Loading processor/model: {args.implementation_model_name}")
    processor = AutoProcessor.from_pretrained(args.implementation_model_name)
    model = LlavaForConditionalGeneration.from_pretrained(args.implementation_model_name, **model_kwargs)
    configure_llava_processor(processor, model.config)
    model.eval()

    summary = {
        "base_teacher_dir": str(base_dir),
        "output_dir": str(output_dir),
        "implementation_model_name": args.implementation_model_name,
        "n_samples_requested": len(record_paths),
        "n_samples_succeeded": 0,
        "n_samples_failed": 0,
        "failures": [],
    }

    with (output_dir / "metadata.jsonl").open("w") as meta_f:
        for record_path in tqdm(record_paths, desc="Building 4-teacher records"):
            sample_id = record_path.stem
            try:
                base_record = load_pt_record(record_path)
                out_record = build_teacher_record(
                    model=model,
                    processor=processor,
                    base_record=base_record,
                    max_new_tokens=args.max_new_tokens,
                    storage_dtype=storage_dtype,
                )
                out_path = records_out / f"{sample_id}.pt"
                torch.save(out_record, out_path)

                metadata = {
                    "sample_id": sample_id,
                    "record_path": str(out_path),
                    "n_image_tokens": int(out_record["image_pos_mm"].numel()),
                    "att_only_answer_shape": list(out_record["att_only_answer"].shape),
                    "splus_answer_shape": list(out_record["splus_answer"].shape),
                    "att_only_postvision_shape": list(out_record["att_only_postvision"].shape),
                    "splus_postvision_shape": list(out_record["splus_postvision"].shape),
                }
                meta_f.write(json.dumps(metadata) + "\n")
                summary["n_samples_succeeded"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["n_samples_failed"] += 1
                failure = {"sample_id": sample_id, "error": repr(exc)}
                summary["failures"].append(failure)
                meta_f.write(json.dumps(failure) + "\n")
                if not args.continue_on_error:
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (output_dir / "validation_report.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("build complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
