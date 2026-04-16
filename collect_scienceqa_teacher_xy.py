#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from build_scienceqa_manifest import iter_scienceqa_samples
from build_scienceqa_manifest import OPTION_LETTERS
from kvzap.llava_extractor import _collect_postvision_minimal_sample, _parse_torch_dtype, configure_llava_processor


SCIENCEQA_COMMON_PROMPT_TEMPLATE = "USER: <image>\n{prompt_body}\nASSISTANT:"
SCIENCEQA_COMMON_PROMPT_BODY_TEMPLATE = (
    "Context: {hint}\n"
    "Question: {question}\n"
    "Options:\n"
    "{options}\n"
    "Select the best answer based on the image and text."
)


def _resolve_shard_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    shard_dir = output_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir


def _parse_storage_dtype(storage_dtype: str | None) -> torch.dtype | None:
    if storage_dtype in (None, "", "none", "None"):
        return None
    parsed = _parse_torch_dtype(storage_dtype)
    if isinstance(parsed, str):
        raise ValueError(f"Invalid storage dtype: {storage_dtype}")
    return parsed


def _parse_target_dtype(target_dtype: str) -> torch.dtype:
    parsed = _parse_torch_dtype(target_dtype)
    if isinstance(parsed, str):
        raise ValueError(f"Invalid target dtype: {target_dtype}")
    return parsed


def _apply_target_transform(scores: torch.Tensor, target_transform: str, target_eps: float) -> torch.Tensor:
    if target_transform == "none":
        return scores
    if target_transform == "log":
        return torch.log(scores.clamp_min(0) + target_eps)
    raise ValueError(f"Unsupported target_transform: {target_transform}")


def _target_dim(y: torch.Tensor) -> int:
    if y.ndim == 1:
        return 1
    if y.ndim == 2:
        return int(y.shape[-1])
    raise ValueError(f"Unsupported target tensor shape: {tuple(y.shape)}")


def _select_token_indices_top_random(
    scores: torch.Tensor,
    rng: np.random.Generator,
    top_fraction: float,
    random_fraction: float,
    max_tokens_per_layer: int | None,
    min_tokens_per_layer: int,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D score tensor, got {tuple(scores.shape)}")

    n_tokens = int(scores.shape[0])
    if n_tokens == 0:
        return torch.empty(0, dtype=torch.long)

    top_k = int(math.ceil(n_tokens * top_fraction)) if top_fraction > 0 else 0
    rand_k = int(math.ceil(n_tokens * random_fraction)) if random_fraction > 0 else 0
    top_k = min(n_tokens, top_k)

    top_idx = torch.empty(0, dtype=torch.long)
    if top_k > 0:
        top_idx = torch.topk(scores, k=top_k, largest=True, sorted=True).indices.cpu().long()

    remaining_mask = torch.ones(n_tokens, dtype=torch.bool)
    remaining_mask[top_idx] = False
    remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()

    rand_idx = torch.empty(0, dtype=torch.long)
    if rand_k > 0 and remaining_idx.numel() > 0:
        rand_k = min(int(remaining_idx.numel()), rand_k)
        chosen = rng.choice(remaining_idx.numpy(), size=rand_k, replace=False)
        rand_idx = torch.from_numpy(np.asarray(chosen)).long()

    selected = torch.cat([top_idx, rand_idx], dim=0)
    if selected.numel() == 0 and min_tokens_per_layer > 0:
        keep_k = min(n_tokens, min_tokens_per_layer)
        selected = torch.topk(scores, k=keep_k, largest=True, sorted=True).indices.cpu().long()

    if max_tokens_per_layer is not None and selected.numel() > max_tokens_per_layer:
        keep_top = top_idx[: min(int(top_idx.numel()), max_tokens_per_layer)]
        remaining = max_tokens_per_layer - int(keep_top.numel())
        keep_rand = rand_idx[: max(remaining, 0)]
        selected = torch.cat([keep_top, keep_rand], dim=0)

    if selected.numel() == 0:
        return selected

    return torch.unique(selected, sorted=True)


class XYShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        split_name: str,
        max_rows_per_shard: int,
        include_sample_ids: bool,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.shard_dir = _resolve_shard_dir(self.output_dir)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.split_name = split_name
        self.max_rows_per_shard = max_rows_per_shard
        self.include_sample_ids = include_sample_ids

        self._x_parts: list[torch.Tensor] = []
        self._y_parts: list[torch.Tensor] = []
        self._layer_parts: list[torch.Tensor] = []
        self._sample_ids: list[str] = []

        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._layer_min: int | None = None
        self._layer_max: int | None = None
        self._x_dtype: str | None = None
        self._y_dtype: str | None = None

        self.shards_written = 0
        self.rows_written = 0

    @property
    def pending_rows(self) -> int:
        return sum(int(part.shape[0]) for part in self._layer_parts)

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        layer_ids: torch.Tensor,
        sample_ids: list[str],
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected X to be [N, D], got {tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be [N], got {tuple(y.shape)}")
        if layer_ids.ndim != 1:
            raise ValueError(f"Expected layer ids to be [N], got {tuple(layer_ids.shape)}")
        if x.shape[0] != y.shape[0] or x.shape[0] != layer_ids.shape[0]:
            raise ValueError(
                f"Row mismatch: X={tuple(x.shape)} y={tuple(y.shape)} layer={tuple(layer_ids.shape)}"
            )
        if self.include_sample_ids and len(sample_ids) != int(x.shape[0]):
            raise ValueError(f"sample_ids length mismatch: {len(sample_ids)} vs {x.shape[0]}")
        if x.shape[0] == 0:
            return

        self._x_parts.append(x)
        self._y_parts.append(y)
        self._layer_parts.append(layer_ids.long())
        if self.include_sample_ids:
            self._sample_ids.extend(sample_ids)

        if self.pending_rows >= self.max_rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._x_parts:
            return

        x = torch.cat(self._x_parts, dim=0)
        y = torch.cat(self._y_parts, dim=0)
        layer = torch.cat(self._layer_parts, dim=0)

        shard_path = self.shard_dir / f"shard_{self.shards_written:04d}.pt"
        payload: dict[str, Any] = {
            "X": x,
            "y": y,
            "layer": layer,
        }
        if self.include_sample_ids:
            payload["sample_id"] = list(self._sample_ids)

        torch.save(payload, shard_path)

        output_dim = _target_dim(y)
        layer_min = int(layer.min().item())
        layer_max = int(layer.max().item())

        if self._input_dim is None:
            self._input_dim = int(x.shape[-1])
            self._output_dim = int(output_dim)
            self._layer_min = layer_min
            self._layer_max = layer_max
            self._x_dtype = str(x.dtype)
            self._y_dtype = str(y.dtype)
        else:
            if self._input_dim != int(x.shape[-1]):
                raise ValueError(f"Input dim mismatch across shards: {self._input_dim} vs {x.shape[-1]}")
            if self._output_dim != int(output_dim):
                raise ValueError(f"Output dim mismatch across shards: {self._output_dim} vs {output_dim}")
            self._layer_min = min(int(self._layer_min), layer_min)
            self._layer_max = max(int(self._layer_max), layer_max)

        metadata = {
            "shard_path": str(shard_path),
            "shard_name": shard_path.name,
            "split": self.split_name,
            "n_rows": int(layer.shape[0]),
            "input_dim": int(x.shape[-1]),
            "output_dim": int(output_dim),
            "layer_min": layer_min,
            "layer_max": layer_max,
            "n_layers_present": int(layer.unique().numel()),
            "x_dtype": str(x.dtype),
            "y_dtype": str(y.dtype),
        }
        with self.metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

        self.rows_written += int(layer.shape[0])
        self.shards_written += 1
        self._x_parts.clear()
        self._y_parts.clear()
        self._layer_parts.clear()
        self._sample_ids.clear()

    def write_dataset_spec(self) -> None:
        if self._input_dim is None or self._output_dim is None or self._layer_min is None or self._layer_max is None:
            spec = {
                "split": self.split_name,
                "n_rows": int(self.rows_written),
                "n_shards": int(self.shards_written),
                "input_dim": None,
                "output_dim": None,
                "layer_min": None,
                "layer_max": None,
                "n_layers": 0,
                "include_sample_ids": bool(self.include_sample_ids),
            }
        else:
            spec = {
                "split": self.split_name,
                "n_rows": int(self.rows_written),
                "n_shards": int(self.shards_written),
                "input_dim": int(self._input_dim),
                "output_dim": int(self._output_dim),
                "layer_min": int(self._layer_min),
                "layer_max": int(self._layer_max),
                "n_layers": int(self._layer_max) + 1,
                "x_dtype": self._x_dtype,
                "y_dtype": self._y_dtype,
                "include_sample_ids": bool(self.include_sample_ids),
            }
        with (self.output_dir / "dataset_spec.json").open("w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)




def _build_scienceqa_options(choices: list[str]) -> str:
    return "\n".join(f"{OPTION_LETTERS[idx]}. {str(choice).strip()}" for idx, choice in enumerate(choices))


def _build_scienceqa_prompt_body(question: str, hint: str, options: str) -> str:
    lines = []
    hint = str(hint or "").strip()
    question = str(question or "").strip()
    options = str(options or "").strip()
    if hint:
        lines.append(f"Context: {hint}")
    lines.append(f"Question: {question}")
    lines.append("Options:")
    if options:
        lines.append(options)
    lines.append("Select the best answer based on the image and text.")
    return "\n".join(lines)


def _build_run_config_payload(args: argparse.Namespace, samples: list[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(vars(args))
    sample_keys = sorted(key for key in samples[0].keys() if key != "raw") if samples else []
    prompt_mode = "scienceqa_common_v1" if args.prompt_template == SCIENCEQA_COMMON_PROMPT_TEMPLATE else "custom"

    prompt_spec: dict[str, Any] = {
        "mode": prompt_mode,
        "template": args.prompt_template,
    }
    if prompt_mode == "scienceqa_common_v1":
        prompt_spec.update(
            {
                "prompt_template_expanded": (
                    "USER: <image>\n"
                    + SCIENCEQA_COMMON_PROMPT_BODY_TEMPLATE
                    + "\nASSISTANT:"
                ),
                "prompt_body_template": SCIENCEQA_COMMON_PROMPT_BODY_TEMPLATE,
                "prompt_body_rules": {
                    "hint_optional": True,
                    "omit_empty_hint_line": True,
                },
                "input_fields": ["hint", "question", "options"],
                "derived_fields": ["prompt_body"],
            }
        )

    payload["prompt_spec"] = prompt_spec
    payload["runtime"] = {
        "device": args.device,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "storage_dtype": args.storage_dtype,
        "target_dtype": args.target_dtype,
    }
    payload["data_spec"] = {
        "base_dir": args.base_dir,
        "split": args.split,
        "limit": args.limit,
        "sample_fields": sample_keys,
    }
    payload["teacher_spec"] = {
        "teacher_type": args.teacher_type,
        "teacher_head_reduction": args.teacher_head_reduction,
        "target_transform": args.target_transform,
        "target_eps": args.target_eps,
    }
    payload["sampling_spec"] = {
        "sample_image_tokens": args.sample_image_tokens,
        "top_fraction": args.top_fraction,
        "random_fraction": args.random_fraction,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "min_tokens_per_layer": args.min_tokens_per_layer,
        "shard_size": args.shard_size,
    }
    return payload

def _load_scienceqa_samples(base_dir: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    out = []
    for sample in iter_scienceqa_samples(base_dir=base_dir, split=split, limit=limit):
        question = str(sample.get("question", "") or "").strip()
        hint = str(sample.get("hint", "") or "").strip()
        choices = [str(choice).strip() for choice in sample.get("choices", [])]
        options = _build_scienceqa_options(choices)
        prompt_body = _build_scienceqa_prompt_body(question=question, hint=hint, options=options)
        out.append(
            {
                "sample_id": sample["sample_id"],
                "question": question,
                "hint": hint,
                "choices": choices,
                "options": options,
                "prompt_body": prompt_body,
                "image_path": sample["image_path"],
                "image_paths": [sample["image_path"]],
                "raw": sample,
            }
        )
    return out


def _resolve_teacher_keys(teacher_type: str) -> tuple[str, str, str]:
    teacher_att_key = "teacher_att_only_postvision"
    teacher_splus_key = "teacher_splus_postvision"
    if teacher_type == "att_only_postvision":
        return teacher_att_key, teacher_splus_key, teacher_att_key
    if teacher_type == "splus_postvision":
        return teacher_att_key, teacher_splus_key, teacher_splus_key
    raise ValueError(f"Unsupported teacher_type: {teacher_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "trainval", "minitrain", "minival", "minitest"], required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--teacher_type", choices=["att_only_postvision", "splus_postvision"], default="splus_postvision")
    parser.add_argument("--implementation_model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--prompt_template", type=str, default=SCIENCEQA_COMMON_PROMPT_TEMPLATE)
    parser.add_argument("--sample_image_tokens", choices=["all", "top_random"], default="all")
    parser.add_argument("--top_fraction", type=float, default=0.2)
    parser.add_argument("--random_fraction", type=float, default=0.2)
    parser.add_argument("--max_tokens_per_layer", type=int, default=None)
    parser.add_argument("--min_tokens_per_layer", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=50000)
    parser.add_argument("--target_transform", choices=["none", "log"], default="log")
    parser.add_argument("--target_eps", type=float, default=1e-8)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--storage_dtype", type=str, default="float16")
    parser.add_argument("--target_dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--teacher_head_reduction", choices=["none", "mean"], default="mean")
    parser.add_argument("--include_sample_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.top_fraction < 0 or args.random_fraction < 0:
        raise ValueError("top_fraction and random_fraction must be non-negative")
    if args.shard_size <= 0:
        raise ValueError("shard_size must be positive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(out_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {out_dir}")

    samples = _load_scienceqa_samples(base_dir=args.base_dir, split=args.split, limit=args.limit)
    teacher_att_key, teacher_splus_key, target_key = _resolve_teacher_keys(args.teacher_type)
    storage_dtype = _parse_storage_dtype(args.storage_dtype)
    target_dtype = _parse_target_dtype(args.target_dtype)

    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": _parse_torch_dtype(args.torch_dtype),
    }
    if args.device is None and args.device_map is not None:
        model_kwargs["device_map"] = args.device_map

    run_config_payload = _build_run_config_payload(args, samples)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config_payload, f, indent=2)

    print(f"Loading LLaVA processor and model: {args.implementation_model_name}")
    processor = AutoProcessor.from_pretrained(args.implementation_model_name)
    model = LlavaForConditionalGeneration.from_pretrained(args.implementation_model_name, **model_kwargs)
    configure_llava_processor(processor, model.config)
    if args.device is not None:
        model = model.to(args.device)
    model.eval()

    writer = XYShardWriter(
        output_dir=out_dir,
        split_name=args.split,
        max_rows_per_shard=args.shard_size,
        include_sample_ids=args.include_sample_ids,
    )
    rng = np.random.default_rng(args.seed)

    summary = {
        "base_dir": str(Path(args.base_dir).resolve()),
        "split": args.split,
        "teacher_type": args.teacher_type,
        "n_samples_requested": len(samples),
        "n_samples_succeeded": 0,
        "n_samples_failed": 0,
        "n_rows_written": 0,
        "n_shards_written": 0,
        "failures": [],
    }

    for sample in tqdm(samples, desc=f"Collecting ScienceQA {args.split} XY shards"):
        try:
            record = _collect_postvision_minimal_sample(
                model=model,
                processor=processor,
                sample=sample,
                prompt_template=args.prompt_template,
                capture_vproj=True,
                save_hidden_image=True,
                save_teacher_scores=True,
                teacher_head_reduction=args.teacher_head_reduction,
                teacher_att_key=teacher_att_key,
                teacher_splus_key=teacher_splus_key,
                storage_dtype=storage_dtype,
            )

            hidden_image = record["hidden_image"]
            teacher = record[target_key]
            if not isinstance(hidden_image, torch.Tensor) or hidden_image.ndim != 3:
                raise ValueError(
                    f"hidden_image must be [L, I, D], got {type(hidden_image)} {getattr(hidden_image, 'shape', None)}"
                )
            if not isinstance(teacher, torch.Tensor) or teacher.ndim != 2:
                raise ValueError(
                    f"teacher must be [L, I], got {type(teacher)} {getattr(teacher, 'shape', None)}"
                )
            if hidden_image.shape[:2] != teacher.shape:
                raise ValueError(
                    f"hidden_image/teacher mismatch: hidden={tuple(hidden_image.shape)}, teacher={tuple(teacher.shape)}"
                )

            x_all = []
            y_all = []
            layer_all = []
            sample_id_all: list[str] = []

            for layer_idx in range(int(hidden_image.shape[0])):
                layer_hidden = hidden_image[layer_idx].detach().cpu()
                layer_teacher = teacher[layer_idx].detach().cpu().float()

                if args.sample_image_tokens == "all":
                    selected = torch.arange(layer_hidden.shape[0], dtype=torch.long)
                else:
                    selected = _select_token_indices_top_random(
                        scores=layer_teacher,
                        rng=rng,
                        top_fraction=args.top_fraction,
                        random_fraction=args.random_fraction,
                        max_tokens_per_layer=args.max_tokens_per_layer,
                        min_tokens_per_layer=args.min_tokens_per_layer,
                    )

                if selected.numel() == 0:
                    continue

                x = layer_hidden.index_select(0, selected).contiguous()
                y = layer_teacher.index_select(0, selected).contiguous()
                y = _apply_target_transform(y, target_transform=args.target_transform, target_eps=args.target_eps)
                layer_ids = torch.full((selected.shape[0],), layer_idx, dtype=torch.long)

                x_all.append(x)
                y_all.append(y.to(dtype=target_dtype))
                layer_all.append(layer_ids)
                if args.include_sample_ids:
                    sample_id_all.extend([str(sample["sample_id"])] * int(selected.shape[0]))

            if x_all:
                writer.add(
                    x=torch.cat(x_all, dim=0),
                    y=torch.cat(y_all, dim=0),
                    layer_ids=torch.cat(layer_all, dim=0),
                    sample_ids=sample_id_all,
                )

            summary["n_samples_succeeded"] += 1
        except Exception as exc:
            summary["n_samples_failed"] += 1
            summary["failures"].append({"sample_id": sample["sample_id"], "error": str(exc)})
            if not args.continue_on_error:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    writer.flush()
    writer.write_dataset_spec()
    summary["n_rows_written"] = writer.rows_written
    summary["n_shards_written"] = writer.shards_written

    with (out_dir / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
