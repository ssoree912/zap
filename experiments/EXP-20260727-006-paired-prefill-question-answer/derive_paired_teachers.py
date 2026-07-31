#!/usr/bin/env python3
"""Build strictly paired P+A and Q+A teacher records for LLaVA-1.5.

The saved QA50 records are the source of truth for sample identity and for the
answer-time component.  Q+A is recomposed from the saved question and answer
components.  P+A performs one new prompt-only forward pass and combines its
all-prefill attention with the exact same saved answer component.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
WORKSPACE_ROOT = ZAP_ROOT.parent
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
QVIK_ROOT = Path(os.environ.get("QVIK_ROOT", WORKSPACE_ROOT / "Q-ViK")).resolve()
for path in (QVIK_ROOT, ZAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from transformers import AutoTokenizer  # noqa: E402

from qvik.llava15.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from qvik.llava15.mm_utils import tokenizer_image_token  # noqa: E402
from qvik.llava15.model.language_model.llava_llama import (  # noqa: E402
    LlavaLlamaForCausalLM,
)


SOURCE_ROOT_DEFAULT = (
    WORKSPACE_ROOT / "data/train/teacher/zap_llava15_qa50_n600_seed0"
)
PA_ROOT_DEFAULT = (
    WORKSPACE_ROOT
    / "data/train/teacher/zap_llava15_prefill_answer50_paired_n600_seed0"
)
QA_ROOT_DEFAULT = (
    WORKSPACE_ROOT
    / "data/train/teacher/zap_llava15_question_answer50_paired_n600_seed0"
)
MODEL_DEFAULT = WORKSPACE_ROOT / "models/llava-v1.5-7b"
DATASETS = ("gqa", "textvqa", "scienceqa")
EXPECTED_SHAPE = (32, 576)
EPS = 1e-8


def _patch_generation_config_nested_dicts() -> None:
    from transformers.generation import configuration_utils

    original = configuration_utils.GenerationConfig.from_model_config.__func__

    @classmethod
    def patched(cls, model_config):
        for attr in ("decoder", "encoder", "text_config", "vision_config"):
            value = getattr(model_config, attr, None)
            if isinstance(value, dict):
                namespace = SimpleNamespace(**value)
                namespace.to_dict = lambda payload=value: payload
                setattr(model_config, attr, namespace)
        return original(cls, model_config)

    configuration_utils.GenerationConfig.from_model_config = patched


def _patch_llava_arch_for_dynamic_cache() -> None:
    import qvik.llava15.model.llava_arch as llava_arch

    original = llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal

    def patched(self, input_ids, attention_mask, past_key_values, labels, images):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                if hasattr(past_key_values, "get_seq_length"):
                    past_len = past_key_values.get_seq_length()
                else:
                    past_len = max(cache[-1].shape[-2] for cache in past_key_values)
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_len + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels
        return original(
            self,
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
        )

    llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = patched


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return value / value.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(record: dict[str, Any], source: Path) -> None:
    required = (
        "sample_id",
        "dataset",
        "prompt_text",
        "image_path",
        "image_token_indices",
        "question_token_indices",
        "teacher_question_norm",
        "teacher_answer_raw",
        "teacher_answer_norm",
        "prompt_len_mm",
        "T",
        "n_img",
        "max_new_tokens",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{source}: missing keys {missing}")
    if int(record["max_new_tokens"]) != 64:
        raise ValueError(f"{source}: expected max_new_tokens=64")
    if int(record["n_img"]) != EXPECTED_SHAPE[1]:
        raise ValueError(f"{source}: expected n_img=576")
    for key in (
        "teacher_question_norm",
        "teacher_answer_raw",
        "teacher_answer_norm",
    ):
        if tuple(record[key].shape) != EXPECTED_SHAPE:
            raise ValueError(
                f"{source}: {key} shape={tuple(record[key].shape)}, "
                f"expected={EXPECTED_SHAPE}"
            )
        if not torch.isfinite(record[key].float()).all():
            raise ValueError(f"{source}: {key} contains non-finite values")
    if not torch.allclose(
        record["teacher_answer_norm"].float().sum(dim=-1),
        torch.ones(EXPECTED_SHAPE[0]),
        atol=2e-3,
        rtol=0,
    ):
        raise ValueError(f"{source}: answer teacher is not layer-wise normalized")


def _atomic_torch_save(record: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(record, temporary)
    os.replace(temporary, destination)


def _paired_metadata(
    record: dict[str, Any],
    source_path: Path,
    first_signal: str,
) -> dict[str, Any]:
    return {
        "paired_control_schema": 1,
        "paired_source_path": str(source_path),
        "paired_source_file_sha256": _file_sha256(source_path),
        "paired_answer_norm_sha256": _tensor_sha256(record["teacher_answer_norm"]),
        "paired_answer_query_rows": "[y_1, ..., y_T]",
        "paired_answer_max_new_tokens": 64,
        "paired_first_signal": first_signal,
        "paired_first_weight": 0.5,
        "paired_answer_weight": 0.5,
        "paired_normalization": (
            "independent visual-token L1 per layer, then 0.5/0.5 mix"
        ),
        "paired_student_conditioning_indices": (
            "unchanged post-image prompt-tail question_token_indices"
        ),
    }


def _compose_qa(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    question_norm = _normalize(record["teacher_question_norm"])
    answer_norm = _normalize(record["teacher_answer_norm"])
    mixed = _normalize(0.5 * question_norm + 0.5 * answer_norm)
    output = dict(record)
    output.update(
        {
            "teacher_raw": mixed.to(torch.float16),
            "teacher_norm": mixed.to(torch.float16),
            "teacher_question_norm": question_norm.to(torch.float16),
            "teacher_question_weight": 0.5,
            "teacher_answer_weight": 0.5,
            "teacher_signal": "paired_question_answer_normalized_mix",
            "teacher_question_source": (
                "all post-image prompt-tail causal query rows"
            ),
            **_paired_metadata(record, source_path, "question_prompt_tail"),
        }
    )
    return output


def _load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, int]:
    _patch_generation_config_nested_dicts()
    _patch_llava_arch_for_dynamic_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    if image_feature_len != EXPECTED_SHAPE[1]:
        raise ValueError(f"Expected 576 visual tokens, got {image_feature_len}")
    return tokenizer, model, vision_tower.image_processor, image_feature_len


@torch.no_grad()
def _compose_pa(
    record: dict[str, Any],
    source_path: Path,
    *,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    image_path = Path(str(record["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        image_tensor = image_processor.preprocess(
            image.convert("RGB"),
            return_tensors="pt",
        )["pixel_values"]
    image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)
    input_ids = tokenizer_image_token(
        str(record["prompt_text"]),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    raw_ids = input_ids[0].detach().cpu()
    placeholders = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholders) != 1:
        raise ValueError(f"{source_path}: expected one image placeholder")
    image_start = int(placeholders[0])
    prompt_len_mm = int(raw_ids.numel() - 1 + EXPECTED_SHAPE[1])
    image_positions = torch.arange(
        image_start,
        image_start + EXPECTED_SHAPE[1],
        dtype=torch.long,
    )
    if prompt_len_mm != int(record["prompt_len_mm"]):
        raise ValueError(
            f"{source_path}: prompt length drift "
            f"{prompt_len_mm} != {record['prompt_len_mm']}"
        )
    if not torch.equal(
        image_positions,
        record["image_token_indices"].cpu().long(),
    ):
        raise ValueError(f"{source_path}: image-token position drift")

    output = model(
        input_ids=input_ids,
        images=image_tensor,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    if output.attentions is None or len(output.attentions) != EXPECTED_SHAPE[0]:
        raise RuntimeError(f"{source_path}: missing or malformed prompt attentions")
    visual_indices = image_positions.to(device)
    prefill_raw = torch.empty(EXPECTED_SHAPE, dtype=torch.float32)
    for layer_index, attention in enumerate(output.attentions):
        if attention.shape[-2:] != (prompt_len_mm, prompt_len_mm):
            raise ValueError(
                f"{source_path}: layer {layer_index} attention shape "
                f"{tuple(attention.shape)}"
            )
        all_prompt_to_visual = attention[0].index_select(
            dim=-1,
            index=visual_indices,
        )
        prefill_raw[layer_index] = (
            all_prompt_to_visual.float().mean(dim=(0, 1)).detach().cpu()
        )
    prefill_norm = _normalize(prefill_raw)
    answer_norm = _normalize(record["teacher_answer_norm"])
    mixed = _normalize(0.5 * prefill_norm + 0.5 * answer_norm)

    result = dict(record)
    result.update(
        {
            "teacher_raw": mixed.to(torch.float16),
            "teacher_norm": mixed.to(torch.float16),
            "teacher_prefill_raw": prefill_raw.to(torch.float16),
            "teacher_prefill_norm": prefill_norm.to(torch.float16),
            "teacher_prefill_weight": 0.5,
            "teacher_answer_weight": 0.5,
            "teacher_signal": "paired_all_prefill_answer_normalized_mix",
            "teacher_prefill_source": (
                "all causal expanded-multimodal prompt query rows"
            ),
            "prefill_query_indices": torch.arange(prompt_len_mm, dtype=torch.long),
            **_paired_metadata(record, source_path, "all_prefill"),
        }
    )
    del output, input_ids, image_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _selected_paths(
    source_root: Path,
    dataset: str,
    shard_index: int,
    num_shards: int,
) -> list[Path]:
    paths = sorted((source_root / dataset).glob("*.pt"))
    if len(paths) != 600:
        raise ValueError(f"{dataset}: expected 600 source records, found {len(paths)}")
    return [
        path
        for index, path in enumerate(paths)
        if index % num_shards == shard_index
    ]


def _write_worker_summary(
    output_root: Path,
    mode: str,
    dataset: str,
    shard_index: int,
    num_shards: int,
    saved: int,
    elapsed: float,
) -> None:
    summary = {
        "mode": mode,
        "dataset": dataset,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_saved": saved,
        "elapsed_seconds": elapsed,
    }
    path = (
        output_root
        / dataset
        / f"_worker_{mode}_{shard_index:02d}_of_{num_shards:02d}.json"
    )
    path.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("question_answer", "prefill_answer"),
        required=True,
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument("--qa-output-root", type=Path, default=QA_ROOT_DEFAULT)
    parser.add_argument("--pa-output-root", type=Path, default=PA_ROOT_DEFAULT)
    parser.add_argument("--model-path", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    if args.mode == "question_answer" and (
        args.shard_index != 0 or args.num_shards != 1
    ):
        raise ValueError("Q+A recomposition is CPU-only and unsharded")

    tokenizer = model = image_processor = None
    device = torch.device(args.device)
    if args.mode == "prefill_answer":
        tokenizer, model, image_processor, _ = _load_model(
            args.model_path,
            device,
        )
    output_root = (
        args.qa_output_root
        if args.mode == "question_answer"
        else args.pa_output_root
    )

    for dataset in args.datasets:
        paths = _selected_paths(
            args.source_root,
            dataset,
            args.shard_index,
            args.num_shards,
        )
        started = time.time()
        saved = 0
        print(
            f"[start] mode={args.mode} dataset={dataset} "
            f"shard={args.shard_index}/{args.num_shards} n={len(paths)}",
            flush=True,
        )
        for index, source_path in enumerate(paths, 1):
            destination = output_root / dataset / source_path.name
            if destination.exists() and not args.overwrite:
                saved += 1
                continue
            record = torch.load(
                source_path,
                map_location="cpu",
                weights_only=False,
            )
            _validate_source(record, source_path)
            if str(record["dataset"]) != dataset:
                raise ValueError(f"{source_path}: dataset metadata mismatch")
            if args.mode == "question_answer":
                derived = _compose_qa(record, source_path)
            else:
                derived = _compose_pa(
                    record,
                    source_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    device=device,
                )
            _atomic_torch_save(derived, destination)
            saved += 1
            if index == 1 or index % 25 == 0 or index == len(paths):
                rate = index / max(time.time() - started, 1e-6)
                print(
                    f"[progress] mode={args.mode} dataset={dataset} "
                    f"{index}/{len(paths)} saved={saved} rate={rate:.3f}/s",
                    flush=True,
                )
        elapsed = time.time() - started
        _write_worker_summary(
            output_root,
            args.mode,
            dataset,
            args.shard_index,
            args.num_shards,
            saved,
            elapsed,
        )
        print(
            f"[done] mode={args.mode} dataset={dataset} "
            f"saved={saved} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
