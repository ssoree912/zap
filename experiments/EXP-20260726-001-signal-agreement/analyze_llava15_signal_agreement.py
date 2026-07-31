# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare LLaVA-1.5 prefill attention and Q-ViK scores to future utility.

The extraction path intentionally keeps every comparison in visual-token space:

* prefill: user-question query rows -> visual-key columns;
* future: the exact saved Q-ViK future-attention teacher;
* Q-ViK: the saved base student's per-layer visual-token predictions.

Run one extraction process per dataset/GPU, then invoke ``--summarize`` once.
Each sample is saved independently, making interrupted extraction resumable.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from scipy.stats import rankdata


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
QVIK_ROOT = Path(os.environ.get("QVIK_ROOT", WORKSPACE_ROOT / "Q-ViK")).resolve()
for _path in (QVIK_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _patch_dependency_version_checks() -> None:
    """Match the compatibility shim used by the LLaVA-1.5 student trainer."""

    original_version = importlib_metadata.version

    def version(package_name: str) -> str:
        if package_name == "tokenizers":
            return "0.20.3"
        if package_name == "huggingface-hub":
            return "0.26.5"
        return original_version(package_name)

    importlib_metadata.version = version


_patch_dependency_version_checks()

from transformers import AutoTokenizer  # noqa: E402

from qvik.llava15.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from qvik.llava15.mm_utils import tokenizer_image_token  # noqa: E402
from qvik.llava15.model.language_model.llava_llama import (  # noqa: E402
    LlavaLlamaForCausalLM,
)

from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402


DEFAULT_TEACHER_ROOT = WORKSPACE_ROOT / "data/train/teacher/zap_llava15_n600_seed0"
DEFAULT_STUDENT_DIR = (
    REPO_ROOT
    / "artifacts/rebuttal_tradeoff_llava15_zap_teacher_n600/checkpoints/base"
)
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models/llava-v1.5-7b"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts/rebuttal_signal_agreement_llava15_base_n600"
)
DEFAULT_DATASETS = ("textvqa", "scienceqa", "gqa")
PREFILL_SIGNALS = (
    "prefill_question_mean",
    "prefill_question_mean_smooth3x3",
    "prefill_question_last",
    "prefill_last_prompt",
)
PAIR_NAMES = tuple(f"{name}_vs_future" for name in PREFILL_SIGNALS) + (
    "qvik_vs_future",
    "qvik_vs_prefill_question_mean",
    "qvik_vs_prefill_question_mean_smooth3x3",
)
METRIC_NAMES = ("spearman", "cosine", "jaccard_10", "jaccard_20", "jaccard_50")
PRIMARY_PREFILL_PAIR = "prefill_question_mean_vs_future"
PRIMARY_QVIK_PAIR = "qvik_vs_future"


def _patch_generation_config_nested_dicts() -> None:
    """Support the nested dicts in the original LLaVA checkpoint config."""

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
    """Make the vendored original-LLaVA path accept a DynamicCache."""

    import qvik.llava15.model.llava_arch as llava_arch

    original = llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal
    if getattr(original, "_qvik_dynamic_cache_compatible", False):
        return

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
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (input_ids.shape[0], past_len + 1),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                else:
                    attention_mask = torch.ones(
                        (attention_mask.shape[0], past_len + 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
            return input_ids, attention_mask, past_key_values, None, labels
        return original(self, input_ids, attention_mask, past_key_values, labels, images)

    patched._qvik_dynamic_cache_compatible = True
    llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--student-dir", type=Path, default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _resolve_image_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((REPO_ROOT / path, WORKSPACE_ROOT / path))
    path_text = str(path)
    if path_text.startswith("/workspace/zap/"):
        candidates.append(REPO_ROOT / path_text.removeprefix("/workspace/zap/"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve image path {path_value}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def list_teacher_files(
    teacher_root: Path,
    datasets: Sequence[str],
    n_per_dataset: int | None,
    seed: int,
) -> list[Path]:
    """Reproduce the trainer's exact deterministic teacher-file listing."""

    rng = random.Random(seed)
    files: list[Path] = []
    for dataset in datasets:
        dataset_files = sorted((teacher_root / dataset).glob("*.pt"))
        if n_per_dataset is not None and len(dataset_files) > n_per_dataset:
            dataset_files = sorted(rng.sample(dataset_files, n_per_dataset))
        files.extend(dataset_files)
    return files


def recover_validation_paths(student_dir: Path) -> set[Path]:
    """Recover the exact held-out paths used while training the checkpoint."""

    config = _load_json(student_dir / "train_config.json")
    teacher_root = Path(config["teacher_root"]).resolve()
    all_files = list_teacher_files(
        teacher_root,
        config["datasets"],
        config.get("n_per_dataset"),
        int(config["seed"]),
    )
    random.Random(int(config["seed"])).shuffle(all_files)
    n_val = max(1, int(round(len(all_files) * float(config["val_ratio"]))))
    return {path.resolve() for path in all_files[-n_val:]}


def infer_image_positions(
    input_ids: torch.Tensor,
    image_feature_len: int,
) -> tuple[torch.Tensor, int, int]:
    raw_ids = input_ids[0].detach().cpu()
    placeholders = torch.where(raw_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if len(placeholders) != 1:
        raise ValueError(f"Expected one image placeholder, found {len(placeholders)}")
    placeholder = int(placeholders[0])
    positions = torch.arange(
        placeholder,
        placeholder + image_feature_len,
        dtype=torch.long,
    )
    expanded_length = int(raw_ids.numel() - 1 + image_feature_len)
    return positions, expanded_length, placeholder


def _tokenize_prompt(text: str, tokenizer: Any) -> torch.Tensor:
    return tokenizer_image_token(
        text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).to(dtype=torch.long)


def infer_user_question_positions(
    *,
    prompt: str,
    question: str,
    tokenizer: Any,
    full_input_ids: torch.Tensor,
    image_placeholder: int,
    image_feature_len: int,
) -> torch.Tensor:
    """Locate only the user-question tokens in the expanded multimodal prompt."""

    char_start = prompt.find(question)
    if char_start < 0:
        raise ValueError("Question text is not a substring of prompt_text")
    if prompt.find(question, char_start + 1) >= 0:
        raise ValueError("Question text occurs more than once in prompt_text")
    char_end = char_start + len(question)

    prefix_ids = _tokenize_prompt(prompt[:char_start], tokenizer)
    question_end_ids = _tokenize_prompt(prompt[:char_end], tokenizer)
    raw_full = full_input_ids[0].detach().cpu()
    if not torch.equal(raw_full[: prefix_ids.numel()], prefix_ids):
        raise ValueError("Tokenized question prefix is not a prefix of the full prompt")
    if not torch.equal(raw_full[: question_end_ids.numel()], question_end_ids):
        raise ValueError("Tokenized question end is not a prefix of the full prompt")

    raw_start = int(prefix_ids.numel())
    raw_end = int(question_end_ids.numel())
    raw_positions = torch.arange(raw_start, raw_end, dtype=torch.long)
    raw_positions = raw_positions[raw_positions != image_placeholder]
    expanded = torch.where(
        raw_positions < image_placeholder,
        raw_positions,
        raw_positions + image_feature_len - 1,
    )
    if expanded.numel() == 0:
        raise ValueError("Question token span is empty")
    return expanded


def normalize_nonnegative(values: torch.Tensor, eps: float) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    if torch.any(values < 0):
        raise ValueError("Expected a nonnegative attention signal")
    shifted = values + eps
    return shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(eps)


def smooth_visual_grid(
    values: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Apply edge-corrected 3x3 average smoothing to each layer."""

    if values.ndim != 2 or values.shape[1] != grid_h * grid_w:
        raise ValueError(
            f"Expected [L, {grid_h * grid_w}], got {tuple(values.shape)}"
        )
    grids = values.reshape(values.shape[0], 1, grid_h, grid_w)
    valid = torch.ones_like(grids)
    summed = torch_f.avg_pool2d(
        grids,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=True,
        divisor_override=1,
    )
    counts = torch_f.avg_pool2d(
        valid,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=True,
        divisor_override=1,
    )
    return (summed / counts.clamp_min(1)).reshape_as(values)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(np.float64, copy=False)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _topk_indices(values: np.ndarray, ratio: float) -> np.ndarray:
    k = max(1, int(round(values.size * ratio)))
    return np.argsort(values, kind="stable")[-k:]


def _jaccard(first: np.ndarray, second: np.ndarray, ratio: float) -> float:
    first_top = set(_topk_indices(first, ratio).tolist())
    second_top = set(_topk_indices(second, ratio).tolist())
    return len(first_top & second_top) / len(first_top | second_top)


def compute_pair_metrics(
    first: torch.Tensor | np.ndarray,
    second: torch.Tensor | np.ndarray,
) -> dict[str, list[float]]:
    """Compute each metric independently at every layer."""

    first_np = np.asarray(
        first.detach().cpu().numpy() if isinstance(first, torch.Tensor) else first,
        dtype=np.float64,
    )
    second_np = np.asarray(
        second.detach().cpu().numpy() if isinstance(second, torch.Tensor) else second,
        dtype=np.float64,
    )
    if first_np.shape != second_np.shape or first_np.ndim != 2:
        raise ValueError(
            f"Expected matching [L, N_visual] signals, got "
            f"{first_np.shape} and {second_np.shape}"
        )
    output = {metric: [] for metric in METRIC_NAMES}
    for layer in range(first_np.shape[0]):
        first_layer = first_np[layer]
        second_layer = second_np[layer]
        output["spearman"].append(_spearman(first_layer, second_layer))
        output["cosine"].append(_cosine(first_layer, second_layer))
        output["jaccard_10"].append(_jaccard(first_layer, second_layer, 0.10))
        output["jaccard_20"].append(_jaccard(first_layer, second_layer, 0.20))
        output["jaccard_50"].append(_jaccard(first_layer, second_layer, 0.50))
    return output


def _load_model_and_student(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, VisualUtilityStudent, int]:
    _patch_generation_config_nested_dicts()
    _patch_llava_arch_for_dynamic_cache()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    image_feature_len = int(getattr(vision_tower, "num_patches", 576))
    student = VisualUtilityStudent.from_pretrained(
        args.student_dir,
        map_location="cpu",
    ).to(device).eval()
    return tokenizer, model, image_processor, student, image_feature_len


@torch.inference_mode()
def extract_one(
    *,
    record_path: Path,
    is_validation: bool,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    student: VisualUtilityStudent,
    image_feature_len: int,
    device: torch.device,
    eps: float,
) -> dict[str, Any]:
    record = torch.load(record_path, weights_only=False, map_location="cpu")
    image_path = _resolve_image_path(record["image_path"])
    with Image.open(image_path) as image:
        image_tensor = image_processor.preprocess(
            image.convert("RGB"),
            return_tensors="pt",
        )["pixel_values"]
    image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)
    input_ids = _tokenize_prompt(record["prompt_text"], tokenizer).unsqueeze(0).to(device)
    image_positions, prompt_len, placeholder = infer_image_positions(
        input_ids,
        image_feature_len,
    )
    question_positions = infer_user_question_positions(
        prompt=record["prompt_text"],
        question=record["question"],
        tokenizer=tokenizer,
        full_input_ids=input_ids,
        image_placeholder=placeholder,
        image_feature_len=image_feature_len,
    )
    student_question_positions = record["question_token_indices"].to(
        dtype=torch.long
    )

    output = model(
        input_ids=input_ids,
        images=image_tensor,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if output.attentions is None or output.hidden_states is None:
        raise RuntimeError("Backbone did not return attentions and hidden states")
    actual_prompt_len = int(output.hidden_states[-1].shape[1])
    if actual_prompt_len != prompt_len:
        raise ValueError(
            f"Expanded prompt length mismatch: inferred={prompt_len}, "
            f"actual={actual_prompt_len}"
        )
    if not torch.equal(
        image_positions,
        record["image_token_indices"].to(dtype=torch.long),
    ):
        raise ValueError("Image-token positions differ from the teacher record")
    if int(question_positions.max()) >= actual_prompt_len:
        raise ValueError("User-question token index exceeds expanded prompt length")

    image_idx = image_positions.to(device=device)
    user_q_idx = question_positions.to(device=device)
    student_q_idx = student_question_positions.to(device=device)
    last_user_q_idx = user_q_idx[-1:]
    last_prompt_idx = torch.tensor(
        [actual_prompt_len - 1],
        dtype=torch.long,
        device=device,
    )

    prefill_mean: list[torch.Tensor] = []
    prefill_last: list[torch.Tensor] = []
    prefill_last_prompt: list[torch.Tensor] = []
    qvik_logits: list[torch.Tensor] = []
    for layer_index, attention in enumerate(output.attentions):
        def aggregate(query_indices: torch.Tensor) -> torch.Tensor:
            selected = attention[0, :, query_indices, :].index_select(
                dim=-1,
                index=image_idx,
            )
            return selected.float().mean(dim=(0, 1)).detach().cpu()

        prefill_mean.append(aggregate(user_q_idx))
        prefill_last.append(aggregate(last_user_q_idx))
        prefill_last_prompt.append(aggregate(last_prompt_idx))
        hidden = output.hidden_states[layer_index + 1].to(dtype=torch.float32)
        logits = student.forward_layer(
            layer_index,
            hidden,
            image_idx,
            student_q_idx,
        ).squeeze(0)
        qvik_logits.append(logits.detach().cpu())

    signals: dict[str, torch.Tensor] = {}
    signals["prefill_question_mean"] = normalize_nonnegative(
        torch.stack(prefill_mean),
        eps,
    )
    signals["prefill_question_last"] = normalize_nonnegative(
        torch.stack(prefill_last),
        eps,
    )
    signals["prefill_last_prompt"] = normalize_nonnegative(
        torch.stack(prefill_last_prompt),
        eps,
    )
    signals["prefill_question_mean_smooth3x3"] = normalize_nonnegative(
        smooth_visual_grid(
            signals["prefill_question_mean"],
            grid_h=student.grid_h,
            grid_w=student.grid_w,
        ),
        eps,
    )
    signals["qvik"] = torch.softmax(torch.stack(qvik_logits), dim=-1)
    signals["future"] = normalize_nonnegative(
        record["teacher_norm"].to(dtype=torch.float32),
        eps,
    )

    pairs: dict[str, dict[str, list[float]]] = {}
    for prefill_name in PREFILL_SIGNALS:
        pairs[f"{prefill_name}_vs_future"] = compute_pair_metrics(
            signals[prefill_name],
            signals["future"],
        )
    pairs["qvik_vs_future"] = compute_pair_metrics(
        signals["qvik"],
        signals["future"],
    )
    pairs["qvik_vs_prefill_question_mean"] = compute_pair_metrics(
        signals["qvik"],
        signals["prefill_question_mean"],
    )
    pairs["qvik_vs_prefill_question_mean_smooth3x3"] = compute_pair_metrics(
        signals["qvik"],
        signals["prefill_question_mean_smooth3x3"],
    )

    return {
        "schema_version": 1,
        "dataset": str(record["dataset"]),
        "sample_id": str(record["sample_id"]),
        "source_record": str(record_path.resolve()),
        "split": "validation" if is_validation else "train",
        "n_layers": len(output.attentions),
        "n_visual": int(image_positions.numel()),
        "n_user_question_tokens": int(question_positions.numel()),
        "n_student_question_tokens": int(student_question_positions.numel()),
        "prompt_len_mm": actual_prompt_len,
        "teacher_steps": int(record["T"]),
        "pairs": pairs,
    }


def run_extraction(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaVA-1.5 signal extraction")
    validation_paths = recover_validation_paths(args.student_dir)
    requested_files: list[Path] = []
    for dataset in args.datasets:
        dataset_files = sorted((args.teacher_root / dataset).glob("*.pt"))
        if args.limit is not None:
            dataset_files = dataset_files[: args.limit]
        requested_files.extend(dataset_files)
    if not requested_files:
        raise RuntimeError(f"No teacher files found under {args.teacher_root}")

    print(
        f"[load] model={args.model_path} student={args.student_dir} "
        f"device={args.device}",
        flush=True,
    )
    tokenizer, model, image_processor, student, image_feature_len = (
        _load_model_and_student(args)
    )
    print(
        f"[load-ok] layers={model.config.num_hidden_layers} "
        f"visual_tokens={image_feature_len} samples={len(requested_files)}",
        flush=True,
    )

    failures: list[dict[str, str]] = []
    start = time.time()
    completed = 0
    skipped = 0
    for index, record_path in enumerate(requested_files, start=1):
        output_path = (
            args.output_dir
            / "samples"
            / record_path.parent.name
            / f"{record_path.stem}.json"
        )
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            payload = extract_one(
                record_path=record_path,
                is_validation=record_path.resolve() in validation_paths,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                student=student,
                image_feature_len=image_feature_len,
                device=torch.device(args.device),
                eps=args.eps,
            )
            _dump_json(output_path, payload)
            completed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "record": str(record_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[failed] {record_path}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            torch.cuda.empty_cache()
        if index == 1 or index % 25 == 0 or index == len(requested_files):
            elapsed = time.time() - start
            print(
                f"[progress] {index}/{len(requested_files)} completed={completed} "
                f"resumed={skipped} failed={len(failures)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    _dump_json(
        args.output_dir / "logs" / f"extract_{'_'.join(args.datasets)}.json",
        {
            "datasets": args.datasets,
            "requested": len(requested_files),
            "completed": completed,
            "resumed": skipped,
            "failed": failures,
            "elapsed_seconds": time.time() - start,
            "device": args.device,
            "teacher_root": str(args.teacher_root),
            "student_dir": str(args.student_dir),
            "model_path": str(args.model_path),
        },
    )
    return 1 if failures else 0


def _bootstrap_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    estimate = float(values.mean())
    if values.size == 1 or replicates <= 0:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    cursor = 0
    while cursor < replicates:
        chunk = min(256, replicates - cursor)
        indices = rng.integers(
            0,
            values.size,
            size=(chunk, values.size),
        )
        bootstrap_means[cursor : cursor + chunk] = values[indices].mean(axis=1)
        cursor += chunk
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return estimate, float(low), float(high)


def _sample_layer_mean(
    records: Sequence[dict[str, Any]],
    pair: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [np.mean(record["pairs"][pair][metric]) for record in records],
        dtype=np.float64,
    )


def _format_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def _write_summary_for_scope(
    *,
    records: Sequence[dict[str, Any]],
    scope_name: str,
    output_dir: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    import csv

    summary_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, Sequence[dict[str, Any]]]] = [("all", records)]
    groups.extend(
        (
            dataset,
            [record for record in records if record["dataset"] == dataset],
        )
        for dataset in DEFAULT_DATASETS
    )
    for group_name, group_records in groups:
        if not group_records:
            continue
        for pair_index, pair in enumerate(PAIR_NAMES):
            for metric_index, metric in enumerate(METRIC_NAMES):
                values = _sample_layer_mean(group_records, pair, metric)
                mean, low, high = _bootstrap_ci(
                    values,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + pair_index * 100 + metric_index,
                )
                summary_rows.append(
                    {
                        "scope": scope_name,
                        "group": group_name,
                        "n_samples": len(group_records),
                        "pair": pair,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

        for prefill_index, prefill_signal in enumerate(PREFILL_SIGNALS):
            prefill_pair = f"{prefill_signal}_vs_future"
            delta_pair = f"qvik_minus_{prefill_signal}_vs_future"
            for metric_index, metric in enumerate(METRIC_NAMES):
                qvik = _sample_layer_mean(
                    group_records,
                    PRIMARY_QVIK_PAIR,
                    metric,
                )
                prefill = _sample_layer_mean(
                    group_records,
                    prefill_pair,
                    metric,
                )
                mean, low, high = _bootstrap_ci(
                    qvik - prefill,
                    replicates=bootstrap_replicates,
                    seed=(
                        bootstrap_seed
                        + 10_000
                        + prefill_index * 100
                        + metric_index
                    ),
                )
                summary_rows.append(
                    {
                        "scope": scope_name,
                        "group": group_name,
                        "n_samples": len(group_records),
                        "pair": delta_pair,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

    summary_path = output_dir / f"summary_{scope_name}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    layer_rows: list[dict[str, Any]] = []
    n_layers = int(records[0]["n_layers"])
    layer_bootstrap_replicates = min(bootstrap_replicates, 2_000)
    for pair in (PRIMARY_PREFILL_PAIR, PRIMARY_QVIK_PAIR):
        for metric in ("spearman", "jaccard_20"):
            matrix = np.asarray(
                [record["pairs"][pair][metric] for record in records],
                dtype=np.float64,
            )
            for layer in range(n_layers):
                mean, low, high = _bootstrap_ci(
                    matrix[:, layer],
                    replicates=layer_bootstrap_replicates,
                    seed=bootstrap_seed + layer,
                )
                layer_rows.append(
                    {
                        "scope": scope_name,
                        "n_samples": len(records),
                        "pair": pair,
                        "metric": metric,
                        "layer": layer,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_replicates": layer_bootstrap_replicates,
                    }
                )
    with (output_dir / f"layerwise_{scope_name}.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    lookup = {
        (row["group"], row["pair"], row["metric"]): row for row in summary_rows
    }
    table_pairs = (
        ("Prefill attention (all user-question tokens)", PRIMARY_PREFILL_PAIR),
        (
            "Smoothed prefill attention (3x3)",
            "prefill_question_mean_smooth3x3_vs_future",
        ),
        ("Prefill attention (last user-question token)", "prefill_question_last_vs_future"),
        ("Prefill attention (last prompt token)", "prefill_last_prompt_vs_future"),
        ("Q-ViK prediction", PRIMARY_QVIK_PAIR),
    )
    lines = [
        f"# LLaVA-1.5 signal agreement ({scope_name})",
        "",
        f"Samples: {len(records)}. Values are sample-level means over 32 layers; "
        f"95% CIs use {bootstrap_replicates:,} sample bootstrap replicates.",
        "",
        "| Compared signal | Spearman vs. Future | Cosine vs. Future | "
        "Jaccard@10% | Jaccard@20% | Jaccard@50% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, pair in table_pairs:
        cells = []
        for metric in METRIC_NAMES:
            row = lookup[("all", pair, metric)]
            cells.append(
                _format_ci(row["mean"], row["ci95_low"], row["ci95_high"])
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| Direct comparison | Spearman | Cosine |",
            "|---|---:|---:|",
        ]
    )
    direct_pair = "qvik_vs_prefill_question_mean"
    direct_cells = []
    for metric in ("spearman", "cosine"):
        row = lookup[("all", direct_pair, metric)]
        direct_cells.append(
            _format_ci(row["mean"], row["ci95_low"], row["ci95_high"])
        )
    lines.append(
        "| Q-ViK prediction vs. prefill attention | "
        + " | ".join(direct_cells)
        + " |"
    )
    lines.extend(
        [
            "",
            "| Q-ViK − Prefill agreement with Future | Delta (95% CI) |",
            "|---|---:|",
        ]
    )
    delta_pair = "qvik_minus_prefill_question_mean_vs_future"
    for metric in METRIC_NAMES:
        row = lookup[("all", delta_pair, metric)]
        lines.append(
            f"| {metric} | "
            f"{_format_ci(row['mean'], row['ci95_low'], row['ci95_high'])} |"
        )
    lines.extend(
        [
            "",
            "| Alternative prefill baseline | Spearman delta | Cosine delta | "
            "Jaccard@20% delta |",
            "|---|---:|---:|---:|",
        ]
    )
    alternative_deltas = (
        (
            "Smoothed all-question",
            "qvik_minus_prefill_question_mean_smooth3x3_vs_future",
        ),
        ("Last user-question token", "qvik_minus_prefill_question_last_vs_future"),
        ("Last prompt token", "qvik_minus_prefill_last_prompt_vs_future"),
    )
    for label, pair in alternative_deltas:
        cells = []
        for metric in ("spearman", "cosine", "jaccard_20"):
            row = lookup[("all", pair, metric)]
            cells.append(
                _format_ci(row["mean"], row["ci95_low"], row["ci95_high"])
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (output_dir / f"table_{scope_name}.md").write_text("\n".join(lines) + "\n")

    _make_layer_figure(
        layer_rows=layer_rows,
        n_samples=len(records),
        scope_name=scope_name,
        output_dir=output_dir,
    )


def _make_layer_figure(
    *,
    layer_rows: Sequence[dict[str, Any]],
    n_samples: int,
    scope_name: str,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    specifications = (
        ("spearman", "Spearman vs. Future"),
        ("jaccard_20", "Jaccard@20% vs. Future"),
    )
    pairs = (
        (PRIMARY_PREFILL_PAIR, "Prefill vs. Future", "#4C78A8"),
        (PRIMARY_QVIK_PAIR, "Q-ViK vs. Future", "#E45756"),
    )
    for axis, (metric, title) in zip(axes, specifications, strict=True):
        for pair, label, color in pairs:
            selected = sorted(
                (
                    row
                    for row in layer_rows
                    if row["pair"] == pair and row["metric"] == metric
                ),
                key=lambda row: row["layer"],
            )
            layers = np.asarray([row["layer"] for row in selected])
            means = np.asarray([row["mean"] for row in selected])
            lows = np.asarray([row["ci95_low"] for row in selected])
            highs = np.asarray([row["ci95_high"] for row in selected])
            axis.plot(layers, means, label=label, color=color, linewidth=2)
            axis.fill_between(layers, lows, highs, color=color, alpha=0.16)
        axis.set_title(title)
        axis.set_xlabel("Layer")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Agreement")
    axes[0].legend(frameon=False)
    figure.suptitle(f"LLaVA-1.5 visual-token agreement ({scope_name}, n={n_samples})")
    figure.tight_layout()
    figure.savefig(output_dir / f"layerwise_{scope_name}.png", dpi=220)
    figure.savefig(output_dir / f"layerwise_{scope_name}.pdf")
    plt.close(figure)


def run_summary(args: argparse.Namespace) -> int:
    sample_paths = sorted((args.output_dir / "samples").glob("*/*.json"))
    records = [_load_json(path) for path in sample_paths]
    if not records:
        raise RuntimeError(f"No sample results under {args.output_dir / 'samples'}")
    expected_pairs = set(PAIR_NAMES)
    for record in records:
        if set(record["pairs"]) != expected_pairs:
            raise ValueError(
                f"{record['dataset']}:{record['sample_id']} has unexpected pair schema"
            )

    summary_dir = args.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    validation = [record for record in records if record["split"] == "validation"]
    _write_summary_for_scope(
        records=validation,
        scope_name="heldout",
        output_dir=summary_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_summary_for_scope(
        records=records,
        scope_name="all1800",
        output_dir=summary_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    manifest = {
        "schema_version": 1,
        "n_samples": len(records),
        "n_validation": len(validation),
        "datasets": {
            dataset: {
                "all": sum(record["dataset"] == dataset for record in records),
                "validation": sum(
                    record["dataset"] == dataset for record in validation
                ),
            }
            for dataset in DEFAULT_DATASETS
        },
        "teacher_root": str(args.teacher_root),
        "student_dir": str(args.student_dir),
        "model_path": str(args.model_path),
        "primary_scope": "heldout",
        "primary_prefill": "mean attention from actual user-question tokens",
        "bootstrap_unit": "sample",
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    _dump_json(summary_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    args.teacher_root = args.teacher_root.resolve()
    args.student_dir = args.student_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.summarize:
        return run_summary(args)
    return run_extraction(args)


if __name__ == "__main__":
    raise SystemExit(main())
