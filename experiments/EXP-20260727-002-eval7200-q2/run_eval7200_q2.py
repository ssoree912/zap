#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Q2 selector comparison for standard eval and all local MileBench tasks.

The numerical implementation is inherited from the audited eval-700 runner.
This wrapper adds:

* 29 collision-free ``milebench__*`` datasets;
* LOOK-compatible prompt construction/scoring;
* explicit MileBench visual-vs-total budget modes;
* per-layer forward hooks that compress long prefill attention immediately,
  avoiding retention of 32 full ``N x N`` matrices.

For a standalone prefill attention matrix ``A`` the hook stores only two rows:

``row_0 = mean(A[MileBench semantic user-prompt rows])``
``row_1 = sum(A[all rows]) - row_0``

The inherited runner therefore recovers the exact H2O cumulative score by
summing the two compact rows and the exact question score from row 0. During
the second long forward (``generate`` prefill), only the final prompt-query row
is retained, which is exactly the first Future-utility step. Autoregressive
one-row attention is left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
WORKSPACE_ROOT = ZAP_ROOT.parent
QVIK_ROOT = WORKSPACE_ROOT / "Q-ViK"
for _path in (QVIK_ROOT, ZAP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BASE_EXP = ZAP_ROOT / "experiments" / "EXP-20260727-001-eval700-q2"
DEFAULT_OUTPUT = (
    ZAP_ROOT / "artifacts" / "rebuttal_eval7200_q2_llava15_keep0p2"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "eval7200_base_runner",
    BASE_EXP / "run_eval700_q2.py",
)
data = load_module("eval7200_data", EXP_DIR / "eval7200_data.py")
compaction = load_module(
    "eval7200_attention_compaction",
    EXP_DIR / "attention_compaction.py",
)

DEFAULT_DATASETS = tuple(data.MILEBENCH_DATASETS)
ALL_DATASETS = tuple(data.DATASETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=data.DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--milebench-root",
        type=Path,
        default=data.DEFAULT_MILEBENCH_ROOT,
    )
    parser.add_argument(
        "--student-dir",
        type=Path,
        default=base.q2.agreement.DEFAULT_STUDENT_DIR,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=base.q2.agreement.DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=(
            "Prefixed dataset names, original MileBench task names, or one of "
            "{milebench, standard, all}."
        ),
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--keep-ratio",
        "--total-keep-ratio",
        dest="keep_ratio",
        type=float,
        default=0.2,
        help=(
            "Retention ratio. Standard tasks always interpret it as total "
            "prompt-cache retention. MileBench follows "
            "--milebench-budget-mode."
        ),
    )
    parser.add_argument(
        "--milebench-budget-mode",
        choices=("visual", "total"),
        default="visual",
        help=(
            "visual retains ceil(r*N_visual) visual KVs (default); total uses "
            "ceil(r*N_prompt)-N_text and fails if no visual KV fits."
        ),
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=data.DEFAULT_MAX_PROMPT_TOKENS,
    )
    parser.add_argument(
        "--milebench-max-new-tokens",
        type=int,
        help=(
            "Override per-task generation caps. Default: 64 for short tasks "
            "and 128 for IEdit/Spot-the-Diff/Needle/MMCoQA."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--summary-seed", type=int, default=20260727)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    # The inherited summary API reads this historical attribute.
    args.total_keep_ratio = args.keep_ratio
    args.datasets = expand_dataset_names(args.datasets)
    return args


def expand_dataset_names(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        if value == "milebench":
            expanded.extend(data.MILEBENCH_DATASETS)
        elif value == "standard":
            expanded.extend(data.STANDARD_DATASETS)
        elif value == "all":
            expanded.extend(data.DATASETS)
        elif value in data.LOCAL_MILEBENCH_TASKS:
            expanded.append(f"{data.MILEBENCH_PREFIX}{value}")
        else:
            expanded.append(value)
    output: list[str] = []
    seen: set[str] = set()
    for value in expanded:
        if value not in seen:
            output.append(value)
            seen.add(value)
    invalid = sorted(set(output) - set(ALL_DATASETS))
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")
    return output


def preflight_milebench_budget(
    *,
    sample: Any,
    tokenizer: Any,
    image_feature_len: int,
    ratio: float,
    budget_mode: str,
    max_prompt_tokens: int,
) -> tuple[int, int, int]:
    context = str(base.sample_value(sample, "context"))
    _raw, prompt_len = data.expanded_prompt_length(
        context,
        tokenizer,
        image_feature_len=image_feature_len,
    )
    if prompt_len > max_prompt_tokens:
        raise ValueError(
            f"{base.sample_value(sample, 'dataset')}/"
            f"{base.sample_value(sample, 'sample_id')}: expanded prompt "
            f"{prompt_len} exceeds cap {max_prompt_tokens}"
        )
    n_visual = int(image_feature_len)
    n_text = prompt_len - n_visual
    if budget_mode == "visual":
        n_keep = min(
            n_visual,
            max(0, int(math.ceil(float(ratio) * n_visual))),
        )
    else:
        n_keep = base.q2.exact_total_budget(
            n_visual=n_visual,
            prompt_len=prompt_len,
            total_keep_ratio=ratio,
        )
    if n_keep <= 0:
        raise ValueError(
            "MileBench eviction budget is infeasible and is not clamped: "
            f"dataset={base.sample_value(sample, 'dataset')}, "
            f"sample_id={base.sample_value(sample, 'sample_id')}, "
            f"mode={budget_mode}, ratio={ratio}, prompt={prompt_len}, "
            f"N_text={n_text}, N_visual={n_visual}, K_visual={n_keep}. "
            "Use --milebench-budget-mode visual for visual-token 20%."
        )
    return prompt_len, n_text, n_keep


@torch.inference_mode()
def run_one(
    *,
    sample: Any,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    student: Any,
    image_feature_len: int,
    device: torch.device,
    keep_ratio: float,
    milebench_budget_mode: str,
    max_prompt_tokens: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    dataset_name = str(base.sample_value(sample, "dataset"))
    if not data.is_milebench_dataset(dataset_name):
        return base.run_one(
            sample=sample,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            student=student,
            image_feature_len=image_feature_len,
            device=device,
            total_keep_ratio=keep_ratio,
        )

    prompt_len, n_text, expected_keep = preflight_milebench_budget(
        sample=sample,
        tokenizer=tokenizer,
        image_feature_len=image_feature_len,
        ratio=keep_ratio,
        budget_mode=milebench_budget_mode,
        max_prompt_tokens=max_prompt_tokens,
    )
    original_budget = base.q2.exact_total_budget
    original_score = base.score_one_prediction

    def milebench_budget(
        *,
        n_visual: int,
        prompt_len: int,
        total_keep_ratio: float,
    ) -> int:
        if milebench_budget_mode == "visual":
            return min(
                int(n_visual),
                max(
                    0,
                    int(math.ceil(float(total_keep_ratio) * int(n_visual))),
                ),
            )
        result = original_budget(
            n_visual=n_visual,
            prompt_len=prompt_len,
            total_keep_ratio=total_keep_ratio,
        )
        if result <= 0:
            raise ValueError(
                "MileBench total-cache budget leaves zero visual tokens; "
                "refusing to clamp silently"
            )
        return result

    base.q2.exact_total_budget = milebench_budget
    base.score_one_prediction = data.score_prediction
    compaction_state = None
    try:
        with compaction.compact_llava15_attentions(
            model,
            base.q2.agreement,
        ) as compaction_state:
            result, masks = base.run_one(
                sample=sample,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                student=student,
                image_feature_len=image_feature_len,
                device=device,
                total_keep_ratio=keep_ratio,
            )
    finally:
        base.q2.exact_total_budget = original_budget
        base.score_one_prediction = original_score

    if compaction_state is None:
        raise RuntimeError("Attention compaction state was not initialized")
    call_counts = dict(compaction_state.full_prefill_call_counts)
    incomplete = {
        layer: count for layer, count in call_counts.items() if count != 2
    }
    if incomplete:
        raise RuntimeError(
            "Lossless attention compaction expected exactly two full-prefill "
            f"calls per layer, got {incomplete}"
        )
    actual_question_count = compaction_state.actual_question_count
    if int(result["n_visual_kept_per_head"]) != expected_keep:
        raise AssertionError(
            f"Budget preflight/run mismatch: {expected_keep} vs "
            f"{result['n_visual_kept_per_head']}"
        )
    metadata = base.sample_value(sample, "metadata", {})
    result.update(
        {
            "budget_mode": milebench_budget_mode,
            "requested_keep_ratio": keep_ratio,
            "requested_total_keep_ratio": (
                keep_ratio if milebench_budget_mode == "total" else None
            ),
            "requested_visual_keep_ratio": (
                keep_ratio if milebench_budget_mode == "visual" else None
            ),
            "question_token_count": actual_question_count,
            "attention_storage": (
                "stream-compressed per layer: [question_mean, "
                "all_prefill_sum-question_mean]; generate prefill last row"
            ),
            "attention_output_compaction_lossless": True,
            "attention_output_compaction_full_prefill_calls": call_counts,
            "milebench_task": metadata.get("milebench_task"),
            "prompt_was_left_truncated": metadata.get(
                "prompt_was_left_truncated"
            ),
            "original_expanded_prompt_tokens": metadata.get(
                "original_expanded_prompt_tokens"
            ),
            "expanded_prompt_tokens": prompt_len,
            "preflight_n_text": n_text,
        }
    )
    return result, masks


def run_config_path(args: argparse.Namespace) -> Path:
    identity = "\n".join(args.datasets)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    worker = base.safe_name(args.worker_id or args.device)
    return args.output_dir / "run_configs" / f"{worker}_{digest}.json"


def load_dataset_samples(
    dataset_name: str,
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    image_feature_len: int,
) -> list[Any]:
    samples = data.load_samples(
        dataset_name,
        args.eval_root,
        args.n_samples,
        args.sample_seed,
        milebench_root=args.milebench_root,
        tokenizer=tokenizer,
        # MileBench composite images can occupy many gigabytes once decoded
        # (TextNeedle is especially large).  Keep only paths here and decode
        # one image immediately before its sample is evaluated.
        load_images=(
            not args.manifest_only
            and not data.is_milebench_dataset(dataset_name)
        ),
        max_prompt_tokens=args.max_prompt_tokens,
        image_feature_len=image_feature_len,
        max_new_tokens=args.milebench_max_new_tokens,
    )
    if len(samples) != args.n_samples:
        raise RuntimeError(
            f"{dataset_name}: expected {args.n_samples}, got {len(samples)}"
        )
    return samples


def extract(args: argparse.Namespace) -> int:
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.sample_start < 0:
        raise ValueError("--sample-start must be nonnegative")
    sample_end = (
        args.sample_end if args.sample_end is not None else args.n_samples
    )
    if not args.sample_start <= sample_end <= args.n_samples:
        raise ValueError(
            f"Invalid sample slice [{args.sample_start}:{sample_end}] "
            f"for n={args.n_samples}"
        )

    if args.manifest_only:
        tokenizer = data.load_tokenizer(str(args.model_path.resolve()))
        model = image_processor = student = None
        image_feature_len = data.DEFAULT_IMAGE_FEATURE_LEN
    else:
        tokenizer, model, image_processor, student, image_feature_len = (
            base.q2.load_exact_model_and_student(args)
        )
    tokenizer.model_max_length = max(
        int(getattr(tokenizer, "model_max_length", 0)),
        int(args.max_prompt_tokens),
    )
    if int(image_feature_len) != data.DEFAULT_IMAGE_FEATURE_LEN:
        raise ValueError(
            f"This adapter expects 576 LLaVA-1.5 visual tokens, got "
            f"{image_feature_len}"
        )

    config = {
        "datasets": list(args.datasets),
        "n_samples_per_dataset": args.n_samples,
        "sample_seed": args.sample_seed,
        "sample_slice": [args.sample_start, args.sample_end],
        "model_path": str(args.model_path.resolve()),
        "student_dir": str(args.student_dir.resolve()),
        "device": args.device,
        "keep_ratio": args.keep_ratio,
        "standard_budget_mode": "total",
        "milebench_budget_mode": args.milebench_budget_mode,
        "max_prompt_tokens": args.max_prompt_tokens,
        "milebench_max_new_tokens_override": args.milebench_max_new_tokens,
        "image_preprocessing": (
            "qvik.llava15.mm_utils.process_images; combined_1_images"
        ),
    }
    config_path = run_config_path(args)
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise RuntimeError(
                "Existing run-config fingerprint does not match this run: "
                f"{config_path}. Use a fresh --output-dir."
            )
    else:
        base.atomic_json(config_path, config)

    total_expected = (sample_end - args.sample_start) * len(args.datasets)
    completed = skipped = failures = 0
    started = time.time()
    for dataset_name in args.datasets:
        samples = load_dataset_samples(
            dataset_name,
            args,
            tokenizer,
            image_feature_len=image_feature_len,
        )
        data.write_manifest(
            samples,
            args.output_dir / "manifests" / f"{dataset_name}.json",
        )
        if args.manifest_only:
            print(
                f"[manifest] dataset={dataset_name} n={len(samples)}",
                flush=True,
            )
            continue

        assert model is not None
        work = samples[args.sample_start:sample_end]
        for sample in work:
            stem = (
                f"{int(base.sample_value(sample, 'row_index')):06d}_"
                f"{base.safe_name(base.sample_value(sample, 'sample_id'))}"
            )
            result_path = (
                args.output_dir / "samples" / dataset_name / f"{stem}.json"
            )
            mask_path = (
                args.output_dir / "masks" / dataset_name / f"{stem}.npz"
            )
            failure_path = (
                args.output_dir / "failures" / dataset_name / f"{stem}.txt"
            )
            if (
                result_path.exists()
                and mask_path.exists()
                and not args.overwrite
            ):
                skipped += 1
                continue
            try:
                if (
                    data.is_milebench_dataset(dataset_name)
                    and base.sample_value(sample, "image") is None
                ):
                    image_path = Path(
                        base.sample_value(sample, "metadata", {})[
                            "combined_image_path"
                        ]
                    )
                    with Image.open(image_path) as opened:
                        sample.image = opened.convert("RGB").copy()
                result, masks = run_one(
                    sample=sample,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    student=student,
                    image_feature_len=image_feature_len,
                    device=torch.device(args.device),
                    keep_ratio=args.keep_ratio,
                    milebench_budget_mode=args.milebench_budget_mode,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                result["mask_file"] = str(
                    mask_path.relative_to(args.output_dir)
                )
                result["qvik_checkpoint"] = str(args.student_dir.resolve())
                base.atomic_json(result_path, result)
                base.q2.save_packed_masks(
                    mask_path,
                    masks,
                    n_visual=int(result["n_visual"]),
                    n_keep=int(result["n_visual_kept_per_head"]),
                )
                failure_path.unlink(missing_ok=True)
                completed += 1
            except Exception as error:  # noqa: BLE001
                # A failed/OOM sample is excluded as one indivisible unit.
                # Never leave a JSON without its matching packed mask (or
                # vice versa), since summary treats a JSON as a completed row.
                result_path.unlink(missing_ok=True)
                mask_path.unlink(missing_ok=True)
                result_path.with_suffix(result_path.suffix + ".tmp").unlink(
                    missing_ok=True
                )
                mask_path.with_suffix(mask_path.suffix + ".tmp").unlink(
                    missing_ok=True
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(
                    f"{type(error).__name__}: {error}\n"
                )
                gc.collect()
                torch.cuda.empty_cache()
                failures += 1
                print(
                    f"[failure] dataset={dataset_name} "
                    f"id={base.sample_value(sample, 'sample_id')}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            finally:
                if data.is_milebench_dataset(dataset_name):
                    sample.image = None
            processed = completed + skipped + failures
            if processed == 1 or processed % 10 == 0:
                print(
                    f"[progress] {processed}/{total_expected} "
                    f"completed={completed} skipped={skipped} "
                    f"failures={failures} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            torch.cuda.empty_cache()
        del samples, work
        gc.collect()
        torch.cuda.empty_cache()

    if args.manifest_only:
        print(
            f"[manifest-done] datasets={len(args.datasets)} "
            f"samples={args.n_samples * len(args.datasets)}",
            flush=True,
        )
    else:
        print(
            f"[done] completed={completed} skipped={skipped} "
            f"failures={failures} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return 1 if failures else 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect_exclusions(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Return manifest rows that lack a complete JSON+mask output pair."""

    exclusions: dict[str, list[dict[str, Any]]] = {}
    for dataset_name in args.datasets:
        manifest_path = args.output_dir / "manifests" / f"{dataset_name}.json"
        if not manifest_path.exists():
            exclusions[dataset_name] = [
                {
                    "sample_id": "<manifest-missing>",
                    "row_index": None,
                    "reason": "manifest missing",
                }
            ]
            continue
        payload = json.loads(manifest_path.read_text())
        missing: list[dict[str, Any]] = []
        for row in payload.get("samples", []):
            stem = (
                f"{int(row['row_index']):06d}_"
                f"{base.safe_name(row['sample_id'])}"
            )
            result_path = (
                args.output_dir / "samples" / dataset_name / f"{stem}.json"
            )
            mask_path = (
                args.output_dir / "masks" / dataset_name / f"{stem}.npz"
            )
            if result_path.exists() and mask_path.exists():
                continue
            failure_path = (
                args.output_dir / "failures" / dataset_name / f"{stem}.txt"
            )
            reason = (
                failure_path.read_text().strip()
                if failure_path.exists()
                else "output pair missing"
            )
            missing.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "row_index": int(row["row_index"]),
                    "reason": reason,
                }
            )
        if missing:
            exclusions[dataset_name] = missing
    return exclusions


def write_suite_report(args: argparse.Namespace) -> None:
    summary_dir = args.output_dir / "summary"
    downstream = _read_csv(summary_dir / "downstream.csv")
    records = [
        json.loads(path.read_text())
        for path in sorted((args.output_dir / "samples").glob("*/*.json"))
        if path.parent.name in args.datasets
    ]
    means = {
        (row["group"], row["selector"]): float(row["mean"])
        for row in downstream
    }
    selectors = (
        "full_cache_generate",
        "full_cache_manual",
        "h2o_prefill",
        "question_prefill",
        "qvik",
        "future_oracle",
    )
    labels = (
        "Full",
        "Matched Full",
        "H2O",
        "Semantic user-prompt",
        "Q-ViK",
        "Oracle",
    )

    present = {
        record["dataset"] for record in records
    }
    standard = [
        name for name in data.STANDARD_DATASETS if name in present
    ]
    official = [
        f"{data.MILEBENCH_PREFIX}{name}"
        for name in data.OFFICIAL_MILEBENCH_TASKS
        if f"{data.MILEBENCH_PREFIX}{name}" in present
    ]
    local = [
        name for name in data.MILEBENCH_DATASETS if name in present
    ]
    def macro(names: list[str], selector: str) -> float | None:
        if not names:
            return None
        return 100.0 * float(
            np.mean([means[(name, selector)] for name in names])
        )

    lines = [
        "# Eval-7200 Q2 suite report",
        "",
        f"Samples: {len(records)} across {len(present)} datasets. Standard "
        "datasets use total-cache retention; MileBench uses "
        f"`{args.milebench_budget_mode}` retention at ratio "
        f"{args.keep_ratio:.2f}.",
        "",
        "The official MileBench macro contains 28 tasks. `nuscenes` is shown "
        "separately through the local 29-task extension.",
        "",
        "For MileBench, the `Semantic user-prompt` rows aggregate the complete "
        "retained semantic user prompt; they are not H2O's all-prefill rows.",
        "",
        "## Suite macro downstream scores",
        "",
        "| Suite | n tasks | " + " | ".join(labels) + " |",
        "|---|---:|" + "---:|" * len(labels),
    ]
    for label, names in (
        ("Standard eval", standard),
        ("MileBench official", official),
        ("MileBench local extension", local),
    ):
        if not names:
            continue
        values = [macro(names, selector) for selector in selectors]
        lines.append(
            f"| {label} | {len(names)} | "
            + " | ".join(f"{value:.2f}" for value in values if value is not None)
            + " |"
        )

    if (
        f"{data.MILEBENCH_PREFIX}nuscenes" in present
    ):
        lines.extend(
            [
                "",
                "## Local-only extension",
                "",
                "| Dataset | " + " | ".join(labels) + " |",
                "|---|" + "---:|" * len(labels),
                "| nuscenes | "
                + " | ".join(
                    f"{100 * means[(f'{data.MILEBENCH_PREFIX}nuscenes', selector)]:.2f}"
                    for selector in selectors
                )
                + " |",
            ]
        )

    exclusions = collect_exclusions(args)
    exclusion_count = sum(len(rows) for rows in exclusions.values())
    base.atomic_json(summary_dir / "exclusions.json", exclusions)
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            f"Excluded incomplete/OOM samples: {exclusion_count}.",
            "",
            "| Dataset | Count | Sample IDs (row index) |",
            "|---|---:|---|",
        ]
    )
    if exclusions:
        for dataset_name, rows in exclusions.items():
            identifiers = ", ".join(
                (
                    f"{row['sample_id']} ({row['row_index']})"
                    if row["row_index"] is not None
                    else str(row["sample_id"])
                )
                for row in rows
            )
            lines.append(f"| {dataset_name} | {len(rows)} | {identifiers} |")
    else:
        lines.append("| — | 0 | — |")

    budget_counts: dict[str, int] = {}
    for record in records:
        mode = str(record.get("budget_mode", "total"))
        budget_counts[mode] = budget_counts.get(mode, 0) + 1
    lines.extend(
        [
            "",
            "## Budget provenance",
            "",
            ", ".join(
                f"{mode}={count}" for mode, count in sorted(budget_counts.items())
            )
            + ".",
            "",
            "Detailed agreement, exact keep-mask, downstream, response-"
            "preservation, and layer-wise tables are saved as CSV files in "
            "this directory.",
        ]
    )
    suite_report = "\n".join(lines) + "\n"
    legacy_report = summary_dir / "report.md"
    if legacy_report.exists():
        (summary_dir / "legacy_detailed_report.md").write_text(
            legacy_report.read_text()
        )
    legacy_report.write_text(suite_report)
    (summary_dir / "suite_report.md").write_text(suite_report)

    provenance_path = summary_dir / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.exists()
        else {}
    )
    provenance.update(
        {
            "standard_budget_mode": "total",
            "milebench_budget_mode": args.milebench_budget_mode,
            "keep_ratio": args.keep_ratio,
            "official_milebench_tasks_present": official,
            "local_milebench_tasks_present": local,
            "nuscenes_reported_as_local_extension": True,
            "attention_storage": "stream-compressed long prefill",
            "milebench_question_score_scope": "semantic user-prompt",
            "excluded_incomplete_samples": exclusion_count,
            "exclusion_file": "summary/exclusions.json",
            "cross_budget_all_dataset_macro_reported": False,
        }
    )
    base.atomic_json(provenance_path, provenance)


def summarize(args: argparse.Namespace) -> int:
    original_datasets = base.DEFAULT_DATASETS
    try:
        base.DEFAULT_DATASETS = tuple(data.DATASETS)
        status = base.summarize(args)
    finally:
        base.DEFAULT_DATASETS = original_datasets
    write_suite_report(args)
    print((args.output_dir / "summary" / "suite_report.md").read_text())
    return status


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        return summarize(args)
    return extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
