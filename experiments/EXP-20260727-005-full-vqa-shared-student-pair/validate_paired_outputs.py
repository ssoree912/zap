#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validation and summary for the paired full-VQA runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterable, Mapping


STUDENT_LABELS = (
    "last_prompt_plus_answer_steps",
    "question_answer_50_50",
)
BASELINE_LABEL = STUDENT_LABELS[0]
QA_LABEL = STUDENT_LABELS[1]
EXPECTED_BUDGET_MODE = "exact_total_ceil"
EXPECTED_MASK_POLICY = (
    "one_layerwise_visual_topk_mask_shared_across_all_kv_heads"
)
EXPECTED_HIDDEN_STATE = "post_block_hidden_states_layer_plus_1"
PRIMARY_MACRO_METRICS = {
    "gqa_local": "exact_match,none",
    "textvqa_local": "exact_match,none",
    "docvqa_local": "anls,none",
    "chartqa_local": "relaxed_overall,none",
}
EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
LOCAL_EXP = (
    ZAP_ROOT
    / "experiments/EXP-20260502-024-llava15-original-teacher-extract"
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load JSON object from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} under {root}, found {matches}"
        )
    return matches[0]


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_model_args(value: Any, source: Path) -> dict[str, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: config.model_args must be a nonempty string")
    parsed: dict[str, str] = {}
    for item in value.split(","):
        key, separator, item_value = item.partition("=")
        if not separator or not key or key in parsed:
            raise ValueError(f"{source}: malformed config.model_args item={item!r}")
        parsed[key] = item_value
    return parsed


def _sample_identity(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    required = ("doc_id", "doc", "prompt_hash", "target_hash", "input")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{source}: sample identity fields missing={missing}")
    try:
        doc_id = int(payload["doc_id"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid doc_id={payload['doc_id']!r}") from error
    return {
        "doc_id": doc_id,
        "doc_sha256": _canonical_hash(payload["doc"]),
        "prompt_hash": str(payload["prompt_hash"]),
        "target_hash": str(payload["target_hash"]),
        "input_sha256": _canonical_hash(payload["input"]),
    }


def _read_sample_identities(path: Path, expected: int) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSONL") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            identities.append(_sample_identity(payload, path))
    if len(identities) != expected:
        raise ValueError(
            f"{path}: expected {expected} samples, found {len(identities)}"
        )
    doc_ids = [row["doc_id"] for row in identities]
    if len(set(doc_ids)) != expected:
        raise ValueError(f"{path}: duplicate doc_id values")
    expected_ids = list(range(expected))
    if sorted(doc_ids) != expected_ids:
        raise ValueError(
            f"{path}: doc_id set is not the complete range [0, {expected - 1}]"
        )
    return sorted(identities, key=lambda row: row["doc_id"])


def _validate_results(
    *,
    path: Path,
    task: str,
    expected: int,
    primary_metrics: Iterable[str],
    expected_model_path: str,
    expected_model_name: str,
    expected_conv_template: str,
    expected_student_path: str,
    expected_stats_path: Path,
    keep_ratio: float,
) -> dict[str, float]:
    payload = _load_object(path)
    try:
        counts = payload["n-samples"][task]
        effective = int(counts["effective"])
        original = int(counts["original"])
        task_results = payload["results"][task]
        run_config = payload["config"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: malformed LMMS result structure for {task}") from error
    if effective != expected or original != expected:
        raise ValueError(
            f"{path}: expected original=effective={expected}, "
            f"found original={original} effective={effective}"
        )
    if run_config.get("model") != "llava15_original_student":
        raise ValueError(f"{path}: wrong LMMS model={run_config.get('model')!r}")
    if str(run_config.get("batch_size")) != "1" or run_config.get("limit") is not None:
        raise ValueError(f"{path}: expected full evaluation with batch_size=1 and limit=None")
    model_args = _parse_model_args(run_config.get("model_args"), path)
    expected_model_args = {
        "pretrained": expected_model_path,
        "student_path": expected_student_path,
        "keep_ratio": str(keep_ratio),
        "keep_budget_mode": EXPECTED_BUDGET_MODE,
        "student_failure_policy": "raise",
        "conv_template": expected_conv_template,
        "model_name": expected_model_name,
        "device": "cuda:0",
        "device_map": "cuda:0",
        "stats_output_dir": str(expected_stats_path),
    }
    if model_args != expected_model_args:
        raise ValueError(
            f"{path}: model_args do not exactly match the run contract; "
            f"expected={expected_model_args} found={model_args}"
        )
    metrics: dict[str, float] = {}
    for metric in primary_metrics:
        if metric not in task_results:
            raise ValueError(f"{path}: missing primary metric {metric!r}")
        value = task_results[metric]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path}: primary metric {metric!r} is not finite")
        metrics[metric] = float(value)
    return metrics


def _validate_budget_stats(
    *,
    path: Path,
    task: str,
    expected: int,
    expected_student_path: str,
    keep_ratio: float,
) -> tuple[list[tuple[int, int, int, int]], str]:
    payload = _load_object(path)
    expected_top_level = {
        "task": task,
        "student_path": expected_student_path,
        "keep_ratio_basis": "total_prompt_cache",
        "keep_budget_mode": EXPECTED_BUDGET_MODE,
        "head_mask_policy": EXPECTED_MASK_POLICY,
        "hidden_state_convention": EXPECTED_HIDDEN_STATE,
        "student_failure_policy": "raise",
    }
    for key, expected_value in expected_top_level.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"{path}: expected {key}={expected_value!r}, "
                f"found {payload.get(key)!r}"
            )
    if not math.isclose(float(payload.get("keep_ratio", -1)), keep_ratio):
        raise ValueError(f"{path}: incorrect keep_ratio={payload.get('keep_ratio')!r}")
    if int(payload.get("n_samples", -1)) != expected:
        raise ValueError(
            f"{path}: expected n_samples={expected}, found {payload.get('n_samples')!r}"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != expected:
        raise ValueError(f"{path}: expected {expected} per-sample budget records")

    trace: list[tuple[int, int, int, int]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{path}: budget sample {index} is not an object")
        try:
            prompt_len = int(sample["prompt_len"])
            n_image = int(sample["n_image_original"])
            n_text = int(sample["n_text"])
            n_kept = int(sample["n_image_kept"])
            requested = int(sample["requested_total_token_budget"])
            actual = int(sample["actual_total_tokens_kept"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: malformed budget sample {index}") from error
        if prompt_len != n_image + n_text:
            raise ValueError(
                f"{path}: sample {index} prompt/image/text lengths are inconsistent"
            )
        exact_total = int(
            (Decimal(str(keep_ratio)) * Decimal(prompt_len)).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        exact_visual = min(n_image, max(0, exact_total - n_text))
        if requested != exact_total or n_kept != exact_visual:
            raise ValueError(
                f"{path}: sample {index} violates exact total budget: "
                f"prompt={prompt_len} text={n_text} image={n_image} "
                f"requested={requested} kept={n_kept} "
                f"expected_total={exact_total} expected_visual={exact_visual}"
            )
        if actual != n_text + n_kept:
            raise ValueError(f"{path}: sample {index} actual token count is inconsistent")
        trace.append((prompt_len, n_text, n_image, n_kept))
    return trace, _canonical_hash(trace)


def _model_task_artifacts(
    *,
    run_root: Path,
    label: str,
    task: str,
    expected: int,
    primary_metrics: Iterable[str],
    expected_student_path: str,
    base_model: Mapping[str, Any],
    keep_ratio: float,
) -> dict[str, Any]:
    output_root = run_root / "outputs" / label / task
    if not output_root.is_dir():
        raise ValueError(f"Missing output directory: {output_root}")
    result_path = _find_one(output_root, "**/*_results.json")
    sample_path = _find_one(output_root, f"**/*_samples_{task}.jsonl")
    stats_path = _find_one(output_root, f"**/{task}_keep_ratio_stats.json")
    metrics = _validate_results(
        path=result_path,
        task=task,
        expected=expected,
        primary_metrics=primary_metrics,
        expected_model_path=str(base_model["path"]),
        expected_model_name=str(base_model["model_name"]),
        expected_conv_template=str(base_model["conversation_template"]),
        expected_student_path=expected_student_path,
        expected_stats_path=stats_path.parent.resolve(),
        keep_ratio=keep_ratio,
    )
    identities = _read_sample_identities(sample_path, expected)
    budget_trace, budget_sha256 = _validate_budget_stats(
        path=stats_path,
        task=task,
        expected=expected,
        expected_student_path=expected_student_path,
        keep_ratio=keep_ratio,
    )
    return {
        "result_path": result_path,
        "sample_path": sample_path,
        "stats_path": stats_path,
        "metrics": metrics,
        "identities": identities,
        "identity_sha256": _canonical_hash(identities),
        "budget_trace": budget_trace,
        "budget_sha256": budget_sha256,
    }


def _validate_frozen_sources(manifest: Mapping[str, Any]) -> None:
    runtime = manifest.get("runtime_provenance")
    if not isinstance(runtime, dict):
        raise ValueError("Run manifest has no runtime_provenance")
    expected_hashes = runtime.get("source_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Run manifest has no source_sha256 map")
    source_paths = {
        "run_config": EXP_DIR / "run_config.json",
        "student_evaluator": ZAP_ROOT / "foresight/eval/lmms_llava15_original_student.py",
        "cache_budget": ZAP_ROOT / "foresight/eval/cache_budget.py",
        "local_dataset_wrapper": LOCAL_EXP / "lmms_eval_original_llava15_local_run.py",
        "prepare_run": EXP_DIR / "prepare_run.py",
        "validator": EXP_DIR / "validate_paired_outputs.py",
        "run_worker": EXP_DIR / "run_worker.sh",
        "launcher": EXP_DIR / "run_3gpu.sh",
        "finalizer": EXP_DIR / "finalize_after_workers.sh",
    }
    tasks = manifest.get("evaluation", {}).get("tasks", {})
    source_paths.update(
        {
            f"task_yaml:{task}": LOCAL_EXP / "tasks" / f"{task}.yaml"
            for task in tasks
        }
    )
    if set(expected_hashes) != set(source_paths):
        raise ValueError("Run manifest source hash keys do not match the validator contract")
    for label, source in source_paths.items():
        if not source.is_file():
            raise ValueError(f"Required frozen source disappeared: {source}")
        actual = _sha256_file(source)
        if actual != expected_hashes[label]:
            raise ValueError(
                f"Frozen source changed during the run: {source}; "
                f"expected={expected_hashes[label]} actual={actual}"
            )


def validate_run(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = run_root.resolve(strict=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = _load_object(manifest_path)
    _validate_frozen_sources(manifest)
    evaluation = manifest.get("evaluation")
    students = manifest.get("students")
    base_model = manifest.get("base_model")
    if (
        not isinstance(evaluation, dict)
        or not isinstance(students, dict)
        or not isinstance(base_model, dict)
    ):
        raise ValueError(
            f"{manifest_path}: missing evaluation/students/base_model configuration"
        )
    if tuple(students) != STUDENT_LABELS:
        raise ValueError(
            f"{manifest_path}: expected student labels/order={STUDENT_LABELS}, "
            f"found={tuple(students)}"
        )
    for label, student in students.items():
        weights = Path(student["path"]) / "pytorch_model.bin"
        if not weights.is_file():
            raise ValueError(f"{manifest_path}: checkpoint disappeared: {weights}")
        actual_sha256 = _sha256_file(weights)
        if actual_sha256 != student.get("pytorch_model_sha256"):
            raise ValueError(
                f"{manifest_path}: checkpoint changed for {label}; "
                f"expected={student.get('pytorch_model_sha256')} "
                f"actual={actual_sha256}"
            )
    if evaluation.get("keep_budget_mode") != EXPECTED_BUDGET_MODE:
        raise ValueError(f"{manifest_path}: exact budget mode is not configured")
    if evaluation.get("hidden_state_convention") != (
        "post_block_hidden_states[layer_idx + 1]"
    ):
        raise ValueError(f"{manifest_path}: hidden-state convention is incorrect")
    if evaluation.get("physical_gpu_allowlist") != [0, 1, 2]:
        raise ValueError(f"{manifest_path}: physical GPU allowlist must be [0,1,2]")
    if evaluation.get("local_dataset_root") != "/workspace/nips/data/eval":
        raise ValueError(f"{manifest_path}: local dataset root is incorrect")
    if evaluation.get("metric_policy") != (
        "Use the official aggregate metrics emitted by LMMS-eval. "
        "Do not rescore DocVQA rows with the eval700 continuous ANLS implementation."
    ):
        raise ValueError(f"{manifest_path}: official LMMS metric policy is missing")
    tasks = evaluation.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"{manifest_path}: no task configuration")
    if set(tasks) != set(PRIMARY_MACRO_METRICS):
        raise ValueError(
            f"{manifest_path}: expected exactly the four VQA tasks "
            f"{tuple(PRIMARY_MACRO_METRICS)}, found={tuple(tasks)}"
        )
    keep_ratio = float(evaluation["requested_total_keep_ratio"])

    task_summary: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for task, task_config in tasks.items():
        expected = int(task_config["expected_samples"])
        primary_metrics = tuple(task_config["primary_metrics"])
        by_label = {
            label: _model_task_artifacts(
                run_root=run_root,
                label=label,
                task=task,
                expected=expected,
                primary_metrics=primary_metrics,
                expected_student_path=str(students[label]["path"]),
                base_model=base_model,
                keep_ratio=keep_ratio,
            )
            for label in STUDENT_LABELS
        }
        base = by_label[BASELINE_LABEL]
        qa = by_label[QA_LABEL]
        if base["identities"] != qa["identities"]:
            raise ValueError(
                f"{task}: sample identity sets/order differ between the two students"
            )
        if base["budget_trace"] != qa["budget_trace"]:
            raise ValueError(
                f"{task}: per-sample budget traces differ between the two students"
            )
        for identity in base["identities"]:
            paired_rows.append({"task": task, **identity})

        model_payload: dict[str, Any] = {}
        for label, artifacts in by_label.items():
            model_payload[label] = {
                "checkpoint_path": students[label]["path"],
                "result_file": str(artifacts["result_path"].relative_to(run_root)),
                "sample_file": str(artifacts["sample_path"].relative_to(run_root)),
                "stats_file": str(artifacts["stats_path"].relative_to(run_root)),
                "metrics": artifacts["metrics"],
                "sample_identity_sha256": artifacts["identity_sha256"],
                "budget_trace_sha256": artifacts["budget_sha256"],
            }
        task_summary[task] = {
            "paired_sample_count": expected,
            "sample_identity_sha256": base["identity_sha256"],
            "budget_trace_sha256": base["budget_sha256"],
            "students": model_payload,
            "question_answer_minus_last_prompt_plus_answer_steps": {
                metric: qa["metrics"][metric] - base["metrics"][metric]
                for metric in primary_metrics
            },
        }

    macro_by_student = {
        label: sum(
            task_summary[task]["students"][label]["metrics"][metric]
            for task, metric in PRIMARY_MACRO_METRICS.items()
        )
        / len(PRIMARY_MACRO_METRICS)
        for label in STUDENT_LABELS
    }
    summary = {
        "schema_version": 1,
        "experiment_id": manifest.get("experiment_id"),
        "validation_status": "passed",
        "paired_sample_policy": (
            "full expected sample set only; exact identity match; "
            "zero silent fallback or unmatched exclusions"
        ),
        "total_paired_samples_per_student": sum(
            int(config["expected_samples"]) for config in tasks.values()
        ),
        "evaluation_contract": {
            "keep_ratio": keep_ratio,
            "budget_mode": EXPECTED_BUDGET_MODE,
            "mask_policy": EXPECTED_MASK_POLICY,
            "hidden_state_convention": EXPECTED_HIDDEN_STATE,
            "last_prompt_query_preserved_in_baseline_teacher": True,
            "milebench_excluded": True,
        },
        "four_vqa_primary_macro": {
            "metric_selection": PRIMARY_MACRO_METRICS,
            **macro_by_student,
            "question_answer_minus_last_prompt_plus_answer_steps": (
                macro_by_student[QA_LABEL] - macro_by_student[BASELINE_LABEL]
            ),
        },
        "tasks": task_summary,
    }
    return summary, paired_rows


def write_outputs(
    run_root: Path,
    summary: Mapping[str, Any],
    paired_rows: Iterable[Mapping[str, Any]],
) -> None:
    summary_root = run_root / "summary"
    summary_root.mkdir(exist_ok=True)
    (summary_root / "paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (summary_root / "paired_samples.jsonl").open("w") as handle:
        for row in paired_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (summary_root / "paired_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "metric",
                BASELINE_LABEL,
                QA_LABEL,
                "question_answer_minus_baseline",
            ),
        )
        writer.writeheader()
        for task, task_payload in summary["tasks"].items():
            base_metrics = task_payload["students"][BASELINE_LABEL]["metrics"]
            qa_metrics = task_payload["students"][QA_LABEL]["metrics"]
            for metric, base_value in base_metrics.items():
                writer.writerow(
                    {
                        "task": task,
                        "metric": metric,
                        BASELINE_LABEL: base_value,
                        QA_LABEL: qa_metrics[metric],
                        "question_answer_minus_baseline": qa_metrics[metric]
                        - base_value,
                    }
                )
        macro = summary["four_vqa_primary_macro"]
        writer.writerow(
            {
                "task": "four_vqa_macro",
                "metric": "primary_metric_arithmetic_mean",
                BASELINE_LABEL: macro[BASELINE_LABEL],
                QA_LABEL: macro[QA_LABEL],
                "question_answer_minus_baseline": (
                    macro["question_answer_minus_last_prompt_plus_answer_steps"]
                ),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, paired_rows = validate_run(args.run_root)
    write_outputs(args.run_root.resolve(), summary, paired_rows)
    print(
        f"Validated {summary['total_paired_samples_per_student']} paired samples "
        f"per student across {len(summary['tasks'])} tasks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
