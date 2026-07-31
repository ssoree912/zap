#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validation and official-metric summary for paired full VQA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterable, Mapping


STUDENT_LABELS = ("prefill_answer_50_50", "question_answer_50_50")
PA_LABEL, QA_LABEL = STUDENT_LABELS
DELTA_LABEL = "question_answer_50_50_minus_prefill_answer_50_50"
EXPECTED_BUDGET_MODE = "exact_total_ceil"
EXPECTED_MASK_POLICY = "one_layerwise_visual_topk_mask_shared_across_all_kv_heads"
EXPECTED_HIDDEN_STATE = "post_block_hidden_states_layer_plus_1"
EXPECTED_GPU_ASSIGNMENT = {
    "0": ["prefill_answer_50_50:gqa_local"],
    "1": ["question_answer_50_50:gqa_local"],
    "2": [
        "prefill_answer_50_50:textvqa_local",
        "prefill_answer_50_50:docvqa_local",
        "prefill_answer_50_50:chartqa_local",
    ],
    "3": [
        "question_answer_50_50:textvqa_local",
        "question_answer_50_50:docvqa_local",
        "question_answer_50_50:chartqa_local",
    ],
}
PRIMARY_MACRO_METRICS = {
    "gqa_local": "exact_match,none",
    "textvqa_local": "exact_match,none",
    "docvqa_local": "anls,none",
    "chartqa_local": "relaxed_overall,none",
}
EXPECTED_TASK_CONFIGS = {
    "gqa_local": {
        "expected_samples": 12578,
        "primary_metrics": ["exact_match,none"],
    },
    "textvqa_local": {
        "expected_samples": 5000,
        "primary_metrics": ["exact_match,none"],
    },
    "docvqa_local": {
        "expected_samples": 5349,
        "primary_metrics": ["anls,none"],
    },
    "chartqa_local": {
        "expected_samples": 2500,
        "primary_metrics": [
            "relaxed_overall,none",
            "relaxed_human_split,none",
            "relaxed_augmented_split,none",
        ],
    },
}
EXPECTED_SPLIT_HASHES = {
    "all": "2ad770606721e126de78129e6d4d200a3badbdae8482a74befd43847546fbf7f",
    "train": "75922bff1346cfd0171231d20786768bbb08140a6031c820ca775adfd715f66e",
    "val": "3f62d3b757012ea2abc32fe37c6be0094a475e8072e4acd501fb21af108211f4",
}
REQUIRED_HASHED_CHECKPOINT_FILES = (
    "pytorch_model.bin",
    "config.json",
    "train_config.json",
    "train_log.jsonl",
)
REQUIRED_PRESENCE_CHECKPOINT_FILES = ("last_checkpoint.pt",)

EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
LOCAL_EXP = (
    ZAP_ROOT
    / "experiments/EXP-20260502-024-llava15-original-teacher-extract"
)
PAIRED_EXP = (
    ZAP_ROOT
    / "experiments/EXP-20260727-006-paired-prefill-question-answer"
)
TEACHER_VERIFICATION_PATH = (
    ZAP_ROOT
    / "artifacts/paired_prefill_question_answer/runs/20260727_065959"
    / "teacher_verification.json"
)
TRAINING_VERIFICATION_PATH = (
    ZAP_ROOT
    / "artifacts/paired_prefill_question_answer/training_runs/20260727_070541"
    / "training_verification.json"
)
TASK_ROOT = LOCAL_EXP / "tasks"
SOURCE_PATHS_BASE = {
    "run_config": EXP_DIR / "run_config.json",
    "student_evaluator": ZAP_ROOT / "foresight/eval/lmms_llava15_original_student.py",
    "cache_budget": ZAP_ROOT / "foresight/eval/cache_budget.py",
    "local_dataset_wrapper": LOCAL_EXP / "lmms_eval_original_llava15_local_run.py",
    "prepare_run": EXP_DIR / "prepare_run.py",
    "validator": EXP_DIR / "validate_paired_outputs.py",
    "run_worker": EXP_DIR / "run_worker.sh",
    "launcher": EXP_DIR / "run_4gpu.sh",
    "post_training_launcher": EXP_DIR / "launch_after_training.sh",
    "finalizer": EXP_DIR / "finalize_after_workers.sh",
    "paired_teacher_deriver": PAIRED_EXP / "derive_paired_teachers.py",
    "paired_target_recomposer": PAIRED_EXP / "recompose_saved_targets.py",
    "paired_teacher_validator": PAIRED_EXP / "verify_paired_teachers.py",
    "paired_training_launcher": PAIRED_EXP / "run_train_2gpu.sh",
    "paired_training_validator": PAIRED_EXP / "verify_paired_training.py",
    "student_trainer": LOCAL_EXP / "train_original_llava15_student.py",
    "teacher_verification_artifact": TEACHER_VERIFICATION_PATH,
    "training_verification_artifact": TRAINING_VERIFICATION_PATH,
}


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
        raise ValueError(f"{path}: expected {expected} samples, found {len(identities)}")
    doc_ids = [row["doc_id"] for row in identities]
    if len(set(doc_ids)) != expected:
        raise ValueError(f"{path}: duplicate doc_id values")
    if sorted(doc_ids) != list(range(expected)):
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
        raise ValueError(
            f"{path}: expected full evaluation with batch_size=1 and limit=None"
        )
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
            f"{path}: expected n_samples={expected}, "
            f"found {payload.get('n_samples')!r}"
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
            raise ValueError(
                f"{path}: sample {index} actual token count is inconsistent"
            )
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


def _source_paths(tasks: Mapping[str, Any]) -> dict[str, Path]:
    paths = dict(SOURCE_PATHS_BASE)
    paths.update(
        {f"task_yaml:{task}": TASK_ROOT / f"{task}.yaml" for task in tasks}
    )
    return paths


def _validate_frozen_sources(manifest: Mapping[str, Any]) -> None:
    runtime = manifest.get("runtime_provenance")
    if not isinstance(runtime, dict):
        raise ValueError("Run manifest has no runtime_provenance")
    expected_hashes = runtime.get("source_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Run manifest has no source_sha256 map")
    tasks = manifest.get("evaluation", {}).get("tasks", {})
    source_paths = _source_paths(tasks)
    if set(expected_hashes) != set(source_paths):
        raise ValueError("Run manifest source hash keys do not match the contract")
    for label, source in source_paths.items():
        if not source.is_file():
            raise ValueError(f"Required frozen source disappeared: {source}")
        actual = _sha256_file(source)
        if actual != expected_hashes[label]:
            raise ValueError(
                f"Frozen source changed during the run: {source}; "
                f"expected={expected_hashes[label]} actual={actual}"
            )


def _validate_lineage_verification(manifest: Mapping[str, Any]) -> None:
    configured = manifest.get("paired_training_contract", {}).get(
        "verification_artifacts"
    )
    expected_paths = {
        "teacher": str(TEACHER_VERIFICATION_PATH),
        "training": str(TRAINING_VERIFICATION_PATH),
    }
    if configured != expected_paths:
        raise ValueError("Paired verification artifact paths are incorrect")
    frozen = manifest.get("lineage_verification_artifacts")
    if not isinstance(frozen, dict) or set(frozen) != {"teacher", "training"}:
        raise ValueError("Run manifest has no validated lineage artifacts")

    live_payloads: dict[str, dict[str, Any]] = {}
    for key, path in (
        ("teacher", TEACHER_VERIFICATION_PATH),
        ("training", TRAINING_VERIFICATION_PATH),
    ):
        entry = frozen.get(key)
        if not isinstance(entry, dict) or entry.get("path") != str(path):
            raise ValueError(f"Lineage artifact entry is malformed: {key}")
        if not path.is_file():
            raise ValueError(f"Lineage artifact disappeared: {path}")
        actual_sha = _sha256_file(path)
        if entry.get("sha256") != actual_sha:
            raise ValueError(
                f"Lineage artifact changed: {path}; "
                f"expected={entry.get('sha256')} actual={actual_sha}"
            )
        payload = _load_object(path)
        if entry.get("validated_payload") != payload:
            raise ValueError(f"Validated lineage payload changed: {path}")
        live_payloads[key] = payload

    teacher = live_payloads["teacher"]
    students = manifest["students"]
    expected_teacher = {
        "schema_version": 1,
        "source_root": manifest["paired_training_contract"]["source_teacher_root"],
        "pa_root": students[PA_LABEL]["teacher_root"],
        "qa_root": students[QA_LABEL]["teacher_root"],
        "records_per_dataset": 600,
        "total_records": 1800,
        "answer_component_byte_identical": True,
        "paired_target_difference": (
            "all-prefill rows versus post-image prompt-tail rows only"
        ),
        "answer_query_rows": "[y_1, ..., y_T]",
        "max_new_tokens": 64,
        "split_hashes": EXPECTED_SPLIT_HASHES,
    }
    for key, expected in expected_teacher.items():
        if teacher.get(key) != expected:
            raise ValueError(
                f"{TEACHER_VERIFICATION_PATH}: expected {key}={expected!r}, "
                f"found={teacher.get(key)!r}"
            )
    error = teacher.get("max_fp16_recomposition_error")
    difference = teacher.get("mean_absolute_pa_qa_target_difference")
    if (
        not isinstance(error, (int, float))
        or not math.isfinite(float(error))
        or float(error) > 3e-4
    ):
        raise ValueError("Teacher verification has invalid recomposition error")
    if (
        not isinstance(difference, (int, float))
        or not math.isfinite(float(difference))
        or float(difference) <= 0
    ):
        raise ValueError("Teacher verification does not distinguish P+A and Q+A")

    training = live_payloads["training"]
    if training.get("schema_version") != 1 or training.get("paired_training") is not True:
        raise ValueError("Training verification has an invalid header")
    controlled = manifest["controlled_training_properties"]
    for artifact_key, label in (
        ("prefill_answer", PA_LABEL),
        ("question_answer", QA_LABEL),
    ):
        payload = training.get(artifact_key)
        if not isinstance(payload, dict):
            raise ValueError(f"Training verification missing {artifact_key}")
        checkpoint = Path(str(students[label]["path"])).resolve()
        expected = {
            "directory": str(checkpoint),
            "epochs": int(controlled["epochs"]),
            "train_seen_each_epoch": int(controlled["expected_train_samples"]),
            "val_seen_each_epoch": int(controlled["expected_val_samples"]),
            "best_weights_bytes": (checkpoint / "pytorch_model.bin").stat().st_size,
        }
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"Training verification {artifact_key}.{key} drifted"
                )


def _validate_checkpoint_provenance(
    label: str,
    student: Mapping[str, Any],
    controlled: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(str(student.get("path", ""))).resolve()
    provenance = student.get("checkpoint_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label}: dynamic checkpoint_provenance is missing")
    if provenance.get("checkpoint_path") != str(checkpoint):
        raise ValueError(f"{label}: checkpoint provenance path mismatch")
    if provenance.get("teacher_root") != str(
        Path(str(student.get("teacher_root", ""))).resolve()
    ):
        raise ValueError(f"{label}: checkpoint teacher-root provenance mismatch")

    hashes = provenance.get("files_sha256")
    sizes = provenance.get("file_sizes_bytes")
    if not isinstance(hashes, dict) or set(hashes) != set(
        REQUIRED_HASHED_CHECKPOINT_FILES
    ):
        raise ValueError(f"{label}: incomplete dynamic checkpoint hash map")
    if not isinstance(sizes, dict) or set(sizes) != set(
        (*REQUIRED_HASHED_CHECKPOINT_FILES, *REQUIRED_PRESENCE_CHECKPOINT_FILES)
    ):
        raise ValueError(f"{label}: incomplete checkpoint size map")

    for filename in REQUIRED_HASHED_CHECKPOINT_FILES:
        path = checkpoint / filename
        if not path.is_file():
            raise ValueError(f"{label}: checkpoint file disappeared: {path}")
        if path.stat().st_size != int(sizes[filename]):
            raise ValueError(f"{label}: checkpoint file size changed: {path}")
        actual = _sha256_file(path)
        if actual != hashes[filename]:
            raise ValueError(
                f"{label}: checkpoint file changed: {path}; "
                f"expected={hashes[filename]} actual={actual}"
            )
    for filename in REQUIRED_PRESENCE_CHECKPOINT_FILES:
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size != int(sizes[filename]):
            raise ValueError(f"{label}: required checkpoint state changed: {path}")

    summary = provenance.get("training_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{label}: missing training completion summary")
    expected_completion = {
        "epochs": int(controlled["epochs"]),
        "final_epoch": int(controlled["epochs"]),
        "train_seen_each_epoch": int(controlled["expected_train_samples"]),
        "val_seen_each_epoch": int(controlled["expected_val_samples"]),
    }
    for key, expected in expected_completion.items():
        if int(summary.get(key, -1)) != expected:
            raise ValueError(
                f"{label}: training completion {key} mismatch; "
                f"expected={expected} actual={summary.get(key)!r}"
            )
    return {
        "checkpoint_path": str(checkpoint),
        "pytorch_model_sha256": hashes["pytorch_model.bin"],
        "config_sha256": hashes["config.json"],
        "train_config_sha256": hashes["train_config.json"],
        "train_log_sha256": hashes["train_log.jsonl"],
        "training_summary": summary,
    }


def _read_state_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"{path}:{line_number}: malformed state line")
        result[key] = value
    return result


def _validate_job_states(run_root: Path) -> None:
    state_root = run_root / "state"
    failures = sorted(state_root.glob("*.failed"))
    if failures:
        raise ValueError(f"Run contains failed job/worker state: {failures}")
    expected: dict[str, tuple[str, str, str]] = {}
    for gpu, jobs in EXPECTED_GPU_ASSIGNMENT.items():
        for job in jobs:
            label, task = job.split(":", 1)
            expected[f"{label}__{task}.done"] = (gpu, label, task)
    actual = {path.name for path in state_root.glob("*.done")}
    if actual != set(expected):
        raise ValueError(
            f"Completed job-state set mismatch; expected={sorted(expected)} "
            f"actual={sorted(actual)}"
        )
    for filename, (gpu, label, task) in expected.items():
        payload = _read_state_file(state_root / filename)
        expected_fields = {"physical_gpu": gpu, "label": label, "task": task}
        for key, value in expected_fields.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"{filename}: expected {key}={value!r}, "
                    f"found={payload.get(key)!r}"
                )


def validate_run(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = run_root.resolve(strict=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = _load_object(manifest_path)
    _validate_frozen_sources(manifest)
    _validate_lineage_verification(manifest)

    evaluation = manifest.get("evaluation")
    students = manifest.get("students")
    base_model = manifest.get("base_model")
    controlled = manifest.get("controlled_training_properties")
    if not all(
        isinstance(value, dict)
        for value in (evaluation, students, base_model, controlled)
    ):
        raise ValueError(
            f"{manifest_path}: missing evaluation/students/base_model/training config"
        )
    if tuple(students) != STUDENT_LABELS:
        raise ValueError(
            f"{manifest_path}: expected student labels/order={STUDENT_LABELS}, "
            f"found={tuple(students)}"
        )
    if manifest.get("comparison_is_oracle") is not False:
        raise ValueError(f"{manifest_path}: comparison must not be labeled oracle")

    paired = manifest.get("paired_training_contract")
    if not isinstance(paired, dict) or paired.get("split_hashes") != EXPECTED_SPLIT_HASHES:
        raise ValueError(f"{manifest_path}: paired split provenance is incorrect")
    if evaluation.get("keep_budget_mode") != EXPECTED_BUDGET_MODE:
        raise ValueError(f"{manifest_path}: exact budget mode is not configured")
    if evaluation.get("hidden_state_convention") != (
        "post_block_hidden_states[layer_idx + 1]"
    ):
        raise ValueError(f"{manifest_path}: hidden-state convention is incorrect")
    if evaluation.get("physical_gpu_allowlist") != [0, 1, 2, 3]:
        raise ValueError(f"{manifest_path}: physical GPU allowlist must be [0,1,2,3]")
    if evaluation.get("job_assignment") != EXPECTED_GPU_ASSIGNMENT:
        raise ValueError(f"{manifest_path}: physical GPU job assignment is incorrect")
    if evaluation.get("local_dataset_root") != "/workspace/nips/data/eval":
        raise ValueError(f"{manifest_path}: local dataset root is incorrect")
    if evaluation.get("student_failure_policy") != (
        "raise; never silently replace a failed pruning sample with full-cache generation"
    ):
        raise ValueError(f"{manifest_path}: strict failure policy is missing")
    if evaluation.get("metric_policy") != (
        "Use the official aggregate metrics emitted by LMMS-eval. "
        "Do not rescore DocVQA rows with the eval700 continuous ANLS implementation."
    ):
        raise ValueError(f"{manifest_path}: official LMMS metric policy is missing")

    checkpoint_summary = {
        label: _validate_checkpoint_provenance(label, students[label], controlled)
        for label in STUDENT_LABELS
    }
    if (
        checkpoint_summary[PA_LABEL]["config_sha256"]
        != checkpoint_summary[QA_LABEL]["config_sha256"]
    ):
        raise ValueError("Paired student architecture config.json files are not identical")
    _validate_job_states(run_root)

    tasks = evaluation.get("tasks")
    if not isinstance(tasks, dict) or tasks != EXPECTED_TASK_CONFIGS:
        raise ValueError(
            f"{manifest_path}: full task counts/official metrics drifted; "
            f"expected={EXPECTED_TASK_CONFIGS} found={tasks}"
        )
    keep_ratio = float(evaluation["requested_total_keep_ratio"])
    if not math.isclose(keep_ratio, 0.2):
        raise ValueError(f"{manifest_path}: requested keep ratio must be 0.2")

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
        pa = by_label[PA_LABEL]
        qa = by_label[QA_LABEL]
        if pa["identities"] != qa["identities"]:
            raise ValueError(
                f"{task}: sample identities/order differ between P+A and Q+A"
            )
        if pa["budget_trace"] != qa["budget_trace"]:
            raise ValueError(
                f"{task}: per-sample budget traces differ between P+A and Q+A"
            )
        paired_rows.extend({"task": task, **row} for row in pa["identities"])

        model_payload: dict[str, Any] = {}
        for label, artifacts in by_label.items():
            model_payload[label] = {
                "checkpoint_path": students[label]["path"],
                "checkpoint_pytorch_model_sha256": checkpoint_summary[label][
                    "pytorch_model_sha256"
                ],
                "result_file": str(artifacts["result_path"].relative_to(run_root)),
                "sample_file": str(artifacts["sample_path"].relative_to(run_root)),
                "stats_file": str(artifacts["stats_path"].relative_to(run_root)),
                "metrics": artifacts["metrics"],
                "sample_identity_sha256": artifacts["identity_sha256"],
                "budget_trace_sha256": artifacts["budget_sha256"],
            }
        task_summary[task] = {
            "paired_sample_count": expected,
            "sample_identity_sha256": pa["identity_sha256"],
            "budget_trace_sha256": pa["budget_sha256"],
            "students": model_payload,
            DELTA_LABEL: {
                metric: qa["metrics"][metric] - pa["metrics"][metric]
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
            "full expected sample set only; exact identity and budget-trace match; "
            "zero silent fallback or unmatched exclusions"
        ),
        "total_paired_samples_per_student": sum(
            int(config["expected_samples"]) for config in tasks.values()
        ),
        "checkpoint_provenance": checkpoint_summary,
        "evaluation_contract": {
            "keep_ratio": keep_ratio,
            "budget_mode": EXPECTED_BUDGET_MODE,
            "mask_policy": EXPECTED_MASK_POLICY,
            "hidden_state_convention": EXPECTED_HIDDEN_STATE,
            "physical_gpu_job_assignment": EXPECTED_GPU_ASSIGNMENT,
            "paired_target_difference_only": (
                "all-prefill rows versus post-image prompt-tail rows"
            ),
            "milebench_excluded": True,
        },
        "four_vqa_primary_macro": {
            "metric_selection": PRIMARY_MACRO_METRICS,
            **macro_by_student,
            DELTA_LABEL: macro_by_student[QA_LABEL] - macro_by_student[PA_LABEL],
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
            fieldnames=("task", "metric", PA_LABEL, QA_LABEL, DELTA_LABEL),
        )
        writer.writeheader()
        for task, task_payload in summary["tasks"].items():
            pa_metrics = task_payload["students"][PA_LABEL]["metrics"]
            qa_metrics = task_payload["students"][QA_LABEL]["metrics"]
            for metric, pa_value in pa_metrics.items():
                writer.writerow(
                    {
                        "task": task,
                        "metric": metric,
                        PA_LABEL: pa_value,
                        QA_LABEL: qa_metrics[metric],
                        DELTA_LABEL: qa_metrics[metric] - pa_value,
                    }
                )
        macro = summary["four_vqa_primary_macro"]
        writer.writerow(
            {
                "task": "four_vqa_macro",
                "metric": "primary_metric_arithmetic_mean",
                PA_LABEL: macro[PA_LABEL],
                QA_LABEL: macro[QA_LABEL],
                DELTA_LABEL: macro[DELTA_LABEL],
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
