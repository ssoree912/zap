#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create an immutable run manifest with dynamically captured checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
CONFIG_PATH = EXP_DIR / "run_config.json"
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
EVALUATOR_PATH = ZAP_ROOT / "foresight/eval/lmms_llava15_original_student.py"
BUDGET_PATH = ZAP_ROOT / "foresight/eval/cache_budget.py"
LOCAL_WRAPPER_PATH = LOCAL_EXP / "lmms_eval_original_llava15_local_run.py"
TASK_ROOT = LOCAL_EXP / "tasks"
TRAINER_PATH = LOCAL_EXP / "train_original_llava15_student.py"

STUDENT_LABELS = ("prefill_answer_50_50", "question_answer_50_50")
EXPECTED_STUDENT_PATHS = {
    "prefill_answer_50_50": (
        "/workspace/nips/zap/artifacts/original_llava_teacher/"
        "student_llava15_prefill_answer50_paired_1800_e15_seed0"
    ),
    "question_answer_50_50": (
        "/workspace/nips/zap/artifacts/original_llava_teacher/"
        "student_llava15_question_answer50_paired_1800_e15_seed0"
    ),
}
EXPECTED_TEACHER_ROOTS = {
    "prefill_answer_50_50": (
        "/workspace/nips/data/train/teacher/"
        "zap_llava15_prefill_answer50_paired_n600_seed0"
    ),
    "question_answer_50_50": (
        "/workspace/nips/data/train/teacher/"
        "zap_llava15_question_answer50_paired_n600_seed0"
    ),
}
EXPECTED_FIRST_COMPONENT_ROWS = {
    "prefill_answer_50_50": "[0, ..., N_prompt_mm - 1]",
    "question_answer_50_50": (
        "[last_visual_token + 1, ..., N_prompt_mm - 1]"
    ),
}
EXPECTED_GPU_ASSIGNMENT = {
    "0": ["prefill_answer_50_50:coco2017_cap_local"],
    "1": ["question_answer_50_50:coco2017_cap_local"],
    "2": [
        "prefill_answer_50_50:nocaps_local",
        "prefill_answer_50_50:textcaps_local",
    ],
    "3": [
        "question_answer_50_50:nocaps_local",
        "question_answer_50_50:textcaps_local",
    ],
}
EXPECTED_TASK_COUNTS = {
    "coco2017_cap_local": 5000,
    "nocaps_local": 4500,
    "textcaps_local": 3166,
}
EXPECTED_TASK_CONFIGS = {
    "coco2017_cap_local": {
        "expected_samples": 5000,
        "primary_metrics": ["coco_ROUGE_L,none"],
    },
    "nocaps_local": {
        "expected_samples": 4500,
        "primary_metrics": ["nocaps_ROUGE_L,none"],
    },
    "textcaps_local": {
        "expected_samples": 3166,
        "primary_metrics": ["textcaps_ROUGE_L,none"],
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

CONTRACT_SOURCE_PATHS = {
    "run_config": CONFIG_PATH,
    "student_evaluator": EVALUATOR_PATH,
    "cache_budget": BUDGET_PATH,
    "local_dataset_wrapper": LOCAL_WRAPPER_PATH,
    "caption_metric_implementation": TASK_ROOT / "caption_rouge_utils.py",
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
    "student_trainer": TRAINER_PATH,
    "teacher_verification_artifact": TEACHER_VERIFICATION_PATH,
    "training_verification_artifact": TRAINING_VERIFICATION_PATH,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load JSON object from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ZAP_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def validate_static_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXP_DIR.name:
        raise ValueError("run_config experiment_id does not match its directory")
    if config.get("comparison_is_oracle") is not False:
        raise ValueError("The paired student comparison must not be labeled as an oracle")
    students = config.get("students")
    if not isinstance(students, dict) or tuple(students) != STUDENT_LABELS:
        raise ValueError(f"Expected ordered student labels {STUDENT_LABELS}")
    if any("pytorch_model_sha256" in student for student in students.values()):
        raise ValueError("Checkpoint SHA256 must be captured dynamically, not configured")
    for label, student in students.items():
        if str(Path(str(student.get("path", ""))).resolve()) != (
            EXPECTED_STUDENT_PATHS[label]
        ):
            raise ValueError(f"{label}: checkpoint path drifted")
        if str(Path(str(student.get("teacher_root", ""))).resolve()) != (
            EXPECTED_TEACHER_ROOTS[label]
        ):
            raise ValueError(f"{label}: paired teacher root drifted")
        if int(student.get("teacher_extraction_max_new_tokens", -1)) != 64:
            raise ValueError(f"{label}: expected teacher max_new_tokens=64")
        if student.get("answer_query_rows") != "[y_1, ..., y_T]":
            raise ValueError(f"{label}: answer query-row definition drifted")
        if (
            student.get("first_component_query_rows")
            != EXPECTED_FIRST_COMPONENT_ROWS[label]
        ):
            raise ValueError(f"{label}: first-component query rows drifted")

    paired = config.get("paired_training_contract")
    if not isinstance(paired, dict) or paired.get("split_hashes") != EXPECTED_SPLIT_HASHES:
        raise ValueError("Paired trainer split hashes are missing or incorrect")
    expected_verification_artifacts = {
        "teacher": str(TEACHER_VERIFICATION_PATH),
        "training": str(TRAINING_VERIFICATION_PATH),
    }
    if paired.get("verification_artifacts") != expected_verification_artifacts:
        raise ValueError("Paired verification artifact paths are missing or incorrect")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Missing evaluation configuration")
    if evaluation.get("physical_gpu_allowlist") != [0, 1, 2, 3]:
        raise ValueError("Physical GPU allowlist must be [0,1,2,3]")
    if evaluation.get("job_assignment") != EXPECTED_GPU_ASSIGNMENT:
        raise ValueError("Four-GPU job assignment does not match the run contract")
    if evaluation.get("keep_budget_mode") != "exact_total_ceil":
        raise ValueError("Evaluation must use exact_total_ceil")
    if not math.isclose(
        float(evaluation.get("requested_total_keep_ratio", -1)), 0.2
    ):
        raise ValueError("Evaluation keep ratio must be 0.2")
    tasks = evaluation.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(EXPECTED_TASK_COUNTS):
        raise ValueError("Evaluation must contain exactly the three local caption tasks")
    if tasks != EXPECTED_TASK_CONFIGS:
        raise ValueError(
            f"Full task/metric config drifted: expected={EXPECTED_TASK_CONFIGS} "
            f"actual={tasks}"
        )


def load_config() -> dict[str, Any]:
    config = _load_json_object(CONFIG_PATH)
    validate_static_config(config)
    return config


def _validate_train_config(
    *,
    label: str,
    checkpoint: Path,
    student: Mapping[str, Any],
    base_model: Mapping[str, Any],
    controlled: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> None:
    expected = {
        "teacher_root": str(Path(str(student["teacher_root"])).resolve()),
        "datasets": list(controlled["datasets"]),
        "llava_path": str(Path(str(base_model["path"])).resolve()),
        "model_name": str(base_model["model_name"]),
        "epochs": int(controlled["epochs"]),
        "seed": int(controlled["seed"]),
        "n_per_dataset": int(controlled["samples_per_dataset"]),
        "val_ratio": float(controlled["val_ratio"]),
        "lr": float(controlled["learning_rate"]),
        "weight_decay": float(controlled["weight_decay"]),
        "lambda_rank": float(controlled["lambda_rank"]),
        "rank_margin": float(controlled["rank_margin"]),
        "rank_top_ratio": float(controlled["rank_top_ratio"]),
        "rank_bottom_ratio": float(controlled["rank_bottom_ratio"]),
        "max_grad_norm": float(controlled["max_grad_norm"]),
        "log_every": int(controlled["log_every"]),
        "student_variant": str(controlled["student_variant"]),
        "conv_dim": int(controlled["conv_dim"]),
        "proj_dim": int(controlled["proj_dim"]),
        "mlp_dim": int(controlled["mlp_dim"]),
        "num_conv_blocks": int(controlled["num_conv_blocks"]),
        "kernel_size": int(controlled["kernel_size"]),
        "grid_h": int(controlled["grid_h"]),
        "grid_w": int(controlled["grid_w"]),
        "output_dir": str(checkpoint),
        "vision_tower_path": "",
        "device": "cuda:0",
        "device_map": "cuda:0",
    }
    actual = dict(train_config)
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key in {"teacher_root", "llava_path", "output_dir"} and actual_value is not None:
            actual_value = str(Path(str(actual_value)).resolve())
        if actual_value != expected_value:
            raise ValueError(
                f"{label}: train_config {key} mismatch; "
                f"expected={expected_value!r} actual={actual_value!r}"
            )
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise ValueError(
            f"{label}: train_config has unexpected keys={sorted(unexpected)}"
        )


def _validate_student_config(
    *,
    label: str,
    controlled: Mapping[str, Any],
    student_config: Mapping[str, Any],
) -> None:
    expected = {
        "layer_indices": list(range(32)),
        "hidden_dim": int(controlled["hidden_dim"]),
        "conv_dim": int(controlled["conv_dim"]),
        "proj_dim": int(controlled["proj_dim"]),
        "mlp_dim": int(controlled["mlp_dim"]),
        "num_conv_blocks": int(controlled["num_conv_blocks"]),
        "kernel_size": int(controlled["kernel_size"]),
        "grid_h": int(controlled["grid_h"]),
        "grid_w": int(controlled["grid_w"]),
        "variant": str(controlled["student_variant"]),
    }
    for key, expected_value in expected.items():
        if student_config.get(key) != expected_value:
            raise ValueError(
                f"{label}: student config {key} mismatch; "
                f"expected={expected_value!r} actual={student_config.get(key)!r}"
            )
    unexpected = set(student_config) - set(expected)
    if unexpected:
        raise ValueError(
            f"{label}: student config has unexpected keys={sorted(unexpected)}"
        )


def _validate_train_log(
    *,
    label: str,
    path: Path,
    controlled: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(payload)

    epochs = int(controlled["epochs"])
    if len(records) != epochs:
        raise ValueError(f"{label}: expected {epochs} train-log rows, found {len(records)}")
    expected_train = int(controlled["expected_train_samples"])
    expected_val = int(controlled["expected_val_samples"])
    for index, record in enumerate(records, 1):
        if int(record.get("epoch", -1)) != index:
            raise ValueError(f"{label}: train-log epoch sequence is incomplete")
        if int(record.get("train_seen", -1)) != expected_train:
            raise ValueError(f"{label}: epoch {index} train_seen is not {expected_train}")
        if int(record.get("val_seen", -1)) != expected_val:
            raise ValueError(f"{label}: epoch {index} val_seen is not {expected_val}")
        for metric in ("train_loss", "train_mse", "train_rank", "val_loss"):
            value = record.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{label}: epoch {index} has invalid {metric}")
    best = min(records, key=lambda row: float(row["val_loss"]))
    return {
        "epochs": len(records),
        "final_epoch": int(records[-1]["epoch"]),
        "best_validation_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["val_loss"]),
        "train_seen_each_epoch": expected_train,
        "val_seen_each_epoch": expected_val,
    }


def resolve_checkpoint_provenance(
    *,
    label: str,
    student: Mapping[str, Any],
    base_model: Mapping[str, Any],
    controlled: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(str(student["path"])).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"{label}: missing checkpoint directory: {checkpoint}")

    required = [
        checkpoint / filename
        for filename in (
            *REQUIRED_HASHED_CHECKPOINT_FILES,
            *REQUIRED_PRESENCE_CHECKPOINT_FILES,
        )
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"{label}: missing or empty checkpoint files: {missing}")

    train_config = _load_json_object(checkpoint / "train_config.json")
    student_config = _load_json_object(checkpoint / "config.json")
    _validate_train_config(
        label=label,
        checkpoint=checkpoint,
        student=student,
        base_model=base_model,
        controlled=controlled,
        train_config=train_config,
    )
    _validate_student_config(
        label=label,
        controlled=controlled,
        student_config=student_config,
    )
    training_summary = _validate_train_log(
        label=label,
        path=checkpoint / "train_log.jsonl",
        controlled=controlled,
    )

    hashed_paths = {
        filename: checkpoint / filename
        for filename in REQUIRED_HASHED_CHECKPOINT_FILES
    }
    presence_paths = {
        filename: checkpoint / filename
        for filename in REQUIRED_PRESENCE_CHECKPOINT_FILES
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint),
        "files_sha256": {
            filename: sha256_file(path) for filename, path in hashed_paths.items()
        },
        "file_sizes_bytes": {
            filename: path.stat().st_size
            for filename, path in {**hashed_paths, **presence_paths}.items()
        },
        "training_summary": training_summary,
        "teacher_root": str(Path(str(student["teacher_root"])).resolve()),
    }


def validate_lineage_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    teacher = _load_json_object(TEACHER_VERIFICATION_PATH)
    expected_teacher = {
        "schema_version": 1,
        "source_root": str(config["paired_training_contract"]["source_teacher_root"]),
        "pa_root": str(config["students"]["prefill_answer_50_50"]["teacher_root"]),
        "qa_root": str(config["students"]["question_answer_50_50"]["teacher_root"]),
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
    recomposition_error = teacher.get("max_fp16_recomposition_error")
    target_difference = teacher.get("mean_absolute_pa_qa_target_difference")
    if (
        not isinstance(recomposition_error, (int, float))
        or not math.isfinite(float(recomposition_error))
        or float(recomposition_error) > 3e-4
    ):
        raise ValueError(
            f"{TEACHER_VERIFICATION_PATH}: invalid recomposition error"
        )
    if (
        not isinstance(target_difference, (int, float))
        or not math.isfinite(float(target_difference))
        or float(target_difference) <= 0
    ):
        raise ValueError(
            f"{TEACHER_VERIFICATION_PATH}: paired targets are not distinguished"
        )

    training = _load_json_object(TRAINING_VERIFICATION_PATH)
    if training.get("schema_version") != 1 or training.get("paired_training") is not True:
        raise ValueError(
            f"{TRAINING_VERIFICATION_PATH}: invalid paired-training header"
        )
    training_keys = {
        "prefill_answer": "prefill_answer_50_50",
        "question_answer": "question_answer_50_50",
    }
    controlled = config["controlled_training_properties"]
    for artifact_key, label in training_keys.items():
        payload = training.get(artifact_key)
        if not isinstance(payload, dict):
            raise ValueError(
                f"{TRAINING_VERIFICATION_PATH}: missing {artifact_key} record"
            )
        checkpoint = Path(str(config["students"][label]["path"])).resolve()
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
                    f"{TRAINING_VERIFICATION_PATH}: {artifact_key}.{key} "
                    f"expected={expected_value!r} found={payload.get(key)!r}"
                )

    return {
        "teacher": {
            "path": str(TEACHER_VERIFICATION_PATH),
            "sha256": sha256_file(TEACHER_VERIFICATION_PATH),
            "validated_payload": teacher,
        },
        "training": {
            "path": str(TRAINING_VERIFICATION_PATH),
            "sha256": sha256_file(TRAINING_VERIFICATION_PATH),
            "validated_payload": training,
        },
    }


def verify_required_noncheckpoint_paths(config: Mapping[str, Any]) -> None:
    required_directories = [
        Path(str(config["base_model"]["path"])),
        Path(str(config["evaluation"]["local_dataset_root"])),
    ]
    missing_directories = [
        str(path) for path in required_directories if not path.is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(f"Missing required directories: {missing_directories}")

    required_sources = list(CONTRACT_SOURCE_PATHS.values())
    required_sources.extend(
        TASK_ROOT / f"{task}.yaml" for task in config["evaluation"]["tasks"]
    )
    missing_sources = [str(path) for path in required_sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing required source files: {missing_sources}")


def build_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_required_noncheckpoint_paths(config)
    manifest = json.loads(json.dumps(config))
    students = manifest["students"]
    for label in STUDENT_LABELS:
        students[label]["checkpoint_provenance"] = resolve_checkpoint_provenance(
            label=label,
            student=students[label],
            base_model=manifest["base_model"],
            controlled=manifest["controlled_training_properties"],
        )
    manifest["lineage_verification_artifacts"] = validate_lineage_artifacts(
        manifest
    )

    source_paths = dict(CONTRACT_SOURCE_PATHS)
    source_paths.update(
        {
            f"task_yaml:{task}": TASK_ROOT / f"{task}.yaml"
            for task in manifest["evaluation"]["tasks"]
        }
    )
    manifest["runtime_provenance"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "source_sha256": {
            label: sha256_file(path) for label, path in source_paths.items()
        },
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_root}")

    config = load_config()
    manifest = build_manifest(config)
    run_root.mkdir(parents=True)
    for child in ("logs", "outputs", "state", "summary"):
        (run_root / child).mkdir()
    destination = run_root / "run_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
