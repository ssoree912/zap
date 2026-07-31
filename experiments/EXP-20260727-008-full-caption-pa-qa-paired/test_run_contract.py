# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the paired P+A versus Q+A full-caption contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
if str(ZAP_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_ROOT))

from foresight.eval.cache_budget import (  # noqa: E402
    EXACT_TOTAL_CEIL,
    LEGACY_TOTAL_ROUND,
    compute_image_keep_count,
    requested_total_token_budget,
)


def _load_module(filename: str, module_name: str):
    path = EXP_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validator():
    return _load_module(
        "validate_paired_outputs.py", "pa_qa_full_caption_validator"
    )


def _load_prepare():
    return _load_module("prepare_run.py", "pa_qa_full_caption_prepare")


def test_exact_total_budget_uses_ceil_and_preserves_text() -> None:
    n_image = 576
    n_text = 50
    prompt_len = n_image + n_text
    requested = requested_total_token_budget(
        prompt_len=prompt_len,
        keep_ratio=0.2,
        mode=EXACT_TOTAL_CEIL,
    )
    n_keep = compute_image_keep_count(
        n_image=n_image,
        n_text=n_text,
        keep_ratio=0.2,
        mode=EXACT_TOTAL_CEIL,
    )
    assert requested == math.ceil(0.2 * prompt_len) == 126
    assert n_keep == 76
    assert n_text + n_keep == requested


def test_exact_budget_can_keep_zero_visual_tokens_when_text_saturates() -> None:
    assert (
        compute_image_keep_count(
            n_image=5,
            n_text=95,
            keep_ratio=0.2,
            mode=EXACT_TOTAL_CEIL,
        )
        == 0
    )


def test_legacy_mode_is_distinct() -> None:
    assert (
        compute_image_keep_count(
            n_image=5,
            n_text=95,
            keep_ratio=0.2,
            mode=LEGACY_TOTAL_ROUND,
        )
        == 1
    )


def test_static_config_is_checkpoint_independent_and_pins_four_gpu_map() -> None:
    prepare = _load_prepare()
    config = prepare.load_config()
    assert tuple(config["students"]) == (
        "prefill_answer_50_50",
        "question_answer_50_50",
    )
    assert all(
        "pytorch_model_sha256" not in student
        for student in config["students"].values()
    )
    evaluation = config["evaluation"]
    assert evaluation["physical_gpu_allowlist"] == [0, 1, 2, 3]
    assert evaluation["job_assignment"] == prepare.EXPECTED_GPU_ASSIGNMENT
    assert evaluation["keep_budget_mode"] == EXACT_TOTAL_CEIL
    assert {
        task: value["expected_samples"]
        for task, value in evaluation["tasks"].items()
    } == prepare.EXPECTED_TASK_COUNTS
    assert evaluation["tasks"] == prepare.EXPECTED_TASK_CONFIGS
    assert (
        prepare.CONTRACT_SOURCE_PATHS["caption_metric_implementation"]
        == prepare.TASK_ROOT / "caption_rouge_utils.py"
    )


def _controlled() -> dict:
    return {
        "datasets": ["gqa", "textvqa", "scienceqa"],
        "seed": 0,
        "epochs": 15,
        "samples_per_dataset": 600,
        "val_ratio": 0.1,
        "expected_train_samples": 1620,
        "expected_val_samples": 180,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "lambda_rank": 0.1,
        "rank_margin": 0.05,
        "rank_top_ratio": 0.2,
        "rank_bottom_ratio": 0.4,
        "max_grad_norm": 1.0,
        "log_every": 25,
        "student_variant": "full",
        "conv_dim": 256,
        "proj_dim": 256,
        "mlp_dim": 512,
        "num_conv_blocks": 2,
        "kernel_size": 7,
        "grid_h": 24,
        "grid_w": 24,
        "hidden_dim": 4096,
    }


def _make_checkpoint(
    *,
    root: Path,
    label: str,
    teacher_root: Path,
    base_model: Path,
) -> dict:
    checkpoint = root / label
    checkpoint.mkdir(parents=True)
    (checkpoint / "pytorch_model.bin").write_bytes(f"weights:{label}".encode())
    student_config = {
        "layer_indices": list(range(32)),
        "hidden_dim": 4096,
        "conv_dim": 256,
        "proj_dim": 256,
        "mlp_dim": 512,
        "num_conv_blocks": 2,
        "kernel_size": 7,
        "grid_h": 24,
        "grid_w": 24,
        "variant": "full",
    }
    (checkpoint / "config.json").write_text(json.dumps(student_config))
    train_config = {
        "teacher_root": str(teacher_root),
        "datasets": ["gqa", "textvqa", "scienceqa"],
        "llava_path": str(base_model),
        "model_name": "llava-v1.5-7b",
        "vision_tower_path": "",
        "device": "cuda:0",
        "device_map": "cuda:0",
        "epochs": 15,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "lambda_rank": 0.1,
        "rank_margin": 0.05,
        "rank_top_ratio": 0.2,
        "rank_bottom_ratio": 0.4,
        "max_grad_norm": 1.0,
        "seed": 0,
        "n_per_dataset": 600,
        "val_ratio": 0.1,
        "log_every": 25,
        "student_variant": "full",
        "conv_dim": 256,
        "proj_dim": 256,
        "mlp_dim": 512,
        "num_conv_blocks": 2,
        "kernel_size": 7,
        "grid_h": 24,
        "grid_w": 24,
        "output_dir": str(checkpoint),
    }
    (checkpoint / "train_config.json").write_text(json.dumps(train_config))
    with (checkpoint / "train_log.jsonl").open("w") as handle:
        for epoch in range(1, 16):
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": 1.0 / epoch,
                        "train_mse": 0.01,
                        "train_rank": 0.02,
                        "val_loss": 1.0 / epoch,
                        "val_mse": 0.01,
                        "val_rank": 0.02,
                        "train_seen": 1620,
                        "val_seen": 180,
                    }
                )
                + "\n"
            )
    (checkpoint / "last_checkpoint.pt").write_bytes(b"complete")
    return {"path": str(checkpoint), "teacher_root": str(teacher_root)}


def test_dynamic_checkpoint_provenance_validates_training_completion(
    tmp_path: Path,
) -> None:
    prepare = _load_prepare()
    base_model = tmp_path / "model"
    teacher_root = tmp_path / "teacher"
    student = _make_checkpoint(
        root=tmp_path,
        label="prefill_answer_50_50",
        teacher_root=teacher_root,
        base_model=base_model,
    )
    provenance = prepare.resolve_checkpoint_provenance(
        label="prefill_answer_50_50",
        student=student,
        base_model={"path": str(base_model), "model_name": "llava-v1.5-7b"},
        controlled=_controlled(),
    )
    weights = Path(student["path"]) / "pytorch_model.bin"
    assert provenance["files_sha256"]["pytorch_model.bin"] == hashlib.sha256(
        weights.read_bytes()
    ).hexdigest()
    assert provenance["training_summary"]["epochs"] == 15
    assert provenance["training_summary"]["best_validation_epoch"] == 15


def test_dynamic_checkpoint_provenance_fails_closed_when_missing(
    tmp_path: Path,
) -> None:
    prepare = _load_prepare()
    with pytest.raises(FileNotFoundError, match="missing checkpoint directory"):
        prepare.resolve_checkpoint_provenance(
            label="prefill_answer_50_50",
            student={
                "path": str(tmp_path / "missing"),
                "teacher_root": str(tmp_path / "teacher"),
            },
            base_model={
                "path": str(tmp_path / "model"),
                "model_name": "llava-v1.5-7b",
            },
            controlled=_controlled(),
        )


def test_lmms_result_validation_requires_exact_model_args(tmp_path: Path) -> None:
    validator = _load_validator()
    task = "coco2017_cap_local"
    metric = "coco_ROUGE_L,none"
    stats_path = tmp_path / "keep_stats" / f"{task}_keep_ratio_stats.json"
    stats_path.parent.mkdir()
    expected_args = {
        "pretrained": "/model",
        "student_path": "/student",
        "keep_ratio": "0.2",
        "keep_budget_mode": "exact_total_ceil",
        "student_failure_policy": "raise",
        "conv_template": "vicuna_v1",
        "model_name": "llava-v1.5-7b",
        "device": "cuda:0",
        "device_map": "cuda:0",
        "stats_output_dir": str(stats_path.parent),
    }
    payload = {
        "n-samples": {task: {"original": 2, "effective": 2}},
        "results": {
            task: {
                metric: 0.5,
                "coco_Bleu_4,none": [],
            }
        },
        "config": {
            "model": "llava15_original_student",
            "model_args": ",".join(
                f"{key}={value}" for key, value in expected_args.items()
            ),
            "batch_size": "1",
            "limit": None,
        },
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(payload))
    metrics = validator._validate_results(
        path=result_path,
        task=task,
        expected=2,
        primary_metrics=(metric,),
        expected_model_path="/model",
        expected_model_name="llava-v1.5-7b",
        expected_conv_template="vicuna_v1",
        expected_student_path="/student",
        expected_stats_path=stats_path.parent,
        keep_ratio=0.2,
    )
    assert metrics == {metric: 0.5}

    payload["config"]["model_args"] = payload["config"]["model_args"].replace(
        "student_failure_policy=raise,", ""
    )
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model_args do not exactly match"):
        validator._validate_results(
            path=result_path,
            task=task,
            expected=2,
            primary_metrics=(metric,),
            expected_model_path="/model",
            expected_model_name="llava-v1.5-7b",
            expected_conv_template="vicuna_v1",
            expected_student_path="/student",
            expected_stats_path=stats_path.parent,
            keep_ratio=0.2,
        )


def _write_synthetic_model_task(
    *,
    root: Path,
    validator,
    label: str,
    student_path: Path,
    task: str,
    metric: str,
    value: float,
) -> None:
    output_root = root / "outputs" / label / task
    model_root = output_root / "models__llava-v1.5-7b"
    stats_root = output_root / "keep_stats"
    model_root.mkdir(parents=True)
    stats_root.mkdir()
    model_args = {
        "pretrained": "/synthetic/model",
        "student_path": str(student_path),
        "keep_ratio": "0.2",
        "keep_budget_mode": "exact_total_ceil",
        "student_failure_policy": "raise",
        "conv_template": "vicuna_v1",
        "model_name": "llava-v1.5-7b",
        "device": "cuda:0",
        "device_map": "cuda:0",
        "stats_output_dir": str(stats_root),
    }
    result = {
        "n-samples": {task: {"original": 2, "effective": 2}},
        "results": {task: {metric: value}},
        "config": {
            "model": "llava15_original_student",
            "model_args": ",".join(
                f"{key}={item}" for key, item in model_args.items()
            ),
            "batch_size": "1",
            "limit": None,
        },
    }
    (model_root / "000_results.json").write_text(json.dumps(result))
    with (model_root / f"000_samples_{task}.jsonl").open("w") as handle:
        for doc_id in range(2):
            sample = {
                "doc_id": doc_id,
                "doc": {"id": doc_id, "question": f"question {doc_id}"},
                "prompt_hash": f"prompt-{doc_id}",
                "target_hash": f"target-{doc_id}",
                "input": f"question {doc_id}",
                "resps": [[f"{label}-{doc_id}"]],
            }
            handle.write(json.dumps(sample) + "\n")
    budget_sample = {
        "prompt_len": 20,
        "n_image_original": 18,
        "n_text": 2,
        "n_image_kept": 2,
        "requested_total_token_budget": 4,
        "actual_total_tokens_kept": 4,
    }
    stats = {
        "task": task,
        "student_path": str(student_path),
        "keep_ratio": 0.2,
        "keep_ratio_basis": "total_prompt_cache",
        "keep_budget_mode": validator.EXPECTED_BUDGET_MODE,
        "head_mask_policy": validator.EXPECTED_MASK_POLICY,
        "hidden_state_convention": validator.EXPECTED_HIDDEN_STATE,
        "student_failure_policy": "raise",
        "n_samples": 2,
        "samples": [budget_sample, budget_sample],
    }
    (stats_root / f"{task}_keep_ratio_stats.json").write_text(json.dumps(stats))


def _write_done_states(run_root: Path, validator) -> None:
    state_root = run_root / "state"
    state_root.mkdir()
    for gpu, jobs in validator.EXPECTED_GPU_ASSIGNMENT.items():
        for job in jobs:
            label, task = job.split(":", 1)
            (state_root / f"{label}__{task}.done").write_text(
                f"physical_gpu={gpu}\n"
                f"label={label}\n"
                f"task={task}\n"
                "completed_at=synthetic\n"
            )


def test_synthetic_full_pair_validation_and_official_macro(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = _load_prepare()
    validator = _load_validator()
    monkeypatch.setattr(validator, "_validate_frozen_sources", lambda manifest: None)
    monkeypatch.setattr(
        validator,
        "_validate_lineage_verification",
        lambda manifest: None,
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "summary").mkdir()
    _write_done_states(run_root, validator)

    controlled = _controlled()
    base_model = {"path": "/synthetic/model", "model_name": "llava-v1.5-7b"}
    students = {}
    for label in validator.STUDENT_LABELS:
        student = _make_checkpoint(
            root=tmp_path,
            label=label,
            teacher_root=tmp_path / f"teacher_{label}",
            base_model=Path(base_model["path"]),
        )
        student["checkpoint_provenance"] = prepare.resolve_checkpoint_provenance(
            label=label,
            student=student,
            base_model=base_model,
            controlled=controlled,
        )
        students[label] = student

    task_values = {
        "coco2017_cap_local": ("coco_ROUGE_L,none", 0.1),
        "nocaps_local": ("nocaps_ROUGE_L,none", 0.2),
        "textcaps_local": ("textcaps_ROUGE_L,none", 0.3),
    }
    for task, (metric, pa_value) in task_values.items():
        _write_synthetic_model_task(
            root=run_root,
            validator=validator,
            label=validator.PA_LABEL,
            student_path=Path(students[validator.PA_LABEL]["path"]),
            task=task,
            metric=metric,
            value=pa_value,
        )
        _write_synthetic_model_task(
            root=run_root,
            validator=validator,
            label=validator.QA_LABEL,
            student_path=Path(students[validator.QA_LABEL]["path"]),
            task=task,
            metric=metric,
            value=pa_value + 0.1,
        )

    metric_policy = (
        "Use only the official local ROUGE-L aggregate emitted by LMMS-eval "
        "for each caption task."
    )
    manifest = {
        "experiment_id": "synthetic",
        "comparison_is_oracle": False,
        "students": students,
        "base_model": {
            **base_model,
            "conversation_template": "vicuna_v1",
        },
        "controlled_training_properties": controlled,
        "paired_training_contract": {
            "split_hashes": validator.EXPECTED_SPLIT_HASHES
        },
        "evaluation": {
            "local_dataset_root": "/workspace/nips/data/eval",
            "keep_budget_mode": "exact_total_ceil",
            "hidden_state_convention": "post_block_hidden_states[layer_idx + 1]",
            "physical_gpu_allowlist": [0, 1, 2, 3],
            "job_assignment": validator.EXPECTED_GPU_ASSIGNMENT,
            "requested_total_keep_ratio": 0.2,
            "student_failure_policy": (
                "raise; never silently replace a failed pruning sample "
                "with full-cache generation"
            ),
            "metric_policy": metric_policy,
            "tasks": {
                task: {"expected_samples": 2, "primary_metrics": [metric]}
                for task, (metric, _) in task_values.items()
            },
        },
    }
    monkeypatch.setattr(
        validator,
        "EXPECTED_TASK_CONFIGS",
        manifest["evaluation"]["tasks"],
    )
    (run_root / "run_manifest.json").write_text(json.dumps(manifest))

    summary, paired_rows = validator.validate_run(run_root)
    assert summary["validation_status"] == "passed"
    assert summary["total_paired_samples_per_student"] == 6
    assert len(paired_rows) == 6
    macro = summary["three_caption_rouge_l_macro"]
    assert macro[validator.PA_LABEL] == pytest.approx(0.2)
    assert macro[validator.QA_LABEL] == pytest.approx(0.3)
    assert macro[validator.DELTA_LABEL] == pytest.approx(0.1)


def test_wrong_gpu_state_is_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_done_states(run_root, validator)
    path = (
        run_root
        / "state"
        / "prefill_answer_50_50__coco2017_cap_local.done"
    )
    path.write_text(
        "physical_gpu=3\n"
        "label=prefill_answer_50_50\n"
        "task=coco2017_cap_local\n"
    )
    with pytest.raises(ValueError, match="expected physical_gpu"):
        validator._validate_job_states(run_root)
