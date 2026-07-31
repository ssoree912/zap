# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the exact full-VQA checkpoint pair."""

from __future__ import annotations

import importlib.util
import hashlib
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


def _load_validator():
    path = EXP_DIR / "validate_paired_outputs.py"
    spec = importlib.util.spec_from_file_location("full_vqa_pair_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_legacy_mode_preserves_minimum_one_visual_token() -> None:
    assert (
        compute_image_keep_count(
            n_image=5,
            n_text=95,
            keep_ratio=0.2,
            mode=LEGACY_TOTAL_ROUND,
        )
        == 1
    )


def test_static_config_names_target_and_controls_without_oracle_claim() -> None:
    config = json.loads((EXP_DIR / "run_config.json").read_text())
    assert config["comparison_is_oracle"] is False
    assert tuple(config["students"]) == (
        "last_prompt_plus_answer_steps",
        "question_answer_50_50",
    )
    base = config["students"]["last_prompt_plus_answer_steps"]
    qa = config["students"]["question_answer_50_50"]
    assert base["last_prompt_query_preserved"] is True
    assert base["teacher_extraction_max_new_tokens"] == 32
    assert qa["teacher_extraction_max_new_tokens"] == 64
    evaluation = config["evaluation"]
    assert evaluation["local_dataset_root"] == "/workspace/nips/data/eval"
    assert evaluation["keep_budget_mode"] == EXACT_TOTAL_CEIL
    assert evaluation["physical_gpu_allowlist"] == [0, 1, 2]
    assert set(evaluation["tasks"]) == {
        "gqa_local",
        "textvqa_local",
        "docvqa_local",
        "chartqa_local",
    }


def test_lmms_result_validation_requires_exact_model_args(tmp_path: Path) -> None:
    validator = _load_validator()
    stats_path = tmp_path / "keep_stats" / "gqa_local_keep_ratio_stats.json"
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
        "n-samples": {"gqa_local": {"original": 2, "effective": 2}},
        "results": {"gqa_local": {"exact_match,none": 0.5}},
        "config": {
            "model": "llava15_original_student",
            "model_args": ",".join(f"{key}={value}" for key, value in expected_args.items()),
            "batch_size": "1",
            "limit": None,
        },
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(payload))
    metrics = validator._validate_results(
        path=result_path,
        task="gqa_local",
        expected=2,
        primary_metrics=("exact_match,none",),
        expected_model_path="/model",
        expected_model_name="llava-v1.5-7b",
        expected_conv_template="vicuna_v1",
        expected_student_path="/student",
        expected_stats_path=stats_path.parent,
        keep_ratio=0.2,
    )
    assert metrics == {"exact_match,none": 0.5}

    payload["config"]["model_args"] = payload["config"]["model_args"].replace(
        "keep_budget_mode=exact_total_ceil,",
        "",
    )
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model_args do not exactly match"):
        validator._validate_results(
            path=result_path,
            task="gqa_local",
            expected=2,
            primary_metrics=("exact_match,none",),
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
            "model_args": ",".join(f"{key}={item}" for key, item in model_args.items()),
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


def test_synthetic_full_pair_validation_and_macro(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "_validate_frozen_sources", lambda manifest: None)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "summary").mkdir()

    students = {}
    for label in validator.STUDENT_LABELS:
        checkpoint = tmp_path / label
        checkpoint.mkdir()
        weights = checkpoint / "pytorch_model.bin"
        weights.write_bytes(label.encode())
        students[label] = {
            "path": str(checkpoint),
            "pytorch_model_sha256": hashlib.sha256(label.encode()).hexdigest(),
        }

    task_values = {
        "gqa_local": ("exact_match,none", 0.1),
        "textvqa_local": ("exact_match,none", 0.2),
        "docvqa_local": ("anls,none", 0.3),
        "chartqa_local": ("relaxed_overall,none", 0.4),
    }
    for task, (metric, base_value) in task_values.items():
        _write_synthetic_model_task(
            root=run_root,
            validator=validator,
            label=validator.BASELINE_LABEL,
            student_path=Path(students[validator.BASELINE_LABEL]["path"]),
            task=task,
            metric=metric,
            value=base_value,
        )
        _write_synthetic_model_task(
            root=run_root,
            validator=validator,
            label=validator.QA_LABEL,
            student_path=Path(students[validator.QA_LABEL]["path"]),
            task=task,
            metric=metric,
            value=base_value + 0.1,
        )

    metric_policy = (
        "Use the official aggregate metrics emitted by LMMS-eval. "
        "Do not rescore DocVQA rows with the eval700 continuous ANLS implementation."
    )
    manifest = {
        "experiment_id": "synthetic",
        "students": students,
        "base_model": {
            "path": "/synthetic/model",
            "model_name": "llava-v1.5-7b",
            "conversation_template": "vicuna_v1",
        },
        "evaluation": {
            "local_dataset_root": "/workspace/nips/data/eval",
            "keep_budget_mode": "exact_total_ceil",
            "hidden_state_convention": "post_block_hidden_states[layer_idx + 1]",
            "physical_gpu_allowlist": [0, 1, 2],
            "requested_total_keep_ratio": 0.2,
            "metric_policy": metric_policy,
            "tasks": {
                task: {"expected_samples": 2, "primary_metrics": [metric]}
                for task, (metric, _) in task_values.items()
            },
        },
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest))

    summary, paired_rows = validator.validate_run(run_root)
    assert summary["validation_status"] == "passed"
    assert summary["total_paired_samples_per_student"] == 8
    assert len(paired_rows) == 8
    macro = summary["four_vqa_primary_macro"]
    assert macro[validator.BASELINE_LABEL] == pytest.approx(0.25)
    assert macro[validator.QA_LABEL] == pytest.approx(0.35)
    assert macro["question_answer_minus_last_prompt_plus_answer_steps"] == pytest.approx(
        0.1
    )
