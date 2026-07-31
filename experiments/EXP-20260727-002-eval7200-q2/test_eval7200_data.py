# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the MileBench data/scoring adapter."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

import eval7200_data as data


def test_dataset_namespace_has_official_28_plus_local_extension() -> None:
    assert len(data.OFFICIAL_MILEBENCH_TASKS) == 28
    assert data.LOCAL_EXTENSION_TASKS == ("nuscenes",)
    assert len(data.MILEBENCH_DATASETS) == 29
    assert len(set(data.MILEBENCH_DATASETS)) == 29
    assert all(
        name.startswith("milebench__") for name in data.MILEBENCH_DATASETS
    )
    assert "docvqa" in data.STANDARD_DATASETS
    assert "milebench__DocVQA" in data.MILEBENCH_DATASETS


def test_seeded_selection_preserves_original_row_indices() -> None:
    tokenizer = data.load_tokenizer()
    first = data.load_samples(
        "milebench__GPR1200",
        n=8,
        seed=42,
        tokenizer=tokenizer,
        load_images=False,
    )
    second = data.load_samples(
        "milebench__GPR1200",
        n=8,
        seed=42,
        tokenizer=tokenizer,
        load_images=False,
    )
    third = data.load_samples(
        "milebench__GPR1200",
        n=8,
        seed=43,
        tokenizer=tokenizer,
        load_images=False,
    )
    first_ids = [(sample.row_index, sample.sample_id) for sample in first]
    assert first_ids == [
        (sample.row_index, sample.sample_id) for sample in second
    ]
    assert first_ids != [
        (sample.row_index, sample.sample_id) for sample in third
    ]
    assert any(sample.row_index >= 200 for sample in first)
    assert all(sample.max_new_tokens == 64 for sample in first)
    # LOOK-M intentionally omits alphabetical labels for GPR1200.
    assert "\nA. " not in first[0].context


def test_long_prompt_is_left_truncated_in_expanded_space() -> None:
    tokenizer = data.load_tokenizer()
    samples = data.load_samples(
        "milebench__TextNeedleInAHaystack",
        n=12,
        seed=42,
        tokenizer=tokenizer,
        load_images=False,
        max_prompt_tokens=4096,
    )
    sample = next(
        value
        for value in samples
        if value.metadata["prompt_was_left_truncated"]
    )
    assert sample.metadata["prompt_was_left_truncated"] is True
    assert sample.metadata["removed_task_context_tokens"] > 0
    assert sample.metadata["expanded_prompt_tokens"] <= 4096
    _raw, expanded = data.expanded_prompt_length(sample.context, tokenizer)
    assert expanded == sample.metadata["expanded_prompt_tokens"]
    assert sample.max_new_tokens == 128


def test_look_compatible_sample_scores() -> None:
    open_sample = data.EvalSample(
        dataset="milebench__ALFRED",
        sample_id="open",
        row_index=0,
        image=None,
        raw_question="question",
        context="question",
        references=("pick up the cup",),
        max_new_tokens=64,
        metadata={
            "milebench_task": "ALFRED",
            "milebench_question_type": "open-ended",
        },
    )
    assert data.score_milebench_prediction(
        open_sample,
        "pick up the cup",
    ) == pytest.approx(1.0)
    assert data.score_milebench_prediction(open_sample, "") == 0.0

    choice_sample = data.EvalSample(
        dataset="milebench__ActionPrediction",
        sample_id="choice",
        row_index=0,
        image=None,
        raw_question="question",
        context="question",
        references=("turn left",),
        max_new_tokens=64,
        metadata={
            "milebench_task": "ActionPrediction",
            "milebench_question_type": "multi-choice",
            "choice_list": ["go straight", "turn left", "stop"],
        },
    )
    assert data.score_milebench_prediction(choice_sample, "B") == 1.0
    assert data.score_milebench_prediction(choice_sample, "A") == 0.0

    needle_sample = data.EvalSample(
        dataset="milebench__TextNeedleInAHaystack",
        sample_id="needle",
        row_index=0,
        image=None,
        raw_question="question",
        context="question",
        references=("banana",),
        max_new_tokens=128,
        metadata={
            "milebench_task": "TextNeedleInAHaystack",
            "milebench_question_type": "open-ended",
        },
    )
    assert data.score_milebench_prediction(
        needle_sample,
        "The answer is banana.",
    ) == 1.0


def test_every_local_task_has_annotation_and_combined_images() -> None:
    root = Path(data.DEFAULT_MILEBENCH_ROOT)
    for task in data.LOCAL_MILEBENCH_TASKS:
        task_root = root / task
        assert (task_root / f"{task}.json").is_file()
        assert (task_root / "combined_1_images").is_dir()


def test_milebench_manifest_keeps_namespaced_dataset_count(
    tmp_path: Path,
) -> None:
    sample = data.EvalSample(
        dataset="milebench__ALFRED",
        sample_id="one",
        row_index=7,
        image=None,
        raw_question="question",
        context="question",
        references=("answer",),
        max_new_tokens=128,
        metadata={"task_type": "vqa"},
    )
    path = tmp_path / "manifest.json"
    data.write_manifest([sample], path)
    payload = json.loads(path.read_text())
    assert payload["datasets"] == {"milebench__ALFRED": 1}
