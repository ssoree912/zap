# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

DEFAULT_PROMPT_TEMPLATE = "USER: <image>\n{question}\nASSISTANT:"
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\{image#\d+\}")
LEGACY_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"<ImageHere>", re.IGNORECASE)


def _sanitize_sample_id(value: Any, idx: int) -> str:
    text = str(value) if value is not None else f"sample-{idx:06d}"
    text = text.strip() or f"sample-{idx:06d}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:128]


def resolve_records_dir(input_dir: str | Path, subdir_name: str = "records") -> Path:
    path = Path(input_dir).resolve()
    subdir = path / subdir_name
    if subdir.is_dir():
        return subdir
    return path


def resolve_teacher_dir(input_dir: str | Path) -> Path:
    path = Path(input_dir).resolve()
    for subdir_name in ("teacher_records", "records"):
        subdir = path / subdir_name
        if subdir.is_dir():
            return subdir
    return path


def list_pt_files(input_dir: str | Path, glob_pattern: str = "*.pt") -> list[Path]:
    directory = Path(input_dir).resolve()
    return sorted(directory.glob(glob_pattern))


def load_pt_record(path: str | Path) -> dict[str, Any]:
    rec = torch.load(Path(path), map_location="cpu")
    if not isinstance(rec, dict):
        raise ValueError(f"Expected dict in {path}, got {type(rec)}")
    return rec


def _normalize_prompt_template(prompt_template: str, image_token: str = "<image>") -> str:
    template = prompt_template
    template = template.replace("{image_tokens}\n", "")
    template = template.replace("{image_tokens}", "")
    if f"{image_token}\n" in template:
        template = template.replace(f"{image_token}\n", "", 1)
    elif image_token in template:
        template = template.replace(image_token, "", 1)
    template = re.sub(r"\n{3,}", "\n\n", template)
    return template.strip()


def _inject_image_tokens(question: str, image_count: int, image_token: str = "<image>") -> str:
    question = str(question).strip()
    image_count = max(0, int(image_count))

    marker = "__KVZAP_IMAGE_MARKER__"

    # Normalize all known placeholder variants to a temporary marker.
    question = IMAGE_PLACEHOLDER_PATTERN.sub(marker, question)
    question = LEGACY_IMAGE_PLACEHOLDER_PATTERN.sub(marker, question)
    question = question.replace(image_token, marker)

    marker_count = question.count(marker)
    if marker_count < image_count:
        missing = image_count - marker_count
        prefix = "\n".join([marker] * missing)
        question = f"{prefix}\n{question}" if question else prefix
    elif marker_count > image_count:
        extras = marker_count - image_count
        # Drop extra placeholders from left to keep the most recent context.
        question = question.replace(marker, "", extras)

    question = question.replace(marker, f"{image_token}\n")
    if image_count > 0 and image_token not in question:
        prefix = "\n".join([image_token] * image_count)
        question = f"{prefix}\n{question}" if question else prefix

    question = re.sub(r"[ \t]+\n", "\n", question)
    question = re.sub(r"\n{3,}", "\n\n", question)
    return question.strip()

def build_prompt(
    question: str | dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    image_count: int = 1,
) -> str:
    if isinstance(question, dict):
        fields: dict[str, str] = {}
        for key, value in question.items():
            if value is None:
                fields[key] = ""
            elif isinstance(value, list):
                fields[key] = "\n".join(str(item) for item in value)
            else:
                fields[key] = str(value)

        fields.setdefault("question", "")
        fields.setdefault("hint", "")
        fields.setdefault("options", "")
        fields.setdefault("prompt_body", fields["question"])
        fields.setdefault("image_tokens", "\n".join(["<image>"] * image_count))

        template_has_image = ("<image>" in prompt_template) or ("{image_tokens}" in prompt_template)
        if not template_has_image:
            fields["question"] = _inject_image_tokens(fields["question"], image_count=image_count)
        elif IMAGE_PLACEHOLDER_PATTERN.search(fields["question"]) or LEGACY_IMAGE_PLACEHOLDER_PATTERN.search(fields["question"]):
            fields["question"] = _inject_image_tokens(fields["question"], image_count=image_count)

        formatted = prompt_template.format(**fields).strip()
        if not template_has_image and "<image>" not in formatted:
            formatted = _inject_image_tokens(formatted, image_count=image_count)
        return re.sub(r"\n{3,}", "\n\n", formatted).strip()

    question_with_images = _inject_image_tokens(question, image_count=image_count)
    normalized_template = _normalize_prompt_template(prompt_template)
    return normalized_template.format(question=question_with_images).strip()


def normalize_answer(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(" .,!?:;\"'")
    return text


def _load_json_records(dataset_path: Path) -> list[dict[str, Any]]:
    if dataset_path.suffix == ".jsonl":
        records = []
        with dataset_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if dataset_path.suffix == ".json":
        with dataset_path.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("JSON dataset must be a list or a dict with a `data` list")

    raise ValueError("Unsupported dataset format; expected .json or .jsonl")


def _build_question_from_milebench(record: dict[str, Any]) -> str:
    task = record["task_instance"]
    question = str(task["context"]).strip()
    question = re.sub(r"[ \t]+", " ", question)
    question = re.sub(r"\n{3,}", "\n\n", question)
    choice_list = task.get("choice_list")
    if choice_list:
        choices = [str(choice).strip() for choice in choice_list if str(choice).strip()]
        if choices:
            question = question + "\nChoices: " + " | ".join(choices)
    return question.strip()


def _resolve_answer(record: dict[str, Any], answer_column: str | None = None) -> Optional[str]:
    candidate_columns = [answer_column, "answer", "response", "target", "label"]
    for column in candidate_columns:
        if column and column in record and record[column] is not None:
            return str(record[column])

    task = record.get("task_instance")
    if isinstance(task, dict):
        for column in ("answer", "answers", "label"):
            value = task.get(column)
            if value is None:
                continue
            if isinstance(value, list) and value:
                return str(value[0])
            return str(value)
    return None


def _resolve_image_paths(image_value: Any, image_base: Path) -> list[str]:
    if isinstance(image_value, (list, tuple)):
        raw_paths = [str(value).strip() for value in image_value if str(value).strip()]
    else:
        raw_paths = [str(image_value).strip()] if str(image_value).strip() else []

    if not raw_paths:
        raise ValueError("Could not resolve any image paths")

    image_paths: list[str] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = image_base / path
        image_paths.append(str(path.resolve()))
    return image_paths


def load_vlm_samples(
    dataset_path: str,
    image_root: str | None = None,
    question_column: str = "question",
    image_column: str = "image_path",
    id_column: str = "sample_id",
    answer_column: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset_file = Path(dataset_path).resolve()
    image_base = Path(image_root).resolve() if image_root is not None else dataset_file.parent
    raw_records = _load_json_records(dataset_file)

    samples: list[dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        if "task_instance" in record:
            question = _build_question_from_milebench(record)
            task = record["task_instance"]
            image_value = None

            # Prefer the caller-selected task image field when available so
            # MileBench records can choose between raw multi-image paths and
            # pre-combined single-image variants such as `combined_1_images`.
            task_image_candidates = [image_column, "image_path", "image", "images", "image_paths"]
            for column in task_image_candidates:
                if column and column in task and task[column]:
                    image_value = task[column]
                    break

            if image_value is None and "images_path" in task and task["images_path"]:
                image_value = task["images_path"]
            if image_value is None and "combined_1_images" in task and task["combined_1_images"]:
                image_value = task["combined_1_images"]
            if image_value is None:
                raise KeyError(f"Could not resolve image path in MileBench sample #{idx}")
            answer = _resolve_answer(record, answer_column=answer_column)
            sample_id = _sanitize_sample_id(record.get(id_column), idx)
        else:
            if question_column not in record:
                raise KeyError(f"Missing `{question_column}` in sample #{idx}")
            question = str(record[question_column])
            image_value = record.get(image_column)
            if image_value is None and image_column != "image_path":
                image_value = record.get("image_path")
            if image_value is None and image_column != "image":
                image_value = record.get("image")
            if image_value is None and image_column != "images":
                image_value = record.get("images")
            if image_value is None and image_column != "image_paths":
                image_value = record.get("image_paths")
            if image_value is None:
                raise KeyError(f"Missing image field `{image_column}` in sample #{idx}")
            answer = _resolve_answer(record, answer_column=answer_column)
            sample_id = _sanitize_sample_id(record.get(id_column), idx)

        image_paths = _resolve_image_paths(image_value, image_base)

        samples.append(
            {
                "sample_id": sample_id,
                "question": question,
                "image_paths": image_paths,
                "image_path": image_paths[0],
                "answer": answer,
                "raw": record,
            }
        )
        if limit is not None and len(samples) >= limit:
            break

    return samples


def split_sample_ids(sample_ids: Iterable[str], train_fraction: float, seed: int = 42) -> tuple[list[str], list[str]]:
    sample_ids = sorted(set(sample_ids))
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    if not sample_ids:
        return [], []
    n_train = max(1, min(len(sample_ids) - 1, int(round(len(sample_ids) * train_fraction)))) if len(sample_ids) > 1 else 1
    train_ids = sorted(sample_ids[:n_train])
    test_ids = sorted(sample_ids[n_train:])
    if not test_ids:
        test_ids = train_ids[-1:]
        train_ids = train_ids[:-1] or train_ids
    return train_ids, test_ids


def iter_common_sample_ids(extractor_dir: str | Path, teacher_dir: str | Path) -> list[str]:
    extractor_records = resolve_records_dir(extractor_dir)
    teacher_records = resolve_teacher_dir(teacher_dir)
    extractor_ids = {path.stem for path in list_pt_files(extractor_records)}
    teacher_ids = {path.stem for path in list_pt_files(teacher_records)}
    return sorted(extractor_ids & teacher_ids)
