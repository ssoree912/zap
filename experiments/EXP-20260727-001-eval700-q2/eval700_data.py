#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic local data and scoring helpers for the seven-dataset Q2 run.

The sampling protocol intentionally reproduces the historical eval-700 dump:
each dataset independently applies ``np.random.default_rng(seed).choice`` to
its ordered evaluation rows.  With ``seed=42`` and ``n=100``, the selected ID
sets match ``EXP-20260505-001-mismatch-analysis/mismatch_eval_700.json``.

Public API
----------
``DATASETS``
    Canonical dataset names in paper order.
``load_samples(name, root, n, seed)``
    Load deterministic samples, including RGB PIL images.
``score_prediction(sample, pred)``
    Per-sample score for the four VQA datasets.
``caption_corpus_rouge_l(samples, predictions)``
    Corpus ROUGE-L for the three caption datasets.
``write_manifest(samples, path)``
    Write a stable, image-free JSON manifest.

The CLI is manifest-only and therefore does not decode image payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import statistics
import string
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_ROOT = Path("/workspace/nips/data/eval")
DEFAULT_REFERENCE = Path(
    "/workspace/nips/zap/experiments/"
    "EXP-20260505-001-mismatch-analysis/mismatch_eval_700.json"
)

DATASETS = (
    "gqa",
    "textvqa",
    "docvqa",
    "chartqa",
    "coco_caption",
    "nocaps",
    "textcaps",
)

HISTORICAL_DATASET_NAMES = {
    "gqa": "gqa",
    "textvqa": "textvqa_val",
    "docvqa": "docvqa_val",
    "chartqa": "chartqa",
    "coco_caption": "coco2017_cap_val",
    "nocaps": "nocaps_val",
    "textcaps": "textcaps_val",
}

MAX_NEW_TOKENS = {
    "gqa": 16,
    # The installed official TextVQA YAML specifies an until-string but no
    # max_new_tokens.  The Q2 protocol uses the same 32-token fallback as the
    # teacher collection code.
    "textvqa": 32,
    "docvqa": 32,
    "chartqa": 16,
    "coco_caption": 64,
    "nocaps": 64,
    "textcaps": 64,
}

SINGLE_PHRASE_SUFFIX = "\nAnswer the question using a single word or phrase."
CHARTQA_SUFFIX = "\nAnswer the question with a single word."
CAPTION_CONTEXT = "Provide a one-sentence caption for the provided image."

_ALIASES = {
    "gqa": "gqa",
    "textvqa": "textvqa",
    "textvqa_val": "textvqa",
    "docvqa": "docvqa",
    "docvqa_val": "docvqa",
    "chartqa": "chartqa",
    "coco": "coco_caption",
    "coco_cap": "coco_caption",
    "coco_caption": "coco_caption",
    "coco_caption2017": "coco_caption",
    "coco2017_cap": "coco_caption",
    "coco2017_cap_val": "coco_caption",
    "nocaps": "nocaps",
    "nocaps_val": "nocaps",
    "textcaps": "textcaps",
    "textcaps_val": "textcaps",
}

_PARQUET_PATTERNS = {
    "textvqa": "TextVQA/data/validation-*.parquet",
    "docvqa": "DocVQA/DocVQA/validation-*.parquet",
    "chartqa": "ChartQA/data/test-*.parquet",
    "coco_caption": "COCO-Caption2017/data/val-*.parquet",
    "nocaps": "NoCaps/data/validation-*.parquet",
    "textcaps": "TextCaps/data/val-*.parquet",
}


@dataclass(slots=True)
class EvalSample:
    """One deterministic evaluation sample.

    ``image`` is an RGB :class:`PIL.Image.Image` for the public
    :func:`load_samples` path.  The private manifest-only path leaves it
    ``None`` so that checking all 700 IDs never decompresses image payloads.
    Dataset-specific fields, including the source ``type``, live in
    ``metadata``.
    """

    dataset: str
    sample_id: str
    row_index: int
    image: Image.Image | None
    raw_question: str
    context: str
    references: tuple[str, ...]
    max_new_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """Dataset source type, falling back to ``vqa`` or ``caption``."""

        return str(self.metadata.get("type", self.metadata.get("task_type", "")))

    @property
    def question_span(self) -> str:
        """Exact semantic span as it appears inside ``context``.

        TextVQA's official formatter applies ``capitalize()``.  Caption tasks
        have no separate interrogative span, so their complete official prompt
        is the semantic span.
        """

        if self.metadata.get("task_type") == "caption":
            return self.context
        if self.dataset == "textvqa":
            return self.raw_question.capitalize()
        return self.raw_question


def _canonical_name(name: str) -> str:
    key = re.sub(r"_+", "_", str(name).strip().lower().replace("-", "_").replace(" ", "_"))
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}; expected one of {DATASETS}"
        ) from exc


def _parquet_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching {Path(root) / pattern}")
    return files


def _row_count(files: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def _sample_indices(total: int, n: int, seed: int) -> list[int]:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    count = min(int(n), int(total))
    if count == 0:
        return []
    rng = np.random.default_rng(int(seed))
    return [
        int(index)
        for index in rng.choice(total, size=count, replace=False).tolist()
    ]


def _read_column(files: Sequence[Path], column: str) -> list[Any]:
    """Read one lightweight column in stable shard/row order."""

    values: list[Any] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=[column], batch_size=4096):
            values.extend(batch.column(0).to_pylist())
    return values


def _read_rows(
    files: Sequence[Path],
    row_indices: Sequence[int],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Read arbitrary global rows while preserving ``row_indices`` order.

    Only row groups containing selected rows are decompressed.  This matters
    for the embedded image column in DocVQA and the caption datasets.
    """

    ordered = [int(index) for index in row_indices]
    if len(set(ordered)) != len(ordered):
        raise ValueError("row_indices must be unique")
    if not ordered:
        return []
    if min(ordered) < 0:
        raise IndexError("row_indices cannot be negative")

    wanted = set(ordered)
    found: dict[int, dict[str, Any]] = {}
    cursor = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(row_group).num_rows
            end = cursor + count
            hits = sorted(index for index in wanted if cursor <= index < end)
            if hits:
                table = parquet.read_row_group(
                    row_group,
                    columns=list(columns),
                )
                offsets = pa.array(
                    [index - cursor for index in hits],
                    type=pa.int64(),
                )
                rows = table.take(offsets).to_pylist()
                found.update(zip(hits, rows))
            cursor = end

    missing = [index for index in ordered if index not in found]
    if missing:
        raise IndexError(
            f"Rows outside parquet range or unreadable: {missing[:10]}"
        )
    return [found[index] for index in ordered]


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported parquet image value: {type(value).__name__}")

    payload = value.get("bytes")
    if payload:
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB").copy()
    raise ValueError(
        f"Embedded image bytes are missing (parquet path={value.get('path')!r})"
    )


def _references(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _base_metadata(
    dataset: str,
    *,
    task_type: str,
    source_type: str,
    seed: int,
    n: int,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "type": source_type,
        "historical_dataset": HISTORICAL_DATASET_NAMES[dataset],
        "selection_seed": int(seed),
        "selection_size_requested": int(n),
    }


def _load_gqa(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    instruction_files = _parquet_files(
        root,
        "GQA/testdev_balanced_instructions/testdev-*.parquet",
    )
    image_files = _parquet_files(
        root,
        "GQA/testdev_balanced_images/testdev-*.parquet",
    )

    image_ids = [str(value) for value in _read_column(image_files, "id")]
    image_id_set = set(image_ids)
    instruction_image_ids = [
        str(value) for value in _read_column(instruction_files, "imageId")
    ]
    eligible = [
        row_index
        for row_index, image_id in enumerate(instruction_image_ids)
        if image_id in image_id_set
    ]
    selected_offsets = _sample_indices(len(eligible), n, seed)
    selected_indices = [eligible[offset] for offset in selected_offsets]
    rows = _read_rows(
        instruction_files,
        selected_indices,
        (
            "id",
            "imageId",
            "question",
            "answer",
            "fullAnswer",
            "isBalanced",
            "types",
        ),
    )

    image_lookup: dict[str, Image.Image] = {}
    if load_images:
        image_row_by_id = {
            image_id: row_index for row_index, image_id in enumerate(image_ids)
        }
        needed_ids = list(dict.fromkeys(str(row["imageId"]) for row in rows))
        needed_rows = [image_row_by_id[image_id] for image_id in needed_ids]
        image_rows = _read_rows(
            image_files,
            needed_rows,
            ("id", "image"),
        )
        image_lookup = {
            str(row["id"]): _decode_image(row["image"])
            for row in image_rows
        }

    samples: list[EvalSample] = []
    for row_index, row in zip(selected_indices, rows):
        raw_question = str(row["question"])
        gqa_types = row.get("types") or {}
        detailed_type = (
            str(gqa_types.get("detailed", "vqa"))
            if isinstance(gqa_types, Mapping)
            else "vqa"
        )
        metadata = _base_metadata(
            "gqa",
            task_type="vqa",
            source_type=detailed_type,
            seed=seed,
            n=n,
        )
        metadata.update(
            {
                "image_id": str(row["imageId"]),
                "full_answer": str(row.get("fullAnswer") or ""),
                "is_balanced": bool(row.get("isBalanced", False)),
                "gqa_types": dict(gqa_types)
                if isinstance(gqa_types, Mapping)
                else {},
            }
        )
        samples.append(
            EvalSample(
                dataset="gqa",
                sample_id=str(row["id"]),
                row_index=int(row_index),
                image=image_lookup.get(str(row["imageId"])),
                raw_question=raw_question,
                context=raw_question + SINGLE_PHRASE_SUFFIX,
                references=_references(row.get("answer")),
                max_new_tokens=MAX_NEW_TOKENS["gqa"],
                metadata=metadata,
            )
        )
    return samples


def _selected_rows(
    root: Path,
    dataset: str,
    n: int,
    seed: int,
    columns: Sequence[str],
    *,
    load_images: bool,
) -> tuple[list[int], list[dict[str, Any]]]:
    files = _parquet_files(root, _PARQUET_PATTERNS[dataset])
    selected_indices = _sample_indices(_row_count(files), n, seed)
    requested_columns = list(columns)
    if load_images and "image" not in requested_columns:
        requested_columns.append("image")
    return selected_indices, _read_rows(
        files,
        selected_indices,
        requested_columns,
    )


def _load_textvqa(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    indices, rows = _selected_rows(
        root,
        "textvqa",
        n,
        seed,
        ("question_id", "image_id", "question", "answers", "ocr_tokens"),
        load_images=load_images,
    )
    samples = []
    for row_index, row in zip(indices, rows):
        raw_question = str(row["question"])
        metadata = _base_metadata(
            "textvqa",
            task_type="vqa",
            source_type="vqa",
            seed=seed,
            n=n,
        )
        metadata.update(
            {
                "image_id": str(row["image_id"]),
                "ocr_tokens": [
                    str(token) for token in (row.get("ocr_tokens") or [])
                ],
                "max_new_tokens_source": "Q2 fallback; official YAML omits it",
            }
        )
        samples.append(
            EvalSample(
                dataset="textvqa",
                sample_id=str(row["question_id"]),
                row_index=int(row_index),
                image=_decode_image(row["image"]) if load_images else None,
                raw_question=raw_question,
                # This exact capitalize() call mirrors textvqa_doc_to_text.
                context=raw_question.capitalize() + SINGLE_PHRASE_SUFFIX,
                references=_references(row.get("answers")),
                max_new_tokens=MAX_NEW_TOKENS["textvqa"],
                metadata=metadata,
            )
        )
    return samples


def _load_docvqa(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    indices, rows = _selected_rows(
        root,
        "docvqa",
        n,
        seed,
        ("questionId", "question", "answers", "question_types", "docId"),
        load_images=load_images,
    )
    samples = []
    for row_index, row in zip(indices, rows):
        raw_question = str(row["question"])
        metadata = _base_metadata(
            "docvqa",
            task_type="vqa",
            source_type="vqa",
            seed=seed,
            n=n,
        )
        metadata.update(
            {
                "doc_id": int(row["docId"]),
                "question_types": [
                    str(value) for value in (row.get("question_types") or [])
                ],
            }
        )
        samples.append(
            EvalSample(
                dataset="docvqa",
                sample_id=str(row["questionId"]),
                row_index=int(row_index),
                image=_decode_image(row["image"]) if load_images else None,
                raw_question=raw_question,
                context=raw_question + SINGLE_PHRASE_SUFFIX,
                references=_references(row.get("answers")),
                max_new_tokens=MAX_NEW_TOKENS["docvqa"],
                metadata=metadata,
            )
        )
    return samples


def _load_chartqa(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    indices, rows = _selected_rows(
        root,
        "chartqa",
        n,
        seed,
        ("type", "question", "answer"),
        load_images=load_images,
    )
    samples = []
    for row_index, row in zip(indices, rows):
        raw_question = str(row["question"])
        source_type = str(row.get("type") or "chartqa")
        metadata = _base_metadata(
            "chartqa",
            task_type="vqa",
            source_type=source_type,
            seed=seed,
            n=n,
        )
        samples.append(
            EvalSample(
                dataset="chartqa",
                sample_id=f"chartqa_{row_index}",
                row_index=int(row_index),
                image=_decode_image(row["image"]) if load_images else None,
                raw_question=raw_question,
                context=raw_question + CHARTQA_SUFFIX,
                references=_references(row.get("answer")),
                max_new_tokens=MAX_NEW_TOKENS["chartqa"],
                metadata=metadata,
            )
        )
    return samples


def _load_coco_caption(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    indices, rows = _selected_rows(
        root,
        "coco_caption",
        n,
        seed,
        ("question_id", "question", "answer", "id", "file_name"),
        load_images=load_images,
    )
    samples = []
    for row_index, row in zip(indices, rows):
        metadata = _base_metadata(
            "coco_caption",
            task_type="caption",
            source_type="caption",
            seed=seed,
            n=n,
        )
        metadata.update(
            {
                "image_id": int(row["id"]),
                "file_name": str(row["file_name"]),
            }
        )
        samples.append(
            EvalSample(
                dataset="coco_caption",
                sample_id=str(row["question_id"]),
                row_index=int(row_index),
                image=_decode_image(row["image"]) if load_images else None,
                raw_question=str(row.get("question") or ""),
                context=CAPTION_CONTEXT,
                references=_references(row.get("answer")),
                max_new_tokens=MAX_NEW_TOKENS["coco_caption"],
                metadata=metadata,
            )
        )
    return samples


def _load_nocaps(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    indices, rows = _selected_rows(
        root,
        "nocaps",
        n,
        seed,
        (
            "image_id",
            "image_file_name",
            "image_open_images_id",
            "annotations_captions",
        ),
        load_images=load_images,
    )
    samples = []
    for row_index, row in zip(indices, rows):
        metadata = _base_metadata(
            "nocaps",
            task_type="caption",
            source_type="caption",
            seed=seed,
            n=n,
        )
        metadata.update(
            {
                "image_id": int(row["image_id"]),
                "file_name": str(row["image_file_name"]),
                "open_images_id": str(row["image_open_images_id"]),
            }
        )
        samples.append(
            EvalSample(
                dataset="nocaps",
                sample_id=str(row["image_id"]),
                row_index=int(row_index),
                image=_decode_image(row["image"]) if load_images else None,
                raw_question="",
                context=CAPTION_CONTEXT,
                references=_references(row.get("annotations_captions")),
                max_new_tokens=MAX_NEW_TOKENS["nocaps"],
                metadata=metadata,
            )
        )
    return samples


def _load_textcaps(
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool,
) -> list[EvalSample]:
    indices, rows = _selected_rows(
        root,
        "textcaps",
        n,
        seed,
        (
            "question_id",
            "question",
            "image_id",
            "image_name",
            "caption_str",
            "reference_strs",
        ),
        load_images=load_images,
    )
    samples = []
    for row_index, row in zip(indices, rows):
        metadata = _base_metadata(
            "textcaps",
            task_type="caption",
            source_type="caption",
            seed=seed,
            n=n,
        )
        metadata.update(
            {
                "image_id": str(row["image_id"]),
                "image_name": str(row["image_name"]),
            }
        )
        # The local paper YAML explicitly overrides doc_to_target to
        # caption_str.  reference_strs is retained only as a fallback.
        refs = row.get("caption_str") or row.get("reference_strs") or []
        samples.append(
            EvalSample(
                dataset="textcaps",
                sample_id=str(row["question_id"]),
                row_index=int(row_index),
                image=_decode_image(row["image"]) if load_images else None,
                raw_question=str(row.get("question") or ""),
                context=CAPTION_CONTEXT,
                references=_references(refs),
                max_new_tokens=MAX_NEW_TOKENS["textcaps"],
                metadata=metadata,
            )
        )
    return samples


_LOADERS = {
    "gqa": _load_gqa,
    "textvqa": _load_textvqa,
    "docvqa": _load_docvqa,
    "chartqa": _load_chartqa,
    "coco_caption": _load_coco_caption,
    "nocaps": _load_nocaps,
    "textcaps": _load_textcaps,
}


def load_samples(
    name: str,
    root: Path,
    n: int,
    seed: int,
    *,
    load_images: bool = True,
) -> list[EvalSample]:
    """Load ``n`` deterministic samples from one local evaluation dataset.

    The first four positional arguments form the integration API.  The
    keyword-only ``load_images=False`` mode is intended solely for manifest
    validation; ordinary callers receive RGB PIL images.
    """

    dataset = _canonical_name(name)
    return _LOADERS[dataset](
        Path(root),
        int(n),
        int(seed),
        load_images=bool(load_images),
    )


def load_eval700(
    root: Path = DEFAULT_ROOT,
    *,
    n: int = 100,
    seed: int = 42,
    load_images: bool = True,
    datasets: Sequence[str] = DATASETS,
) -> list[EvalSample]:
    """Load and concatenate the seven deterministic subsets in paper order."""

    return [
        sample
        for dataset in datasets
        for sample in load_samples(
            dataset,
            root,
            n,
            seed,
            load_images=load_images,
        )
    ]


def _gqa_normalize(value: str) -> str:
    """HF exact-match normalization for ignore_case+ignore_punctuation."""

    table = str.maketrans("", "", string.punctuation)
    return str(value).lower().translate(table)


@lru_cache(maxsize=1)
def _textvqa_answer_processor() -> Any:
    # Use the exact processor shipped with the local official TextVQA task.
    from lmms_eval.tasks._task_utils.vqa_eval_metric import (
        EvalAIAnswerProcessor,
    )

    return EvalAIAnswerProcessor()


def gqa_exact_score(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = _gqa_normalize(prediction)
    return float(
        any(
            normalized_prediction == _gqa_normalize(reference)
            for reference in references
        )
    )


def textvqa_soft_score(prediction: str, references: Sequence[str]) -> float:
    """Official leave-one-annotator-out TextVQA soft accuracy."""

    if not references:
        return 0.0
    processor = _textvqa_answer_processor()
    predicted = processor(str(prediction))
    ground_truth = [processor(str(reference)) for reference in references]
    per_annotator = []
    for index in range(len(ground_truth)):
        other_answers = (
            ground_truth[:index] + ground_truth[index + 1 :]
        )
        matches = sum(answer == predicted for answer in other_answers)
        per_annotator.append(min(1.0, float(matches) / 3.0))
    return float(statistics.mean(per_annotator))


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_char in enumerate(right, start=1):
        current = [row]
        for column, left_char in enumerate(left, start=1):
            substitution = previous[column - 1] + (
                left_char != right_char
            )
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def docvqa_anls_score(
    prediction: str,
    references: Sequence[str],
    *,
    threshold: float = 0.5,
) -> float:
    """Official continuous max-ANLS, thresholded below 0.5."""

    if not references:
        return 0.0
    raw_prediction = str(prediction)
    normalized_prediction = " ".join(
        raw_prediction.strip().lower().split()
    )
    similarities = []
    for reference in references:
        raw_reference = str(reference)
        normalized_reference = " ".join(
            raw_reference.strip().lower().split()
        )
        distance = _levenshtein_distance(
            normalized_reference,
            normalized_prediction,
        )
        # This denominator mirrors lmms_eval.api.metrics.anls, including its
        # use of the original strings rather than normalized strings.
        length = max(len(raw_reference.upper()), len(raw_prediction.upper()))
        normalized_distance = (
            0.0 if length == 0 else float(distance) / float(length)
        )
        similarities.append(1.0 - normalized_distance)
    score = max(similarities)
    return float(score if score >= threshold else 0.0)


def chartqa_relaxed_score(
    prediction: str,
    references: Sequence[str],
    *,
    max_relative_change: float = 0.05,
) -> float:
    """ChartQA relaxed accuracy from the installed official task."""

    if not references:
        return 0.0
    target = str(references[0])
    prediction = str(prediction)

    def to_float(text: str) -> float | None:
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None

    prediction_float = to_float(prediction)
    target_float = to_float(target)
    if prediction_float is not None and target_float:
        relative_change = (
            abs(prediction_float - target_float) / abs(target_float)
        )
        return float(relative_change <= max_relative_change)
    return float(prediction.lower() == target.lower())


def score_prediction(sample: EvalSample, pred: str) -> float:
    """Score one VQA prediction using the dataset's paper metric.

    Caption metrics are corpus-level; pass caption samples and predictions to
    :func:`caption_corpus_rouge_l` instead.
    """

    dataset = _canonical_name(sample.dataset)
    if dataset == "gqa":
        return gqa_exact_score(pred, sample.references)
    if dataset == "textvqa":
        return textvqa_soft_score(pred, sample.references)
    if dataset == "docvqa":
        return docvqa_anls_score(pred, sample.references)
    if dataset == "chartqa":
        return chartqa_relaxed_score(pred, sample.references)
    raise ValueError(
        f"{dataset} is a caption dataset; use caption_corpus_rouge_l"
    )


_CAPTION_PUNCTUATION = {
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
}


@lru_cache(maxsize=1)
def _caption_tokenizer() -> Any:
    from nltk.tokenize import TreebankWordTokenizer

    return TreebankWordTokenizer()


def _tokenize_caption(caption: str) -> str:
    tokens = _caption_tokenizer().tokenize(
        str(caption).replace("\n", " ").lower()
    )
    return " ".join(
        token for token in tokens if token not in _CAPTION_PUNCTUATION
    )


def caption_results_rouge_l(
    results: Iterable[Mapping[str, Any]],
) -> float:
    """ROUGE-L for ``{"answer": refs, "pred": text}`` result dictionaries.

    Tokenization and scoring are byte-for-byte equivalent in behavior to the
    local paper helper ``tasks/caption_rouge_utils.py``.
    """

    materialized = list(results)
    if not materialized:
        return 0.0
    from pycocoevalcap.rouge.rouge import Rouge

    ground_truths = {}
    predictions = {}
    for index, result in enumerate(materialized):
        ground_truths[index] = [
            _tokenize_caption(answer)
            for answer in result["answer"]
        ]
        predictions[index] = [_tokenize_caption(result["pred"])]
    score, _ = Rouge().compute_score(ground_truths, predictions)
    return float(score)


def caption_corpus_rouge_l(
    samples: Sequence[EvalSample],
    predictions: Sequence[str] | Mapping[str, str],
) -> float:
    """Compute the local-paper corpus ROUGE-L for caption samples."""

    if isinstance(predictions, Mapping):
        prediction_list = [
            str(predictions[sample.sample_id]) for sample in samples
        ]
    else:
        prediction_list = [str(value) for value in predictions]
        if len(prediction_list) != len(samples):
            raise ValueError(
                f"Got {len(prediction_list)} predictions for "
                f"{len(samples)} samples"
            )
    for sample in samples:
        if sample.metadata.get("task_type") != "caption":
            raise ValueError(
                f"{sample.dataset}/{sample.sample_id} is not a caption sample"
            )
    return caption_results_rouge_l(
        {
            "answer": list(sample.references),
            "pred": prediction,
        }
        for sample, prediction in zip(samples, prediction_list)
    )


def sample_manifest_record(sample: EvalSample) -> dict[str, Any]:
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "row_index": int(sample.row_index),
        "raw_question": sample.raw_question,
        "question_span": sample.question_span,
        "context": sample.context,
        "references": list(sample.references),
        "max_new_tokens": int(sample.max_new_tokens),
        "metadata": sample.metadata,
    }


def manifest_payload(samples: Sequence[EvalSample]) -> dict[str, Any]:
    records = [sample_manifest_record(sample) for sample in samples]
    counts = Counter(sample.dataset for sample in samples)
    return {
        "schema_version": 1,
        "sample_count": len(samples),
        "datasets": {
            dataset: counts.get(dataset, 0)
            for dataset in DATASETS
            if counts.get(dataset, 0)
        },
        "samples": records,
    }


def write_manifest(samples: Sequence[EvalSample], path: Path) -> None:
    """Atomically write a deterministic image-free JSON manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest_payload(samples),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    temporary.replace(path)


def manifest_sha256(samples: Sequence[EvalSample]) -> str:
    encoded = json.dumps(
        manifest_payload(samples),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_reference_ids(
    samples: Sequence[EvalSample],
    reference_path: Path,
) -> dict[str, int]:
    reference = json.loads(Path(reference_path).read_text())
    expected: dict[str, set[str]] = {
        dataset: set() for dataset in DATASETS
    }
    historical_to_canonical = {
        historical: canonical
        for canonical, historical in HISTORICAL_DATASET_NAMES.items()
    }
    for row in reference.get("samples", []):
        historical = str(row["dataset"])
        if historical in historical_to_canonical:
            expected[historical_to_canonical[historical]].add(
                str(row["sample_id"])
            )

    actual: dict[str, set[str]] = {
        dataset: {
            sample.sample_id
            for sample in samples
            if sample.dataset == dataset
        }
        for dataset in DATASETS
    }
    checked = {}
    for dataset in DATASETS:
        if not actual[dataset]:
            continue
        if actual[dataset] != expected[dataset]:
            missing = sorted(expected[dataset] - actual[dataset])[:5]
            extra = sorted(actual[dataset] - expected[dataset])[:5]
            raise AssertionError(
                f"{dataset}: historical ID mismatch; "
                f"missing={missing}, extra={extra}"
            )
        checked[dataset] = len(actual[dataset])
    return checked


def _scorer_self_test() -> None:
    assert gqa_exact_score("Red!", ("red",)) == 1.0
    assert gqa_exact_score("blue", ("red",)) == 0.0
    assert textvqa_soft_score("two", ("two",) * 10) == 1.0
    assert textvqa_soft_score("three", ("two",) * 10) == 0.0
    assert abs(docvqa_anls_score("hallo", ("hello",)) - 0.8) < 1e-12
    assert docvqa_anls_score("zzzzz", ("hello",)) == 0.0
    assert chartqa_relaxed_score("104", ("100",)) == 1.0
    assert chartqa_relaxed_score("106", ("100",)) == 0.0
    assert chartqa_relaxed_score("YES", ("yes",)) == 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/check the deterministic eval-700 manifest only."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-mismatch",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Historical JSON whose ID sets are checked when it exists.",
    )
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Explicit marker; the CLI never decodes images in any mode.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    del args.manifest_only
    canonical_datasets = tuple(
        _canonical_name(dataset) for dataset in args.datasets
    )
    if len(set(canonical_datasets)) != len(canonical_datasets):
        raise ValueError("Duplicate datasets after alias normalization")

    samples = load_eval700(
        args.root,
        n=args.n,
        seed=args.seed,
        load_images=False,
        datasets=canonical_datasets,
    )
    counts = Counter(sample.dataset for sample in samples)
    for dataset in canonical_datasets:
        expected = min(args.n, counts[dataset])
        if counts[dataset] != expected or counts[dataset] != args.n:
            raise AssertionError(
                f"{dataset}: expected {args.n}, got {counts[dataset]}"
            )
        ids = [
            sample.sample_id
            for sample in samples
            if sample.dataset == dataset
        ]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{dataset}: duplicate sample IDs")

    reference_match: dict[str, int] = {}
    if (
        not args.skip_reference_check
        and args.reference_mismatch.exists()
        and args.n == 100
        and args.seed == 42
    ):
        reference_match = _check_reference_ids(
            samples,
            args.reference_mismatch,
        )

    if args.self_test:
        _scorer_self_test()
        for sample in samples:
            expected_context = {
                "gqa": sample.raw_question + SINGLE_PHRASE_SUFFIX,
                "textvqa": (
                    sample.raw_question.capitalize()
                    + SINGLE_PHRASE_SUFFIX
                ),
                "docvqa": sample.raw_question + SINGLE_PHRASE_SUFFIX,
                "chartqa": sample.raw_question + CHARTQA_SUFFIX,
                "coco_caption": CAPTION_CONTEXT,
                "nocaps": CAPTION_CONTEXT,
                "textcaps": CAPTION_CONTEXT,
            }[sample.dataset]
            assert sample.context == expected_context
            assert sample.max_new_tokens == MAX_NEW_TOKENS[sample.dataset]
            assert sample.references

    if args.output is not None:
        write_manifest(samples, args.output)

    print(
        json.dumps(
            {
                "mode": "manifest-only",
                "root": str(args.root.resolve()),
                "counts": dict(counts),
                "seed": args.seed,
                "n_per_dataset": args.n,
                "manifest_sha256": manifest_sha256(samples),
                "historical_id_set_match": reference_match,
                "output": str(args.output.resolve())
                if args.output is not None
                else None,
                "self_test": bool(args.self_test),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CAPTION_CONTEXT",
    "DATASETS",
    "DEFAULT_ROOT",
    "EvalSample",
    "HISTORICAL_DATASET_NAMES",
    "MAX_NEW_TOKENS",
    "caption_corpus_rouge_l",
    "caption_results_rouge_l",
    "chartqa_relaxed_score",
    "docvqa_anls_score",
    "gqa_exact_score",
    "load_eval700",
    "load_samples",
    "manifest_payload",
    "manifest_sha256",
    "sample_manifest_record",
    "score_prediction",
    "textvqa_soft_score",
    "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
