#!/usr/bin/env python3
"""Fail-closed validation for the paired P+A and Q+A teacher roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

EXPECTED_DATASETS = ("gqa", "textvqa", "scienceqa")
EXPECTED_SHAPE = (32, 576)
EXPECTED_SPLIT_HASHES = {
    "all": "2ad770606721e126de78129e6d4d200a3badbdae8482a74befd43847546fbf7f",
    "train": "75922bff1346cfd0171231d20786768bbb08140a6031c820ca775adfd715f66e",
    "val": "3f62d3b757012ea2abc32fe37c6be0094a475e8072e4acd501fb21af108211f4",
}


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_files(root: Path) -> dict[str, list[Path]]:
    result = {}
    for dataset in EXPECTED_DATASETS:
        paths = sorted((root / dataset).glob("*.pt"))
        if len(paths) != 600:
            raise ValueError(f"{root}/{dataset}: expected 600 records, found {len(paths)}")
        result[dataset] = paths
    return result


def _ordered_hash(paths: list[Path]) -> str:
    payload = "\n".join(str(path) for path in paths).encode()
    return hashlib.sha256(payload).hexdigest()


def _relative_order_hash(paths: list[Path]) -> str:
    payload = "\n".join(f"{path.parent.name}/{path.name}" for path in paths).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pa-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = _list_files(args.source_root)
    pa = _list_files(args.pa_root)
    qa = _list_files(args.qa_root)
    ordered_relative: list[Path] = []
    max_mix_error = 0.0
    target_absolute_difference = 0.0
    target_elements = 0

    for dataset in EXPECTED_DATASETS:
        source_names = [path.name for path in source[dataset]]
        if source_names != [path.name for path in pa[dataset]]:
            raise ValueError(f"{dataset}: P+A file identity mismatch")
        if source_names != [path.name for path in qa[dataset]]:
            raise ValueError(f"{dataset}: Q+A file identity mismatch")
        ordered_relative.extend(source[dataset])
        for source_path, pa_path, qa_path in zip(
            source[dataset],
            pa[dataset],
            qa[dataset],
            strict=True,
        ):
            s = torch.load(source_path, map_location="cpu", weights_only=False)
            p = torch.load(pa_path, map_location="cpu", weights_only=False)
            q = torch.load(qa_path, map_location="cpu", weights_only=False)
            identity_keys = (
                "sample_id",
                "dataset",
                "prompt_text",
                "image_path",
                "prompt_len_mm",
                "T",
                "n_img",
                "max_new_tokens",
            )
            for key in identity_keys:
                if s[key] != p[key] or s[key] != q[key]:
                    raise ValueError(f"{source_path}: identity drift at {key}")
            for key in ("image_token_indices", "question_token_indices"):
                if not torch.equal(s[key], p[key]) or not torch.equal(s[key], q[key]):
                    raise ValueError(f"{source_path}: token-index drift at {key}")
            answer_sha = _tensor_sha256(s["teacher_answer_norm"])
            if (
                answer_sha != _tensor_sha256(p["teacher_answer_norm"])
                or answer_sha != _tensor_sha256(q["teacher_answer_norm"])
            ):
                raise ValueError(f"{source_path}: answer component is not byte-identical")
            if p.get("paired_answer_norm_sha256") != answer_sha:
                raise ValueError(f"{pa_path}: bad paired answer SHA")
            if q.get("paired_answer_norm_sha256") != answer_sha:
                raise ValueError(f"{qa_path}: bad paired answer SHA")
            source_sha = _file_sha256(source_path)
            if p.get("paired_source_file_sha256") != source_sha:
                raise ValueError(f"{pa_path}: bad paired source-file SHA")
            if q.get("paired_source_file_sha256") != source_sha:
                raise ValueError(f"{qa_path}: bad paired source-file SHA")
            expected_prefill_indices = torch.arange(
                int(s["prompt_len_mm"]),
                dtype=torch.long,
            )
            if not torch.equal(
                p.get("prefill_query_indices"),
                expected_prefill_indices,
            ):
                raise ValueError(f"{pa_path}: all-prefill query range is incorrect")
            if p.get("paired_first_signal") != "all_prefill":
                raise ValueError(f"{pa_path}: wrong first-signal metadata")
            if q.get("paired_first_signal") != "question_prompt_tail":
                raise ValueError(f"{qa_path}: wrong first-signal metadata")
            if p.get("teacher_signal") != "paired_all_prefill_answer_normalized_mix":
                raise ValueError(f"{pa_path}: wrong teacher signal")
            if q.get("teacher_signal") != "paired_question_answer_normalized_mix":
                raise ValueError(f"{qa_path}: wrong teacher signal")
            expected_weights = (
                p.get("teacher_prefill_weight") == 0.5
                and p.get("teacher_answer_weight") == 0.5
                and q.get("teacher_question_weight") == 0.5
                and q.get("teacher_answer_weight") == 0.5
                and p.get("paired_first_weight") == 0.5
                and q.get("paired_first_weight") == 0.5
                and p.get("paired_answer_weight") == 0.5
                and q.get("paired_answer_weight") == 0.5
            )
            if not expected_weights:
                raise ValueError(f"{source_path}: paired 0.5/0.5 weight metadata drift")
            for record, key in (
                (p, "teacher_prefill_norm"),
                (q, "teacher_question_norm"),
            ):
                if tuple(record[key].shape) != EXPECTED_SHAPE:
                    raise ValueError(f"{record['sample_id']}: bad {key} shape")
                if not torch.isfinite(record[key].float()).all():
                    raise ValueError(f"{record['sample_id']}: non-finite {key}")
                if not torch.allclose(
                    record[key].float().sum(dim=-1),
                    torch.ones(EXPECTED_SHAPE[0]),
                    atol=2e-3,
                    rtol=0,
                ):
                    raise ValueError(f"{record['sample_id']}: {key} not normalized")
            expected_pa = _normalize(
                0.5 * _normalize(p["teacher_prefill_norm"])
                + 0.5 * _normalize(s["teacher_answer_norm"])
            )
            expected_qa = _normalize(
                0.5 * _normalize(q["teacher_question_norm"])
                + 0.5 * _normalize(s["teacher_answer_norm"])
            )
            for record, expected in ((p, expected_pa), (q, expected_qa)):
                if record.get("paired_component_l1_reapplied_in_float32") is not True:
                    raise ValueError(
                        f"{record['sample_id']}: float32 component L1 marker missing"
                    )
                target = record["teacher_norm"].float()
                error = float((target - expected).abs().max())
                max_mix_error = max(max_mix_error, error)
                if error > 3e-4:
                    raise ValueError(
                        f"{record['sample_id']}: target recomposition error={error}"
                    )
                if not torch.allclose(
                    target.sum(dim=-1),
                    torch.ones(EXPECTED_SHAPE[0]),
                    atol=2e-3,
                    rtol=0,
                ):
                    raise ValueError(f"{record['sample_id']}: target not normalized")
            target_absolute_difference += float(
                (p["teacher_norm"].float() - q["teacher_norm"].float()).abs().sum()
            )
            target_elements += p["teacher_norm"].numel()

    shuffled = list(ordered_relative)
    random.Random(0).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * 0.1)))
    train, val = shuffled[:-n_val], shuffled[-n_val:]
    split_hashes = {
        "all": _relative_order_hash(ordered_relative),
        "train": _relative_order_hash(train),
        "val": _relative_order_hash(val),
    }
    if split_hashes != EXPECTED_SPLIT_HASHES:
        raise ValueError(
            f"Trainer split hash drift: expected={EXPECTED_SPLIT_HASHES}, "
            f"actual={split_hashes}"
        )
    mean_target_absolute_difference = target_absolute_difference / target_elements
    if mean_target_absolute_difference <= 1e-7:
        raise ValueError("P+A and Q+A targets are unexpectedly identical")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(args.source_root.resolve()),
        "pa_root": str(args.pa_root.resolve()),
        "qa_root": str(args.qa_root.resolve()),
        "records_per_dataset": 600,
        "total_records": 1800,
        "answer_component_byte_identical": True,
        "paired_target_difference": "all-prefill rows versus post-image prompt-tail rows only",
        "answer_query_rows": "[y_1, ..., y_T]",
        "max_new_tokens": 64,
        "split_hashes": split_hashes,
        "max_fp16_recomposition_error": max_mix_error,
        "mean_absolute_pa_qa_target_difference": mean_target_absolute_difference,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
