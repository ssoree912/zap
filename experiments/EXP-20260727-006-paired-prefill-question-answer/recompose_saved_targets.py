#!/usr/bin/env python3
"""Recompose both paired targets in float32 from their saved fp16 components."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

DATASETS = ("gqa", "textvqa", "scienceqa")


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _atomic_save(record: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(record, temporary)
    os.replace(temporary, path)


def _recompose(path: Path, first_key: str) -> None:
    record = torch.load(path, map_location="cpu", weights_only=False)
    first = _normalize(record[first_key])
    answer = _normalize(record["teacher_answer_norm"])
    target = _normalize(0.5 * first + 0.5 * answer)
    record["teacher_raw"] = target.to(torch.float16)
    record["teacher_norm"] = target.to(torch.float16)
    record["paired_component_l1_reapplied_in_float32"] = True
    record["paired_target_recomposition"] = (
        f"normalize(0.5*normalize({first_key})"
        "+0.5*normalize(teacher_answer_norm))"
    )
    _atomic_save(record, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pa-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for dataset in DATASETS:
        pa_paths = sorted((args.pa_root / dataset).glob("*.pt"))
        qa_paths = sorted((args.qa_root / dataset).glob("*.pt"))
        if len(pa_paths) != 600 or len(qa_paths) != 600:
            raise ValueError(
                f"{dataset}: expected 600/600 records, "
                f"found {len(pa_paths)}/{len(qa_paths)}"
            )
        if [path.name for path in pa_paths] != [path.name for path in qa_paths]:
            raise ValueError(f"{dataset}: paired filenames differ")
        for index, (pa_path, qa_path) in enumerate(
            zip(pa_paths, qa_paths, strict=True),
            1,
        ):
            _recompose(pa_path, "teacher_prefill_norm")
            _recompose(qa_path, "teacher_question_norm")
            if index % 100 == 0:
                print(f"[progress] {dataset} {index}/600", flush=True)
    print("[done] recomposed 1,800 P+A and 1,800 Q+A targets", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
