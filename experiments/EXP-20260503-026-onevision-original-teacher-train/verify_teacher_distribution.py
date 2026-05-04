#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify OneVision teacher label distributions."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch


def summarize_dataset(root: Path, dataset: str) -> dict[str, float | int | str]:
    files = sorted((root / dataset).glob("*.pt"))
    if not files:
        return {"dataset": dataset, "n": 0}
    t_values: list[int] = []
    n_img_values: list[int] = []
    hit_max = 0
    bad_rows = 0
    nonfinite = 0
    bad_layer_counts: collections.Counter[int] = collections.Counter()
    for path in files:
        rec = torch.load(path, weights_only=False, map_location="cpu")
        t = int(rec.get("T", 0))
        t_values.append(t)
        n_img_values.append(int(rec.get("n_img", 0)))
        if t == int(rec.get("max_new_tokens", -1)):
            hit_max += 1
        bad_rows += int(rec.get("bad_teacher_rows", 0))
        raw = rec.get("teacher_raw")
        if isinstance(raw, torch.Tensor):
            bad_layers = (raw.float().sum(dim=-1) <= 1e-12).nonzero(as_tuple=False).flatten()
            bad_layer_counts.update(int(x) for x in bad_layers.tolist())
        teacher = rec.get("teacher_norm")
        if not isinstance(teacher, torch.Tensor) or not torch.isfinite(teacher).all():
            nonfinite += 1
    return {
        "dataset": dataset,
        "n": len(files),
        "t_min": min(t_values),
        "t_max": max(t_values),
        "t_mean": sum(t_values) / len(t_values),
        "hit_max_new_tokens": hit_max,
        "n_img_min": min(n_img_values),
        "n_img_max": max(n_img_values),
        "n_img_mean": sum(n_img_values) / len(n_img_values),
        "bad_teacher_rows": bad_rows,
        "bad_layer_counts": {str(k): v for k, v in sorted(bad_layer_counts.items())},
        "nonfinite_files": nonfinite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", default="/workspace/zap/artifacts/original_onevision_teacher/future_decode_qwen2_7b")
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "gqa", "scienceqa"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.teacher_root)
    summaries = [summarize_dataset(root, dataset) for dataset in args.datasets]
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
