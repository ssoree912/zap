#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize a deterministic MM-Vet subset for decode benchmarking."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


PRE_PROMPT = (
    "First please perform reasoning, and think step by step to provide the best "
    "answer to the following question:\n\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        "lmms-lab/MMVet",
        split="test",
        cache_dir=str(args.cache_dir),
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.count]

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in indices:
        record = dataset[index]
        sample_id = str(record["question_id"])
        image_path = image_dir / f"{sample_id}.png"
        record["image"].convert("RGB").save(image_path)
        samples.append(
            {
                "dataset": "MM-Vet",
                "sample_id": sample_id,
                "question": PRE_PROMPT + str(record["question"]),
                "answer": record["answer"],
                "image_path": str(image_path.resolve()),
                "capability": record["capability"],
            }
        )

    manifest = {
        "dataset": "lmms-lab/MMVet",
        "split": "test",
        "seed": args.seed,
        "count": len(samples),
        "prompt": "official lmms-eval MM-Vet pre_prompt + dataset question",
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
