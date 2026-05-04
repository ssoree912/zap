#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small GQA train-balanced sample with images.")
    parser.add_argument("--repo-id", default="lmms-lab/GQA")
    parser.add_argument("--output-dir", default="data/gqa")
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-shards", type=int, default=21)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def download_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


def copy_to_sources(src: Path, sources_dir: Path, name: str) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    dst = sources_dir / name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def image_id_set_from_parquet(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["id"])
    return {str(value) for value in table.column("id").to_pylist()}


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    record = dict(row)
    record["id"] = str(record["id"])
    record["imageId"] = str(record["imageId"])
    return record


def load_matching_questions(instructions_path: Path, image_ids: set[str]) -> list[dict[str, Any]]:
    table = pq.read_table(instructions_path)
    candidates: list[dict[str, Any]] = []
    for row in table.to_pylist():
        image_id = str(row.get("imageId", ""))
        question = str(row.get("question", "")).strip()
        if image_id in image_ids and question:
            candidates.append(row_to_record(row))
    return candidates


def write_images(image_shards: list[Path], needed_image_ids: set[str], images_dir: Path) -> dict[str, str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    remaining = set(needed_image_ids)
    for shard in image_shards:
        if not remaining:
            break
        table = pq.read_table(shard, columns=["id", "image"])
        for row in table.to_pylist():
            image_id = str(row["id"])
            if image_id not in remaining:
                continue
            image = row["image"] or {}
            data = image.get("bytes")
            if data is None:
                raise ValueError(f"Image {image_id} in {shard} has no embedded bytes")
            out_path = images_dir / f"{image_id}.jpg"
            out_path.write_bytes(data)
            written[image_id] = str(out_path)
            remaining.remove(image_id)
    if remaining:
        missing = ", ".join(sorted(remaining)[:10])
        raise RuntimeError(f"Missing {len(remaining)} requested images; first missing: {missing}")
    return written


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / ".hf_cache"
    sources_dir = output_dir / "source_parquets"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions_src = download_file(
        args.repo_id,
        "train_balanced_instructions/train-00000-of-00001.parquet",
        cache_dir,
    )
    instructions_path = copy_to_sources(
        instructions_src,
        sources_dir,
        "train_balanced_instructions-00000-of-00001.parquet",
    )

    image_shards: list[Path] = []
    available_image_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for shard_idx in range(args.max_image_shards):
        shard_name = f"train_balanced_images/train-{shard_idx:05d}-of-00021.parquet"
        shard_src = download_file(args.repo_id, shard_name, cache_dir)
        shard_path = copy_to_sources(
            shard_src,
            sources_dir,
            f"train_balanced_images-{shard_idx:05d}-of-00021.parquet",
        )
        image_shards.append(shard_path)
        available_image_ids.update(image_id_set_from_parquet(shard_path))

        candidates = load_matching_questions(instructions_path, available_image_ids)
        if len(candidates) >= args.sample_size:
            rng.shuffle(candidates)
            selected = candidates[: args.sample_size]
            break

    if len(selected) < args.sample_size:
        raise RuntimeError(f"Only found {len(selected)} candidates, need {args.sample_size}")

    needed_image_ids = {row["imageId"] for row in selected}
    image_paths = write_images(image_shards, needed_image_ids, images_dir)

    questions = {}
    jsonl_rows = []
    for row in selected:
        qid = row["id"]
        row["local_image_path"] = image_paths[row["imageId"]]
        questions[qid] = row
        jsonl_rows.append(row)

    questions_path = output_dir / f"train_balanced_questions_{args.sample_size}.json"
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    jsonl_path = output_dir / f"train_balanced_sample_{args.sample_size}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "repo_id": args.repo_id,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "questions_json": str(questions_path),
        "jsonl": str(jsonl_path),
        "images_dir": str(images_dir),
        "n_unique_images": len(needed_image_ids),
        "image_shards_used": [str(path) for path in image_shards],
    }
    metadata_path = output_dir / f"train_balanced_sample_{args.sample_size}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
