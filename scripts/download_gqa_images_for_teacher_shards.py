#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize GQA images referenced by teacher shards.")
    parser.add_argument("--teacher-dir", default="artifacts/future_decode_llava15_7b/gqa")
    parser.add_argument("--output-dir", default="data/gqa")
    parser.add_argument("--repo-id", default="lmms-lab/GQA")
    parser.add_argument("--max-image-shards", type=int, default=21)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def image_ids_from_teacher(teacher_dir: Path) -> set[str]:
    image_ids: set[str] = set()
    for path in sorted(teacher_dir.glob("*.pt")):
        rec = torch.load(path, weights_only=False, map_location="cpu")
        image_ids.add(Path(str(rec["image_path"])).stem)
    if not image_ids:
        raise RuntimeError(f"No teacher records found under {teacher_dir}")
    return image_ids


def download_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


def cached_source_path(sources_dir: Path, shard_idx: int) -> Path:
    return sources_dir / f"train_balanced_images-{shard_idx:05d}-of-00021.parquet"


def ensure_source_shard(repo_id: str, cache_dir: Path, sources_dir: Path, shard_idx: int) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    dst = cached_source_path(sources_dir, shard_idx)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    filename = f"train_balanced_images/train-{shard_idx:05d}-of-00021.parquet"
    src = download_file(repo_id, filename, cache_dir)
    shutil.copy2(src, dst)
    return dst


def extract_images_from_shard(shard_path: Path, remaining: set[str], images_dir: Path) -> list[str]:
    if not remaining:
        return []
    written: list[str] = []
    table = pq.read_table(shard_path, columns=["id", "image"])
    for row in table.to_pylist():
        image_id = str(row["id"])
        if image_id not in remaining:
            continue
        image = row["image"] or {}
        data = image.get("bytes")
        if data is None:
            raise ValueError(f"Image {image_id} in {shard_path} has no embedded bytes")
        out_path = images_dir / f"{image_id}.jpg"
        out_path.write_bytes(data)
        remaining.remove(image_id)
        written.append(image_id)
    return written


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    teacher_dir = Path(args.teacher_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / ".hf_cache"
    sources_dir = output_dir / "source_parquets"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    needed = image_ids_from_teacher(teacher_dir)
    existing = {path.stem for path in images_dir.glob("*.jpg")}
    remaining = set(needed - existing)
    written_by_shard: dict[str, int] = {}

    for shard_idx in range(args.max_image_shards):
        if not remaining:
            break
        shard_path = ensure_source_shard(args.repo_id, cache_dir, sources_dir, shard_idx)
        written = extract_images_from_shard(shard_path, remaining, images_dir)
        if written:
            written_by_shard[f"{shard_idx:05d}"] = len(written)
            print(
                f"[shard {shard_idx:05d}] wrote={len(written)} remaining={len(remaining)}",
                flush=True,
            )

    summary = {
        "teacher_dir": str(teacher_dir),
        "images_dir": str(images_dir),
        "needed": len(needed),
        "already_present": len(existing & needed),
        "written": sum(written_by_shard.values()),
        "missing": len(remaining),
        "written_by_shard": written_by_shard,
        "first_missing": sorted(remaining)[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if remaining:
        raise RuntimeError(f"Missing {len(remaining)} teacher images after scanning shards")


if __name__ == "__main__":
    main()
