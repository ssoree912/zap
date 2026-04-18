#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download derek-thomas/ScienceQA and convert it to the local manifest layout.

Output layout:
  {out_dir}/problems.json
  {out_dir}/pid_splits.json
  {out_dir}/images/{split}/{qid}/image.png

This matches the structure expected by build_scienceqa_manifest.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_image(image: Any, path: Path) -> bool:
    if image is None:
        return False
    if not isinstance(image, Image.Image):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return True


def _qid_for(split: str, index: int) -> str:
    return f"{split}_{index:06d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Destination directory, e.g. /workspace/zap/data/scienceqa",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    problems: dict[str, dict[str, Any]] = {}
    pid_splits: dict[str, list[str]] = {value: [] for value in split_map.values()}

    for hf_split, out_split in split_map.items():
        ds = load_dataset("derek-thomas/ScienceQA", split=hf_split)
        for index, sample in enumerate(tqdm(ds, desc=f"scienceqa/{out_split}")):
            qid = _qid_for(out_split, index)
            image_name = "image.png"
            image_path = out_dir / "images" / out_split / qid / image_name
            has_image = _save_image(sample.get("image"), image_path)

            problems[qid] = {
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "answer": sample.get("answer", -1),
                "hint": sample.get("hint", ""),
                "task": sample.get("task", ""),
                "grade": sample.get("grade", ""),
                "subject": sample.get("subject", ""),
                "topic": sample.get("topic", ""),
                "category": sample.get("category", ""),
                "skill": sample.get("skill", ""),
                "lecture": sample.get("lecture", ""),
                "solution": sample.get("solution", ""),
                "image": image_name if has_image else "",
            }
            pid_splits[out_split].append(qid)

        print(f"{out_split}: {len(pid_splits[out_split])} samples")

    _write_json(out_dir / "problems.json", problems)
    _write_json(out_dir / "pid_splits.json", pid_splits)
    print(f"Wrote ScienceQA to {out_dir}")


if __name__ == "__main__":
    main()
