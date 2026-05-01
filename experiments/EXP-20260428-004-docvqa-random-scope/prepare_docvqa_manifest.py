#!/usr/bin/env python3
"""Prepare a flat 200-sample DocVQA manifest for eval_ppl.py/eval_rouge.py."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


IMAGE_PLACEHOLDER = re.compile(r"\{image#\d+\}")


def build_question(record: dict[str, Any]) -> str:
    task = record["task_instance"]
    question = IMAGE_PLACEHOLDER.sub("", str(task.get("context", ""))).strip()
    question = re.sub(r"[ \t]+", " ", question)
    question = re.sub(r"\n{3,}", "\n\n", question)
    choices = [str(x).strip() for x in task.get("choice_list", []) if str(x).strip()]
    if choices:
        question = question + "\nChoices: " + " | ".join(choices)
    return question.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/workspace/zap/data/MileBench/DocVQA/DocVQA.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.input)
    raw = json.loads(input_path.read_text())
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    samples: list[dict[str, Any]] = []
    for rec in records[: args.limit]:
        task = rec["task_instance"]
        combined = task.get("combined_1_images") or []
        images = combined or task.get("images_path") or []
        if not images:
            continue
        samples.append(
            {
                "id": str(rec.get("sample_id", len(samples))),
                "image": str(images[0]),
                "question": build_question(rec),
                "answer": str(rec.get("response", "")),
                "source_old_sample_id": rec.get("old_sample_id"),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    print(f"[manifest] wrote {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()

