#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_problem(problems: Dict[Any, Dict[str, Any]], qid: str) -> Dict[str, Any] | None:
    if qid in problems:
        return problems[qid]
    if str(qid) in problems:
        return problems[str(qid)]
    try:
        qid_int = int(qid)
    except ValueError:
        return None
    return problems.get(qid_int)


def build_scienceqa_prompt(problem: Dict[str, Any]) -> str:
    question = str(problem["question"]).strip()
    choices: List[str] = [str(choice).strip() for choice in problem["choices"]]
    hint = str(problem.get("hint", "") or "").strip()

    lines = ["USER: <image>"]
    if hint:
        lines.append(f"Context: {hint}")

    lines.append(f"Question: {question}")
    lines.append("Options:")

    for idx, choice in enumerate(choices):
        lines.append(f"{OPTION_LETTERS[idx]}. {choice}")

    lines.append("Select the best answer based on the image and text.")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _resolve_image_path(base_dir: Path, split: str, qid: str, image_name: str) -> Path:
    candidates = [
        base_dir / "image" / split / qid / image_name,
        base_dir / "images" / split / qid / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def iter_scienceqa_samples(
    base_dir: str | Path,
    split: str,
    closed_choice_only: bool = True,
    require_image: bool = True,
    limit: int | None = None,
) -> Iterator[Dict[str, Any]]:
    base = Path(base_dir).resolve()
    problems = load_json(base / "problems.json")
    pid_splits = load_json(base / "pid_splits.json")

    if split not in pid_splits:
        raise KeyError(f"Split `{split}` not found in pid_splits.json")

    n_yielded = 0
    for raw_qid in pid_splits[split]:
        qid = str(raw_qid)
        problem = _get_problem(problems, qid)
        if problem is None:
            continue

        image_name = problem.get("image")
        if require_image and (image_name in (None, "", "none")):
            continue

        if closed_choice_only and problem.get("task") != "closed choice":
            continue

        choices = problem.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            continue
        if len(choices) > len(OPTION_LETTERS):
            continue

        image_path = _resolve_image_path(base, split, qid, str(image_name))
        if require_image and not image_path.exists():
            continue

        prompt_text = build_scienceqa_prompt(problem)
        sample = {
            "sample_id": qid,
            "image_path": str(image_path),
            "prompt_text": prompt_text,
            "question": str(problem.get("question", "")),
            "hint": str(problem.get("hint", "") or ""),
            "choices": [str(choice) for choice in choices],
            "answer_idx": int(problem.get("answer", -1)),
            "split": split,
        }

        yield sample
        n_yielded += 1
        if limit is not None and n_yielded >= limit:
            break


def write_manifest(
    base_dir: str | Path,
    split: str,
    out_path: str | Path,
    closed_choice_only: bool = True,
    require_image: bool = True,
    limit: int | None = None,
) -> dict[str, int]:
    output = Path(out_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with output.open("w", encoding="utf-8") as f:
        for sample in iter_scienceqa_samples(
            base_dir=base_dir,
            split=split,
            closed_choice_only=closed_choice_only,
            require_image=require_image,
            limit=limit,
        ):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1

    return {"split": split, "written": n_written}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/workspace/hd/data/scienceqa")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--closed_choice_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = write_manifest(
        base_dir=args.base_dir,
        split=args.split,
        out_path=args.out_path,
        closed_choice_only=args.closed_choice_only,
        require_image=args.require_image,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
