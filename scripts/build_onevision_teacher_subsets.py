#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the 600-sample seed-42 training subsets consumed by the original
LLaVA-OneVision teacher collector.

`experiments/EXP-20260503-026-onevision-original-teacher-train/collect_original_onevision_teacher.py`
reads three files that are not tracked in the repo:

  data/gqa/train/subset_600_seed42.json
  artifacts/original_llava_teacher/subsets/textvqa_600_seed42_ids.json
  artifacts/original_llava_teacher/subsets/scienceqa_600_seed42_ids.json

This script regenerates them from the raw datasets under `--data-root`
(default `/workspace/hd/data`). Selection is `random.Random(seed).sample(...)`
over the id list sorted deterministically, so re-running reproduces the same
600 ids per dataset.

It also materializes a normalized ScienceQA view, because the raw ScienceQA
release keys `problems.json` by bare pid and stores images under
`image/<split>/<pid>/image.png`, while the collector expects `train_<pid>` keys
and `images/<split>/train_<pid>/image.png`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {path}")


# ── TextVQA ────────────────────────────────────────────────────────────────
def build_textvqa(data_root: Path, out_ids: Path, n: int, seed: int) -> None:
    data_json = data_root / "textvqa" / "train" / "data.json"
    records = json.loads(data_json.read_text())
    usable: list[str] = []
    for rec in records:
        rel = str(rec.get("image_path", ""))
        if not rel:
            continue
        if not (data_root / rel).exists():
            continue
        usable.append(str(rec.get("question_id", rec.get("id", ""))))
    usable = sorted(set(usable), key=lambda x: (len(x), x))
    if len(usable) < n:
        raise RuntimeError(f"textvqa: only {len(usable)} usable samples, need {n}")
    picked = random.Random(seed).sample(usable, n)
    _write_json(out_ids, {"dataset": "textvqa", "seed": seed, "n": n, "sample_ids": picked})


# ── ScienceQA ──────────────────────────────────────────────────────────────
def build_scienceqa(
    data_root: Path, norm_root: Path, out_ids: Path, n: int, seed: int
) -> None:
    src = data_root / "scienceqa"
    problems = json.loads((src / "problems.json").read_text())
    splits = json.loads((src / "pid_splits.json").read_text())

    train_pids = [str(p) for p in splits["train"]]
    usable: list[str] = []
    for pid in train_pids:
        prob = problems.get(pid)
        if not prob:
            continue
        image_name = prob.get("image")
        if not image_name:
            continue
        if not (src / "image" / "train" / pid / image_name).exists():
            continue
        usable.append(pid)
    usable = sorted(usable, key=lambda x: (len(x), x))
    if len(usable) < n:
        raise RuntimeError(f"scienceqa: only {len(usable)} usable samples, need {n}")
    picked = random.Random(seed).sample(usable, n)

    # Normalized view: `train_<pid>` keys + images/<split>/<qid>/<image>.
    norm_problems = {f"train_{pid}": problems[pid] for pid in usable}
    _write_json(norm_root / "problems.json", norm_problems)

    images_train = norm_root / "images" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    linked = 0
    for pid in usable:
        link = images_train / f"train_{pid}"
        target = (src / "image" / "train" / pid).resolve()
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(target, target_is_directory=True)
        linked += 1
    print(f"[link] scienceqa images: {linked} new symlinks under {images_train}")

    _write_json(
        out_ids,
        {
            "dataset": "scienceqa",
            "seed": seed,
            "n": n,
            "sample_ids": [f"train_{pid}" for pid in picked],
        },
    )


# ── GQA ────────────────────────────────────────────────────────────────────
def build_gqa(data_root: Path, out_subset: Path, n: int, seed: int) -> None:
    gqa_dir = data_root / "gqa"
    questions_json = gqa_dir / f"train_balanced_questions_{n}.json"
    if not questions_json.exists():
        raise RuntimeError(
            f"{questions_json} not found — run scripts/download_gqa_train_sample.py first"
        )
    questions = json.loads(questions_json.read_text())
    rows = []
    for qid, rec in questions.items():
        image_path = rec.get("local_image_path")
        if not image_path:
            image_path = str(gqa_dir / "images" / f"{rec['imageId']}.jpg")
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            continue
        rows.append(
            {
                "sample_id": str(qid),
                "question_id": str(qid),
                "image_id": str(rec.get("imageId", "")),
                "question": str(rec.get("question", "")).strip(),
                "image_path": image_path,
                "answer": rec.get("answer"),
            }
        )
    rows.sort(key=lambda r: (len(r["sample_id"]), r["sample_id"]))
    if len(rows) < n:
        raise RuntimeError(f"gqa: only {len(rows)} usable samples, need {n}")
    picked = random.Random(seed).sample(rows, n)
    _write_json(out_subset, picked)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/workspace/hd/data")
    p.add_argument("--repo-root", default="/workspace/zap")
    p.add_argument("--scienceqa-norm-root", default="/workspace/hd/data/scienceqa_onevision")
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", nargs="+", default=["textvqa", "gqa", "scienceqa"])
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    subsets_dir = repo_root / "artifacts" / "original_llava_teacher" / "subsets"

    if "textvqa" in args.datasets:
        build_textvqa(
            data_root,
            subsets_dir / f"textvqa_{args.n_samples}_seed{args.seed}_ids.json",
            args.n_samples,
            args.seed,
        )
    if "scienceqa" in args.datasets:
        build_scienceqa(
            data_root,
            Path(args.scienceqa_norm_root).resolve(),
            subsets_dir / f"scienceqa_{args.n_samples}_seed{args.seed}_ids.json",
            args.n_samples,
            args.seed,
        )
    if "gqa" in args.datasets:
        build_gqa(
            data_root,
            data_root / "gqa" / "train" / f"subset_{args.n_samples}_seed{args.seed}.json",
            args.n_samples,
            args.seed,
        )


if __name__ == "__main__":
    main()
