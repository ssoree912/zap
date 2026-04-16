#!/usr/bin/env python3
"""Download TextVQA and NLVR2 datasets to disk.

TextVQA → {out_dir}/textvqa/{train,val}/
  images/   ← JPEG files named {question_id}.jpg
  data.json ← list of {question_id, question, answers, image_id}

NLVR2   → {out_dir}/nlvr2/{train,val,test}/
  images/   ← JPEG files named {identifier}.jpg  (2 per sample)
  data.json ← list of {identifier, sentence, label, image_paths:[...]}

Usage:
  conda run -n kv python scripts/download_textvqa_nlvr2.py \
      --out_dir /workspace/hd/data \
      --datasets textvqa nlvr2
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm


# ── helpers ────────────────────────────────────────────────────────────────────

def _save_image(img: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path, format="JPEG", quality=95)
    elif isinstance(img, bytes):
        Image.open(io.BytesIO(img)).convert("RGB").save(path, format="JPEG", quality=95)
    elif isinstance(img, dict) and "bytes" in img:
        Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(path, format="JPEG", quality=95)
    else:
        raise TypeError(f"Unknown image type: {type(img)}")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── TextVQA ────────────────────────────────────────────────────────────────────

def download_textvqa(out_dir: Path, splits: list[str] = ["train", "val"]) -> None:
    """Download TextVQA via HuggingFace datasets.

    Saves:
      {out_dir}/textvqa/{split}/images/{question_id}.jpg
      {out_dir}/textvqa/{split}/data.json
    """
    from datasets import load_dataset

    # HuggingFace TextVQA split names
    textvqa_split_map = {"train": "train", "val": "validation", "validation": "validation"}

    print("=== TextVQA ===")
    for split in splits:
        hf_split = textvqa_split_map.get(split, split)
        out_split = "val" if split == "validation" else split
        print(f"  Loading split: {hf_split} …")
        ds = load_dataset("textvqa", split=hf_split, trust_remote_code=True)

        split_dir = out_dir / "textvqa" / out_split
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for sample in tqdm(ds, desc=f"textvqa/{out_split}"):
            qid = str(sample["question_id"])
            img_path = img_dir / f"{qid}.jpg"
            if not img_path.exists():
                _save_image(sample["image"], img_path)
            records.append({
                "question_id": qid,
                "question": sample["question"],
                "answers": sample.get("answers", []),
                "image_id": sample.get("image_id", qid),
                "image_path": str(img_path.relative_to(out_dir)),
            })

        _write_json(split_dir / "data.json", records)
        print(f"  → {len(records)} samples, images: {img_dir}")



# ── NLVR2 ──────────────────────────────────────────────────────────────────────

def download_nlvr2(out_dir: Path, splits: list[str] = ["train", "validation"]) -> None:
    """Download NLVR2 via HuggingFace datasets.

    Saves:
      {out_dir}/nlvr2/{split}/images/{identifier}_0.jpg  (left image)
      {out_dir}/nlvr2/{split}/images/{identifier}_1.jpg  (right image)
      {out_dir}/nlvr2/{split}/data.json
    """
    from datasets import load_dataset

    print("=== NLVR2 ===")
    # HuggingFace split names
    hf_split_map = {"train": "train", "val": "validation", "validation": "validation"}

    for split in splits:
        hf_split = hf_split_map.get(split, split)
        print(f"  Loading split: {hf_split} …")
        # pingzhili/nlvr2 columns: identifier, sentence, label, image0, image1
        ds = load_dataset("pingzhili/nlvr2", split=hf_split, trust_remote_code=True)

        out_split = "val" if split == "validation" else split
        split_dir = out_dir / "nlvr2" / out_split
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i, sample in enumerate(tqdm(ds, desc=f"nlvr2/{out_split}")):
            identifier = sample.get("identifier", str(i))
            safe_id = str(identifier).replace("/", "_").replace("-", "_")

            img_paths = []
            for img_idx, img_key in enumerate(["image0", "image1"]):
                if img_key not in sample:
                    continue
                img_path = img_dir / f"{safe_id}_{img_idx}.jpg"
                if not img_path.exists():
                    try:
                        _save_image(sample[img_key], img_path)
                    except Exception as e:
                        print(f"    [warn] could not save {img_key} for {identifier}: {e}")
                        continue
                img_paths.append(str(img_path.relative_to(out_dir)))
                if len(img_paths) == 2:
                    break

            if len(img_paths) < 2:
                continue  # skip samples without 2 images

            records.append({
                "identifier": str(identifier),
                "sentence": sample["sentence"],
                "label": sample["label"],  # True / False or 0 / 1
                "image_paths": img_paths,
            })

        _write_json(split_dir / "data.json", records)
        print(f"  → {len(records)} samples, images: {img_dir}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Root output directory (e.g. /workspace/hd/data)")
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "nlvr2"],
                        choices=["textvqa", "nlvr2"],
                        help="Which datasets to download")
    parser.add_argument("--textvqa_splits", nargs="+", default=["train", "val"])
    parser.add_argument("--nlvr2_splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_dir}")

    if "textvqa" in args.datasets:
        download_textvqa(out_dir, splits=args.textvqa_splits)

    if "nlvr2" in args.datasets:
        download_nlvr2(out_dir, splits=args.nlvr2_splits)

    print("\nAll done.")


if __name__ == "__main__":
    main()
