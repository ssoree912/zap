# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Collect v2 Future teacher records for MMLongBench-Doc with LLaVA-1.5.

MMLongBench-Doc stores PDF documents plus QA rows. This script renders one page
per QA row to an image, then reuses `collect_future_teacher_v2.collect_one` so
the saved `.pt` schema matches the existing v2 teacher records.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, "/workspace/zap")
from collect_future_teacher_v2 import collect_one  # noqa: E402
from foresight.llava_extractor import configure_llava_processor  # noqa: E402


PROMPT_TEMPLATE = "USER: <image>\nQuestion: {question}\nASSISTANT:"


def safe_id(text: Any, max_len: int = 128) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return (value or "sample")[:max_len]


def first_page_index(evidence_pages: Any) -> int:
    """Return zero-based PDF page index from evidence_pages.

    Dataset evidence pages are one-based. Empty evidence means no grounded page,
    so we use page 1 as a deterministic fallback.
    """
    if isinstance(evidence_pages, np.ndarray):
        evidence_pages = evidence_pages.tolist()
    if isinstance(evidence_pages, (list, tuple)) and evidence_pages:
        try:
            return max(0, int(evidence_pages[0]) - 1)
        except (TypeError, ValueError):
            return 0
    return 0


def render_pdf_page(pdf_path: Path, page_index: int, image_path: Path, zoom: float) -> Path:
    if image_path.exists():
        return image_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        if len(doc) == 0:
            raise ValueError(f"empty PDF: {pdf_path}")
        page_index = min(max(0, page_index), len(doc) - 1)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(image_path)
    return image_path


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    df = pd.read_parquet(args.parquet)
    rows = df.to_dict("records")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_samples]

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        doc_id = str(row["doc_id"])
        page_idx = first_page_index(row.get("evidence_pages"))
        sid = f"mmlongdoc_{idx:04d}_{safe_id(Path(doc_id).stem, 64)}_p{page_idx + 1}"
        question = str(row["question"]).strip()
        out.append(
            {
                "sample_id": sid,
                "doc_id": doc_id,
                "page_index": page_idx,
                "question": question,
                "answer": row.get("answer"),
                "evidence_pages": row.get("evidence_pages"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/zap/data/MMLongBench-Doc"))
    parser.add_argument("--parquet", type=Path, default=Path("/workspace/zap/data/MMLongBench-Doc/data/train-00000-of-00001.parquet"))
    parser.add_argument("--render-root", type=Path, default=Path("/workspace/zap/data/MMLongBench-Doc/rendered_pages_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/zap/data/teacher_v2"))
    parser.add_argument("--dataset-name", default="mmlongbench_doc")
    parser.add_argument("--model", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--trajectory-m", type=int, default=1)
    parser.add_argument("--trajectory-temperature", type=float, default=0.7)
    parser.add_argument("--trajectory-top-p", type=float, default=0.9)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.output_root / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[load] {args.model} dtype=bf16 attn=eager device={device}", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor = configure_llava_processor(processor, model.config)
    print(f"[info] num_hidden_layers={model.config.text_config.num_hidden_layers}", flush=True)

    rows = load_rows(args)
    print(f"[info] loaded {len(rows)} MMLongBench-Doc rows", flush=True)

    saved = 0
    skipped: list[tuple[str, str]] = []
    t_list: list[int] = []
    t0 = time.time()

    for idx, row in enumerate(rows):
        sid = row["sample_id"]
        out_path = out_dir / f"{safe_id(sid)}.pt"
        if out_path.exists():
            saved += 1
            continue
        try:
            pdf_path = args.data_root / "documents" / row["doc_id"]
            page_image = args.render_root / safe_id(row["doc_id"], 96) / f"page_{row['page_index'] + 1:04d}.png"
            render_pdf_page(pdf_path, row["page_index"], page_image, args.zoom)
            image = Image.open(page_image).convert("RGB")
            prompt = PROMPT_TEMPLATE.format(question=row["question"])
            rec = collect_one(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                trajectory_m=args.trajectory_m,
                trajectory_temperature=args.trajectory_temperature,
                trajectory_top_p=args.trajectory_top_p,
            )
            rec.update(
                sample_id=sid,
                dataset=args.dataset_name,
                model="llava-1.5-7b-hf",
                prompt_text=prompt,
                image_path=str(page_image),
                source_pdf=str(pdf_path),
                source_doc_id=row["doc_id"],
                source_page_index=int(row["page_index"]),
                evidence_pages=row.get("evidence_pages"),
                answer=row.get("answer"),
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            torch.save(rec, out_path)
            saved += 1
            t_list.append(int(rec["T"]))
            if idx == 0:
                print(
                    f"[sanity] sid={sid} L={rec['teacher_raw'].shape[0]} "
                    f"N_I={rec['teacher_raw'].shape[1]} T={rec['T']} "
                    f"prompt_len_mm={rec['prompt_len_mm']} "
                    f"|img|={rec['image_token_indices'].numel()} "
                    f"|q|={rec['question_token_indices'].numel()} image={page_image}",
                    flush=True,
                )
        except Exception as exc:  # keep long collection moving
            skipped.append((sid, repr(exc)))
            print(f"[skip] {sid}: {exc}", flush=True)
            continue

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"[progress] {idx + 1}/{len(rows)} | rate={(idx + 1) / max(elapsed, 1e-6):.3f}/s "
                f"| elapsed={elapsed:.1f}s | T_mean={np.mean(t_list):.2f} "
                f"| saved={saved} skipped={len(skipped)}",
                flush=True,
            )
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    summary = dict(
        dataset=args.dataset_name,
        n_requested=args.n_samples,
        n_loaded=len(rows),
        n_saved=saved,
        n_skipped=len(skipped),
        skipped=skipped,
        t_mean=float(np.mean(t_list)) if t_list else 0.0,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        elapsed_seconds=elapsed,
        trajectory_m=args.trajectory_m,
        trajectory_temperature=args.trajectory_temperature,
        trajectory_top_p=args.trajectory_top_p,
        render_root=str(args.render_root),
        output_dir=str(out_dir),
        page_policy="first evidence_pages entry, fallback page 1",
    )
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {summary}", flush=True)
    print(f"[save] {out_dir / '_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
