#!/usr/bin/env python3
"""Evaluate H2O / Q-ViK student presses on the *same* 100 samples used for the
mismatch analysis (textvqa_val, docvqa_val, gqa).

For each (method, keep_ratio) the model is run with the press attached during
prefill, generates an answer, and the answer is scored against the dataset's
ground truth using the standard metric:

    - textvqa: VQA score = avg over GTs of min(matches/3, 1)
    - docvqa:  ANLS = max over GTs of (1 - lev/max_len), thresholded at 0.5
    - gqa:     exact match (case/punctuation normalized)

Pairs naturally with `compute_mismatch.py` output to produce mismatch-vs-accuracy
scatter plots downstream.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# VFlowOpt LLaVA path
VFLOWOPT_LLAVA_ROOT = Path(os.environ.get(
    "VFLOWOPT_LLAVA_ROOT",
    "/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision",
))
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.image_token_press import (  # noqa: E402
    H2OImageOnlyPress,
    VisualUtilityStudentPress,
)
from datasets import load_from_disk  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

_PUNC_RE = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    return _PUNC_RE.sub("", s.lower()).strip()


def vqa_score(pred: str, gts: list[str]) -> float:
    p = _normalize(pred)
    matches = sum(1 for g in gts if _normalize(g) == p)
    return min(matches / 3.0, 1.0)


def _levenshtein(a: str, b: str) -> int:
    if not a: return len(b)
    if not b: return len(a)
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def anls_score(pred: str, gts: list[str], thresh: float = 0.5) -> float:
    pred_n = _normalize(pred)
    best = 0.0
    for g in gts:
        gn = _normalize(g)
        if not gn and not pred_n:
            sim = 1.0
        elif not gn or not pred_n:
            sim = 0.0
        else:
            sim = 1.0 - _levenshtein(pred_n, gn) / max(len(pred_n), len(gn))
        if sim > best:
            best = sim
    return float(best) if best >= thresh else 0.0


def exact_match(pred: str, gts: list[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(_normalize(g) == p for g in gts) else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev = tmp
    return dp[m]


def rouge_l_score(pred: str, gts: list[str]) -> float:
    """Max ROUGE-L F1 across references (token-level LCS, normalized text)."""
    p_toks = _normalize(pred).split()
    if not p_toks:
        return 0.0
    best = 0.0
    for g in gts:
        g_toks = _normalize(g).split()
        if not g_toks:
            continue
        lcs = _lcs_length(p_toks, g_toks)
        if lcs == 0:
            continue
        prec = lcs / len(p_toks)
        rec = lcs / len(g_toks)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return float(best)


METRIC_FN = {
    "textvqa_val": vqa_score,
    "docvqa_val": anls_score,
    "gqa": exact_match,
    "coco2017_cap_val": rouge_l_score,
}


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth lookup (by sample_id)
# ──────────────────────────────────────────────────────────────────────────────

def build_gt_lookup(dataset: str, eval_root: Path) -> dict[str, list[str]]:
    if dataset == "textvqa_val":
        ds = load_from_disk(str(eval_root / "textvqa_val"))
        return {str(ex["question_id"]): list(ex["answers"]) for ex in ds}
    if dataset == "docvqa_val":
        ds = load_from_disk(str(eval_root / "docvqa_val"))
        return {str(ex["questionId"]): list(ex["answers"]) for ex in ds}
    if dataset == "gqa":
        ds = load_from_disk(str(eval_root / "gqa" / "instructions"))
        return {str(ex["id"]): [str(ex["answer"])] for ex in ds}
    if dataset == "coco2017_cap_val":
        ds = load_from_disk(str(eval_root / "coco2017_cap_val"))
        return {str(ex["question_id"]): [str(c) for c in (ex["answer"] or [])] for ex in ds}
    raise ValueError(f"Unsupported dataset for GT lookup: {dataset}")


# ──────────────────────────────────────────────────────────────────────────────
# Generation with press
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_press(
    model, tokenizer, image_processor,
    rec: dict, press, device: torch.device,
    max_new_tokens: int,
) -> str:
    image = Image.open(io.BytesIO(rec["image_bytes"])).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        rec["prompt_text"], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)

    image_indices = rec["image_token_indices"].long().to(device)
    question_indices = rec["question_token_indices"].long().to(device)

    if press is not None:
        press.set_image_positions(image_indices)
        if hasattr(press, "set_question_positions"):
            press.set_question_positions(question_indices)

    needs_attn = press is not None and press.__class__.__name__.startswith("H2O")
    gen_kwargs = dict(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        modalities=["image"],
        do_sample=False, num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        output_attentions=bool(needs_attn),
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    try:
        if press is None:
            out = model.generate(**gen_kwargs)
        else:
            with press(model):
                out = model.generate(**gen_kwargs)
    finally:
        if press is not None:
            press.clear_sample_context()

    seq = out.sequences[0]
    prompt_len = int(input_ids.shape[1])
    if seq.shape[0] > prompt_len:
        answer_ids = seq[prompt_len:].detach().cpu().tolist()
    else:
        answer_ids = seq.detach().cpu().tolist()
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def make_press(method: str, keep_ratio: float, student_ckpt: str | None):
    if method == "h2o":
        return H2OImageOnlyPress(image_keep_ratio=keep_ratio)
    if method == "student":
        if not student_ckpt:
            raise ValueError("student_ckpt required for method=student")
        return VisualUtilityStudentPress(
            image_keep_ratio=keep_ratio,
            student_model_name=student_ckpt,
        )
    if method == "full_cache":
        return None
    raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-dir", required=True, type=Path)
    p.add_argument("--eval-root", default="/mnt/srv/home/dlpc.3842/zap/data/eval", type=Path)
    p.add_argument("--datasets", nargs="+", default=["textvqa_val", "docvqa_val", "gqa"])
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--sample-start", type=int, default=0,
                   help="Start index (inclusive) into the sorted .pt list per dataset.")
    p.add_argument("--sample-end", type=int, default=None,
                   help="End index (exclusive). If None, use n-samples.")
    p.add_argument("--methods", nargs="+", default=["h2o", "student", "full_cache"])
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--keep-ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--model-path", default="/mnt/srv/home/dlpc.3842/zap/ckpts/llava-v1.5-7b")
    p.add_argument("--model-name", default="llava-v1.5-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def load_model(args):
    print(f"[load] {args.model_path} attn=eager", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path, model_base=None, model_name=args.model_name,
        device_map=args.device_map, attn_implementation="eager", multimodal=True,
    )
    model.eval()
    return tokenizer, model, image_processor


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer, model, image_processor = load_model(args)

    # Build GT lookup once per dataset
    gt_lookups = {ds: build_gt_lookup(ds, args.eval_root) for ds in args.datasets}

    # Plan: for each dataset → for each sample → for each method → for each keep_ratio
    # results[(dataset, sample_id, method, keep_ratio)] = {pred, score}
    rows: list[dict] = []
    t_start = time.time()
    n_done = 0
    n_total = 0
    for ds in args.datasets:
        ds_dir = args.teacher_dir / ds
        all_pts = sorted(ds_dir.glob("*.pt"))[: args.n_samples]
        s_start = args.sample_start
        s_end = args.sample_end if args.sample_end is not None else args.n_samples
        pts = all_pts[s_start:s_end]
        n_total += len(pts) * sum(
            1 if m == "full_cache" else len(args.keep_ratios) for m in args.methods
        )

    for ds in args.datasets:
        ds_dir = args.teacher_dir / ds
        all_pts = sorted(ds_dir.glob("*.pt"))[: args.n_samples]
        s_start = args.sample_start
        s_end = args.sample_end if args.sample_end is not None else args.n_samples
        pts = all_pts[s_start:s_end]
        gt_map = gt_lookups[ds]
        metric_fn = METRIC_FN[ds]
        print(f"[ds] {ds}  n_samples={len(pts)}", flush=True)

        for idx, tpath in enumerate(pts, 1):
            rec = torch.load(tpath, map_location="cpu", weights_only=False)
            sid = str(rec["sample_id"])
            gts = gt_map.get(sid, [str(rec.get("answer", "")) or ""])

            for method in args.methods:
                krs = [None] if method == "full_cache" else args.keep_ratios
                for kr in krs:
                    try:
                        press = make_press(method, kr or 1.0, args.student_ckpt)
                        pred = generate_with_press(
                            model, tokenizer, image_processor,
                            rec, press, device, args.max_new_tokens,
                        )
                        score = metric_fn(pred, gts)
                    except Exception as exc:
                        print(f"[skip] {ds}/{sid} {method} kr={kr}: {exc}", flush=True)
                        pred = ""; score = float("nan")

                    rows.append({
                        "dataset": ds, "sample_id": sid,
                        "method": method, "keep_ratio": (kr if kr is not None else 1.0),
                        "pred": pred,
                        "score": float(score) if not np.isnan(score) else None,
                        "gts": gts[:5],
                    })
                    n_done += 1

            if idx % 10 == 0 or idx == len(pts):
                elapsed = time.time() - t_start
                rate = n_done / max(elapsed, 1e-6)
                eta = (n_total - n_done) / max(rate, 1e-6) / 60
                print(f"[progress] {ds} {idx}/{len(pts)}  "
                      f"runs={n_done}/{n_total}  {rate:.2f}/s  ETA {eta:.1f}min",
                      flush=True)

    # Aggregate per (dataset, method, keep_ratio)
    agg: dict[str, dict] = {}
    for r in rows:
        key = (r["dataset"], r["method"], r["keep_ratio"])
        if r["score"] is None:
            continue
        agg.setdefault(key, []).append(r["score"])
    summary = []
    for (ds, m, kr), vals in agg.items():
        summary.append({
            "dataset": ds, "method": m, "keep_ratio": kr,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        })
    summary.sort(key=lambda x: (x["dataset"], x["method"], x["keep_ratio"]))

    payload = {
        "config": {
            "datasets": args.datasets, "methods": args.methods,
            "keep_ratios": args.keep_ratios, "n_samples": args.n_samples,
            "student_ckpt": args.student_ckpt, "model_path": args.model_path,
        },
        "summary": summary,
        "rows": rows,
    }
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {args.output}  rows={len(rows)}", flush=True)
    print()
    print("=== summary ===")
    for s in summary:
        print(f"  {s['dataset']:12s}  {s['method']:10s}  k={s['keep_ratio']:.2f}  "
              f"score={s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
