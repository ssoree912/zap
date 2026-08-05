#!/usr/bin/env python3
"""Random-init (and uniform) baseline on the parity probe's exact 180 val shards.

Reuses the trainer's own helpers and replicates its `run_forward(..., train=False)`
reduction verbatim, so val_mse / val_rank are directly comparable to
`student_onevision_parity_probe/train_log.jsonl`.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/zap")

TRAINER = Path(
    "/workspace/zap/experiments/EXP-20260503-026-onevision-original-teacher-train/"
    "train_original_onevision_student.py"
)
spec = importlib.util.spec_from_file_location("onevision_trainer", TRAINER)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

from kvpress.presses.visual_utility_student_onevision import (  # noqa: E402
    VisualUtilityStudentOneVision,
    pairwise_ranking_loss,
)
from llava.model.builder import load_pretrained_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-root", default="/workspace/zap/artifacts/original_onevision_teacher/future_decode_qwen2_7b")
    p.add_argument("--datasets", nargs="+", default=["textvqa", "gqa", "scienceqa"])
    p.add_argument("--per-ds-limit", type=int, default=600)
    p.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    p.add_argument("--model-name", default="llava_qwen")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--rank-top-ratio", type=float, default=0.2)
    p.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    p.add_argument("--lambda-rank", type=float, default=0.1)
    args = p.parse_args()

    # Same seeding / shuffle / split as the trainer.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    T.patch_siglip_loader_to_local_init()
    tokenizer, lvlm, image_processor, _ = load_pretrained_model(
        model_path=args.model_path, model_base=None, model_name=args.model_name,
        device_map=args.device_map, attn_implementation="sdpa", multimodal=True,
    )
    lvlm.eval()
    for prm in lvlm.parameters():
        prm.requires_grad_(False)

    all_files = T.list_teacher_files(Path(args.teacher_root), args.datasets, args.per_ds_limit)
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_total = len(all_files)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    val_files = all_files[n_train:]
    print(f"[data] total={n_total} train={n_train} val={n_val}", flush=True)

    student = VisualUtilityStudentOneVision().to(device).eval()

    sums = {k: 0.0 for k in ("rand_loss", "rand_mse", "rand_rank", "uni_mse", "uni_rank")}
    n_seen = 0
    for i, path in enumerate(val_files, start=1):
        rec = torch.load(path, weights_only=False, map_location="cpu")
        inputs = T.build_inputs(
            rec=rec, tokenizer=tokenizer, image_processor=image_processor,
            model=lvlm, device=device,
        )
        with torch.no_grad():
            out = lvlm(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        H_all = out.hidden_states
        image_idx = rec["image_token_indices"].to(device, dtype=torch.long)
        q_idx = rec["question_token_indices"].to(device, dtype=torch.long)
        teacher_norm = rec["teacher_norm"].to(device, dtype=torch.float32)
        teacher_raw = rec.get("teacher_raw")
        teacher_valid = None
        if isinstance(teacher_raw, torch.Tensor):
            teacher_valid = teacher_raw.float().sum(dim=-1).to(device) > 1e-12

        tot = {k: 0.0 for k in sums}
        n_used = 0
        with torch.no_grad():
            for li in student.layer_indices:
                if li + 1 >= len(H_all) or li >= teacher_norm.shape[0]:
                    continue
                if teacher_valid is not None and li < teacher_valid.numel() and not bool(teacher_valid[li].item()):
                    continue
                target = teacher_norm[li].unsqueeze(0)
                if not torch.isfinite(target).all():
                    continue
                H_l = H_all[li + 1].to(torch.float32)
                pred = student.forward_layer(li, H_l, image_idx, q_idx)
                pred_norm = F.softmax(pred, dim=-1)
                mse = F.mse_loss(pred_norm, target)
                rank = pairwise_ranking_loss(
                    pred_norm, target, margin=args.rank_margin,
                    top_ratio=args.rank_top_ratio, bottom_ratio=args.rank_bottom_ratio,
                )
                tot["rand_mse"] += float(mse)
                tot["rand_rank"] += float(rank)
                tot["rand_loss"] += float(mse) + args.lambda_rank * float(rank)

                uni = torch.full_like(target, 1.0 / target.shape[-1])
                tot["uni_mse"] += float(F.mse_loss(uni, target))
                tot["uni_rank"] += float(pairwise_ranking_loss(
                    uni, target, margin=args.rank_margin,
                    top_ratio=args.rank_top_ratio, bottom_ratio=args.rank_bottom_ratio,
                ))
                n_used += 1
                del H_l, pred, pred_norm
        if n_used == 0:
            continue
        for k in sums:
            sums[k] += tot[k] / n_used
        n_seen += 1
        if i % 30 == 0:
            print(f"[progress] {i}/{len(val_files)} "
                  + " ".join(f"{k}={sums[k]/n_seen:.6g}" for k in sums), flush=True)

    print("[result] n=%d" % n_seen)
    for k in sums:
        print(f"  {k} = {sums[k]/n_seen:.6g}")


if __name__ == "__main__":
    main()
