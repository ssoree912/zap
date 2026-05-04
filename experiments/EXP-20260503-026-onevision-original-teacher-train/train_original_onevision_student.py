#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train `VisualUtilityStudentOneVision` from original LLaVA-OneVision teachers."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/workspace/zap")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(LLAVA_ONEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))

from kvpress.presses.visual_utility_student_onevision import (  # noqa: E402
    VisualUtilityStudentOneVision,
    pairwise_ranking_loss,
)
from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


def patch_siglip_loader_to_local_init() -> None:
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):  # noqa: ANN001, ARG001
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


def list_teacher_files(teacher_root: Path, datasets: list[str], per_ds_limit: int | None) -> list[Path]:
    files: list[Path] = []
    for dataset in datasets:
        ds_files = sorted((teacher_root / dataset).glob("*.pt"))
        if per_ds_limit is not None:
            ds_files = ds_files[:per_ds_limit]
        files.extend(ds_files)
    return files


class TeacherDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = torch.load(self.files[idx], weights_only=False, map_location="cpu")
        rec["_path"] = str(self.files[idx])
        return rec


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(batch) == 1
    return batch[0]


def _to_image_inputs(image_tensor: Any, device: torch.device) -> Any:
    if isinstance(image_tensor, torch.Tensor):
        return image_tensor.to(device=device, dtype=torch.float16)
    return [tensor.to(device=device, dtype=torch.float16) for tensor in image_tensor]


def build_inputs(
    *,
    rec: dict[str, Any],
    tokenizer: Any,
    image_processor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    prompt_text = rec["prompt_text"]
    image_path = rec["image_path"]
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    image_tensor = _to_image_inputs(image_tensor, device)
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "images": image_tensor,
        "image_sizes": [image_size],
        "modalities": ["image"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", default="/workspace/zap/artifacts/original_onevision_teacher/future_decode_qwen2_7b")
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "gqa", "scienceqa"])
    parser.add_argument("--per-ds-limit", type=int, default=600)
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-top-ratio", type=float, default=0.2)
    parser.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    log_f = (out_dir / "train_log.jsonl").open("w")

    patch_siglip_loader_to_local_init()
    print(
        f"[load] model={args.model_path} model_name={args.model_name} "
        f"device_map={args.device_map} attn=sdpa",
        flush=True,
    )
    tokenizer, lvlm, image_processor, _context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="sdpa",
        multimodal=True,
    )
    lvlm.eval()
    for p in lvlm.parameters():
        p.requires_grad_(False)

    per_ds_limit = args.per_ds_limit if args.per_ds_limit and args.per_ds_limit > 0 else None
    all_files = list_teacher_files(Path(args.teacher_root), args.datasets, per_ds_limit)
    if not all_files:
        raise FileNotFoundError(f"No teacher .pt files under {args.teacher_root} for {args.datasets}")
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_total = len(all_files)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    print(f"[data] total={n_total} train={n_train} val={n_val} datasets={args.datasets}", flush=True)

    train_loader = DataLoader(
        TeacherDataset(all_files[:n_train]),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        TeacherDataset(all_files[n_train:]),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
    )

    student = VisualUtilityStudentOneVision().to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[student] layers={student.layer_indices} params={n_params:,}", flush=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_forward(rec: dict[str, Any], train: bool) -> tuple[float, float, float]:
        inputs = build_inputs(
            rec=rec,
            tokenizer=tokenizer,
            image_processor=image_processor,
            model=lvlm,
            device=device,
        )
        with torch.no_grad():
            out = lvlm(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        H_all = out.hidden_states
        image_idx = rec["image_token_indices"].to(device, dtype=torch.long)
        q_idx = rec["question_token_indices"].to(device, dtype=torch.long)
        teacher_norm = rec["teacher_norm"].to(device, dtype=torch.float32)
        teacher_raw = rec.get("teacher_raw")
        teacher_valid = None
        if isinstance(teacher_raw, torch.Tensor):
            teacher_valid = teacher_raw.float().sum(dim=-1).to(device) > 1e-12

        n_prefill = int(H_all[0].shape[1])
        if int(image_idx.max().item()) >= n_prefill:
            raise ValueError(
                f"image_idx max {int(image_idx.max().item())} exceeds hidden length {n_prefill}"
            )
        if teacher_norm.shape[-1] != image_idx.numel():
            raise ValueError(
                f"teacher width {teacher_norm.shape[-1]} != image tokens {image_idx.numel()}"
            )

        if train:
            optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_mse = 0.0
        total_rank = 0.0
        n_used = 0
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
            loss_mse = F.mse_loss(pred_norm, target)
            loss_rank = pairwise_ranking_loss(
                pred_norm,
                target,
                margin=args.rank_margin,
                top_ratio=args.rank_top_ratio,
                bottom_ratio=args.rank_bottom_ratio,
            )
            loss = loss_mse + args.lambda_rank * loss_rank
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at layer={li}")
            if train:
                (loss / len(student.layer_indices)).backward()
            total_loss += float(loss.detach().item())
            total_mse += float(loss_mse.detach().item())
            total_rank += float(loss_rank.detach().item())
            n_used += 1
            del H_l, pred, pred_norm, loss_mse, loss_rank, loss

        if n_used == 0:
            raise RuntimeError("no finite teacher layers used")
        if train:
            bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in student.parameters()
            )
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("non-finite gradient detected")
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        denom = max(1, n_used)
        return total_loss / denom, total_mse / denom, total_rank / denom

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = train_mse = train_rank = 0.0
        n_seen = 0
        for step, rec in enumerate(train_loader, start=1):
            try:
                loss, mse, rank = run_forward(rec, train=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip train] {rec.get('sample_id', '?')}: {exc}\n{traceback.format_exc()}", flush=True)
                continue
            train_loss += loss
            train_mse += mse
            train_rank += rank
            n_seen += 1
            if step % args.log_every == 0:
                print(
                    f"[epoch {epoch} step {step}/{len(train_loader)}] "
                    f"loss={train_loss/max(1,n_seen):.6f} mse={train_mse/max(1,n_seen):.6f} "
                    f"rank={train_rank/max(1,n_seen):.6f} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
        train_loss /= max(1, n_seen)
        train_mse /= max(1, n_seen)
        train_rank /= max(1, n_seen)

        student.eval()
        val_loss = val_mse = val_rank = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for rec in val_loader:
                try:
                    loss, mse, rank = run_forward(rec, train=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"[skip val] {rec.get('sample_id', '?')}: {exc}", flush=True)
                    continue
                val_loss += loss
                val_mse += mse
                val_rank += rank
                n_val_seen += 1
        val_loss /= max(1, n_val_seen)
        val_mse /= max(1, n_val_seen)
        val_rank /= max(1, n_val_seen)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_rank": train_rank,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_rank": val_rank,
            "elapsed": elapsed,
            "n_train_seen": n_seen,
            "n_val_seen": n_val_seen,
        }
        print(
            f"[epoch {epoch}] train_loss={train_loss:.6f} train_mse={train_mse:.6f} "
            f"train_rank={train_rank:.6f} | val_loss={val_loss:.6f} val_mse={val_mse:.6f} "
            f"val_rank={val_rank:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

        torch.save(
            {
                "epoch": epoch,
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            },
            out_dir / "last_checkpoint.pt",
        )
        if val_loss < best_val:
            best_val = val_loss
            student.save_pretrained(out_dir)
            print(f"[ckpt] saved best val={best_val:.6f} -> {out_dir}", flush=True)

    log_f.close()
    print(f"[done] best_val={best_val:.6f} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
