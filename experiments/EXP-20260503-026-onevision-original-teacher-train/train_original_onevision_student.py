#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train `VisualUtilityStudentOneVision` from original LLaVA-OneVision teachers."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = WORKSPACE_ROOT / "VFlowOpt" / "src" / "LLaVA-OneVision"
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


class HiddenStatesOnlyLMHead(torch.nn.Module):
    """Avoid allocating full-vocabulary logits when only hidden states are used."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_empty((*hidden_states.shape[:-1], 0))


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
    parser.add_argument(
        "--teacher-root",
        default=str(
            REPO_ROOT
            / "artifacts"
            / "original_onevision_teacher"
            / "future_answer_agnostic_1800_seed42"
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "gqa", "scienceqa"])
    parser.add_argument("--per-ds-limit", type=int, default=600)
    parser.add_argument(
        "--model-path",
        default=str(WORKSPACE_ROOT / "models" / "llava-onevision-qwen2-7b-ov"),
    )
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=["adamw8bit", "adamw"],
        default="adamw8bit",
        help="8-bit AdamW keeps the 125M-parameter student within a 24GB GPU.",
    )
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-top-ratio", type=float, default=0.2)
    parser.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to last_checkpoint.pt; resumes at the following epoch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    process_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
        if args.device_map == "auto":
            args.device_map = args.device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    resume_path = Path(args.resume_from) if args.resume_from else None
    if process_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        train_config = dict(vars(args), world_size=world_size)
        (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))
        log_f = (out_dir / "train_log.jsonl").open("a" if resume_path else "w")
    else:
        log_f = None

    patch_siglip_loader_to_local_init()
    print(
        f"[rank {process_rank}] [load] model={args.model_path} model_name={args.model_name} "
        f"device={device} device_map={args.device_map} attn=sdpa",
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
    # Qwen's causal-LM wrapper otherwise materializes [B, T, vocab] logits,
    # which can exceed 2GB for high-resolution OneVision samples.  The student
    # consumes hidden states only, so remove the unused projection and its
    # roughly 1GB fp16 weight.
    lvlm.lm_head = HiddenStatesOnlyLMHead().to(device)
    torch.cuda.empty_cache()

    per_ds_limit = args.per_ds_limit if args.per_ds_limit and args.per_ds_limit > 0 else None
    all_files = list_teacher_files(Path(args.teacher_root), args.datasets, per_ds_limit)
    if not all_files:
        raise FileNotFoundError(f"No teacher .pt files under {args.teacher_root} for {args.datasets}")
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_total = len(all_files)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_files = all_files[:n_train][process_rank::world_size]
    val_files = all_files[n_train:][process_rank::world_size]
    if process_rank == 0:
        print(
            f"[data] total={n_total} train={n_train} val={n_val} "
            f"world_size={world_size} datasets={args.datasets}",
            flush=True,
        )

    train_loader = DataLoader(
        TeacherDataset(train_files),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        generator=torch.Generator().manual_seed(args.seed + process_rank),
    )
    val_loader = DataLoader(
        TeacherDataset(val_files),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
    )

    student = VisualUtilityStudentOneVision().to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if process_rank == 0:
        print(f"[student] layers={student.layer_indices} params={n_params:,}", flush=True)
    if args.optimizer == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer = AdamW8bit(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if process_rank == 0:
        print(f"[optimizer] {args.optimizer} initialized", flush=True)

    start_epoch = 1
    best_val = float("inf")
    elapsed_offset = 0.0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, weights_only=False, map_location="cpu")
        student.load_state_dict(checkpoint["student"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))
        log_path = out_dir / "train_log.jsonl"
        if log_path.exists():
            rows = [
                json.loads(line)
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
            if rows:
                elapsed_offset = float(rows[-1].get("elapsed", 0.0))
                best_val = min(best_val, *(float(row["val_loss"]) for row in rows))
        if process_rank == 0:
            print(
                f"[resume] checkpoint={resume_path} start_epoch={start_epoch} "
                f"best_val={best_val:.6f} elapsed_offset={elapsed_offset:.1f}s",
                flush=True,
            )

    def sync_gradients() -> None:
        if not distributed:
            return
        # Each decoder layer owns an independent student.  Coalescing gradients
        # per layer avoids thousands of tiny NCCL collectives.
        for layer in student.layers.values():
            params = [param for param in layer.parameters() if param.requires_grad]
            grads = [
                param.grad if param.grad is not None else torch.zeros_like(param)
                for param in params
            ]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            synced = torch._utils._unflatten_dense_tensors(flat, grads)
            for param, grad in zip(params, synced):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.copy_(grad)

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
            sync_gradients()
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

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if process_rank == 0:
            print(f"[epoch {epoch}] start local_train_steps={len(train_loader)}", flush=True)
        student.train()
        train_loss = train_mse = train_rank = 0.0
        n_seen = 0
        for step, rec in enumerate(train_loader, start=1):
            try:
                loss, mse, rank = run_forward(rec, train=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip train] {rec.get('sample_id', '?')}: {exc}\n{traceback.format_exc()}", flush=True)
                if distributed:
                    raise
                continue
            train_loss += loss
            train_mse += mse
            train_rank += rank
            n_seen += 1
            if process_rank == 0 and step % args.log_every == 0:
                print(
                    f"[epoch {epoch} step {step}/{len(train_loader)}] "
                    f"loss={train_loss/max(1,n_seen):.6f} mse={train_mse/max(1,n_seen):.6f} "
                    f"rank={train_rank/max(1,n_seen):.6f} "
                    f"elapsed={elapsed_offset+time.time()-t0:.1f}s",
                    flush=True,
                )
        train_stats = torch.tensor(
            [train_loss, train_mse, train_rank, float(n_seen)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        train_loss = float(train_stats[0].item()) / max(1.0, float(train_stats[3].item()))
        train_mse = float(train_stats[1].item()) / max(1.0, float(train_stats[3].item()))
        train_rank = float(train_stats[2].item()) / max(1.0, float(train_stats[3].item()))
        n_seen_global = int(train_stats[3].item())

        student.eval()
        val_loss = val_mse = val_rank = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for rec in val_loader:
                try:
                    loss, mse, rank = run_forward(rec, train=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"[skip val] {rec.get('sample_id', '?')}: {exc}", flush=True)
                    if distributed:
                        raise
                    continue
                val_loss += loss
                val_mse += mse
                val_rank += rank
                n_val_seen += 1
        val_stats = torch.tensor(
            [val_loss, val_mse, val_rank, float(n_val_seen)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        val_loss = float(val_stats[0].item()) / max(1.0, float(val_stats[3].item()))
        val_mse = float(val_stats[1].item()) / max(1.0, float(val_stats[3].item()))
        val_rank = float(val_stats[2].item()) / max(1.0, float(val_stats[3].item()))
        n_val_seen_global = int(val_stats[3].item())
        elapsed = elapsed_offset + time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_rank": train_rank,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_rank": val_rank,
            "elapsed": elapsed,
            "n_train_seen": n_seen_global,
            "n_val_seen": n_val_seen_global,
        }
        if process_rank == 0:
            print(
                f"[epoch {epoch}] train_loss={train_loss:.6f} train_mse={train_mse:.6f} "
                f"train_rank={train_rank:.6f} | val_loss={val_loss:.6f} val_mse={val_mse:.6f} "
                f"val_rank={val_rank:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            assert log_f is not None
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            if val_loss < best_val:
                best_val = val_loss
                student.save_pretrained(out_dir)
                print(f"[ckpt] saved best val={best_val:.6f} -> {out_dir}", flush=True)
            torch.save(
                {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "args": vars(args),
                    "world_size": world_size,
                },
                out_dir / "last_checkpoint.pt",
            )
        else:
            best_val = min(best_val, val_loss)
        if distributed:
            dist.barrier()

    if log_f is not None:
        log_f.close()
    if process_rank == 0:
        print(
            f"[done] best_val={best_val:.6f} "
            f"elapsed={elapsed_offset+time.time()-t0:.1f}s",
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
