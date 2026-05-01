#!/usr/bin/env python3
"""Stage 2: Train 3-branch visual-utility student.

Pipeline per sample: load cached teacher (`teacher_norm [L, N_I]`) and prompt /
image; run frozen LLaVA prefill in bf16 to obtain per-layer hidden states; for
each layer in the scope, compute student score, softmax to a distribution, and
combine MSE + ranking loss against the teacher. Student weights and optimizer
state stay in fp32; only LLaVA forward is bf16.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, "/workspace/zap")
from kvpress.presses.visual_utility_student import (
    VisualUtilityStudent,
    pairwise_ranking_loss,
)
from ..llava_15b_extractor import configure_llava_processor


def list_teacher_files(
    teacher_root: Path,
    datasets: list[str],
    n_per_dataset: int | None = None,
    seed: int = 0,
) -> list[Path]:
    rng = random.Random(seed)
    files: list[Path] = []
    for ds in datasets:
        ds_dir = teacher_root / ds
        if not ds_dir.exists():
            continue
        ds_files = sorted(ds_dir.glob("*.pt"))
        if n_per_dataset is not None and len(ds_files) > n_per_dataset:
            ds_files = rng.sample(ds_files, n_per_dataset)
            ds_files = sorted(ds_files)
        files.extend(ds_files)
    return files


class TeacherCacheDataset(Dataset):
    """Preload all teacher .pt files (and corresponding processed inputs) once.

    600 samples × ~80 KB metadata + ~1 MB processed input ids ≈ 0.6 GB CPU RAM
    — trivial vs the 14 GB LLaVA on GPU and saves a torch.load + processor call
    on every training step.
    """

    def __init__(
        self,
        files: list[Path],
        processor=None,
        verbose: bool = True,
    ) -> None:
        self.files = list(files)
        self.records: list[dict] = []
        for i, p in enumerate(self.files):
            rec = torch.load(p, weights_only=False, map_location="cpu")
            if processor is not None:
                image = Image.open(rec["image_path"]).convert("RGB")
                inputs = processor(images=image, text=rec["prompt_text"], return_tensors="pt")
                rec["_processed_inputs"] = {k: v for k, v in inputs.items()}
            self.records.append(rec)
            if verbose and (i + 1) % 100 == 0:
                print(f"[preload] {i+1}/{len(self.files)}", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_single(batch: list[dict]) -> dict:
    assert len(batch) == 1, "this trainer uses batch size 1"
    return batch[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-root", default="/workspace/zap/data/train/teacher_llava15")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["llava_instruct", "scienceqa", "textvqa", "gqa"],
    )
    p.add_argument("--llava-path", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lambda-rank", type=float, default=0.1)
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--rank-top-ratio", type=float, default=0.2)
    p.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-per-dataset", type=int, default=500)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    print(f"[load] LLaVA={args.llava_path} dtype=bf16 device={device}", flush=True)
    lvlm = LlavaForConditionalGeneration.from_pretrained(
        args.llava_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for p_ in lvlm.parameters():
        p_.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(args.llava_path)
    processor = configure_llava_processor(processor, lvlm.config)

    # ----- dataset -----
    all_files = list_teacher_files(Path(args.teacher_root), args.datasets, args.n_per_dataset, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_total = len(all_files)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    print(f"[data] total={n_total} train={n_train} val={n_val} (preloading...)", flush=True)
    train_ds = TeacherCacheDataset(all_files[:n_train], processor=processor)
    val_ds = TeacherCacheDataset(all_files[n_train:], processor=processor)
    print(f"[data] preloaded train={len(train_ds)} val={len(val_ds)}", flush=True)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_single, generator=loader_gen
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    # ----- student -----
    student = VisualUtilityStudent().to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[student] layers={student.layer_indices} params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_forward(rec: dict, train: bool) -> tuple[float, float, float]:
        inputs = {k: v.to(device) for k, v in rec["_processed_inputs"].items()}

        with torch.no_grad():
            out = lvlm(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # index 0 = embeddings, index l+1 = output of layer l
        H_all = out.hidden_states

        image_idx = rec["image_token_indices"].to(device, dtype=torch.long)
        q_idx = rec["question_token_indices"].to(device, dtype=torch.long)
        teacher_norm = rec["teacher_norm"].to(device, dtype=torch.float32)  # [L, N_I]

        total_mse = 0.0
        total_rank = 0.0
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        for li in student.layer_indices:
            H_l = H_all[li + 1].to(torch.float32)
            s_pred = student.forward_layer(li, H_l, image_idx, q_idx)  # [B, N_I]
            pred_norm = F.softmax(s_pred, dim=-1)
            t = teacher_norm[li].unsqueeze(0)  # [1, N_I]
            loss_mse = F.mse_loss(pred_norm, t)
            loss_rank = pairwise_ranking_loss(
                pred_norm,
                t,
                margin=args.rank_margin,
                top_ratio=args.rank_top_ratio,
                bottom_ratio=args.rank_bottom_ratio,
            )
            total_loss = total_loss + loss_mse + args.lambda_rank * loss_rank
            total_mse += float(loss_mse.detach().item())
            total_rank += float(loss_rank.detach().item())

        if train:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        n_layers = len(student.layer_indices)
        return float(total_loss.detach().item()), total_mse / n_layers, total_rank / n_layers

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        student.train()
        train_loss = train_mse = train_rank = 0.0
        n_seen = 0
        for step, rec in enumerate(train_loader):
            try:
                loss, mse, rank = run_forward(rec, train=True)
            except (RuntimeError, ValueError, IOError) as e:
                print(
                    f"[skip] {rec.get('sample_id', '?')}: {e}\n{traceback.format_exc()}",
                    flush=True,
                )
                continue
            train_loss += loss
            train_mse += mse
            train_rank += rank
            n_seen += 1
            if (step + 1) % args.log_every == 0:
                print(
                    f"[epoch {epoch} step {step+1}/{len(train_loader)}] "
                    f"loss={train_loss/n_seen:.5f} mse={train_mse/n_seen:.5f} "
                    f"rank={train_rank/n_seen:.5f} elapsed={time.time()-t0:.1f}s",
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
                except (RuntimeError, ValueError, IOError) as e:
                    print(
                        f"[skip val] {rec.get('sample_id', '?')}: {e}\n{traceback.format_exc()}",
                        flush=True,
                    )
                    continue
                val_loss += loss
                val_mse += mse
                val_rank += rank
                n_val_seen += 1
        if n_val_seen == 0:
            n_val_seen = 1
        val_loss /= n_val_seen
        val_mse /= n_val_seen
        val_rank /= n_val_seen

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss={train_loss:.5f} train_mse={train_mse:.5f} "
            f"train_rank={train_rank:.5f} | val_loss={val_loss:.5f} val_mse={val_mse:.5f} "
            f"val_rank={val_rank:.5f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        log_f.write(
            json.dumps(
                dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_mse=train_mse,
                    train_rank=train_rank,
                    val_loss=val_loss,
                    val_mse=val_mse,
                    val_rank=val_rank,
                    elapsed=elapsed,
                )
            )
            + "\n"
        )
        log_f.flush()

        if val_loss < best_val:
            best_val = val_loss
            student.save_pretrained(out_dir)
            print(f"[ckpt] saved best (val_loss={val_loss:.5f}) to {out_dir}", flush=True)

    log_f.close()
    print(f"[done] best_val={best_val:.5f} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
