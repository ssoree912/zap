#!/usr/bin/env python3
"""Fast one-pass multi-layer trainer for unified teacher shards.

Uses PostVision-style streaming (one shard scan / epoch) but with per-sample
softmax MSE grouping (method B).

Within each shard-batch:
  * group rows by (sample_id, layer)
  * per group: pred = softmax(mlp_layer(x_group)); label = y_group / y_group.sum()
  * loss = sum_mse_over_groups
  * single backward() across all layers

Shards must come from collect_unified_teacher_shards.py with schema:
  x, y_pv, y_future, layer, sample_id, token_idx

Usage:
  python train_unified_probe_onepass.py \\
      --shard_dirs teacher/unified/textvqa teacher/unified/scienceqa teacher/unified/nlvr2 \\
      --teacher future --out_dir ckpts/future_probe_v3 \\
      --n_layers_model 32 --selected_layers 0 1 2 ... 31 \\
      --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel


# ── shard iteration ────────────────────────────────────────────────────────

def iter_shard_batches(
    shard_paths: list[Path],
    dir_idx_of_path: dict[Path, int],
    sid_offset: int,
    split_mask_fn,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield per-shard row dicts after applying split_mask_fn(shard_dict) → bool mask.

    Also remaps sample_id to global: dir_idx * sid_offset + local_sample_id.
    """
    paths = list(shard_paths)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)
    for sp in paths:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        ds_idx = dir_idx_of_path[sp]
        global_sid = d["sample_id"].int() + int(ds_idx * sid_offset)
        keep = split_mask_fn(global_sid)  # bool tensor
        if not keep.any():
            continue
        out = {
            "x": d["x"][keep],
            "y": d[split_mask_fn.y_key][keep],
            "layer": d["layer"][keep],
            "sample_id": global_sid[keep],
        }
        yield out


class _SplitMask:
    def __init__(self, sid_set: set[int], y_key: str) -> None:
        self.sid_set = sid_set
        self.y_key = y_key

    def __call__(self, global_sids: torch.Tensor) -> torch.Tensor:
        sid_np = global_sids.numpy()
        return torch.from_numpy(np.isin(sid_np, list(self.sid_set)))


# ── training loop ──────────────────────────────────────────────────────────

def compute_batch_loss(
    model: KVzapModel,
    x: torch.Tensor,       # [N, D] on device
    y: torch.Tensor,        # [N] on device
    layer: torch.Tensor,    # [N] uint8 on device
    sample_id: torch.Tensor, # [N] int32 on device
) -> tuple[torch.Tensor, int]:
    """Per-(sample_id, layer) softmax MSE.

    Returns (total_loss, num_groups).
    """
    loss = torch.zeros(1, device=x.device)
    n_groups = 0
    # Efficient grouping: build key = sample_id * 32 + layer (uint64)
    key = sample_id.long() * 64 + layer.long()
    unique_keys, inverse = torch.unique(key, return_inverse=True)
    for k_idx in range(unique_keys.numel()):
        m = inverse == k_idx
        if m.sum() < 2:
            continue
        x_g = x[m]
        y_g = y[m]
        l_id = int(layer[m][0].item())
        pred = torch.softmax(model.layers[l_id](x_g).squeeze(-1), dim=0)
        label = y_g / (y_g.sum() + 1e-8)
        loss = loss + F.mse_loss(pred, label)
        n_groups += 1
    return loss, n_groups


@torch.no_grad()
def eval_all_layers(
    model: KVzapModel,
    shard_paths: list[Path],
    dir_idx_of_path: dict[Path, int],
    sid_offset: int,
    sid_set: set[int],
    y_key: str,
    device: str,
    n_layers: int,
) -> dict[int, dict[str, float]]:
    """Compute per-(sample_id, layer) Spearman and top-k overlap; aggregate per layer."""
    # Accumulate per-sample preds across shards (sample may span shards)
    groups: dict[tuple[int, int], dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"x": [], "y": []})
    mask_fn = _SplitMask(sid_set, y_key)
    for batch in iter_shard_batches(shard_paths, dir_idx_of_path, sid_offset, mask_fn, shuffle=False, seed=0):
        # vectorized: group by (sid, layer)
        key = batch["sample_id"].long() * 64 + batch["layer"].long()
        unique_keys = torch.unique(key)
        for k_val in unique_keys.tolist():
            m = key == k_val
            sid = int(batch["sample_id"][m][0].item())
            l = int(batch["layer"][m][0].item())
            groups[(sid, l)]["x"].append(batch["x"][m])
            groups[(sid, l)]["y"].append(batch["y"][m])

    per_layer_stats = defaultdict(lambda: {"sp": [], "tk": [], "mse": []})
    model.eval()
    for (sid, l), parts in groups.items():
        x = torch.cat(parts["x"], dim=0).float().to(device)
        y = torch.cat(parts["y"], dim=0).float()
        if x.shape[0] < 2:
            continue
        pred = model.layers[l](x).squeeze(-1).cpu()
        pred_soft = torch.softmax(pred, dim=-1)
        label = y / (y.sum() + 1e-8)
        per_layer_stats[l]["mse"].append(float(torch.mean((pred_soft - label) ** 2)))
        rho, _ = spearmanr(label.numpy(), pred_soft.detach().numpy())
        if rho == rho:
            per_layer_stats[l]["sp"].append(float(rho))
        k = max(1, int(label.numel() * 0.5))
        overlap = len(set(label.topk(k).indices.tolist()) & set(pred_soft.topk(k).indices.tolist())) / k
        per_layer_stats[l]["tk"].append(overlap)

    out = {}
    for l in range(n_layers):
        s = per_layer_stats.get(l, {"sp": [], "tk": [], "mse": []})
        out[l] = {
            "spearman": float(np.mean(s["sp"])) if s["sp"] else float("nan"),
            "topk_overlap_50": float(np.mean(s["tk"])) if s["tk"] else float("nan"),
            "mse": float(np.mean(s["mse"])) if s["mse"] else float("nan"),
        }
    return out


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard_dirs", nargs="+", required=True)
    p.add_argument("--teacher", choices=["pv", "future"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_layers_model", type=int, default=32)
    p.add_argument("--selected_layers", type=int, nargs="+", required=True)
    p.add_argument("--train_fraction", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_max_epochs", type=int, default=10)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--progress_file", default=None,
                   help="Optional file to write per-epoch progress (flushed immediately)")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    y_key = f"y_{args.teacher}"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [Path(d).resolve() for d in args.shard_dirs]
    shard_paths: list[Path] = []
    dir_idx_of_path: dict[Path, int] = {}
    for i, d in enumerate(shard_dirs):
        for sp in sorted((d / "shards").glob("shard_*.pt")):
            shard_paths.append(sp)
            dir_idx_of_path[sp] = i
    print(f"Shards: {len(shard_paths)} across {len(shard_dirs)} dirs")
    if not shard_paths:
        raise ValueError("No shards")

    SID_OFFSET = 100000
    # Sample IDs are deterministic per dataset: 0 .. n_samples-1 (from validation_report.json).
    # No need to scan shards (saves ~30min of full-shard disk reads).
    all_global_sids = set()
    for i, d in enumerate(shard_dirs):
        report_path = d / "validation_report.json"
        if report_path.exists():
            rep = json.loads(report_path.read_text())
            n_samples = int(rep.get("saved", rep.get("n_samples_requested", 0)))
        else:
            # fallback: peek shards to infer range
            n_samples = 0
            for sp in sorted((d / "shards").glob("shard_*.pt")):
                row = torch.load(sp, map_location="cpu", weights_only=False)
                n_samples = max(n_samples, int(row["sample_id"].max().item()) + 1)
        for local_sid in range(n_samples):
            all_global_sids.add(i * SID_OFFSET + local_sid)
        print(f"[sids] {d.name}: {n_samples} samples")

    all_global_sids = sorted(all_global_sids)
    rng = random.Random(args.seed)
    rng.shuffle(all_global_sids)
    n_train = max(1, int(len(all_global_sids) * args.train_fraction))
    train_sids = set(all_global_sids[:n_train])
    val_sids = set(all_global_sids[n_train:])
    print(f"Samples: total={len(all_global_sids)} train={len(train_sids)} val={len(val_sids)}")

    # Peek input_dim
    first = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    input_dim = int(first["x"].shape[-1])
    del first

    n_model = args.n_layers_model
    resolved = sorted(set(l % n_model for l in args.selected_layers))
    keep_layer = torch.zeros(n_model, dtype=torch.bool)
    for l in resolved:
        keep_layer[l] = True

    cfg = KVzapConfig(input_dim=input_dim, hidden_dim=args.mlp_hidden_dim, output_dim=1, n_modules=n_model)
    model = KVzapModel(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr)

    train_mask = _SplitMask(train_sids, y_key)
    val_mask = _SplitMask(val_sids, y_key)

    best_mean_spearman = -float("inf")
    best_state = None
    epoch_logs = []

    import time
    for epoch in range(args.mlp_max_epochs):
        model.train()
        n_steps = 0
        total_loss_acc = 0.0
        ep_start = time.time()
        n_total = len(shard_paths)
        last_write = ep_start
        for shard_idx, batch in enumerate(iter_shard_batches(
                shard_paths, dir_idx_of_path, SID_OFFSET, train_mask,
                shuffle=True, seed=args.seed + epoch)):
            now = time.time()
            if args.progress_file and (now - last_write >= 3.0 or shard_idx == n_total - 1):
                last_write = now
                pct = int(100 * (shard_idx + 1) / n_total)
                elapsed = now - ep_start
                eta = elapsed / max(shard_idx + 1, 1) * (n_total - shard_idx - 1)
                bar_fill = pct // 5  # 20-char bar
                bar = "█" * bar_fill + "░" * (20 - bar_fill)
                with open(args.progress_file, "a") as pf:
                    pf.write(f"ep{epoch+1}/{args.mlp_max_epochs} train [{bar}] {pct}% shard={shard_idx+1}/{n_total} elapsed={elapsed:.0f}s eta={eta:.0f}s\n")
                    pf.flush()
            # Filter to selected layers only
            layer = batch["layer"]
            keep = keep_layer[layer.long()]
            if not keep.any():
                continue
            x = batch["x"][keep].float().to(args.device)
            y = batch["y"][keep].float().to(args.device)
            l = layer[keep].to(args.device)
            sid = batch["sample_id"][keep].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            loss, n_groups = compute_batch_loss(model, x, y, l, sid)
            if n_groups == 0:
                continue
            (loss / n_groups).backward()
            optimizer.step()
            total_loss_acc += float(loss.item()) / n_groups
            n_steps += 1

        # Evaluate
        val_stats = eval_all_layers(
            model, shard_paths, dir_idx_of_path, SID_OFFSET, val_sids, y_key,
            args.device, n_model,
        )
        sp_vals = [val_stats[l]["spearman"] for l in resolved if val_stats[l]["spearman"] == val_stats[l]["spearman"]]
        mean_sp = float(np.mean(sp_vals)) if sp_vals else float("nan")
        msg = f"Epoch {epoch + 1}/{args.mlp_max_epochs}: avg_train_loss={total_loss_acc/max(n_steps,1):.6f} val_mean_spearman={mean_sp:.4f}"
        print(msg)
        epoch_logs.append({"epoch": epoch + 1, "train_loss": total_loss_acc/max(n_steps,1), "val_mean_spearman": mean_sp})
        if args.progress_file:
            with open(args.progress_file, "a") as pf:
                pf.write(msg + "\n")
                pf.flush()

        if mean_sp > best_mean_spearman:
            best_mean_spearman = mean_sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})

    # Final eval
    final_stats = eval_all_layers(
        model, shard_paths, dir_idx_of_path, SID_OFFSET, val_sids, y_key,
        args.device, n_model,
    )

    model.cpu()
    model.save_pretrained(str(out_dir))

    run_config = {
        "teacher": args.teacher,
        "shard_dirs": [str(d) for d in shard_dirs],
        "n_layers_model": n_model,
        "selected_layers": args.selected_layers,
        "resolved_layers": resolved,
        "input_dim": input_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_max_epochs": args.mlp_max_epochs,
        "mlp_lr": args.mlp_lr,
        "n_train_samples": len(train_sids),
        "n_val_samples": len(val_sids),
        "seed": args.seed,
        "best_val_mean_spearman": best_mean_spearman,
        "metrics_per_layer": [{"layer": l, **final_stats[l]} for l in resolved],
        "epoch_logs": epoch_logs,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"Saved to: {out_dir}")
    print(f"Best mean spearman: {best_mean_spearman:.4f}")


if __name__ == "__main__":
    main()
