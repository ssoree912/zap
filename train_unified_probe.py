#!/usr/bin/env python3
"""Train a per-layer image-importance MLP using method B (per-sample softmax MSE).

Input: shards produced by collect_unified_teacher_shards.py
  x, y_pv, y_future, layer, sample_id, token_idx

Loss per (sample_id, layer) group:
    pred  = softmax(mlp[layer](x_group), dim=0)     # sum-to-1 over image tokens in sample
    label = y_group / y_group.sum()                 # sum-to-1
    loss  = MSE(pred, label)

--teacher choice controls which scalar to use: pv or future.

Output: KVzapModel checkpoint (same format as existing probes) at --out_dir.

Usage:
  python train_unified_probe.py \\
      --shard_dirs /workspace/zap/artifacts/teacher/unified/textvqa \\
                   /workspace/zap/artifacts/teacher/unified/scienceqa \\
                   /workspace/zap/artifacts/teacher/unified/nlvr2 \\
      --teacher future --out_dir /workspace/zap/ckpts/future_probe_v3 \\
      --selected_layers 0 1 2 ... 31 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel


# ── shard loading ──────────────────────────────────────────────────────────

def load_shard_paths(shard_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for d in shard_dirs:
        sd = d / "shards"
        if not sd.is_dir():
            print(f"[WARN] no shards dir: {sd}")
            continue
        paths.extend(sorted(sd.glob("shard_*.pt")))
    return paths


def load_layer_from_shards(
    shard_paths: list[Path],
    layer_idx: int,
    y_key: str,
    split_sample_ids: set[int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return list of (x_sample [N, D], y_sample [N]) grouped by sample_id for one layer.

    Each entry corresponds to one sample; labels already normalized (sum-to-1).
    Only samples whose sample_id ∈ split_sample_ids are returned (if given).
    """
    groups: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"x": [], "y": []})
    for sp in shard_paths:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        mask = d["layer"] == layer_idx
        if split_sample_ids is not None:
            sid_tensor = d["sample_id"][mask]
            sid_np = sid_tensor.numpy()
            keep = np.isin(sid_np, list(split_sample_ids))
            mask_idx = mask.nonzero(as_tuple=True)[0][torch.from_numpy(keep)]
        else:
            mask_idx = mask.nonzero(as_tuple=True)[0]
        if mask_idx.numel() == 0:
            continue
        xs = d["x"][mask_idx].float()
        ys = d[y_key][mask_idx].float()
        sids = d["sample_id"][mask_idx]
        # group by sample_id
        for sid in torch.unique(sids):
            m = sids == sid
            groups[int(sid)]["x"].append(xs[m])
            groups[int(sid)]["y"].append(ys[m])

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sid, parts in groups.items():
        x = torch.cat(parts["x"], dim=0)
        y = torch.cat(parts["y"], dim=0)
        if x.shape[0] < 2:
            continue
        y_norm = y / (y.sum() + 1e-8)
        out.append((x, y_norm))
    return out


def collect_sample_ids(shard_paths: list[Path]) -> list[int]:
    sids = set()
    for sp in shard_paths:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        sids.update(int(s) for s in torch.unique(d["sample_id"]).tolist())
    return sorted(sids)


# ── metrics & training ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_metrics(mlp: nn.Module, samples, device: str, topk_ratio: float = 0.5) -> dict[str, float]:
    mlp.eval()
    sp_vals, tk_vals, mse_vals = [], [], []
    for x_cpu, y_norm in samples:
        x = x_cpu.to(device)
        pred = mlp(x).squeeze(-1).cpu()
        pred_soft = torch.softmax(pred, dim=-1)
        mse_vals.append(float(torch.mean((pred_soft - y_norm) ** 2)))
        rho, _ = spearmanr(y_norm.numpy(), pred_soft.detach().numpy())
        if rho == rho:
            sp_vals.append(float(rho))
        k = max(1, int(y_norm.numel() * topk_ratio))
        overlap = len(set(y_norm.topk(k).indices.tolist()) & set(pred_soft.topk(k).indices.tolist())) / k
        tk_vals.append(overlap)
    return {
        "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
        "spearman": float(np.mean(sp_vals)) if sp_vals else float("nan"),
        f"topk_overlap_{int(topk_ratio * 100)}": float(np.mean(tk_vals)) if tk_vals else float("nan"),
    }


def train_one_layer(
    train_samples, val_samples, input_dim: int, hidden_dim: int,
    max_epochs: int, batch_size: int, lr: float, device: str,
) -> tuple[nn.Sequential, dict[str, float]]:
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_sp, best_state = -float("inf"), {}

    for epoch in range(max_epochs):
        mlp.train()
        perm = list(range(len(train_samples)))
        random.shuffle(perm)
        for start in range(0, len(perm), batch_size):
            batch = [train_samples[i] for i in perm[start:start + batch_size]]
            optimizer.zero_grad()
            tot = torch.zeros(1, device=device)
            for x_cpu, y_cpu in batch:
                x = x_cpu.to(device); y = y_cpu.to(device)
                pred = torch.softmax(mlp(x).squeeze(-1), dim=-1)
                tot = tot + loss_fn(pred, y)
            (tot / len(batch)).backward()
            optimizer.step()

        if val_samples:
            m = evaluate_metrics(mlp, val_samples, device)
            if m["spearman"] > best_sp:
                best_sp = m["spearman"]
                best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}

    if best_state:
        mlp.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_metrics = evaluate_metrics(mlp, val_samples, device) if val_samples else {}
    return mlp, final_metrics


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--teacher", choices=["pv", "future"], required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_layers_model", type=int, default=32)
    parser.add_argument("--selected_layers", type=int, nargs="+", required=True)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_max_epochs", type=int, default=10)
    parser.add_argument("--mlp_batch_size", type=int, default=32)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [Path(d).resolve() for d in args.shard_dirs]
    shard_paths = load_shard_paths(shard_dirs)
    if not shard_paths:
        raise ValueError("No shards found")
    print(f"Shards: {len(shard_paths)} from {len(shard_dirs)} dirs")

    # Split sample IDs into train/val (deterministic by sample_id).
    # Since sample_id is NOT globally unique across datasets (each dataset starts at 0),
    # split by path of shard dir * sample_id to avoid cross-dataset collisions.
    # Simpler: prefix sample_id by dataset index.
    # For now just collect all unique sample_ids per shard (dataset_dir index added).
    all_sids: list[int] = []
    per_dir_sids: dict[int, set[int]] = {}
    offset = 0
    for i, d in enumerate(shard_dirs):
        dir_paths = sorted((d / "shards").glob("shard_*.pt"))
        for sp in dir_paths:
            loaded = torch.load(sp, map_location="cpu", weights_only=False)
            u = torch.unique(loaded["sample_id"]).tolist()
            shifted = [int(s) + offset for s in u]
            # remap sample_id in shard in-memory (for later loading pass too)
            # easiest: re-save shards with shifted ids
        offset += 10000  # large gap per dataset
    # Actually simpler: let each shard's sample_id be treated with (shard_path → ds_index) pair
    # but for grouping we need unique sids. Remap inline via (dir_idx * 100000 + sid):
    dir_of_path: dict[Path, int] = {}
    for i, d in enumerate(shard_dirs):
        for sp in (d / "shards").glob("shard_*.pt"):
            dir_of_path[sp] = i

    # Build global sid set
    all_global_sids = set()
    per_path_cache: dict[Path, torch.Tensor] = {}
    for sp in shard_paths:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        ds_idx = dir_of_path[sp]
        shifted = d["sample_id"].int() + int(ds_idx * 100000)
        per_path_cache[sp] = shifted
        all_global_sids.update(int(s) for s in torch.unique(shifted).tolist())

    all_global_sids = sorted(all_global_sids)
    rng = random.Random(args.seed)
    rng.shuffle(all_global_sids)
    n_train = max(1, int(len(all_global_sids) * args.train_fraction))
    train_sids = set(all_global_sids[:n_train])
    val_sids = set(all_global_sids[n_train:])
    print(f"Samples: total={len(all_global_sids)} train={len(train_sids)} val={len(val_sids)}")

    def load_layer(layer_idx: int, sid_set: set[int]):
        groups: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"x": [], "y": []})
        for sp in shard_paths:
            d = torch.load(sp, map_location="cpu", weights_only=False)
            gsid = per_path_cache[sp]
            mask = (d["layer"] == layer_idx)
            sids_here = gsid[mask]
            xs_here = d["x"][mask].float()
            ys_here = d[f"y_{args.teacher}"][mask].float()
            keep_mask = torch.tensor([int(s) in sid_set for s in sids_here.tolist()])
            if not keep_mask.any():
                continue
            for sid in torch.unique(sids_here[keep_mask]):
                m = sids_here == sid
                groups[int(sid)]["x"].append(xs_here[m])
                groups[int(sid)]["y"].append(ys_here[m])
        out = []
        for parts in groups.values():
            x = torch.cat(parts["x"], dim=0)
            y = torch.cat(parts["y"], dim=0)
            if x.shape[0] < 2:
                continue
            y_norm = y / (y.sum() + 1e-8)
            out.append((x, y_norm))
        return out

    # Peek dim
    first = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    input_dim = int(first["x"].shape[-1])
    del first

    n_model = args.n_layers_model
    resolved = [l % n_model for l in args.selected_layers]

    kvzap_cfg = KVzapConfig(input_dim=input_dim, hidden_dim=args.mlp_hidden_dim, output_dim=1, n_modules=n_model)
    kvzap_model = KVzapModel(kvzap_cfg)

    all_metrics = []
    for model_layer in tqdm(resolved, desc=f"{args.teacher} layers"):
        train_samples = load_layer(model_layer, train_sids)
        val_samples = load_layer(model_layer, val_sids) if val_sids else []
        mlp, val_metrics = train_one_layer(
            train_samples, val_samples, input_dim, args.mlp_hidden_dim,
            args.mlp_max_epochs, args.mlp_batch_size, args.mlp_lr, args.device,
        )
        print(f"  layer {model_layer}: {val_metrics}")
        kvzap_model.layers[model_layer].load_state_dict({k: v.cpu() for k, v in mlp.state_dict().items()})
        all_metrics.append({"layer": model_layer, **val_metrics})
        del train_samples, val_samples

    kvzap_model.save_pretrained(str(out_dir))

    run_config = {
        "teacher": args.teacher,
        "shard_dirs": [str(d) for d in shard_dirs],
        "n_layers_model": n_model,
        "selected_layers": args.selected_layers,
        "resolved_layers": resolved,
        "input_dim": input_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_max_epochs": args.mlp_max_epochs,
        "mlp_batch_size": args.mlp_batch_size,
        "mlp_lr": args.mlp_lr,
        "n_train_samples": len(train_sids),
        "n_val_samples": len(val_sids),
        "seed": args.seed,
        "metrics_per_layer": all_metrics,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    sp_vals = [m["spearman"] for m in all_metrics if "spearman" in m]
    print(f"\nMean Spearman ρ ({args.teacher}): {np.mean(sp_vals):.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
