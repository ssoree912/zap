#!/usr/bin/env python3
"""Train a future-supervised token importance MLP for image KV pruning.

Input records (from collect_future_supervised_labels.py):
  hidden_image  [n_sel_layers, N_image, D]  prefill hidden at image positions
  future_splus  [n_sel_layers, N_image]     decode→image attention teacher label

The label records contain n_sel_layers (e.g. 4 selected layers), but the trained
KVzapModel has one MLP per full model layer (n_layers_model).  Training maps:
  selected layer index i → full model layer index resolved_layers[i].

Output checkpoint is a KVzapModel saved via HuggingFace (from_pretrained compatible).
Validation metrics: MSE, Spearman ρ, Top-k overlap.

Usage:
  python train_future_supervised_probe.py \\
    --data_dirs /workspace/zap/artifacts/future_teacher/textvqa \\
                /workspace/zap/artifacts/future_teacher/scienceqa \\
    --out_dir /workspace/zap/ckpts/future_probe_v1 \\
    --n_layers_model 32 \\
    --selected_layers -4 -3 -2 -1 \\
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel


# ── dataset ─────────────────────────────────────────────────────────────────

def _load_layer_from_paths(
    paths: list[Path],
    layer_local_idx: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load records from disk extracting only the specified layer's (hidden, label).

    Frees each full record immediately after slicing so peak memory stays at
    ~(num_records × N_image × D × fp16) instead of 32× that.
    """
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for p in paths:
        rec = torch.load(p, map_location="cpu", weights_only=False)
        hidden = rec["hidden_image"][layer_local_idx].float().clone()
        label = rec["future_splus"][layer_local_idx].float().clone()
        del rec
        if hidden.shape[0] != label.shape[0] or hidden.shape[0] < 2:
            continue
        label = label / (label.sum() + 1e-8)
        out.append((hidden, label))
    if not out:
        raise ValueError(f"No valid samples for layer_local_idx={layer_local_idx}")
    return out


class FutureProbeDataset(Dataset):
    """Sample-level dataset over pre-extracted per-layer (hidden, label) pairs."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ── metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_metrics(
    mlp: nn.Module,
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
    topk_ratio: float = 0.5,
) -> dict[str, float]:
    """Per-sample Spearman ρ and Top-k overlap, then averaged.

    samples: list of (hidden [N_image, D], label [N_image] already sum-to-1 normalized).
    """
    mlp.eval()
    spearmans: list[float] = []
    topk_overlaps: list[float] = []
    mse_vals: list[float] = []

    for hidden_cpu, label_norm in samples:
        hidden = hidden_cpu.to(device)
        N = hidden.shape[0]
        if N < 2:
            continue

        pred = mlp(hidden).squeeze(-1).cpu()                              # [N_image]
        pred_soft = torch.softmax(pred, dim=-1)

        mse_vals.append(float(torch.mean((pred_soft - label_norm) ** 2)))

        rho, _ = spearmanr(label_norm.numpy(), pred_soft.detach().numpy())
        spearmans.append(float(rho))

        k = max(1, int(N * topk_ratio))
        teacher_topk = set(label_norm.topk(k).indices.tolist())
        pred_topk = set(pred_soft.topk(k).indices.tolist())
        overlap = len(teacher_topk & pred_topk) / k
        topk_overlaps.append(overlap)

    return {
        "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
        "spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        f"topk_overlap_{int(topk_ratio*100)}": float(np.mean(topk_overlaps)) if topk_overlaps else float("nan"),
    }


# ── training ─────────────────────────────────────────────────────────────────

def _collate_samples(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Keep as a list of (hidden [N_i, D], label [N_i]) — no padding needed since loss is per-sample.
    return batch


def train_one_layer(
    train_samples: list[tuple[torch.Tensor, torch.Tensor]],
    val_samples: list[tuple[torch.Tensor, torch.Tensor]],
    input_dim: int,
    hidden_dim: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[nn.Sequential, dict[str, float]]:
    train_ds = FutureProbeDataset(train_samples)
    # batch_size here = number of samples per gradient step (not number of tokens).
    # Each sample has variable N_image tokens; loss is averaged over tokens within each sample.
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=_collate_samples,
    )

    # Architecture matches KVzapModel.layers[i] exactly (Sequential: Linear→GELU→Linear)
    # so weights can be copied directly without shape mismatch.
    # LayerNorm is intentionally excluded to preserve KVzapModel compatibility.
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_spearman = -float("inf")
    best_state: dict = {}

    for epoch in range(max_epochs):
        mlp.train()
        for sample_batch in train_loader:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for x_sample, y_sample in sample_batch:
                x_sample = x_sample.to(device)  # [N_image, D]
                y_sample = y_sample.to(device)   # [N_image]
                # softmax over N_image tokens within this sample — correct normalization
                pred = torch.softmax(mlp(x_sample).squeeze(-1), dim=-1)  # [N_image]
                total_loss = total_loss + loss_fn(pred, y_sample)
            (total_loss / len(sample_batch)).backward()
            optimizer.step()

        if val_samples:
            val_metrics = evaluate_metrics(mlp, val_samples, device)
            spearman = val_metrics["spearman"]
            if spearman > best_val_spearman:
                best_val_spearman = spearman
                best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}

    if best_state:
        mlp.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_metrics = evaluate_metrics(mlp, val_samples, device) if val_samples else {}
    return mlp, final_metrics


# ── record loading ────────────────────────────────────────────────────────────

def collect_record_paths(
    data_dirs: list[Path],
    limit: int | None = None,
    per_dir_limit: int | None = None,
) -> list[Path]:
    """Return paths to record .pt files (no loading)."""
    paths: list[Path] = []
    for d in data_dirs:
        rec_dir = d / "records"
        if not rec_dir.is_dir():
            print(f"[WARN] records dir not found: {rec_dir}")
            continue
        n_from_dir = 0
        for pt_path in sorted(rec_dir.glob("*.pt")):
            if per_dir_limit is not None and n_from_dir >= per_dir_limit:
                break
            paths.append(pt_path)
            n_from_dir += 1
        print(f"[paths] {d.name}: {n_from_dir} records")
    if limit is not None:
        paths = paths[:limit]
    return paths


def peek_record_shape(path: Path) -> tuple[int, int]:
    """Return (n_sel_layers, input_dim) without holding the full tensor."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    n_sel = int(rec["hidden_image"].shape[0])
    input_dim = int(rec["hidden_image"].shape[2])
    del rec
    return n_sel, input_dim


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_layers_model", type=int, default=32,
                        help="Total number of transformer layers in LLaVA (e.g. 32)")
    parser.add_argument("--selected_layers", type=int, nargs="+", default=[-4, -3, -2, -1],
                        help="Layer indices used during collection (must match collect script)")
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per_dir_limit", type=int, default=None,
                        help="Take at most this many records per data_dir before shuffling")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_max_epochs", type=int, default=10)
    parser.add_argument("--mlp_batch_size", type=int, default=2048)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--topk_ratio_eval", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dirs = [Path(d).resolve() for d in args.data_dirs]
    print(f"Collecting record paths from: {[str(d) for d in data_dirs]}")
    all_paths = collect_record_paths(data_dirs, limit=args.limit, per_dir_limit=args.per_dir_limit)
    if not all_paths:
        raise ValueError("No record paths found")

    print(f"Found {len(all_paths)} records")

    # Peek first record to validate n_sel_layers & input_dim
    n_sel_layers, input_dim = peek_record_shape(all_paths[0])
    if len(args.selected_layers) != n_sel_layers:
        raise ValueError(
            f"selected_layers length {len(args.selected_layers)} != "
            f"record n_sel_layers {n_sel_layers}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(all_paths)
    n_train = max(1, int(len(all_paths) * args.train_fraction))
    train_paths = all_paths[:n_train]
    val_paths = all_paths[n_train:]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Resolved model-layer indices (e.g. -1 → 31 for 32-layer model)
    n_model = args.n_layers_model
    resolved = [l % n_model for l in args.selected_layers]

    # Build KVzapModel (one MLP per model layer; only selected layers are trained)
    # Unselected layers get identity-like default weights and are not used during inference
    # when using HybridImageTeacherPress with layer_idx matching.
    kvzap_cfg = KVzapConfig(
        input_dim=input_dim,
        hidden_dim=args.mlp_hidden_dim,
        output_dim=1,
        n_modules=n_model,
    )
    kvzap_model = KVzapModel(kvzap_cfg)

    all_metrics: list[dict] = []

    for local_idx, model_layer_idx in enumerate(resolved):
        print(f"\n── Layer {model_layer_idx} (local {local_idx}) ──")
        # Lazy per-layer load — frees prev layer's tensors when this one is rebuilt
        train_samples = _load_layer_from_paths(train_paths, local_idx)
        val_samples = _load_layer_from_paths(val_paths, local_idx) if val_paths else []
        mlp, val_metrics = train_one_layer(
            train_samples=train_samples,
            val_samples=val_samples,
            input_dim=input_dim,
            hidden_dim=args.mlp_hidden_dim,
            max_epochs=args.mlp_max_epochs,
            batch_size=args.mlp_batch_size,
            lr=args.mlp_lr,
            device=args.device,
        )
        print(f"  val: {val_metrics}")
        # Free layer tensors before next iteration
        del train_samples, val_samples

        # Copy trained weights into the correct model-layer slot.
        # mlp architecture: Sequential(Linear(d,h)[0], GELU[1], Linear(h,1)[2])
        # KVzapModel.layers[i]: Sequential(Linear(d,h)[0], GELU[1], Linear(h,1)[2])
        # Architectures are identical so we can load state dict directly.
        trained_state = {k: v.cpu() for k, v in mlp.state_dict().items()}
        kvzap_model.layers[model_layer_idx].load_state_dict(trained_state)

        all_metrics.append({"layer": model_layer_idx, "local_idx": local_idx, **val_metrics})

    kvzap_model.save_pretrained(str(out_dir))

    run_config = {
        "data_dirs": [str(d) for d in data_dirs],
        "n_layers_model": n_model,
        "selected_layers": args.selected_layers,
        "resolved_layers": resolved,
        "input_dim": input_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_max_epochs": args.mlp_max_epochs,
        "mlp_batch_size": args.mlp_batch_size,
        "mlp_lr": args.mlp_lr,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "seed": args.seed,
        "metrics_per_layer": all_metrics,
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    # Summary
    spearmans = [m["spearman"] for m in all_metrics if "spearman" in m]
    overlaps = [m.get(f"topk_overlap_{int(args.topk_ratio_eval*100)}", float("nan")) for m in all_metrics]
    print(f"\n── Summary ──")
    print(f"  Mean Spearman ρ : {np.mean(spearmans):.4f}")
    print(f"  Mean Top-k overlap (@{int(args.topk_ratio_eval*100)}%): {np.nanmean(overlaps):.4f}")
    print(f"  Checkpoint saved to: {out_dir}")
    print(f"\n── Press initialization (copy-paste) ──")
    print(f"  FutureSupervisedImagePress(")
    print(f"      probe_model_name='{out_dir}',")
    print(f"      selected_layer_indices={tuple(resolved)},")
    print(f"      image_keep_ratio=0.5,")
    print(f"  )")
    print(f"  HybridImageTeacherPress(")
    print(f"      future_probe_name='{out_dir}',")
    print(f"      selected_layer_indices={tuple(resolved)},")
    print(f"      alpha=0.5,")
    print(f"      image_keep_ratio=0.5,")
    print(f"  )")


if __name__ == "__main__":
    main()
