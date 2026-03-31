#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel
from kvzap.image_teacher_utils import (
    iter_common_sample_ids,
    load_pt_record,
    resolve_records_dir,
    resolve_teacher_dir,
    split_sample_ids,
)


class LayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def _get_input_hidden_for_layer(hidden_prompt: Sequence[torch.Tensor], layer_idx: int, n_layers: int) -> torch.Tensor:
    if len(hidden_prompt) == n_layers + 1:
        return hidden_prompt[layer_idx]
    if len(hidden_prompt) == n_layers:
        return hidden_prompt[layer_idx]
    raise ValueError(f"Unexpected hidden_prompt length: {len(hidden_prompt)} for n_layers={n_layers}")



def load_layer_dataset(
    extractor_dir: str | Path,
    teacher_dir: str | Path,
    sample_ids: Sequence[str],
    layer_idx: int,
    target_score_name: str,
    n_layers: int,
    max_image_tokens_per_sample: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    extractor_records = resolve_records_dir(extractor_dir)
    teacher_records = resolve_teacher_dir(teacher_dir)
    rng = np.random.default_rng(seed + layer_idx)

    all_x = []
    all_y = []
    for sample_id in sample_ids:
        extractor = load_pt_record(extractor_records / f"{sample_id}.pt")
        teacher = load_pt_record(teacher_records / f"{sample_id}.pt")

        image_pos = teacher["image_pos_mm"].long()
        score_tensor = teacher[target_score_name][layer_idx].transpose(0, 1).float()  # [I, H]
        hidden_prompt = extractor["hidden_prompt"]
        hidden_layer = _get_input_hidden_for_layer(hidden_prompt, layer_idx, n_layers).float()
        x = hidden_layer[image_pos]  # [I, D]

        if x.shape[0] != score_tensor.shape[0]:
            raise ValueError(
                f"Hidden/score image length mismatch for sample {sample_id}, layer {layer_idx}: {x.shape[0]} vs {score_tensor.shape[0]}"
            )

        if max_image_tokens_per_sample is not None and x.shape[0] > max_image_tokens_per_sample:
            indices = rng.choice(x.shape[0], size=max_image_tokens_per_sample, replace=False)
            indices = np.sort(indices)
            x = x[indices]
            score_tensor = score_tensor[indices]

        all_x.append(x)
        all_y.append(score_tensor)

    if not all_x:
        raise ValueError("No training samples available")

    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)



def train_linear_layer(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
    model = Ridge()
    model.fit(X_train.float().numpy(), y_train.float().numpy())
    pred = model.predict(X_test.float().numpy())
    mse = float(np.mean((pred - y_test.float().numpy()) ** 2))
    return np.atleast_2d(model.coef_), np.atleast_1d(model.intercept_), mse



def train_mlp_layer(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_dim: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    val_fraction: float = 0.05,
) -> tuple[dict, float]:
    model = LayerMLP(X_train.shape[1], hidden_dim, y_train.shape[1]).to(device)
    dataset = TensorDataset(X_train, y_train)
    if len(dataset) > 1:
        n_val = max(1, int(round(len(dataset) * val_fraction)))
        n_val = min(n_val, len(dataset) - 1)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_val_items = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                batch_loss = loss_fn(pred, yb).item()
                val_loss += batch_loss * xb.shape[0]
                n_val_items += xb.shape[0]
        val_loss = val_loss / max(n_val_items, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device)).cpu()
    mse = float(torch.mean((pred - y_test) ** 2).item())
    net_state = {
        key.removeprefix("net."): value
        for key, value in best_state.items()
    }
    return net_state, mse



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor_dir", type=str, required=True)
    parser.add_argument("--teacher_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_score_name", type=str, default="splus_postvision")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_image_tokens_per_sample", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=["linear", "mlp"])
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_max_epochs", type=int, default=10)
    parser.add_argument("--mlp_batch_size", type=int, default=2048)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = iter_common_sample_ids(args.extractor_dir, args.teacher_dir)
    if not sample_ids:
        raise ValueError("No overlapping sample ids found between extractor and teacher records")
    train_ids, test_ids = split_sample_ids(sample_ids, train_fraction=args.train_fraction, seed=args.seed)

    teacher_root = resolve_teacher_dir(args.teacher_dir)
    extractor_root = resolve_records_dir(args.extractor_dir)
    first_teacher = load_pt_record(teacher_root / f"{sample_ids[0]}.pt")
    first_extractor = load_pt_record(extractor_root / f"{sample_ids[0]}.pt")
    target = first_teacher[args.target_score_name]
    n_layers, output_dim, _ = target.shape
    hidden_prompt = first_extractor["hidden_prompt"]
    input_dim = _get_input_hidden_for_layer(hidden_prompt, 0, n_layers).shape[-1]

    methods = set(args.methods)
    linear_model = None
    mlp_model = None
    if "linear" in methods:
        linear_model = KVzapModel(KVzapConfig(input_dim=input_dim, hidden_dim=None, output_dim=output_dim, n_modules=n_layers))
    if "mlp" in methods:
        mlp_model = KVzapModel(KVzapConfig(input_dim=input_dim, hidden_dim=args.mlp_hidden_dim, output_dim=output_dim, n_modules=n_layers))

    metrics = []
    for layer_idx in tqdm(range(n_layers), desc="Training image-teacher probes"):
        X_train, y_train = load_layer_dataset(
            args.extractor_dir,
            args.teacher_dir,
            train_ids,
            layer_idx,
            args.target_score_name,
            n_layers,
            max_image_tokens_per_sample=args.max_image_tokens_per_sample,
            seed=args.seed,
        )
        X_test, y_test = load_layer_dataset(
            args.extractor_dir,
            args.teacher_dir,
            test_ids,
            layer_idx,
            args.target_score_name,
            n_layers,
            max_image_tokens_per_sample=args.max_image_tokens_per_sample,
            seed=args.seed,
        )

        if linear_model is not None:
            W, b, mse = train_linear_layer(X_train, y_train, X_test, y_test)
            linear_model.layers[layer_idx].weight.data = torch.tensor(W, dtype=torch.float32)
            linear_model.layers[layer_idx].bias.data = torch.tensor(b, dtype=torch.float32)
            metrics.append(
                {
                    "method": "linear",
                    "layer": layer_idx,
                    "n_train_tokens": int(X_train.shape[0]),
                    "n_test_tokens": int(X_test.shape[0]),
                    "test_mse": mse,
                }
            )

        if mlp_model is not None:
            state_dict, mse = train_mlp_layer(
                X_train,
                y_train,
                X_test,
                y_test,
                hidden_dim=args.mlp_hidden_dim,
                max_epochs=args.mlp_max_epochs,
                batch_size=args.mlp_batch_size,
                lr=args.mlp_lr,
                device=args.device,
            )
            mlp_model.layers[layer_idx].load_state_dict(state_dict)
            metrics.append(
                {
                    "method": "mlp",
                    "layer": layer_idx,
                    "n_train_tokens": int(X_train.shape[0]),
                    "n_test_tokens": int(X_test.shape[0]),
                    "test_mse": mse,
                }
            )

    if linear_model is not None:
        linear_dir = output_dir / "linear"
        linear_dir.mkdir(parents=True, exist_ok=True)
        linear_model.save_pretrained(linear_dir)
    if mlp_model is not None:
        mlp_dir = output_dir / "mlp"
        mlp_dir.mkdir(parents=True, exist_ok=True)
        mlp_model.save_pretrained(mlp_dir)

    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "split.json").open("w") as f:
        json.dump({"train_ids": train_ids, "test_ids": test_ids}, f, indent=2)
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved probe models to {output_dir}")
    print(f"Train samples: {len(train_ids)} | Test samples: {len(test_ids)}")


if __name__ == "__main__":
    main()
