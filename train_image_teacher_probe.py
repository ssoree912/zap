#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
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



def _resolve_image_positions(teacher: dict[str, Any]) -> torch.Tensor:
    if "image_pos_mm" in teacher:
        return teacher["image_pos_mm"].long().flatten()
    if "image_indices_mm" in teacher:
        return teacher["image_indices_mm"].long().flatten()
    if "is_image_pos_mm" in teacher:
        return teacher["is_image_pos_mm"].nonzero(as_tuple=False).flatten().long()
    raise KeyError("Teacher record must contain one of: image_pos_mm, image_indices_mm, is_image_pos_mm")



def _format_layer_targets(layer_scores: torch.Tensor, n_image_tokens: int) -> torch.Tensor:
    if layer_scores.ndim == 1:
        if int(layer_scores.shape[0]) != n_image_tokens:
            raise ValueError(
                f"Target/image length mismatch: target={tuple(layer_scores.shape)}, n_image_tokens={n_image_tokens}"
            )
        return layer_scores.unsqueeze(-1)  # [I, 1]

    if layer_scores.ndim == 2:
        if int(layer_scores.shape[0]) == n_image_tokens:
            return layer_scores  # [I, H]
        if int(layer_scores.shape[1]) == n_image_tokens:
            return layer_scores.transpose(0, 1)  # [I, H]
        raise ValueError(
            "Could not align layer target shape with image tokens: "
            f"target={tuple(layer_scores.shape)}, n_image_tokens={n_image_tokens}"
        )

    raise ValueError(f"Unsupported layer target ndim={layer_scores.ndim}; expected 1D or 2D")



def _apply_target_transform(y: torch.Tensor, target_transform: str, target_eps: float) -> torch.Tensor:
    if target_transform == "none":
        return y
    if target_transform == "log":
        return torch.log(y.clamp_min(0) + target_eps)
    raise ValueError(f"Unsupported target_transform: {target_transform}")



def _extract_image_hidden_for_layer(
    extractor: dict[str, Any],
    teacher: dict[str, Any],
    layer_idx: int,
    n_layers: int,
) -> torch.Tensor:
    if "hidden_prompt" in extractor:
        image_pos = _resolve_image_positions(teacher)
        hidden_prompt = extractor["hidden_prompt"]
        hidden_layer = _get_input_hidden_for_layer(hidden_prompt, layer_idx, n_layers).float()
        return hidden_layer[image_pos]

    hidden_image = extractor.get("hidden_image")
    if hidden_image is None:
        hidden_image = teacher.get("hidden_image")
    if isinstance(hidden_image, torch.Tensor):
        if hidden_image.ndim != 3:
            raise ValueError(f"hidden_image must be [L, I, D], got {tuple(hidden_image.shape)}")
        return hidden_image[layer_idx].float()

    raise KeyError(
        "Could not find image-token hidden inputs. Expected extractor.hidden_prompt "
        "or extractor/teacher hidden_image"
    )



def load_layer_dataset(
    extractor_dir: str | Path,
    teacher_dir: str | Path,
    sample_ids: Sequence[str],
    layer_idx: int,
    target_score_name: str,
    n_layers: int,
    max_image_tokens_per_sample: int | None = None,
    seed: int = 42,
    target_transform: str = "none",
    target_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    extractor_records = resolve_records_dir(extractor_dir)
    teacher_records = resolve_teacher_dir(teacher_dir)
    rng = np.random.default_rng(seed + layer_idx)

    all_x = []
    all_y = []
    for sample_id in sample_ids:
        extractor = load_pt_record(extractor_records / f"{sample_id}.pt")
        teacher = load_pt_record(teacher_records / f"{sample_id}.pt")

        x = _extract_image_hidden_for_layer(
            extractor=extractor,
            teacher=teacher,
            layer_idx=layer_idx,
            n_layers=n_layers,
        )  # [I, D]

        layer_scores = teacher[target_score_name][layer_idx].float()
        score_tensor = _format_layer_targets(layer_scores, n_image_tokens=int(x.shape[0]))
        score_tensor = _apply_target_transform(score_tensor, target_transform=target_transform, target_eps=target_eps)

        if x.shape[0] != score_tensor.shape[0]:
            raise ValueError(
                f"Hidden/score image length mismatch for sample {sample_id}, layer {layer_idx}: "
                f"{x.shape[0]} vs {score_tensor.shape[0]}"
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



def train_linear_layer(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with torch.no_grad():
        X_train_f = X_train.float().to(device)
        y_train_f = y_train.float().to(device)
        X_test_f = X_test.float().to(device)
        y_test_f = y_test.float().to(device)

        ones_train = torch.ones((X_train_f.shape[0], 1), device=device, dtype=X_train_f.dtype)
        ones_test = torch.ones((X_test_f.shape[0], 1), device=device, dtype=X_test_f.dtype)
        X_train_aug = torch.cat([X_train_f, ones_train], dim=1)
        X_test_aug = torch.cat([X_test_f, ones_test], dim=1)

        reg = torch.eye(X_train_aug.shape[1], device=device, dtype=X_train_f.dtype) * ridge_alpha
        reg[-1, -1] = 0.0
        beta = torch.linalg.solve(X_train_aug.T @ X_train_aug + reg, X_train_aug.T @ y_train_f)
        pred = X_test_aug @ beta
        mse = float(torch.mean((pred - y_test_f) ** 2).item())

    weight = beta[:-1].transpose(0, 1).contiguous().cpu().numpy()
    bias = beta[-1].contiguous().cpu().numpy()
    return np.atleast_2d(weight), np.atleast_1d(bias), mse



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
    parser.add_argument("--target_transform", choices=["none", "log"], default="none")
    parser.add_argument("--target_eps", type=float, default=1e-8)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_image_tokens_per_sample", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=["linear", "mlp"])
    parser.add_argument("--linear_alpha", type=float, default=1.0)
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
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Teacher target `{args.target_score_name}` must be a tensor")
    if target.ndim not in (2, 3):
        raise ValueError(
            f"Teacher target `{args.target_score_name}` must be [L,I] or [L,H,I], got {tuple(target.shape)}"
        )

    n_layers = int(target.shape[0])

    first_x = _extract_image_hidden_for_layer(
        extractor=first_extractor,
        teacher=first_teacher,
        layer_idx=0,
        n_layers=n_layers,
    )
    first_layer_target = _format_layer_targets(target[0].float(), n_image_tokens=int(first_x.shape[0]))
    first_layer_target = _apply_target_transform(
        first_layer_target,
        target_transform=args.target_transform,
        target_eps=args.target_eps,
    )

    input_dim = int(first_x.shape[-1])
    output_dim = int(first_layer_target.shape[-1])

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
            target_transform=args.target_transform,
            target_eps=args.target_eps,
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
            target_transform=args.target_transform,
            target_eps=args.target_eps,
        )

        if linear_model is not None:
            W, b, mse = train_linear_layer(
                X_train,
                y_train,
                X_test,
                y_test,
                device=args.device,
                ridge_alpha=args.linear_alpha,
            )
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
