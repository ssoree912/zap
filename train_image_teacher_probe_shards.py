#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel


@dataclass
class ShardInfo:
    path: Path
    split: str | None = None
    n_rows: int | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    layer_min: int | None = None
    layer_max: int | None = None
    n_layers_present: int | None = None


@dataclass
class SplitCatalog:
    root_dir: Path
    shard_dir: Path
    shard_infos: list[ShardInfo]
    input_dim: int | None
    output_dim: int | None
    n_layers: int | None
    n_rows: int | None


@dataclass
class DatasetSpec:
    input_dim: int
    output_dim: int
    n_layers: int
    n_rows: int
    n_train_rows: int
    n_val_rows: int
    n_train_shards: int
    n_val_shards: int


@dataclass
class EvalStats:
    sq_error: torch.Tensor
    n_items: torch.Tensor
    n_rows: torch.Tensor


class PerLayerQuota:
    def __init__(self, n_layers: int, max_rows_per_layer: int | None):
        self.max_rows_per_layer = max_rows_per_layer
        if max_rows_per_layer is None:
            self.remaining = None
        else:
            self.remaining = [int(max_rows_per_layer) for _ in range(n_layers)]

    def done(self) -> bool:
        return self.remaining is not None and all(value <= 0 for value in self.remaining)


def _resolve_input_root_and_shard_dir(input_dir: str | Path) -> tuple[Path, Path]:
    path = Path(input_dir).resolve()
    shard_dir = path / "shards"
    if shard_dir.is_dir():
        return path, shard_dir
    return path.parent if path.name == "shards" else path, path


def _resolve_shard_path(root_dir: Path, shard_dir: Path, shard_path: str) -> Path:
    path = Path(shard_path)
    if path.is_absolute():
        return path
    candidate = root_dir / path
    if candidate.exists():
        return candidate.resolve()
    return (shard_dir / path.name).resolve()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_shard(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}, got {type(payload)}")
    return payload


def _normalize_targets(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y.unsqueeze(-1).float()
    if y.ndim == 2:
        return y.float()
    raise ValueError(f"Unsupported y shape: {tuple(y.shape)}")


def _list_shard_files(shard_dir: Path) -> list[Path]:
    return sorted(p.resolve() for p in shard_dir.glob("*.pt") if p.is_file())


def _read_metadata_entries(root_dir: Path, shard_dir: Path) -> list[ShardInfo]:
    metadata_path = root_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return []

    shard_infos: list[ShardInfo] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            shard_infos.append(
                ShardInfo(
                    path=_resolve_shard_path(root_dir, shard_dir, entry["shard_path"]),
                    split=entry.get("split"),
                    n_rows=entry.get("n_rows"),
                    input_dim=entry.get("input_dim"),
                    output_dim=entry.get("output_dim"),
                    layer_min=entry.get("layer_min"),
                    layer_max=entry.get("layer_max"),
                    n_layers_present=entry.get("n_layers_present"),
                )
            )
    return shard_infos


def _infer_from_first_nonempty_shard(shard_infos: list[ShardInfo]) -> dict[str, int]:
    for info in shard_infos:
        payload = _load_shard(info.path)
        x = payload["X"]
        y = payload["y"]
        layer = payload["layer"]
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError(f"Shard X must be [N, D], got {type(x)} {getattr(x, 'shape', None)} in {info.path}")
        if not isinstance(y, torch.Tensor):
            raise ValueError(f"Shard y must be tensor in {info.path}")
        if not isinstance(layer, torch.Tensor) or layer.ndim != 1:
            raise ValueError(f"Shard layer must be [N] tensor in {info.path}")
        if x.shape[0] == 0 or layer.numel() == 0:
            continue
        normalized_y = _normalize_targets(y)
        return {
            "input_dim": int(x.shape[-1]),
            "output_dim": int(normalized_y.shape[-1]),
            "n_layers": int(layer.max().item()) + 1,
        }
    raise ValueError("Could not infer shard dataset spec from any non-empty shard")


def _write_dataset_spec(root_dir: Path, spec: dict[str, Any]) -> None:
    with (root_dir / "dataset_spec.json").open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


def load_split_catalog(input_dir: str | Path) -> SplitCatalog:
    root_dir, shard_dir = _resolve_input_root_and_shard_dir(input_dir)
    shard_infos = _read_metadata_entries(root_dir, shard_dir)
    if not shard_infos:
        shard_infos = [ShardInfo(path=path) for path in _list_shard_files(shard_dir)]
    if not shard_infos:
        raise ValueError(f"No shard files found in {shard_dir}")

    dataset_spec = _load_json(root_dir / "dataset_spec.json") or {}
    input_dim = dataset_spec.get("input_dim")
    output_dim = dataset_spec.get("output_dim")
    n_layers = dataset_spec.get("n_layers")
    n_rows = dataset_spec.get("n_rows")

    metadata_input_dims = {info.input_dim for info in shard_infos if info.input_dim is not None}
    if input_dim is None and len(metadata_input_dims) == 1:
        input_dim = int(next(iter(metadata_input_dims)))

    metadata_output_dims = {info.output_dim for info in shard_infos if info.output_dim is not None}
    if output_dim is None and len(metadata_output_dims) == 1:
        output_dim = int(next(iter(metadata_output_dims)))

    if n_rows is None:
        row_counts = [int(info.n_rows) for info in shard_infos if info.n_rows is not None]
        if row_counts and len(row_counts) == len(shard_infos):
            n_rows = int(sum(row_counts))

    metadata_layer_max = [int(info.layer_max) for info in shard_infos if info.layer_max is not None]
    if n_layers is None and metadata_layer_max:
        n_layers = max(metadata_layer_max) + 1

    if input_dim is None or output_dim is None or n_layers is None:
        inferred = _infer_from_first_nonempty_shard(shard_infos)
        input_dim = inferred["input_dim"] if input_dim is None else int(input_dim)
        output_dim = inferred["output_dim"] if output_dim is None else int(output_dim)
        n_layers = inferred["n_layers"] if n_layers is None else int(n_layers)

    split_name = next((info.split for info in shard_infos if info.split), None)
    if n_rows is None:
        n_rows = -1

    _write_dataset_spec(
        root_dir,
        {
            "split": split_name,
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "n_layers": int(n_layers),
            "n_rows": int(n_rows),
            "n_shards": len(shard_infos),
        },
    )

    return SplitCatalog(
        root_dir=root_dir,
        shard_dir=shard_dir,
        shard_infos=shard_infos,
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        n_layers=int(n_layers),
        n_rows=int(n_rows),
    )


def merge_split_catalogs(catalogs: list[SplitCatalog]) -> SplitCatalog:
    """Merge multiple SplitCatalogs into one by concatenating their shard_infos."""
    if len(catalogs) == 1:
        return catalogs[0]
    all_infos = [info for cat in catalogs for info in cat.shard_infos]
    total_rows = sum(int(cat.n_rows) for cat in catalogs if cat.n_rows is not None and cat.n_rows >= 0)
    return SplitCatalog(
        root_dir=catalogs[0].root_dir,
        shard_dir=catalogs[0].shard_dir,
        shard_infos=all_infos,
        input_dim=catalogs[0].input_dim,
        output_dim=catalogs[0].output_dim,
        n_layers=catalogs[0].n_layers,
        n_rows=total_rows if total_rows > 0 else -1,
    )


def inspect_shards(train_shard_dir: str | Path | list, val_shard_dir: str | Path) -> tuple[DatasetSpec, SplitCatalog, SplitCatalog]:
    if isinstance(train_shard_dir, list):
        train_catalogs = [load_split_catalog(d) for d in train_shard_dir]
        train_catalog = merge_split_catalogs(train_catalogs)
    else:
        train_catalog = load_split_catalog(train_shard_dir)
    val_catalog = load_split_catalog(val_shard_dir)

    if train_catalog.input_dim != val_catalog.input_dim:
        raise ValueError(f"Input dim mismatch: train={train_catalog.input_dim} val={val_catalog.input_dim}")
    if train_catalog.output_dim != val_catalog.output_dim:
        raise ValueError(f"Output dim mismatch: train={train_catalog.output_dim} val={val_catalog.output_dim}")
    if train_catalog.n_layers != val_catalog.n_layers:
        raise ValueError(f"Layer count mismatch: train={train_catalog.n_layers} val={val_catalog.n_layers}")

    train_rows = int(train_catalog.n_rows)
    val_rows = int(val_catalog.n_rows)
    total_rows = train_rows + val_rows if train_rows >= 0 and val_rows >= 0 else -1

    spec = DatasetSpec(
        input_dim=int(train_catalog.input_dim),
        output_dim=int(train_catalog.output_dim),
        n_layers=int(train_catalog.n_layers),
        n_rows=total_rows,
        n_train_rows=train_rows,
        n_val_rows=val_rows,
        n_train_shards=len(train_catalog.shard_infos),
        n_val_shards=len(val_catalog.shard_infos),
    )
    return spec, train_catalog, val_catalog


def _resolve_device(device: str) -> str:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device}, but CUDA is not available")
        return device
    return "cpu"


def _iter_shard_tensors(
    shard_infos: list[ShardInfo],
    desc: str,
    shuffle_shards: bool,
    seed: int,
    max_shards: int | None = None,
) -> Iterator[tuple[ShardInfo, torch.Tensor, torch.Tensor, torch.Tensor]]:
    order = np.arange(len(shard_infos))
    if shuffle_shards:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    if max_shards is not None:
        order = order[:max_shards]

    for index in tqdm(order.tolist(), desc=desc, leave=False):
        info = shard_infos[int(index)]
        payload = _load_shard(info.path)
        x = payload["X"]
        y = _normalize_targets(payload["y"])
        layer = payload["layer"].long()
        if x.ndim != 2:
            raise ValueError(f"Expected shard X to be [N, D], got {tuple(x.shape)} in {info.path}")
        if layer.ndim != 1:
            raise ValueError(f"Expected shard layer to be [N], got {tuple(layer.shape)} in {info.path}")
        if x.shape[0] != y.shape[0] or x.shape[0] != layer.shape[0]:
            raise ValueError(
                f"Shard row mismatch in {info.path}: X={tuple(x.shape)} y={tuple(y.shape)} layer={tuple(layer.shape)}"
            )
        yield info, x.float(), y.float(), layer


def _apply_row_quota(
    x: torch.Tensor,
    y: torch.Tensor,
    layer: torch.Tensor,
    quota: PerLayerQuota,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if quota.remaining is None:
        return x, y, layer
    if quota.done():
        empty_x = x.new_empty((0, x.shape[-1]))
        empty_y = y.new_empty((0, y.shape[-1]))
        empty_layer = layer.new_empty((0,), dtype=layer.dtype)
        return empty_x, empty_y, empty_layer

    kept_indices: list[np.ndarray] = []
    for layer_idx_tensor in torch.unique(layer, sorted=True):
        layer_idx = int(layer_idx_tensor.item())
        remaining = quota.remaining[layer_idx]
        if remaining <= 0:
            continue
        layer_positions = (layer == layer_idx_tensor).nonzero(as_tuple=False).flatten().numpy()
        if layer_positions.size > remaining:
            layer_positions = np.sort(rng.choice(layer_positions, size=remaining, replace=False))
        quota.remaining[layer_idx] -= int(layer_positions.size)
        if layer_positions.size > 0:
            kept_indices.append(layer_positions)

    if not kept_indices:
        empty_x = x.new_empty((0, x.shape[-1]))
        empty_y = y.new_empty((0, y.shape[-1]))
        empty_layer = layer.new_empty((0,), dtype=layer.dtype)
        return empty_x, empty_y, empty_layer

    merged = np.concatenate(kept_indices)
    merged.sort()
    indices = torch.from_numpy(merged).long()
    return x.index_select(0, indices), y.index_select(0, indices), layer.index_select(0, indices)


def _iter_streaming_batches(
    shard_infos: list[ShardInfo],
    batch_size: int,
    desc: str,
    shuffle_shards: bool,
    shuffle_within_shard: bool,
    seed: int,
    n_layers: int,
    max_rows_per_layer: int | None,
    max_shards: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    shard_rng = np.random.default_rng(seed)
    row_rng = np.random.default_rng(seed + 17)
    quota = PerLayerQuota(n_layers=n_layers, max_rows_per_layer=max_rows_per_layer)

    for _, x, y, layer in _iter_shard_tensors(
        shard_infos,
        desc=desc,
        shuffle_shards=shuffle_shards,
        seed=seed,
        max_shards=max_shards,
    ):
        x, y, layer = _apply_row_quota(x=x, y=y, layer=layer, quota=quota, rng=shard_rng)
        n_rows = int(x.shape[0])
        if n_rows == 0:
            if quota.done():
                break
            continue

        if shuffle_within_shard:
            order = row_rng.permutation(n_rows)
        else:
            order = np.arange(n_rows)

        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            batch_indices = torch.from_numpy(order[start:stop].copy()).long()
            yield (
                x.index_select(0, batch_indices),
                y.index_select(0, batch_indices),
                layer.index_select(0, batch_indices),
            )

        if quota.done():
            break


def _empty_eval_stats(n_layers: int) -> EvalStats:
    return EvalStats(
        sq_error=torch.zeros(n_layers, dtype=torch.float64),
        n_items=torch.zeros(n_layers, dtype=torch.long),
        n_rows=torch.zeros(n_layers, dtype=torch.long),
    )


def evaluate_model_streaming(
    model: KVzapModel,
    shard_infos: list[ShardInfo],
    batch_size: int,
    device: str,
    n_layers: int,
    desc: str,
    seed: int,
    max_rows_per_layer: int | None,
    max_shards: int | None,
) -> EvalStats:
    stats = _empty_eval_stats(n_layers)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for x_cpu, y_cpu, layer_cpu in _iter_streaming_batches(
            shard_infos=shard_infos,
            batch_size=batch_size,
            desc=desc,
            shuffle_shards=False,
            shuffle_within_shard=False,
            seed=seed,
            n_layers=n_layers,
            max_rows_per_layer=max_rows_per_layer,
            max_shards=max_shards,
        ):
            for layer_idx_tensor in torch.unique(layer_cpu, sorted=True):
                layer_idx = int(layer_idx_tensor.item())
                mask = layer_cpu == layer_idx_tensor
                xb = x_cpu[mask].to(device)
                yb = y_cpu[mask].to(device)
                pred = model.layers[layer_idx](xb)
                diff = pred - yb
                stats.sq_error[layer_idx] += diff.pow(2).sum().double().cpu()
                stats.n_items[layer_idx] += int(diff.numel())
                stats.n_rows[layer_idx] += int(xb.shape[0])

    return stats


def _stats_to_metrics(method: str, train_rows: torch.Tensor, val_stats: EvalStats) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for layer_idx in range(int(train_rows.shape[0])):
        n_items = int(val_stats.n_items[layer_idx].item())
        val_mse = float(val_stats.sq_error[layer_idx].item() / max(n_items, 1))
        metrics.append(
            {
                "method": method,
                "layer": layer_idx,
                "n_train_rows": int(train_rows[layer_idx].item()),
                "n_val_rows": int(val_stats.n_rows[layer_idx].item()),
                "val_mse": val_mse,
            }
        )
    return metrics


def train_linear_streaming(
    spec: DatasetSpec,
    train_catalog: SplitCatalog,
    val_catalog: SplitCatalog,
    device: str,
    ridge_alpha: float,
    seed: int,
    max_train_rows_per_layer: int | None,
    max_val_rows_per_layer: int | None,
    max_train_shards: int | None,
    max_val_shards: int | None,
) -> tuple[KVzapModel, list[dict[str, Any]]]:
    tqdm.write("[linear] accumulating train statistics from shards")
    model = KVzapModel(
        KVzapConfig(
            input_dim=spec.input_dim,
            hidden_dim=None,
            output_dim=spec.output_dim,
            n_modules=spec.n_layers,
        )
    )

    aug_dim = spec.input_dim + 1
    xtx = [torch.zeros((aug_dim, aug_dim), dtype=torch.float64) for _ in range(spec.n_layers)]
    xty = [torch.zeros((aug_dim, spec.output_dim), dtype=torch.float64) for _ in range(spec.n_layers)]
    train_rows = torch.zeros(spec.n_layers, dtype=torch.long)

    for x_cpu, y_cpu, layer_cpu in _iter_streaming_batches(
        shard_infos=train_catalog.shard_infos,
        batch_size=65536,
        desc="[linear] train shards",
        shuffle_shards=True,
        shuffle_within_shard=False,
        seed=seed,
        n_layers=spec.n_layers,
        max_rows_per_layer=max_train_rows_per_layer,
        max_shards=max_train_shards,
    ):
        for layer_idx_tensor in torch.unique(layer_cpu, sorted=True):
            layer_idx = int(layer_idx_tensor.item())
            mask = layer_cpu == layer_idx_tensor
            xb = x_cpu[mask].to(device)
            yb = y_cpu[mask].to(device)
            ones = torch.ones((xb.shape[0], 1), device=device, dtype=xb.dtype)
            xb_aug = torch.cat([xb, ones], dim=1)
            xtx[layer_idx].add_((xb_aug.T @ xb_aug).double().cpu())
            xty[layer_idx].add_((xb_aug.T @ yb).double().cpu())
            train_rows[layer_idx] += int(xb.shape[0])

    betas: list[torch.Tensor] = []
    tqdm.write("[linear] solving per-layer systems")
    for layer_idx in tqdm(range(spec.n_layers), desc="[linear] solve layers", leave=False):
        reg = torch.eye(aug_dim, dtype=torch.float64, device=device) * ridge_alpha
        reg[-1, -1] = 0.0
        beta = torch.linalg.solve(xtx[layer_idx].to(device) + reg, xty[layer_idx].to(device))
        betas.append(beta.detach().cpu())
        model.layers[layer_idx].weight.data.copy_(beta[:-1].transpose(0, 1).float().cpu())
        model.layers[layer_idx].bias.data.copy_(beta[-1].float().cpu())

    tqdm.write("[linear] evaluating validation shards")
    val_stats = _empty_eval_stats(spec.n_layers)
    for x_cpu, y_cpu, layer_cpu in _iter_streaming_batches(
        shard_infos=val_catalog.shard_infos,
        batch_size=65536,
        desc="[linear] val shards",
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=seed,
        n_layers=spec.n_layers,
        max_rows_per_layer=max_val_rows_per_layer,
        max_shards=max_val_shards,
    ):
        for layer_idx_tensor in torch.unique(layer_cpu, sorted=True):
            layer_idx = int(layer_idx_tensor.item())
            mask = layer_cpu == layer_idx_tensor
            xb = x_cpu[mask].to(device)
            yb = y_cpu[mask].to(device)
            ones = torch.ones((xb.shape[0], 1), device=device, dtype=xb.dtype)
            xb_aug = torch.cat([xb, ones], dim=1)
            beta = betas[layer_idx].to(device=device, dtype=xb.dtype)
            pred = xb_aug @ beta
            diff = pred - yb
            val_stats.sq_error[layer_idx] += diff.pow(2).sum().double().cpu()
            val_stats.n_items[layer_idx] += int(diff.numel())
            val_stats.n_rows[layer_idx] += int(xb.shape[0])

    return model.cpu(), _stats_to_metrics("linear", train_rows=train_rows, val_stats=val_stats)


def train_mlp_streaming(
    spec: DatasetSpec,
    train_catalog: SplitCatalog,
    val_catalog: SplitCatalog,
    hidden_dim: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    max_train_rows_per_layer: int | None,
    max_val_rows_per_layer: int | None,
    max_train_shards: int | None,
    max_val_shards: int | None,
) -> tuple[KVzapModel, list[dict[str, Any]]]:
    model = KVzapModel(
        KVzapConfig(
            input_dim=spec.input_dim,
            hidden_dim=hidden_dim,
            output_dim=spec.output_dim,
            n_modules=spec.n_layers,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_state = None
    best_val_score = float("inf")
    best_val_stats = None
    train_rows_reference = torch.zeros(spec.n_layers, dtype=torch.long)

    for epoch_idx in range(max_epochs):
        epoch_label = f"[mlp][epoch {epoch_idx + 1}/{max_epochs}]"
        tqdm.write(f"{epoch_label} streaming train shards on {device}")
        model.train()
        epoch_train_rows = torch.zeros(spec.n_layers, dtype=torch.long)

        for x_cpu, y_cpu, layer_cpu in _iter_streaming_batches(
            shard_infos=train_catalog.shard_infos,
            batch_size=batch_size,
            desc=f"{epoch_label} train shards",
            shuffle_shards=True,
            shuffle_within_shard=True,
            seed=seed + epoch_idx,
            n_layers=spec.n_layers,
            max_rows_per_layer=max_train_rows_per_layer,
            max_shards=max_train_shards,
        ):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = None
            batch_items = 0

            for layer_idx_tensor in torch.unique(layer_cpu, sorted=True):
                layer_idx = int(layer_idx_tensor.item())
                mask = layer_cpu == layer_idx_tensor
                xb = x_cpu[mask].to(device)
                yb = y_cpu[mask].to(device)
                pred = model.layers[layer_idx](xb)
                layer_loss = F.mse_loss(pred, yb, reduction="sum")
                batch_loss = layer_loss if batch_loss is None else batch_loss + layer_loss
                batch_items += int(yb.numel())
                epoch_train_rows[layer_idx] += int(xb.shape[0])

            if batch_loss is None or batch_items == 0:
                continue

            (batch_loss / batch_items).backward()
            optimizer.step()

        if epoch_idx == 0:
            train_rows_reference = epoch_train_rows.clone()

        tqdm.write(f"{epoch_label} evaluating val shards")
        val_stats = evaluate_model_streaming(
            model=model,
            shard_infos=val_catalog.shard_infos,
            batch_size=batch_size,
            device=device,
            n_layers=spec.n_layers,
            desc=f"{epoch_label} val shards",
            seed=seed,
            max_rows_per_layer=max_val_rows_per_layer,
            max_shards=max_val_shards,
        )
        total_items = int(val_stats.n_items.sum().item())
        total_sq_error = float(val_stats.sq_error.sum().item())
        val_score = total_sq_error / max(total_items, 1)
        tqdm.write(f"{epoch_label} val_mse={val_score:.6f}")

        if val_score < best_val_score:
            best_val_score = val_score
            best_val_stats = EvalStats(
                sq_error=val_stats.sq_error.clone(),
                n_items=val_stats.n_items.clone(),
                n_rows=val_stats.n_rows.clone(),
            )
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_val_stats is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    return model.cpu(), _stats_to_metrics("mlp", train_rows=train_rows_reference, val_stats=best_val_stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_shard_dir", type=str, nargs="+", required=True)
    parser.add_argument("--val_shard_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--methods", nargs="*", default=["linear", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_rows_per_layer", type=int, default=None)
    parser.add_argument("--max_val_rows_per_layer", type=int, default=None)
    parser.add_argument("--max_train_shards", type=int, default=None)
    parser.add_argument("--max_val_shards", type=int, default=None)
    parser.add_argument("--linear_alpha", type=float, default=1.0)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_max_epochs", type=int, default=10)
    parser.add_argument("--mlp_batch_size", type=int, default=2048)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    train_dirs = args.train_shard_dir if len(args.train_shard_dir) > 1 else args.train_shard_dir[0]
    spec, train_catalog, val_catalog = inspect_shards(train_dirs, args.val_shard_dir)
    methods = set(args.methods)

    with (output_dir / "dataset_spec.json").open("w", encoding="utf-8") as f:
        json.dump(spec.__dict__, f, indent=2)

    metrics: list[dict[str, Any]] = []

    if "linear" in methods:
        linear_model, linear_metrics = train_linear_streaming(
            spec=spec,
            train_catalog=train_catalog,
            val_catalog=val_catalog,
            device=device,
            ridge_alpha=args.linear_alpha,
            seed=args.seed,
            max_train_rows_per_layer=args.max_train_rows_per_layer,
            max_val_rows_per_layer=args.max_val_rows_per_layer,
            max_train_shards=args.max_train_shards,
            max_val_shards=args.max_val_shards,
        )
        linear_dir = output_dir / "linear"
        linear_dir.mkdir(parents=True, exist_ok=True)
        linear_model.save_pretrained(linear_dir)
        metrics.extend(linear_metrics)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "mlp" in methods:
        mlp_model, mlp_metrics = train_mlp_streaming(
            spec=spec,
            train_catalog=train_catalog,
            val_catalog=val_catalog,
            hidden_dim=args.mlp_hidden_dim,
            max_epochs=args.mlp_max_epochs,
            batch_size=args.mlp_batch_size,
            lr=args.mlp_lr,
            device=device,
            seed=args.seed,
            max_train_rows_per_layer=args.max_train_rows_per_layer,
            max_val_rows_per_layer=args.max_val_rows_per_layer,
            max_train_shards=args.max_train_shards,
            max_val_shards=args.max_val_shards,
        )
        mlp_dir = output_dir / "mlp"
        mlp_dir.mkdir(parents=True, exist_ok=True)
        mlp_model.save_pretrained(mlp_dir)
        metrics.extend(mlp_metrics)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved probe models to {output_dir}")
    print(
        "Input dim: "
        f"{spec.input_dim} | Output dim: {spec.output_dim} | Layers: {spec.n_layers} | "
        f"Train shards: {spec.n_train_shards} | Val shards: {spec.n_val_shards}"
    )


if __name__ == "__main__":
    main()
