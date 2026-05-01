"""Sub-exp A: Per-layer teacher entropy analysis on ChartQA.

Loads all teacher shards from data/teacher_v2_onevision/chartqa/,
computes per-layer normalized entropy H^(l) / log(N_I) for each sample,
then aggregates (mean ± std) across samples and plots.

Outputs:
  outputs/layer_entropy.csv       — per-sample, per-layer entropy
  outputs/layer_entropy_agg.csv   — mean/std/median aggregated per layer
  outputs/layer_entropy_plot.png  — line plot with std band
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SHARD_DIR = Path(__file__).parents[2] / "data" / "teacher_v2_onevision" / "chartqa"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-row normalized entropy. p: [L, N], assumed non-negative and sum=1 per row."""
    N = p.shape[-1]
    h = -(p * torch.clamp(p, min=EPS).log()).sum(dim=-1)  # [L]
    return h / math.log(N)


def load_shard(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    shard_paths = sorted(SHARD_DIR.glob("*.pt"))
    print(f"Found {len(shard_paths)} shards in {SHARD_DIR}")

    records = []

    first = True
    for path in shard_paths:
        data = load_shard(path)
        sample_id = data["sample_id"]

        # teacher_norm: [L, N_I], already row-normalized (sums to 1 per layer)
        p = data["teacher_norm"].float()          # [28, N_I]
        if first:
            row_sums = p.sum(dim=-1)
            assert row_sums.allclose(torch.ones_like(row_sums), atol=1e-3), \
                f"teacher_norm row sums deviate from 1: {row_sums}"
            first = False
        n_layers, n_img = p.shape

        h = normalized_entropy(p)                 # [28]

        for l, h_val in enumerate(h.tolist()):
            records.append({"sample_id": sample_id, "layer": l, "entropy": h_val, "n_img": n_img})

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "layer_entropy.csv", index=False)
    print(f"Saved per-sample entropy → {OUT_DIR / 'layer_entropy.csv'}")

    # Aggregate per layer
    agg = (
        df.groupby("layer")["entropy"]
        .agg(mean="mean", std="std", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    agg.to_csv(OUT_DIR / "layer_entropy_agg.csv", index=False)
    print(f"Saved aggregated entropy → {OUT_DIR / 'layer_entropy_agg.csv'}")
    print()
    print(agg.to_string(index=False, float_format="{:.4f}".format))

    _plot(agg)


def _plot(agg: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = agg["layer"].values
    mean = agg["mean"].values
    std = agg["std"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, mean, marker="o", markersize=4, linewidth=1.5, color="steelblue", label="mean")
    ax.fill_between(layers, mean - std, mean + std, alpha=0.25, color="steelblue", label="±1 std")
    ax.plot(layers, agg["median"].values, linestyle="--", linewidth=1, color="gray", label="median")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized entropy  H / log(N_I)")
    ax.set_title("Per-layer teacher score entropy — ChartQA")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = OUT_DIR / "layer_entropy_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
