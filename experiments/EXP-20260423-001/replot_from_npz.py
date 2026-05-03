#!/usr/bin/env python3
"""Re-plot the 6x2 future-vs-H2O heatmap grid from a saved raw npz.

Useful after tweaking color normalization — avoids re-running the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_future_vs_h2o_timewise import (  # type: ignore
    build_future_heatmap,
    build_h2o_heatmap,
    plot_grid,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, help="Path to wikivqa_sample*_raw.npz")
    p.add_argument("--out", default=None, help="Output PNG (defaults alongside npz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz).resolve()
    data = np.load(npz_path)

    layers = [int(x) for x in data["layers"].tolist()]
    prompt_len = int(data["prompt_len_mm"][0])
    n_gen = int(data["n_gen"][0])

    per_layer_attn = {li: data[f"attn_L{li}"] for li in layers}
    future_scores = {li: data[f"future_L{li}"] for li in layers}
    T = next(iter(per_layer_attn.values())).shape[0]

    heat_h2o = {li: build_h2o_heatmap(per_layer_attn[li]) for li in layers}
    heat_future = {li: build_future_heatmap(future_scores[li], T) for li in layers}

    out_path = (
        Path(args.out).resolve()
        if args.out
        else npz_path.with_name(npz_path.stem.replace("_raw", "") + "_shared_norm.png")
    )
    plot_grid(
        layers,
        heat_h2o,
        heat_future,
        prompt_len=prompt_len,
        out_path=out_path,
        sample_meta={"sample_id": npz_path.stem, "n_gen": n_gen},
    )
    print(f"[replot] wrote {out_path}")


if __name__ == "__main__":
    main()
