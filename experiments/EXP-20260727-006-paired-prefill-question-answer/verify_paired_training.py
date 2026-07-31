#!/usr/bin/env python3
"""Validate that both paired students completed the identical train protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

PA_ROOT = (
    "/workspace/nips/data/train/teacher/"
    "zap_llava15_prefill_answer50_paired_n600_seed0"
)
QA_ROOT = (
    "/workspace/nips/data/train/teacher/"
    "zap_llava15_question_answer50_paired_n600_seed0"
)
COMMON_CONFIG: dict[str, Any] = {
    "datasets": ["gqa", "textvqa", "scienceqa"],
    "llava_path": "/workspace/nips/models/llava-v1.5-7b",
    "model_name": "llava-v1.5-7b",
    "vision_tower_path": "",
    "device": "cuda:0",
    "device_map": "cuda:0",
    "epochs": 15,
    "lr": 0.0001,
    "weight_decay": 0.0,
    "lambda_rank": 0.1,
    "rank_margin": 0.05,
    "rank_top_ratio": 0.2,
    "rank_bottom_ratio": 0.4,
    "max_grad_norm": 1.0,
    "seed": 0,
    "n_per_dataset": 600,
    "val_ratio": 0.1,
    "log_every": 25,
    "student_variant": "full",
    "conv_dim": 256,
    "proj_dim": 256,
    "mlp_dim": 512,
    "num_conv_blocks": 2,
    "kernel_size": 7,
    "grid_h": 24,
    "grid_w": 24,
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _validate(directory: Path, teacher_root: str) -> dict[str, Any]:
    weights = directory / "pytorch_model.bin"
    last_checkpoint = directory / "last_checkpoint.pt"
    config_path = directory / "train_config.json"
    log_path = directory / "train_log.jsonl"
    for path in (weights, last_checkpoint, config_path, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty training artifact: {path}")
    config = _load_json(config_path)
    expected = {**COMMON_CONFIG, "teacher_root": teacher_root, "output_dir": str(directory)}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{config_path}: expected {key}={value!r}, found {config.get(key)!r}"
            )
    unexpected = set(config) - set(expected)
    if unexpected:
        raise ValueError(f"{config_path}: unexpected config keys={sorted(unexpected)}")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    if [row.get("epoch") for row in rows] != list(range(1, 16)):
        raise ValueError(f"{log_path}: expected exactly epochs 1..15")
    for row in rows:
        if row.get("train_seen") != 1620 or row.get("val_seen") != 180:
            raise ValueError(f"{log_path}: paired sample denominator drift at {row}")
        for metric in (
            "train_loss",
            "train_mse",
            "train_rank",
            "val_loss",
            "val_mse",
            "val_rank",
        ):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{log_path}: non-finite {metric} at {row}")
    checkpoint = torch.load(
        last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("epoch") != 15:
        raise ValueError(f"{last_checkpoint}: last epoch is not 15")
    if checkpoint.get("metrics") != rows[-1]:
        raise ValueError(f"{last_checkpoint}: final metrics do not match train log")
    return {
        "directory": str(directory),
        "epochs": 15,
        "train_seen_each_epoch": 1620,
        "val_seen_each_epoch": 180,
        "best_weights_bytes": weights.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pa-dir", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = {
        "schema_version": 1,
        "paired_training": True,
        "prefill_answer": _validate(args.pa_dir.resolve(), PA_ROOT),
        "question_answer": _validate(args.qa_dir.resolve(), QA_ROOT),
    }
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
