#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create an immutable, provenance-rich run directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parent
ZAP_ROOT = EXP_DIR.parents[1]
CONFIG_PATH = EXP_DIR / "run_config.json"
EVALUATOR_PATH = ZAP_ROOT / "foresight/eval/lmms_llava15_original_student.py"
BUDGET_PATH = ZAP_ROOT / "foresight/eval/cache_budget.py"
LOCAL_WRAPPER_PATH = (
    ZAP_ROOT
    / "experiments/EXP-20260502-024-llava15-original-teacher-extract"
    / "lmms_eval_original_llava15_local_run.py"
)
TASK_ROOT = (
    ZAP_ROOT
    / "experiments/EXP-20260502-024-llava15-original-teacher-extract/tasks"
)
CONTRACT_SOURCE_PATHS = {
    "prepare_run": EXP_DIR / "prepare_run.py",
    "validator": EXP_DIR / "validate_paired_outputs.py",
    "run_worker": EXP_DIR / "run_worker.sh",
    "launcher": EXP_DIR / "run_3gpu.sh",
    "finalizer": EXP_DIR / "finalize_after_workers.sh",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ZAP_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def load_and_verify_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    for label, student in config["students"].items():
        checkpoint = Path(student["path"])
        weights = checkpoint / "pytorch_model.bin"
        if not checkpoint.is_dir() or not weights.is_file():
            raise FileNotFoundError(f"{label}: missing checkpoint weights at {weights}")
        actual = sha256_file(weights)
        expected = student["pytorch_model_sha256"]
        if actual != expected:
            raise ValueError(
                f"{label}: checkpoint SHA256 mismatch; expected={expected} actual={actual}"
            )
    model_path = Path(config["base_model"]["path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing base model directory: {model_path}")
    dataset_root = Path(config["evaluation"]["local_dataset_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing local evaluation dataset root: {dataset_root}")
    required_sources = [EVALUATOR_PATH, BUDGET_PATH, LOCAL_WRAPPER_PATH]
    for task in config["evaluation"]["tasks"]:
        required_sources.append(TASK_ROOT / f"{task}.yaml")
    missing = [str(path) for path in required_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")
    return config


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    source_paths = {
        "run_config": CONFIG_PATH,
        "student_evaluator": EVALUATOR_PATH,
        "cache_budget": BUDGET_PATH,
        "local_dataset_wrapper": LOCAL_WRAPPER_PATH,
        **CONTRACT_SOURCE_PATHS,
    }
    source_paths.update(
        {
            f"task_yaml:{task}": TASK_ROOT / f"{task}.yaml"
            for task in config["evaluation"]["tasks"]
        }
    )
    return {
        **config,
        "runtime_provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_status_short": _git_output("status", "--short"),
            "source_sha256": {
                label: sha256_file(path) for label, path in source_paths.items()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(
            f"Refusing to reuse existing run directory: {run_root}"
        )
    config = load_and_verify_config()
    run_root.mkdir(parents=True)
    for child in ("logs", "outputs", "state", "summary"):
        (run_root / child).mkdir()
    manifest = build_manifest(config)
    destination = run_root / "run_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
