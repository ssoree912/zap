# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Emit a compact live progress log for the ChartQA seed sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from pathlib import Path


PROGRESS_RE = re.compile(
    r"Infer ov7b_baseline_local/ChartQA_TEST.*?"
    r"(?P<pct>\d+)%.*?\|\s*(?P<done>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^<\]]+)<(?P<remaining>[^,\]]+),\s*(?P<rate>[^\]]+)\]"
)
RUN_RE = re.compile(r"\[seed=(?P<seed>\d+) gpu=(?P<gpu>\d+)\] (?P<event>start|done) (?P<time>.+)")

DEFAULT_SEEDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 42, 123, 321, 777,
    1234, 2024, 2026, 3407, 7777, 9999,
]


def parse_duration_seconds(value: str) -> int | None:
    parts = value.strip().split(":")
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60 + seconds
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return hours * 3600 + minutes * 60 + seconds
    return None


def format_seconds(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")


def latest_progress(log_path: Path) -> dict[str, str | int] | None:
    last = None
    for line in read_text(log_path).splitlines():
        match = PROGRESS_RE.search(line)
        if match:
            data: dict[str, str | int] = match.groupdict()
            data["pct"] = int(data["pct"])
            data["done"] = int(data["done"])
            data["total"] = int(data["total"])
            data["remaining_seconds"] = parse_duration_seconds(str(data["remaining"])) or -1
            last = data
    return last


def read_csv_metric(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    return {
        "overall": float(row["Overall"]),
        "test_human": float(row["test_human"]),
        "test_augmented": float(row["test_augmented"]),
    }


def metric_path(root: Path, seed: int) -> Path:
    return (
        root
        / f"seed_{seed}"
        / "ov7b_baseline_local"
        / "ov7b_baseline_local_ChartQA_TEST_acc.csv"
    )


def parse_run_log(run_log: Path) -> tuple[list[int], list[int], dict[int, int], set[int]]:
    seeds = DEFAULT_SEEDS[:]
    gpus: list[int] = []
    running_gpu: dict[int, int] = {}
    done: set[int] = set()
    for line in read_text(run_log).splitlines():
        if line.startswith("EXP_DIR="):
            running_gpu = {}
            done = set()
        elif line.startswith("SEEDS="):
            try:
                seeds = [int(x) for x in line.removeprefix("SEEDS=").split()]
            except ValueError:
                pass
        elif line.startswith("GPUS="):
            try:
                gpus = [int(x) for x in line.removeprefix("GPUS=").split()]
            except ValueError:
                pass
        match = RUN_RE.search(line)
        if not match:
            continue
        seed = int(match.group("seed"))
        gpu = int(match.group("gpu"))
        if match.group("event") == "start":
            running_gpu[seed] = gpu
        else:
            done.add(seed)
            running_gpu.pop(seed, None)
    return seeds, gpus, running_gpu, done


def completed_metrics(root: Path, seeds: list[int]) -> list[tuple[int, dict[str, float]]]:
    rows = []
    for seed in seeds:
        metric = read_csv_metric(metric_path(root, seed))
        if metric is not None:
            rows.append((seed, metric))
    return rows


def print_snapshot(args: argparse.Namespace) -> None:
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    seeds, gpus, running_gpu, done_from_log = parse_run_log(args.run_log)
    metrics = completed_metrics(args.root, seeds)
    completed_seeds = {seed for seed, _ in metrics} | done_from_log
    active = []
    stale = []

    for seed, gpu in sorted(running_gpu.items()):
        if seed in completed_seeds:
            continue
        log_path = args.root / "logs" / f"seed_{seed}.log"
        progress = latest_progress(log_path)
        age = time.time() - log_path.stat().st_mtime if log_path.exists() else None
        row = (seed, gpu, log_path, progress, age)
        if age is not None and age <= args.active_age_seconds:
            active.append(row)
        else:
            stale.append(row)

    print(f"[{now}] OneVision ChartQA full-cache seed sweep")
    print(f"  screen: onevision_chartqa_seed_sweep")
    print(f"  root: {args.root}")
    print(f"  monitor: {args.monitor_log}")
    print(f"  configured_gpus: {' '.join(map(str, gpus)) if gpus else 'unknown'}")
    print(f"  completed: {len(completed_seeds)}/{len(seeds)} seeds")

    if metrics:
        best_seed, best_metric = max(metrics, key=lambda item: item[1]["overall"])
        print(
            "  best: seed={seed} overall={overall:.2f} human={human:.2f} augmented={aug:.2f} target={target:.2f}".format(
                seed=best_seed,
                overall=best_metric["overall"],
                human=best_metric["test_human"],
                aug=best_metric["test_augmented"],
                target=args.target,
            )
        )
    else:
        print(f"  best: n/a target={args.target:.2f}")

    max_remaining = -1
    if active:
        print("  active:")
        for seed, gpu, log_path, progress, _age in active:
            if progress is None:
                print(f"    seed={seed} gpu={gpu}: starting; log={log_path}")
                continue
            remaining = int(progress["remaining_seconds"])
            max_remaining = max(max_remaining, remaining)
            print(
                "    seed={seed} gpu={gpu}: {pct}% {done}/{total}, elapsed={elapsed}, remaining={remaining}, rate={rate}".format(
                    seed=seed,
                    gpu=gpu,
                    pct=progress["pct"],
                    done=progress["done"],
                    total=progress["total"],
                    elapsed=progress["elapsed"],
                    remaining=progress["remaining"],
                    rate=progress["rate"],
                )
            )
        print(f"  current_batch_eta: {format_seconds(max_remaining if max_remaining >= 0 else None)}")
    else:
        print("  active: none detected")

    if stale:
        print("  stale_started_without_metric:")
        for seed, gpu, log_path, _progress, age in stale:
            age_text = "unknown" if age is None else f"{int(age)}s"
            print(f"    seed={seed} gpu={gpu}: log_age={age_text}; log={log_path}")

    print("", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--monitor-log", type=Path, required=True)
    parser.add_argument("--target", type=float, default=80.3)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--active-age-seconds", type=int, default=180)
    args = parser.parse_args()

    while True:
        print_snapshot(args)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
