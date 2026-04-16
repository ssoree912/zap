#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def _slugify(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_common69_datasets(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = sorted({str(row["dataset"]) for row in manifest})
    return datasets


def build_dataset_dir_map(probe_global_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for metrics_path in probe_global_root.glob("*/probe_mlp/keep_0p20/metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ds = payload.get("look_dataset_name")
        if not ds:
            continue
        mapping[str(ds)] = metrics_path.parts[-4]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/workspace/hd/artifacts/probe_global/efficiency_all_datasets/sample_manifest_common69.json")
    parser.add_argument("--milebench_root", default="/workspace/hd/data/MileBench")
    parser.add_argument("--probe_global_root", default="/workspace/hd/artifacts/probe_global")
    parser.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total_keep_ratio", type=float, default=0.20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional per-dataset sample limit for smoke tests")
    parser.add_argument("--truncate_like_lookm", action="store_true", default=True)
    parser.add_argument("--look_max_context_len", type=int, default=2048)
    parser.add_argument("--look_n_tokens_per_image", type=int, default=576)
    parser.add_argument("--conda_env", default="kv")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated MileBench dataset names to run. Default: all datasets in common69 manifest")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--force_rerun", action="store_true", default=False,
                        help="Ignore existing metrics and rerun selected datasets")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    milebench_root = Path(args.milebench_root).resolve()
    probe_global_root = Path(args.probe_global_root).resolve()

    datasets = load_common69_datasets(manifest_path)
    if args.datasets:
        wanted = {name.strip() for name in str(args.datasets).split(",") if name.strip()}
        datasets = [d for d in datasets if d in wanted]
    ds_dir_map = build_dataset_dir_map(probe_global_root)

    summary_rows: list[dict[str, str]] = []
    started = time.time()

    for idx, ds in enumerate(datasets, start=1):
        ds_dir = ds_dir_map.get(ds, _slugify(ds))
        dataset_path = milebench_root / ds / f"{ds}.json"
        image_root = milebench_root / ds / "images"
        out_dir = probe_global_root / ds_dir / "oracle_onthefly" / "keep_0p20"
        metrics_path = out_dir / "metrics.json"

        if not dataset_path.exists():
            print(f"[{idx}/{len(datasets)}] SKIP {ds}: dataset json not found: {dataset_path}", flush=True)
            summary_rows.append({"dataset": ds, "dataset_dir": ds_dir, "status": "missing_dataset", "output_dir": str(out_dir)})
            continue

        if args.skip_existing and (not args.force_rerun) and metrics_path.exists():
            skip_ok = False
            try:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                n_samples = int(existing.get("n_samples", 0) or 0)
                n_predictions = int(existing.get("n_predictions", 0) or 0)
                n_failures = int(existing.get("n_failures", 0) or 0)
                skip_ok = (n_samples > 0 and n_predictions >= n_samples and n_failures == 0)
            except Exception:
                skip_ok = False
            if skip_ok:
                print(f"[{idx}/{len(datasets)}] SKIP {ds}: existing complete metrics at {metrics_path}", flush=True)
                summary_rows.append({"dataset": ds, "dataset_dir": ds_dir, "status": "skipped_existing", "output_dir": str(out_dir)})
                continue
            print(f"[{idx}/{len(datasets)}] RETRY {ds}: metrics exist but incomplete/failing -> {metrics_path}", flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", args.conda_env, "--no-capture-output",
            "python", "/workspace/zap/evaluate_image_teacher_pruning.py",
            "--mode", "oracle_onthefly",
            "--dataset_path", str(dataset_path),
            "--image_root", str(image_root),
            "--image_column", "images_path",
            "--output_dir", str(out_dir),
            "--implementation_model_name", args.model_name,
            "--prompt_style", "look_milebench",
            "--prompt_template", "USER: <image>\n{question}\nASSISTANT:",
            "--max_new_tokens", str(args.max_new_tokens),
            "--torch_dtype", "float16",
            "--attn_implementation", "eager",
            "--total_keep_ratio", str(args.total_keep_ratio),
            "--head_reduce", "amax",
            "--look_dataset_name", ds,
            "--look_model_name", "oracle_onthefly_common69",
            "--look_result_root", str(probe_global_root),
            "--device", args.device,
            "--continue_on_error",
            "--allow_partial_look_eval",
        ]
        if args.truncate_like_lookm:
            cmd.extend([
                "--truncate_like_lookm",
                "--look_max_context_len", str(args.look_max_context_len),
                "--look_n_tokens_per_image", str(args.look_n_tokens_per_image),
            ])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        print(f"[{idx}/{len(datasets)}] RUN {ds} -> {out_dir}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"failed_rc_{proc.returncode}"
        print(f"[{idx}/{len(datasets)}] DONE {ds}: status={status} elapsed={elapsed/60.0:.1f}m", flush=True)
        summary_rows.append(
            {
                "dataset": ds,
                "dataset_dir": ds_dir,
                "status": status,
                "elapsed_sec": f"{elapsed:.1f}",
                "output_dir": str(out_dir),
            }
        )

    summary_path = probe_global_root / "efficiency_all_datasets" / "oracle_onthefly_common69_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "manifest": str(manifest_path),
        "n_datasets": len(datasets),
        "elapsed_sec": round(time.time() - started, 2),
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
