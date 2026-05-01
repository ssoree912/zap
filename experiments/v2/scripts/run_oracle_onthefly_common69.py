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


def list_milebench_datasets(milebench_root: Path) -> list[str]:
    datasets: list[str] = []
    for dataset_dir in sorted(path for path in milebench_root.iterdir() if path.is_dir()):
        name = dataset_dir.name
        dataset_json = dataset_dir / f"{name}.json"
        if dataset_json.is_file():
            datasets.append(name)
    return datasets


def _ratio_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _is_complete_run(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    n_samples = int(metrics.get("n_samples", 0) or 0)
    n_predictions = int(metrics.get("n_predictions", 0) or 0)
    n_failures = int(metrics.get("n_failures", 0) or 0)
    return n_samples > 0 and n_predictions >= n_samples and n_failures == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milebench_root", default="/workspace/zap/data/MileBench")
    parser.add_argument("--artifact_root", default="/workspace/zap/artifacts/oracle")
    parser.add_argument("--look_result_root", default=None,
                        help="Default: {artifact_root}/_look_runs")
    parser.add_argument("--model_name", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total_keep_ratio", type=float, default=0.20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional per-dataset sample limit for smoke tests")
    parser.add_argument("--truncate_like_lookm", action="store_true", default=True)
    parser.add_argument("--look_max_context_len", type=int, default=4096)
    parser.add_argument("--look_n_tokens_per_image", type=int, default=576)
    parser.add_argument("--conda_env", default="kv")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated MileBench dataset names to run. Default: all datasets in milebench_root")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--force_rerun", action="store_true", default=False,
                        help="Ignore existing metrics and rerun selected datasets")
    args = parser.parse_args()

    milebench_root = Path(args.milebench_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    look_result_root = (
        Path(args.look_result_root).resolve()
        if args.look_result_root
        else (artifact_root / "_look_runs").resolve()
    )

    datasets = list_milebench_datasets(milebench_root)
    if args.datasets:
        wanted = {name.strip() for name in str(args.datasets).split(",") if name.strip()}
        datasets = [name for name in datasets if name in wanted]

    ratio_tag = _ratio_tag(args.total_keep_ratio)
    summary_rows: list[dict[str, str]] = []
    started = time.time()

    for idx, ds in enumerate(datasets, start=1):
        ds_dir = _slugify(ds)
        dataset_path = milebench_root / ds / f"{ds}.json"
        image_root = milebench_root / ds / "images"
        out_dir = artifact_root / ds_dir / "oracle_onthefly" / f"keep_{ratio_tag}"
        metrics_path = out_dir / "metrics.json"
        look_model_name = f"zap_{ds_dir}_oracle_onthefly_k{ratio_tag}"

        if not dataset_path.exists():
            print(f"[{idx}/{len(datasets)}] SKIP {ds}: dataset json not found: {dataset_path}", flush=True)
            summary_rows.append({"dataset": ds, "dataset_dir": ds_dir, "status": "missing_dataset", "output_dir": str(out_dir)})
            continue
        if not image_root.exists():
            print(f"[{idx}/{len(datasets)}] SKIP {ds}: image root not found: {image_root}", flush=True)
            summary_rows.append({"dataset": ds, "dataset_dir": ds_dir, "status": "missing_images", "output_dir": str(out_dir)})
            continue

        if args.skip_existing and (not args.force_rerun) and _is_complete_run(metrics_path):
            print(f"[{idx}/{len(datasets)}] SKIP {ds}: existing complete metrics at {metrics_path}", flush=True)
            summary_rows.append({"dataset": ds, "dataset_dir": ds_dir, "status": "skipped_existing", "output_dir": str(out_dir)})
            continue

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
            "--look_model_name", look_model_name,
            "--look_result_root", str(look_result_root),
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

    summary_path = artifact_root / "_run_summaries" / "oracle_onthefly_all_datasets_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "milebench_root": str(milebench_root),
        "artifact_root": str(artifact_root),
        "look_result_root": str(look_result_root),
        "n_datasets": len(datasets),
        "total_keep_ratio": args.total_keep_ratio,
        "truncate_like_lookm": bool(args.truncate_like_lookm),
        "look_max_context_len": int(args.look_max_context_len),
        "look_n_tokens_per_image": int(args.look_n_tokens_per_image),
        "elapsed_sec": round(time.time() - started, 2),
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
