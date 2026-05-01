#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_primary_metric(payload: dict[str, Any]) -> tuple[str, Any]:
    look_eval = payload.get("look_eval", {}) if isinstance(payload, dict) else {}
    if not isinstance(look_eval, dict):
        return "", ""
    for key in ("Accuracy", "ROUGE-L", "BLEU", "F1"):
        if key in look_eval:
            return key, look_eval.get(key, "")
    for key, value in look_eval.items():
        if isinstance(value, (int, float)):
            return str(key), value
    return "", ""


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "none":
        return ""
    return text


def collect_metrics_by_dataset(root: Path, method_subdir: str) -> dict[str, dict[str, Any]]:
    by_ds: dict[str, dict[str, Any]] = {}
    pattern = f"*/{method_subdir}/keep_0p20/metrics.json"
    for path in root.glob(pattern):
        try:
            payload = read_json(path)
        except Exception:
            continue
        ds = payload.get("look_dataset_name")
        if not ds:
            continue
        metric_name, metric_value = extract_primary_metric(payload)
        by_ds[str(ds)] = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "n_samples": payload.get("n_samples", ""),
            "n_predictions": payload.get("n_predictions", ""),
            "n_failures": payload.get("n_failures", ""),
            "path": str(path.parent),
        }
    return by_ds


def collect_oracle_teacher_metrics(root: Path) -> dict[str, dict[str, Any]]:
    by_ds: dict[str, dict[str, Any]] = {}
    for path in root.glob("*_teacher_oracle_sweep/att_only_postvision/keep_0p20/metrics.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        ds = payload.get("look_dataset_name")
        if not ds:
            continue
        metric_name, metric_value = extract_primary_metric(payload)
        by_ds[str(ds)] = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "n_samples": payload.get("n_samples", ""),
            "n_predictions": payload.get("n_predictions", ""),
            "n_failures": payload.get("n_failures", ""),
            "path": str(path.parent),
        }
    return by_ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/workspace/hd/artifacts/probe_global/efficiency_all_datasets/sample_manifest_common69.json")
    parser.add_argument("--probe_global_root", default="/workspace/hd/artifacts/probe_global")
    parser.add_argument("--oracle_root", default="/workspace/hd/artifacts/oracle")
    parser.add_argument("--out_csv", default="/workspace/hd/artifacts/probe_global/efficiency_all_datasets/performance_common69_datasets_keep0p20.csv")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ds_counts = Counter(str(row["dataset"]) for row in manifest)
    datasets = sorted(ds_counts)

    probe_root = Path(args.probe_global_root).resolve()
    oracle_root = Path(args.oracle_root).resolve()

    look_by_ds = collect_metrics_by_dataset(probe_root, "look_m")
    probe_by_ds = collect_metrics_by_dataset(probe_root, "probe_mlp")
    oracle_onthefly_by_ds = collect_metrics_by_dataset(probe_root, "oracle_onthefly")
    oracle_teacher_by_ds = collect_oracle_teacher_metrics(oracle_root)

    rows: list[dict[str, Any]] = []
    for ds in datasets:
        look = look_by_ds.get(ds, {})
        probe = probe_by_ds.get(ds, {})
        oracle_otf = oracle_onthefly_by_ds.get(ds, {})
        oracle_teacher = oracle_teacher_by_ds.get(ds, {})

        oracle_final = oracle_otf if oracle_otf else oracle_teacher
        oracle_source = "oracle_onthefly" if oracle_otf else ("oracle_teacher_precomputed" if oracle_teacher else "")

        metric_name = (
            oracle_final.get("metric_name")
            or probe.get("metric_name")
            or look.get("metric_name")
            or ""
        )

        rows.append(
            {
                "dataset": ds,
                "common69_samples": ds_counts.get(ds, 0),
                "metric_name": metric_name,
                "look_m": look.get("metric_value", ""),
                "probe_mlp_att_only_postvision": probe.get("metric_value", ""),
                "oracle_att_only_postvision": oracle_final.get("metric_value", ""),
                "oracle_source": oracle_source,
                "look_n_failures": look.get("n_failures", ""),
                "probe_n_failures": probe.get("n_failures", ""),
                "oracle_n_failures": oracle_final.get("n_failures", ""),
                "look_n_predictions": look.get("n_predictions", ""),
                "probe_n_predictions": probe.get("n_predictions", ""),
                "oracle_n_predictions": oracle_final.get("n_predictions", ""),
                "look_path": look.get("path", ""),
                "probe_path": probe.get("path", ""),
                "oracle_path": oracle_final.get("path", ""),
            }
        )

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "common69_samples", "metric_name",
        "look_m", "probe_mlp_att_only_postvision", "oracle_att_only_postvision", "oracle_source",
        "look_n_failures", "probe_n_failures", "oracle_n_failures",
        "look_n_predictions", "probe_n_predictions", "oracle_n_predictions",
        "look_path", "probe_path", "oracle_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: clean(row.get(k, "")) for k in fieldnames})

    print(out_path)
    print(f"rows={len(rows)}")
    print(f"oracle_filled={sum(1 for r in rows if clean(r.get('oracle_att_only_postvision',''))!='')}")


if __name__ == "__main__":
    main()
