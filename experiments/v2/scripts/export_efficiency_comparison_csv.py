#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "none":
        return ""
    return text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean(row.get(name, "")) for name in fieldnames})


SUMMARY_METRICS = [
    "prefill_latency_ms_mean",
    "prefill_latency_ms_std",
    "decode_latency_ms_per_token_mean",
    "decode_latency_ms_per_token_std",
    "peak_gpu_memory_gib_mean",
    "peak_gpu_memory_gib_std",
    "r_img_mean",
    "r_img_std",
    "r_eff_prompt_mean",
    "r_eff_prompt_std",
    "r_eff_decode_t_end_mean",
    "r_eff_decode_t_end_std",
]

PER_SAMPLE_METRICS = [
    "prefill_latency_ms",
    "prefill_setup_latency_ms",
    "decode_latency_ms_per_token",
    "peak_gpu_memory_gib",
    "r_img",
    "r_eff_prompt",
    "r_eff_decode_t_end",
    "prompt_full_seq_len",
    "prompt_retained_seq_len",
    "final_cache_seq_len",
    "generated_tokens",
    "decode_steps",
    "image_tokens_total",
    "image_tokens_kept",
    "image_keep_ratio",
]

METHOD_LABELS = {
    "full_cache": "full_cache",
    "probe_att_only_postvision_mlp": "probe_att_only_postvision_mlp",
    "look_m": "look_m",
    "oracle_att_only_postvision": "oracle_att_only_postvision_pass2_only",
    "oracle_att_only_postvision_2pass_total": "oracle_att_only_postvision_2pass_total",
}


def build_combined_summary(zap_rows: list[dict[str, str]], look_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in zap_rows + look_rows:
        method = METHOD_LABELS.get(row.get("method", ""), row.get("method", ""))
        out = {
            "implementation": row.get("implementation", ""),
            "method": method,
            "n_samples": row.get("n_samples", ""),
            "datasets": row.get("datasets", ""),
        }
        for metric in SUMMARY_METRICS:
            out[metric] = row.get(metric, "")
        rows.append(out)
    order = {"full_cache": 0, "probe_att_only_postvision_mlp": 1, "look_m": 2, "oracle_att_only_postvision_2pass_total": 3, "oracle_att_only_postvision_pass2_only": 4}
    rows.sort(key=lambda r: (order.get(r["method"], 99), r["method"]))
    return rows


def build_wide_per_sample(zap_rows: list[dict[str, str]], look_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    wide: dict[str, dict[str, Any]] = {}

    def ensure(sample_key: str, row: dict[str, str]) -> dict[str, Any]:
        if sample_key not in wide:
            wide[sample_key] = {
                "dataset": row.get("dataset", ""),
                "sample_id": row.get("sample_id", ""),
                "sample_key": sample_key,
            }
        return wide[sample_key]

    for row in zap_rows + look_rows:
        sample_key = row.get("sample_key", "")
        out = ensure(sample_key, row)
        method = METHOD_LABELS.get(row.get("method", ""), row.get("method", ""))
        for metric in PER_SAMPLE_METRICS:
            out[f"{method}__{metric}"] = row.get(metric, "")

    return [wide[key] for key in sorted(wide)]


def build_long_per_sample(zap_rows: list[dict[str, str]], look_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in zap_rows + look_rows:
        out = {
            "implementation": row.get("implementation", ""),
            "method": METHOD_LABELS.get(row.get("method", ""), row.get("method", "")),
            "dataset": row.get("dataset", ""),
            "sample_id": row.get("sample_id", ""),
            "sample_key": row.get("sample_key", ""),
        }
        for metric in PER_SAMPLE_METRICS:
            out[metric] = row.get(metric, "")
        rows.append(out)
    order = {"full_cache": 0, "probe_att_only_postvision_mlp": 1, "look_m": 2, "oracle_att_only_postvision_2pass_total": 3, "oracle_att_only_postvision_pass2_only": 4}
    rows.sort(key=lambda r: (r["dataset"], r["sample_id"], order.get(r["method"], 99)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zap_dir", required=True)
    parser.add_argument("--look_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    zap_dir = Path(args.zap_dir).resolve()
    look_dir = Path(args.look_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    zap_summary = read_csv(zap_dir / "summary.csv")
    look_summary = read_csv(look_dir / "summary.csv")
    zap_per_sample = read_csv(zap_dir / "per_sample.csv")
    look_per_sample = read_csv(look_dir / "per_sample.csv")

    combined_summary = build_combined_summary(zap_summary, look_summary)
    long_per_sample = build_long_per_sample(zap_per_sample, look_per_sample)
    wide_per_sample = build_wide_per_sample(zap_per_sample, look_per_sample)

    summary_fields = ["implementation", "method", "n_samples", "datasets", *SUMMARY_METRICS]
    long_fields = ["implementation", "method", "dataset", "sample_id", "sample_key", *PER_SAMPLE_METRICS]
    wide_fields = [
        "dataset",
        "sample_id",
        "sample_key",
        *[f"full_cache__{metric}" for metric in PER_SAMPLE_METRICS],
        *[f"probe_att_only_postvision_mlp__{metric}" for metric in PER_SAMPLE_METRICS],
        *[f"look_m__{metric}" for metric in PER_SAMPLE_METRICS],
        *[f"oracle_att_only_postvision_2pass_total__{metric}" for metric in PER_SAMPLE_METRICS],
        *[f"oracle_att_only_postvision_pass2_only__{metric}" for metric in PER_SAMPLE_METRICS],
    ]

    write_csv(out_dir / "efficiency_random20_combined_summary.csv", combined_summary, summary_fields)
    write_csv(out_dir / "efficiency_random20_combined_per_sample_long.csv", long_per_sample, long_fields)
    write_csv(out_dir / "efficiency_random20_combined_per_sample_wide.csv", wide_per_sample, wide_fields)

    print(out_dir)


if __name__ == "__main__":
    main()
