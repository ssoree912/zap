#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PROB_ROOT = Path('/workspace/hd/artifacts/prob')
LOOK_ROOT = Path('/workspace/LOOK-M/outputs/text_prior_pivot_merge_0.1_0.1_speed')
OUT_DIR = PROB_ROOT


def clean(value: Any) -> Any:
    return '' if value is None else value


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def extract_primary_metric(eval_payload: dict[str, Any]) -> tuple[str, Any]:
    for key in ('Accuracy', 'ROUGE-L', 'Rouge-L f'):
        if key in eval_payload:
            norm = 'ROUGE-L' if key == 'Rouge-L f' else key
            return norm, eval_payload[key]
    for key, value in eval_payload.items():
        if not isinstance(value, dict):
            return key, value
    return '', ''


fieldnames = [
    'source',
    'dataset',
    'method',
    'setting',
    'metric_name',
    'metric_value',
    'n_predictions',
    'n_failures',
    'image_keep_ratio',
    'path',
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, '')) for key in fieldnames})


def collect_probe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_paths = sorted(PROB_ROOT.glob('*_scienceqa_probe_sweep/att_only_postvision/mlp/keep_*/eval.json'))
    for eval_path in eval_paths:
        eval_payload = load_json(eval_path)
        metrics_path = eval_path.parent / 'metrics.json'
        metrics_payload = load_json(metrics_path) if metrics_path.is_file() else {}
        dataset_name = metrics_payload.get('look_dataset_name') or eval_payload.get('look_dataset_name') or eval_path.parts[5]
        keep_ratio = eval_path.parent.name.replace('keep_', '').replace('p', '.')
        metric_name, metric_value = extract_primary_metric(eval_payload)
        rows.append(
            {
                'source': 'probe',
                'dataset': dataset_name,
                'method': 'att_only_postvision_mlp',
                'setting': f'keep_{keep_ratio}',
                'metric_name': metric_name,
                'metric_value': metric_value,
                'n_predictions': metrics_payload.get('n_predictions'),
                'n_failures': metrics_payload.get('n_failures'),
                'image_keep_ratio': metrics_payload.get('image_keep_ratio'),
                'path': str(eval_path.parent),
            }
        )
    return rows


def collect_look_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(LOOK_ROOT.glob('*/eval.json')):
        dataset_name = eval_path.parent.name
        eval_payload = load_json(eval_path)
        metric_name, metric_value = extract_primary_metric(eval_payload)
        rows.append(
            {
                'source': 'look_m',
                'dataset': dataset_name,
                'method': 'text_prior_pivot_merge',
                'setting': 'hh_0.10_recent_0.10',
                'metric_name': metric_name,
                'metric_value': metric_value,
                'n_predictions': '',
                'n_failures': '',
                'image_keep_ratio': '',
                'path': str(eval_path.parent),
            }
        )
    return rows


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float('-inf')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_rows = collect_probe_rows()
    look_rows = collect_look_rows()

    probe_datasets = {str(row['dataset']) for row in probe_rows}
    look_by_dataset = {str(row['dataset']): row for row in look_rows}
    common_datasets = sorted(probe_datasets.intersection(look_by_dataset.keys()))

    probe_common = [row for row in probe_rows if str(row['dataset']) in common_datasets]
    look_common = [look_by_dataset[name] for name in common_datasets]
    all_rows = sorted(probe_common + look_common, key=lambda row: (row['dataset'], row['source'], row['setting']))

    best_probe_rows: list[dict[str, Any]] = []
    for dataset_name in common_datasets:
        candidates = [row for row in probe_common if row['dataset'] == dataset_name]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: to_float(row.get('metric_value')))
        best_probe_rows.append(best)
    comparison_rows = sorted(best_probe_rows + look_common, key=lambda row: (row['dataset'], row['source']))

    write_csv(OUT_DIR / 'probe_lookm_common_datasets_detailed.csv', all_rows)
    write_csv(OUT_DIR / 'probe_lookm_common_datasets_best_probe_vs_lookm.csv', comparison_rows)

    write_csv(OUT_DIR / 'probe_lookm_all_datasets_detailed.csv', all_rows)
    write_csv(OUT_DIR / 'probe_lookm_all_datasets_best_probe_vs_lookm.csv', comparison_rows)

    print(OUT_DIR / 'probe_lookm_all_datasets_detailed.csv')
    print(OUT_DIR / 'probe_lookm_all_datasets_best_probe_vs_lookm.csv')


if __name__ == '__main__':
    main()
