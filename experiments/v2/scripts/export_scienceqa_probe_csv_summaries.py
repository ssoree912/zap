#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ORACLE_ROOT = Path('/workspace/hd/artifacts/oracle')
SQ_ROOT = Path('/workspace/hd/artifacts/sq_teacher')


def _clean(value: Any) -> str | int | float:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return value


def _parse_eval_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rel_parts = path.relative_to(ORACLE_ROOT).parts
    group = rel_parts[0]
    row: dict[str, Any] = {
        'run_type': 'probe' if '_scienceqa_probe_sweep' in group else ('oracle' if '_teacher_oracle_sweep' in group else 'other'),
        'group': group,
        'dataset': payload.get('look_dataset_name'),
        'mode': payload.get('mode'),
        'teacher': '',
        'teacher_score_name': payload.get('teacher_score_name'),
        'method': '',
        'image_keep_ratio': payload.get('image_keep_ratio'),
        'keep_tag': '',
        'look_metric_name': '',
        'look_metric_value': '',
        'look_metric_few': '',
        'look_metric_medium': '',
        'look_metric_many': '',
        'exact_match_accuracy': payload.get('exact_match_accuracy'),
        'n_samples': payload.get('n_samples'),
        'n_predictions': payload.get('n_predictions'),
        'n_failures': payload.get('n_failures'),
        'probe_model_name': payload.get('probe_model_name'),
        'relative_path': '/'.join(rel_parts),
        'path': str(path),
    }

    teacher_match = re.search(r'(att_only_postvision|splus_postvision|att_only_answer|splus_answer)', str(path))
    if teacher_match:
        row['teacher'] = teacher_match.group(1)

    method_match = re.search(r'/(linear|mlp)/', str(path))
    if method_match:
        row['method'] = method_match.group(1)

    keep_match = re.search(r'(keep_0p\d+)', str(path))
    if keep_match:
        row['keep_tag'] = keep_match.group(1)

    look_eval = payload.get('look_eval') or {}
    if 'Accuracy' in look_eval:
        row['look_metric_name'] = 'Accuracy'
        row['look_metric_value'] = look_eval.get('Accuracy')
        lvl = look_eval.get('image_quantity_level-Accuracy') or {}
        row['look_metric_few'] = lvl.get('Few')
        row['look_metric_medium'] = lvl.get('Medium')
        row['look_metric_many'] = lvl.get('Many')
    elif 'ROUGE-L' in look_eval:
        row['look_metric_name'] = 'ROUGE-L'
        row['look_metric_value'] = look_eval.get('ROUGE-L')
        lvl = look_eval.get('image_quantity_level-ROUGE-L') or {}
        row['look_metric_few'] = lvl.get('Few')
        row['look_metric_medium'] = lvl.get('Medium')
        row['look_metric_many'] = lvl.get('Many')

    return {k: _clean(v) for k, v in row.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _clean(row.get(k, '')) for k in fieldnames})


def export_eval_summaries() -> tuple[Path, Path, Path]:
    all_metrics = sorted(ORACLE_ROOT.rglob('metrics.json'))
    eval_rows = [_parse_eval_row(path) for path in all_metrics]
    probe_rows = [row for row in eval_rows if row['run_type'] == 'probe']

    eval_fieldnames = [
        'run_type', 'group', 'dataset', 'mode', 'teacher', 'teacher_score_name', 'method',
        'image_keep_ratio', 'keep_tag', 'look_metric_name', 'look_metric_value',
        'look_metric_few', 'look_metric_medium', 'look_metric_many', 'exact_match_accuracy',
        'n_samples', 'n_predictions', 'n_failures', 'probe_model_name', 'relative_path', 'path',
    ]

    all_csv = ORACLE_ROOT / 'all_eval_results_summary.csv'
    probe_csv = ORACLE_ROOT / 'scienceqa_probe_results_summary.csv'
    best_csv = ORACLE_ROOT / 'scienceqa_probe_best_results_summary.csv'

    _write_csv(all_csv, eval_rows, eval_fieldnames)
    _write_csv(probe_csv, probe_rows, eval_fieldnames)

    best_by_dataset: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in probe_rows:
        dataset = str(row.get('dataset', ''))
        if not dataset:
            continue
        raw_value = row.get('look_metric_value', '')
        value = float(raw_value) if raw_value not in ('', None) else float('-inf')
        if dataset not in best_by_dataset or value > best_by_dataset[dataset][0]:
            best_by_dataset[dataset] = (value, row)
    best_rows = [best_by_dataset[key][1] for key in sorted(best_by_dataset)]
    _write_csv(best_csv, best_rows, eval_fieldnames)
    return all_csv, probe_csv, best_csv


def export_probe_training_summary() -> Path:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(SQ_ROOT.rglob('metrics.csv')):
        probe_root = metrics_path.parent
        run_config = {}
        dataset_spec = {}
        run_config_path = probe_root / 'run_config.json'
        dataset_spec_path = probe_root / 'dataset_spec.json'
        if run_config_path.exists():
            run_config = json.loads(run_config_path.read_text())
        if dataset_spec_path.exists():
            dataset_spec = json.loads(dataset_spec_path.read_text())

        with metrics_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for metric_row in reader:
                row = {
                    'probe_root': str(probe_root),
                    'metrics_path': str(metrics_path),
                    'teacher_target': run_config.get('target_score_name') or run_config.get('train_shard_dir'),
                    'input_dim': dataset_spec.get('input_dim'),
                    'output_dim': dataset_spec.get('output_dim'),
                    'n_layers': dataset_spec.get('n_layers'),
                }
                row.update(metric_row)
                rows.append({k: _clean(v) for k, v in row.items()})

    fieldnames = sorted({key for row in rows for key in row.keys()})
    out_csv = SQ_ROOT / 'probe_training_results_summary.csv'
    _write_csv(out_csv, rows, fieldnames)
    return out_csv


def main() -> None:
    all_csv, probe_csv, best_csv = export_eval_summaries()
    train_csv = export_probe_training_summary()
    print(all_csv)
    print(probe_csv)
    print(best_csv)
    print(train_csv)


if __name__ == '__main__':
    main()
