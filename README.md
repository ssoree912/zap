# zap

`zap` is a trimmed repository for the current LLaVA image-token pruning workflow.

The retained code supports four stages only:

1. Build ScienceQA manifests from `/workspace/hd/data/scienceqa`.
2. Collect image-token hidden states and postvision teacher targets into shard datasets.
3. Train linear or MLP probes on those shards.
4. Evaluate oracle or probe-based image-token pruning on MileBench and measure efficiency against full cache and LOOK-M.

## Retained entrypoints

Core Python entrypoints:
- `build_scienceqa_manifest.py`: ScienceQA sample loading and prompt construction.
- `collect_scienceqa_teacher_xy.py`: collect `(X, y, layer)` shards from ScienceQA.
- `train_image_teacher_probe.py`: shared probe model and training utilities.
- `train_image_teacher_probe_shards.py`: shard-based linear / MLP probe training.
- `build_docvqa_teacher4.py`: build oracle teacher records for MileBench-style datasets.
- `evaluate_image_teacher_pruning.py`: oracle / probe pruning evaluation.

Core helper modules:
- `kvzap/image_teacher_utils.py`: prompt loading, dataset loading, teacher record helpers.
- `kvzap/llava_extractor.py`: LLaVA-specific extraction helpers, including no-forward image-position recovery.
- `kvzap/milebench_look_metrics.py`: LOOK-style metric wrappers.
- `kvpress/presses/base_press.py`: minimal hook-based cache compression base class.
- `kvpress/presses/image_token_press.py`: image-only top-k pruning, oracle press, probe press.
- `kvpress/presses/kvzap_press.py`: retained `KVzapConfig` and `KVzapModel` definitions used by probe training.

Execution scripts:
- `scripts/run_scienceqa_postvision_pipeline.sh`
- `scripts/run_docvqa_build_teacher4.sh`
- `scripts/run_docvqa_oracle_teacher_sweep.sh`
- `scripts/run_docvqa_probe_teacher_sweep.sh`
- `scripts/run_milebench_probe_all.sh`
- `scripts/measure_milebench_efficiency.py`
- `scripts/run_efficiency_successful20_kv_only.sh`
- `scripts/run_efficiency_successful20_look_only.sh`
- `scripts/export_scienceqa_probe_csv_summaries.py`
- `scripts/export_probe_lookm_performance_csv.py`
- `scripts/export_efficiency_comparison_csv.py`

## What was removed

The repository no longer keeps the generic `kvpress` benchmark library surface, legacy presses, notebook demos, old evaluation package, or obsolete pipeline scripts. The remaining code is scoped to the current ScienceQA teacher/probe and MileBench evaluation workflow only.

## Current workflow

### 1. Collect ScienceQA training shards

```bash
bash /workspace/zap/scripts/run_scienceqa_postvision_pipeline.sh
```

This writes shard datasets under `/workspace/hd/artifacts/sq_teacher`.

### 2. Build oracle teacher records for a MileBench dataset

```bash
bash /workspace/zap/scripts/run_docvqa_build_teacher4.sh
```

The script is parameterized through environment variables and can be pointed at datasets other than DocVQA.

### 3. Evaluate oracle or probe pruning

Oracle sweep:

```bash
bash /workspace/zap/scripts/run_docvqa_oracle_teacher_sweep.sh
```

Probe sweep:

```bash
bash /workspace/zap/scripts/run_docvqa_probe_teacher_sweep.sh
```

All-dataset probe sweep:

```bash
bash /workspace/zap/scripts/run_milebench_probe_all.sh
```

### 4. Measure efficiency

`kv` env for full cache and probe:

```bash
bash /workspace/zap/scripts/run_efficiency_successful20_kv_only.sh
```

`look` env for LOOK-M only:

```bash
bash /workspace/zap/scripts/run_efficiency_successful20_look_only.sh
```

Then merge the results:

```bash
/opt/conda/envs/kv/bin/python /workspace/zap/scripts/export_efficiency_comparison_csv.py
```

## Notes

- The active pruning method is image-only top-k pruning.
- `full_cache` does not run image-position inference.
- `probe` uses no-forward LLaVA image-position recovery.
- Long-running jobs should be launched with `screen` and a log file.
