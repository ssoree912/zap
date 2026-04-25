# zap

`zap` is a trimmed repository for the current LLaVA image-token pruning workflow.

The retained code supports four stages:

1. Build ScienceQA manifests from `/workspace/zap/data/scienceqa`.
2. Collect unified teacher shards (PV + Future labels in a single forward pass).
3. Train per-layer MLP probes (PostVision / Future / Hybrid) on those shards.
4. Evaluate oracle / probe-based image-token pruning on MileBench and measure
   efficiency against full cache and LOOK-M; or run PPL on PrefixKV datasets.

## Retained entrypoints

Core Python entrypoints:
- `build_scienceqa_manifest.py`: ScienceQA sample loading and prompt construction.
- `collect_unified_teacher_shards.py`: collect unified `(x, y_pv, y_future)` shards.
- `train_unified_probe_onepass.py`: per-layer softmax-MSE probe training (teacher = pv or future).
- `evaluate_image_teacher_pruning.py`: oracle / probe pruning evaluation on MileBench.
- `eval_ppl.py`: teacher-forcing PPL on PrefixKV-style datasets (detail_1k, mm-vet).

Core helper modules:
- `kvzap/image_teacher_utils.py`: prompt loading, dataset loading, teacher record helpers.
- `kvzap/llava_extractor.py`: LLaVA-specific extraction helpers, including no-forward image-position recovery.
- `kvzap/milebench_look_metrics.py`: LOOK-style metric wrappers.
- `kvpress/presses/base_press.py`: minimal hook-based cache compression base class.
- `kvpress/presses/image_token_press.py`: image-only top-k pruning, oracle / probe / hybrid / future presses.
- `kvpress/presses/kvzap_press.py`: retained `KVzapConfig` and `KVzapModel` definitions used by probe training.

Execution scripts:
- `scripts/run_milebench_probe_all.sh`
- `scripts/measure_milebench_efficiency.py`
- `scripts/run_efficiency_successful20_kv_only.sh`
- `scripts/run_efficiency_successful20_look_only.sh`
- `scripts/export_scienceqa_probe_csv_summaries.py`
- `scripts/export_probe_lookm_performance_csv.py`
- `scripts/export_efficiency_comparison_csv.py`

## Current workflow

### 1. Collect unified teacher shards

```bash
python collect_unified_teacher_shards.py \
  --dataset scienceqa \
  --data_dir /workspace/zap/data/scienceqa \
  --out_dir /workspace/zap/artifacts/teacher/unified/scienceqa \
  --limit 500 --device cuda:0
```

Add `--all_token_targets` to collect labels at image + text prompt positions.

### 2. Train per-layer probe

```bash
python train_unified_probe_onepass.py \
  --shard_dirs /workspace/zap/artifacts/teacher/unified/scienceqa \
               /workspace/zap/artifacts/teacher/unified/textvqa \
               /workspace/zap/artifacts/teacher/unified/nlvr2 \
  --teacher future \
  --out_dir /workspace/zap/ckpts/future_probe \
  --n_layers_model 32 \
  --selected_layers 0 1 2 ... 31 \
  --mlp_max_epochs 10 --device cuda:0
```

`--teacher pv` trains the PostVision probe used by Hybrid at inference.

### 3. Evaluate pruning on MileBench

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

- Active pruning methods: image-only top-k (oracle / probe / hybrid / future).
- `full_cache` does not run image-position inference.
- `probe` uses no-forward LLaVA image-position recovery.
- Long-running jobs should be launched with `screen` (or `nohup`) and a log file.
