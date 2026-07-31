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

## Rebuttal experiments (branch `rebuttal/question+answer`)

Summary of the experiments run for the rebuttal. Result artifacts live under each
experiment directory; model checkpoints and teacher shards are not committed
(`ckpts/`, `artifacts/` are gitignored).

### 1. Teacher-composition ablation — LLaVA-1.5, ChartQA (EXP-20260502-024)

Compares what the teacher aggregates over: attention from generated **answer
tokens only** (`--question-weight 0.0`) vs a 50/50 mix of **question + answer**
token attention (`--question-weight 0.5`). Same student architecture, optimizer,
KV budget, and number of updates; evaluated with the zap press at total prompt-KV
keep 0.2 on `chartqa_local` (lmms-eval, relaxed accuracy).

| Teacher | ChartQA relaxed_overall | human | augmented |
| --- | ---: | ---: | ---: |
| Answer-only (`answer_chartqa_keep02`) | 16.32 | 19.76 | 12.88 |
| Question+answer 0.5/0.5 (`question_answer_chartqa_keep02`) | **16.88** | **20.08** | **13.68** |

Pipelines: `run_answer_reextract_retrain_gpu0.sh` (answer-only) and
`run_qa50_reextract_retrain_gpu0.sh` (0.5/0.5). Earlier commits on this branch
also carry the paired keep-0.1 evaluation and the keep-0.2 zap run.

### 2. Answer-agnostic OneVision teacher + training (EXP-20260503-026)

`collect_original_onevision_teacher.py` now collects **answer-agnostic**
future-attention labels: training samples are drawn deterministically from each
split with no correctness gate on the generated answer (`prediction_correct` is
recorded but `require_correct=False`). Generated answer tokens serve only as
future query positions; the per-layer teacher remains the head/step-averaged
answer→image attention normalized over image tokens.
`train_original_onevision_student.py` gains `--resume-from last_checkpoint.pt`.
Launchers: `run_answer_agnostic_2gpu.sh`, `run_train_gpu0_answer_agnostic.sh`,
`run_resume_train_2gpu.sh`.

### 3. TTFT bench — full cache vs student press (EXP-20260503-027)

`bench_onevision_ttft.py` measures time-to-first-token for LLaVA-OneVision-7B
on GPU 0 (RTX 4090, fp16, sdpa): full cache vs `VisualUtilityStudentOneVisionPress`
at keep 0.2 / 0.05. Synthetic images sweep prompt length (single image at
336/672/1008 px anyres, plus 2/4/8-image prompts); medians of 3 timed runs after
2 warmups; lm_head applied to the last position only (both arms).

| L_p | full | student keep 0.2 | overhead |
| ---: | ---: | ---: | ---: |
| 1,517 | 169 ms | 217 ms | +28.0% |
| 3,003 | 345 ms | 395 ms | +14.6% |
| 3,731 | 440 ms | 499 ms | +13.6% |
| 5,975 | 726 ms | 800 ms | +10.2% |
| 7,403 | 911 ms | 1,000 ms | +9.7% |
| 11,919 | 1,535 ms | 1,652 ms | **+7.6%** |

TTFT cannot improve (pruning follows a full prefill). The student forward is a
constant ~21 ms regardless of prompt length and keep ratio; the remainder is
per-layer KV gather/prune. Relative overhead shrinks as prompts grow — 7.6% at
L_p≈12k — while the method's gains come from decode latency and prompt-KV
memory. Results: `outputs/ttft_gpu0/ttft_results.{json,csv}`.
