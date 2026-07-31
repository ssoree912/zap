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

## RTX 3090 rebuttal experiment snapshot

Snapshot date: 2026-07-31. Unless stated otherwise, all measurements below
used NVIDIA GeForce RTX 3090 24 GB GPUs, batch size 1, BF16, and seed 0.
Scores are percentages. A requested total-prompt keep ratio of 0.2 preserves
all text KVs and chooses the visual-KV count so that the retained text and
visual KVs together are 20% of the prompt cache.

### LLaVA-1.5-7B scorer capacity

The LLaVA experiment collected 1,800 Zap teacher records: 600 each from
TextVQA, ScienceQA, and GQA. Small, base, and large students used the same
teacher set, 90/10 train/validation split, and 15-epoch training schedule.
Evaluation used complete DocVQA, TextVQA, and ChartQA sets at a 0.2 total
prompt-KV keep ratio.

| Scorer | Parameters (M) | Scoring latency (ms) | Added VRAM (MiB) | ChartQA | DocVQA | TextVQA | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 74.43 | 23.15 | 168.7 | 17.80 | 26.81 | 45.76 | **30.12** |
| Base | 156.18 | 24.72 | 449.8 | **17.84** | **26.86** | 45.59 | 30.10 |
| Large | 309.17 | 26.33 | 622.3 | 17.56 | 26.73 | 45.56 | 29.95 |

Increasing capacity beyond the base scorer did not improve the three-task
mean. Small retained the best measured mean while using 52% fewer parameters
than base. Base remains the default architecture because it is the controlled
1x configuration and its task differences from small are negligible.

The base scorer has 156.18M parameters, or 2.21% of the measured
LLaVA-1.5-7B model. One all-32-layer call is 141.15 GFLOPs. On real LLaVA
prefill hidden states it took 27.07 ms on average (26.59 ms median); this
measurement excludes the frozen-backbone prefill.

### LLaVA-1.5-7B offline cost

| Stage | GPU allocation | Samples/work | Wall-clock | GPU-hours | Peak allocated VRAM |
|---|---|---|---:|---:|---:|
| Teacher-signal generation | RTX 3090 x1 | 1,800 (600/dataset) | 0.2101 h (12.61 min) | 0.2101 | 14.71 GiB |
| Base student training | RTX 3090 x1 | 1,620 train / 180 val, 15 epochs, 24,300 steps | 2.9420 h | 2.9420 | 16.28 GiB |
| **Total offline** | RTX 3090 x1 | - | **3.1521 h** | **3.1521** | - |

The backbone was frozen for both stages. Teacher extraction was a one-time
full-cache autoregressive pass. Prefill features were not cached during the
final student run, so the frozen-backbone forward was recomputed during
training and is included in the reported 2.9420 GPU-hours. In this measured
configuration student training, not teacher extraction, dominates offline
cost.

### Qwen2-VL-7B transfer

The Qwen2-VL student was trained from a separately generated 1,800-record
teacher set with the same 600 samples per training dataset. The VQA table uses
full evaluation sets and a 0.2 total-prompt keep ratio. `PrefixKV (all-token)`
is the run that budgets text and image KVs together; Zap preserves text KVs
and uses its student to select visual KVs.

| Method | TextVQA | ChartQA | DocVQA | GQA | Mean |
|---|---:|---:|---:|---:|---:|
| Full Cache | **80.12** | **83.24** | **92.52** | **61.77** | **79.41** |
| PrefixKV (all-token) | 71.42 | 65.12 | 81.08 | 61.42 | 69.76 |
| LOOK-M | 48.90 | 27.76 | 51.24 | 60.50 | 47.10 |
| VisionZip | 51.10 | 43.60 | 37.86 | 53.20 | 46.44 |
| **Zap student** | 79.17 | 77.72 | 81.53 | 61.74 | 75.04 |

Captioning is reported with ROUGE-L x100.

| Method | COCO Caption | NoCaps | TextCaps | Mean |
|---|---:|---:|---:|---:|
| PrefixKV (all-token) | 53.44 | **58.35** | 51.87 | 54.55 |
| LOOK-M | 49.62 | 32.65 | 33.97 | 38.75 |
| VisionZip | 48.44 | 56.67 | 48.32 | 51.14 |
| **Zap student** | **53.89** | 57.90 | **53.04** | **54.95** |

At this snapshot, the Qwen full-cache caption run is still in progress and is
therefore not included as a completed result.

Qwen offline cost was 0.6151 GPU-hours for teacher generation and 3.5019
GPU-hours for 15-epoch student training, for 4.1170 GPU-hours total. Teacher
and student peak allocated VRAM were 16.35 and 17.91 GiB. The 125.05M
parameter, 28-layer scorer took 23.96 ms per prompt and added 393 MiB peak
allocated VRAM in the scorer-only profile.

### Teacher-data and optimization ablations

The LLaVA adaptation sweep randomly initialized every scorer and kept the
backbone frozen. Generated rows use autoregressively generated answer
attention. The reference row teacher-forces the dataset reference response.
All rows below were evaluated at 0.2 total prompt-KV retention.

| Teacher response | Records | Epochs | ChartQA | DocVQA | TextVQA | GQA | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Generated | 300 | 15 | 16.52 | 23.76 | 43.56 | 61.86 | 36.43 |
| Generated | 900 | 15 | 16.44 | 23.93 | **44.04** | **61.87** | **36.57** |
| Generated | 1,800 | 3 | **16.28** | 23.78 | 43.95 | 61.86 | 36.47 |
| Generated | 1,800 | 7 | 15.96 | **23.89** | 43.41 | 61.86 | 36.28 |
| Generated | 1,800 | 10 | 16.12 | 23.72 | 43.51 | 61.85 | 36.30 |
| Reference | 1,800 | 15 | **17.80** | **27.01** | **45.72** | **61.89** | **38.11** |

Generated-answer performance saturated early: neither more records nor more
than three epochs produced a consistent gain in this sweep. Reference-response
supervision improved the four-task mean by about 1.5 points over the best
generated row. For the completed generated 1,800-record caption runs, the
ROUGE-L means were 52.82 at 7 epochs and 52.92 at 10 epochs.

The OneVision warm-start experiment transferred the LLaVA scorer by relative
layer depth and evaluated ChartQA after epoch 1 at 10% image-KV retention.
Only this first checkpoint is treated as complete in this snapshot.

| OneVision teacher records | Scratch | LLaVA warm start | Warm-start delta |
|---:|---:|---:|---:|
| 450 | 77.92 | 78.44 | +0.52 |
| 900 | 78.36 | 77.92 | -0.44 |

The epoch-1 result does not show a consistent warm-start benefit.

### Teacher-signal construction

Prefill+answer and question+answer teachers were trained and evaluated as a
paired LLaVA comparison at 0.2 total prompt-KV retention.

| Teacher construction | GQA | TextVQA | DocVQA | ChartQA | VQA mean |
|---|---:|---:|---:|---:|---:|
| Prefill + answer | 61.81 | 45.11 | **25.82** | **17.48** | **37.55** |
| Question + answer | **61.89** | **45.12** | 24.90 | 16.56 | 37.12 |

| Teacher construction | COCO ROUGE-L | NoCaps ROUGE-L | TextCaps ROUGE-L | Caption mean |
|---|---:|---:|---:|---:|
| Prefill + answer | **56.08** | **58.96** | 46.73 | **53.92** |
| Question + answer | 55.70 | 58.27 | **46.79** | 53.59 |

Question+answer did not improve the aggregate result: it was 0.44 points lower
on the VQA mean and 0.34 points lower on the caption ROUGE-L mean.

### Selector diagnostics

These experiments test whether the learned signal behaves differently from
prefill attention and H2O. They are diagnostic analyses and use different
sample scopes and budget definitions, so their absolute scores should not be
combined into one benchmark average.

| Experiment | Main result |
|---|---|
| LLaVA held-out teacher split, 180 samples, exact 20% total budget | Q-ViK/Future Jaccard was 0.1667 versus 0.1606 for Prefill/Future, a paired +0.0061. Downstream Q-ViK was 0.5407 versus Prefill 0.5463; the -0.0056 difference was not resolved. |
| LLaVA H2O comparison, 180 samples | Future Top-K recall was 0.4969 for Q-ViK and 0.3863 for H2O; Jaccard was 0.3431 versus 0.2521. Downstream scores were 0.5574 and 0.5519, with a paired CI that included zero. |
| OneVision MileBench, 800 samples, 20% total budget | Prefill/Future Spearman was 0.7321 and Q-ViK/Future was 0.4673. Downstream macro was 41.80 for Prefill, 41.46 for Q-ViK, and 41.89 for Full Cache. |
| LLaVA MileBench, 5,535 shared samples, 20% visual-KV budget | Official-28 macro was 31.1990 for H2O and 31.0670 for Q-ViK. Q-ViK minus H2O was -0.1320 with 95% CI [-0.2403, -0.0253]. |

The diagnostics show mixed evidence. Q-ViK selects more Future top-K tokens
than H2O on the held-out training-domain sample, but this does not establish a
downstream advantage, and the large MileBench visual-budget run slightly
favored H2O.

### Post-eviction decode latency

The timed region begins after prefill, scorer execution, and KV eviction. It
therefore measures cached autoregressive decode only. The 32-step VQA run uses
100 samples; the long run uses 50 MM-Vet samples and 256 fixed decode steps.

| Backbone and budget | Decode length | Full Cache (ms/token) | Zap (ms/token) | Reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B, 20% total prompt-KV | 32 | 23.85 | 22.35 | 6.26% | 1.069x |
| LLaVA-1.5-7B, 20% total prompt-KV | 256 | 24.54 | 22.76 | 7.24% | 1.078x |
| LLaVA-OneVision-Qwen2-7B, 10% image-KV | 32 | 36.76 | 24.59 | 32.15% | 1.492x |
| LLaVA-OneVision-Qwen2-7B, 10% image-KV | 256 | 36.67 | 25.08 | 29.59% | 1.456x |

Because scoring and eviction are excluded, these numbers quantify the decode
benefit after compression rather than end-to-end request latency.

### Checkpoints, teacher data, and exports

Runtime artifacts are intentionally not tracked by Git.

| Artifact | Local path |
|---|---|
| LLaVA small/base/large checkpoints | `artifacts/rebuttal_tradeoff_llava15_zap_teacher_n600/checkpoints/` |
| LLaVA teacher records | `/workspace/nips/data/train/teacher/zap_llava15_n600_seed0/` |
| LLaVA base student archive | `artifacts/rebuttal_tradeoff_llava15_zap_teacher_n600/exports/llava15_student_base_n600.tar.gz` |
| LLaVA teacher archive | `artifacts/rebuttal_tradeoff_llava15_zap_teacher_n600/exports/llava15_teacher_n600x3.tar.gz` |
| Qwen base checkpoint | `artifacts/rebuttal_qwen2vl_zap_cost_n600/checkpoints/base/` |
| Qwen teacher records | `/workspace/nips/data/train/teacher/zap_qwen2vl_n600_seed0/` |
| Qwen student archive | `artifacts/rebuttal_qwen2vl_zap_cost_n600/exports/qwen2vl_student_base_n600.tar.gz` |
| Qwen teacher archive | `artifacts/rebuttal_qwen2vl_zap_cost_n600/exports/qwen2vl_teacher_n600x3.tar.gz` |

Each `tar.gz` export has a matching `.sha256` file and passed `gzip -t` and
checksum verification.
