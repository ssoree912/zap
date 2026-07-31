# ZAP OneVision — Video KV-Eviction Evaluation

Run LLaVA-OneVision-7B with **student-scorer KV eviction** (image tokens kept at
`image_keep_ratio`) through `lmms_eval`, scored by the Video-MME and SEED-Bench
graders ported from `../look_rebuttal`.

- Model: `llava-onevision-qwen2-7b-ov`
- Eviction: per-layer visual-utility **student scorer**
  (`student_onevision_answer_n1800_e15_seed0`), image tokens pruned to
  `image_keep_ratio` (default **0.1**); text tokens kept in full. The first
  decoded token comes from a post-eviction prompt replay.

## Layout

```
onevision_zap/
  run_lmms.py          # entry: registers the `zap_onevision_student` lmms model
  lmms_model.py        # lmms_eval adapter (generate_until only)
  model.py             # prefill → student scoring → evict_image_kv → replay → decode
  eviction.py          # evict_image_kv + EvictionStats
  student.py           # VisualUtilityStudent (per-layer scorer)
  decode.py            # prompt-replay first token + greedy decode
  lmms_tasks/
    videomme_local/    # Video-MME grader (yaml + utils.py)
    seedbench_local/   # SEED-Bench grader (yaml + utils.py, video-only filter)
```

Run scripts live at the repo root: `run_videomme_gpu2.sh`, `run_seedbench_gpu3.sh`.

## How to run

```bash
# Video-MME on physical GPU 2
./run_videomme_gpu2.sh

# SEED-Bench (video questions only) on physical GPU 3
./run_seedbench_gpu3.sh

# Quick smoke test (N samples) — works with either script
SMOKE_LIMIT=2 ./run_videomme_gpu2.sh
```

Both scripts pin one physical GPU (`CUDA_VISIBLE_DEVICES`) exposed as logical
`cuda:0`, use `/opt/conda/envs/VFlowOpt/bin/python`, and set `PYTHONPATH` across
`zap`, `Q-ViK`, `VFlowOpt`, `lmms_eval-0.2.4`, and `transformers-4.46.0`.

Results are written under `results/lmms_zap_onevision_keep0.1/<task>/full`
(git-ignored); the adapter prints one `[zap-onevision]` line reporting the
realized `image_keep_ratio` and kept token counts.

## Data paths (this machine)

| Dataset | Source |
|---|---|
| Video-MME parquet | `/workspace/nips/Q-ViK/data/Video-MME/videomme/test-00000-of-00001.parquet` (2700 q) |
| Video-MME videos | `/workspace/nips/Q-ViK/data/Video-MME/data/<videoID>.mp4` (900 mp4, `VIDEO_ROOT` in the task utils) |
| SEED-Bench parquet | `/workspace/nips/.cache/huggingface_official/hub/datasets--lmms-lab--SEED-Bench/.../data/*.parquet` (273 files, ~18k q; 3757 are `data_type == "video"`) |

## Notes / gotchas

- **CUDA-before-decord segfault (fixed).** `lmms_model.py` imports `decord`/PyAV
  at import time. If those load before torch's CUDA context, accelerate's
  `device_map="auto"` balanced-memory probe segfaults (exit 139) in
  `torch.cuda._lazy_init` right after "Loading vision tower". `run_lmms.py` now
  forces the CUDA context first (`torch.cuda.init()` + a device tensor).
- **Explicit parquet list.** This `datasets` version does not expand a
  `*.parquet` glob string in `data_files` (yields 0 examples → 
  `DatasetGenerationError`); the SEED yaml lists every parquet path explicitly.
- **SEED video-only.** `seed_filter_video` (a `process_docs` hook) keeps only
  `data_type == "video"`, reading just the `data_type` column
  (`input_columns=[...]`) so the image bytes are never decoded during filtering.
- **Datasets cache location.** The HF `datasets` loader must materialize parquet
  into an Arrow cache before it can be indexed. The default `~/.cache` lives on a
  small/full root overlay, so `run_seedbench_gpu3.sh` points `HF_DATASETS_CACHE`
  at `data/hf_datasets_cache` on the roomy `/workspace` volume.
- **Frames.** Videos are uniformly sampled to `max_frames_num` (default 32).
