# Worklog — ZAP OneVision video KV-eviction evaluation

Goal: run LLaVA-OneVision-7B with **student-scorer KV eviction** at
`image_keep_ratio=0.1` on **Video-MME** and **SEED-Bench**, using the grading
(scoring) tasks from `../look_rebuttal`, with the checkpoint
`student_onevision_answer_n1800_e15_seed0`. One dataset per GPU (2 and 3).

## What was built

1. **Graders ported from `../look_rebuttal`** into `onevision_zap/lmms_tasks/`:
   - `videomme_local/` — Video-MME (`videomme_local.yaml`, `videomme.yaml`,
     `utils.py`). `VIDEO_ROOT` repointed to the local mp4 directory.
   - `seedbench_local/` — SEED-Bench (`seedbench_local.yaml`, `utils.py`), with a
     `seed_filter_video` `process_docs` hook so only video questions are scored.
2. **Run scripts** at repo root: `run_videomme_gpu2.sh` (GPU 2),
   `run_seedbench_gpu3.sh` (GPU 3). `SMOKE_LIMIT=N` adds `--limit N`.
3. **Adapter fix** in `run_lmms.py` (CUDA context before decord/av — see below).

The eviction path itself (`model.py` → `eviction.py` → student scorer in
`student.py`) was already present and is used as-is; the run confirms
`image_keep_ratio ≈ 0.1` via the printed `[zap-onevision]` line.

## Problems hit and fixes

| Problem | Root cause | Fix |
|---|---|---|
| Segfault (exit 139) right after "Loading vision tower" | `decord`/PyAV imported before torch's CUDA context; accelerate `device_map="auto"` then segfaults in `torch.cuda._lazy_init` | Force CUDA context at top of `run_lmms.py` (`torch.cuda.init()` + device tensor) |
| SEED load → `DatasetGenerationError`, 0 examples | this `datasets` version does not expand a `*.parquet` glob string | list all 273 parquet paths explicitly in the yaml |
| Single video-subset parquet unreadable | 22 GB single file → `ArrowNotImplementedError: Nested data conversions ... chunked` | dropped the subset copy; read the original parquet and filter in-memory instead |
| Root filesystem hit 100% (0 B) | HF Arrow build cache (~26 GB) landing on the small root overlay; a 22 GB subset copy | set `HF_DATASETS_CACHE` to `/workspace` (roomy `sdb1`); deleted regenerable HF caches (kept siglip + Video-MME cache) |
| SEED `process_docs` filter took ~7 min and looped | `.filter` decoded every image to read `data_type` | filter with `input_columns=["data_type"]` — no image decode, runs in seconds |

## Verification

- Video-MME `--limit 2` smoke: model loads past the vision tower, student-scorer
  eviction runs at keep-ratio 0.1, grader writes aggregate + per-sample results.
- SEED (video-only) `--limit 2` smoke: video path (`is_video=True`, e.g.
  `image_tokens=1569 → kept 157`) works; `seed_video`/`seed_all` computed.

## Status at handoff

- **Video-MME (GPU 2)**: running full (2700 questions, `max_frames_num=32`,
  `image_keep_ratio=0.1`).
- **SEED-Bench (GPU 3)**: stopped at the user's request — not being run.

## Not committed (large local artifacts, git-ignored)

`student_onevision_answer_n1800_e15_seed0/`, `onevision_st.tar.gz`, `data/`,
`.hf_datasets_cache/`, `videomme/`, `results/`.
