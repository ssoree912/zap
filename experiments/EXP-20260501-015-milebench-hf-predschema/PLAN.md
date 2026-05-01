## Experiment Plan

**ID**: EXP-20260501-015-milebench-hf-predschema
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation
Regenerate MileBench predictions instead of reusing existing `pred.json`, with
the LOOK-M-style prediction record from `scripts/evaluate_image_teacher_pruning.py`.

### 2. Hypothesis
The patched `pred.json` schema is compatible with `/workspace/look-m/evaluate.py`,
which now calls the repo-local `LookMileBenchEvaluator`.

### 3. What We Change
- Fresh generation output directory.
- `pred.json` schema includes `image`, `question`, `gen_model_id`, and `gen_kwargs`.

### 4. What We Measure
- MileBench per-dataset LOOK-M-compatible metrics.
- Generation success/failure through logs and `pred.json` presence.

### 5. Fixed Conditions
- Data: `/workspace/zap/data/MileBench`
- Model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Student: `/workspace/zap/ckpts/student_llava15_instruct_2000_lr1e4`
- Image input: `combined_1_images`
- Prompt style: `look_milebench`

### 6. Baseline
- Existing old outputs under `EXP-20260501-007...` are not overwritten.

### 7. Expected Result
All generated datasets should have LOOK-M-style `pred.json`, `eval.json`, and
`eval_score.json`.

### 8. Success Criteria
The run completes all MileBench datasets for keep=0.5 with zero script-level
crashes, and the evaluator writes scores for each dataset.

### 9. Caveats
- This run uses the HF LLaVA-1.5 path with combined images, not the currently
  broken `/workspace/look-m/LLaVA-mix_merge_v1` import path.
- Comparison against raw LOOK-M multi-image runs should account for input path
  differences.

### 10. Runtime / Resources
- GPU: one RTX 4090
- Expected time: hours for full MileBench
- Disk: prediction JSON and logs under this experiment directory
