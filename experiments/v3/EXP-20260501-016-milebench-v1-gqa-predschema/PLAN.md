## Experiment Plan

**ID**: EXP-20260501-016-milebench-v1-gqa-predschema
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation
Run fresh MileBench generation with the requested v1 GQA student checkpoint,
using the same LOOK-M-style `pred.json` schema added for the current rerun.

### 2. Hypothesis
`/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4` can be evaluated through the
HF LLaVA-1.5 MileBench path and scored by patched `/workspace/look-m/evaluate.py`.

### 3. What We Change
- Student checkpoint: `/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4`
- GPU: physical GPU 2

### 4. What We Measure
- MileBench per-dataset LOOK-M-compatible metrics.
- Generation completion and `pred.json` schema correctness.

### 5. Fixed Conditions
- Backbone: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Data: `/workspace/zap/data/MileBench`
- Keep ratio: `total_keep_ratio=0.5`
- Image input: `combined_1_images`
- Prompt style: `look_milebench`

### 6. Baseline
- Existing fresh run with `/workspace/zap/ckpts/student_llava15_instruct_2000_lr1e4`
  in `EXP-20260501-015-milebench-hf-predschema`.

### 7. Expected Result
The run writes full MileBench outputs under `outputs/keep05` with LOOK-M-style
prediction records and evaluation files.

### 8. Success Criteria
All datasets complete without script-level crashes and produce `pred.json` plus
`eval.json`.

### 9. Caveats
- This uses HF LLaVA-1.5 with combined images, not the currently broken
  `/workspace/look-m/LLaVA-mix_merge_v1` import path.

### 10. Runtime / Resources
- GPU: RTX 4090 GPU 2
- Expected time: hours for full MileBench
- Disk: JSON outputs and logs under this experiment directory
