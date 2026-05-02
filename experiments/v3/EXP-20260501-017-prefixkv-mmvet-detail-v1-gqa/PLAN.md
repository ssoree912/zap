## Experiment Plan

**ID**: EXP-20260501-017-prefixkv-mmvet-detail-v1-gqa
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
Run the PrefixKV-style free-generation evaluation on `mm-vet` and `detail_1k`
with the v1 GQA LLaVA-1.5 visual utility student checkpoint, separate from the
MileBench generation path.

### 2. 가설 (Hypothesis)
The v1 GQA student should produce usable ROUGE outputs on the single-image
PrefixKV datasets when evaluated with the compatible LLaVA-1.5 HF inference
stack.

### 3. 독립변수 (What we change)
- Student checkpoint: `/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4`
- Dataset: `mm-vet`, `detail_1k`
- Total keep ratio: `0.5`

### 4. 종속변수 (What we measure)
- Free-generation ROUGE-L F1 against dataset answers
- Per-sample generated predictions in `result.json`
- Failure count and runtime

### 5. 고정 조건 (What stays the same)
- Base model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Eval script: `scripts/eval_rouge.py`
- Method: `visual_utility_student`
- Generation: greedy, max new tokens default from script
- Hardware: GPU0 via `CUDA_VISIBLE_DEVICES=0`

### 6. 베이스라인
- Prior v2 PrefixKV-style runs under `experiments/v2/EXP-20260425-002/`
- Full-cache results can be added separately if direct baseline comparison is needed.

### 7. 예상 결과
`detail_1k` should be more stable than MileBench multi-image tasks because it is
single-image. `mm-vet` may have noisy ROUGE because answers are short and diverse.

### 8. 판단 기준 (Success Criteria)
- `result.json` is produced for both datasets.
- `n_failures` is zero or small enough to inspect directly.

### 9. 이 실험으로 증명할 수 없는 것
This does not isolate original LLaVA-vs-HF implementation effects, and it does
not evaluate raw multi-image MileBench behavior.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090 on GPU0
- 예상 시간: mm-vet minutes, detail_1k tens of minutes
- 디스크: JSON result files and logs
