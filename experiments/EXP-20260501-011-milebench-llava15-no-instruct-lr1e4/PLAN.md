## Experiment Plan

**ID**: EXP-20260501-011-milebench-llava15-no-instruct-lr1e4
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
`lr=1e-4`로 재훈련한 LLaVA-1.5 no-instruct student checkpoint의 MileBench downstream 성능을 확인한다.

### 2. 가설 (Hypothesis)
훈련 loss가 낮아진 `student_llava15_no_instruct_1800_lr1e4`는 이전 `lr=1e-3`/LLaVA-Instruct 포함 checkpoint보다 안정적인 pruning 성능을 보일 것이다.

### 3. 독립변수 (What we change)
- Student checkpoint: `/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4`
- Keep ratios: `0.5`, `0.1`

### 4. 종속변수 (What we measure)
- MileBench task별 `eval.json` metric
- task별 `pred.json`
- token keep statistics

### 5. 고정 조건 (What stays the same)
- Base model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Evaluation script: `foresight/eval/milebench_llava15_student.py`
- Dataset: all MileBench tasks under `/workspace/zap/data/MileBench`
- Generation: greedy, `MAX_NEW_TOKENS=32`
- KV eviction semantics: direct-generate compatible first-token behavior
- Hardware: GPU0

### 6. 베이스라인
- Previous `student_llava15_v2`
- Previous `student_llava15_mmvet_instruct`
- LOOK-M / PrefixKV results from existing experiments

### 7. 예상 결과
`keep_ratio=0.5`가 `keep_ratio=0.1`보다 안정적일 것이며, loss가 낮아진 checkpoint가 이전 student보다 개선될 가능성이 있다.

### 8. 판단 기준 (Success Criteria)
전체 task 추론이 실패 없이 완료되고, task별 `pred.json` 및 `eval.json`이 생성된다.

### 9. 이 실험으로 증명할 수 없는 것
훈련 loss만으로 downstream 성능이 보장되지는 않는다. 또한 image-only pruning이므로 global cache compression과 compression scope가 다르다.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090, GPU0
- 예상 시간: 수십 분 이상
- 디스크: prediction/eval JSON 파일
