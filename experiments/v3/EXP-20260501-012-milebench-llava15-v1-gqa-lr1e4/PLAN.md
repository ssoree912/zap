## Experiment Plan

**ID**: EXP-20260501-012-milebench-llava15-v1-gqa-lr1e4
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
기존 성능이 좋았던 `student_v2_A_gqa_lr1e4` checkpoint의 MileBench downstream 성능을 현재 direct-generate-compatible KV eviction 로직으로 다시 확인한다.

### 2. 가설 (Hypothesis)
`lr=1e-4`로 학습된 GQA 기반 v1 student가 최근 no-instruct/instruct student와 비교 가능한 pruning 성능을 보일 수 있다.

### 3. 독립변수 (What we change)
- Student checkpoint: `/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4`
- Keep ratios: `0.5`, `0.1`

### 4. 종속변수 (What we measure)
- MileBench task별 `pred.json`
- MileBench task별 `eval.json`
- token keep statistics

### 5. 고정 조건 (What stays the same)
- Base model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Evaluation script: `foresight/eval/milebench_llava15_student.py`
- Dataset: all MileBench tasks
- Generation: greedy, `MAX_NEW_TOKENS=32`
- KV eviction semantics: direct-generate-compatible first-token behavior
- Hardware: GPU0

### 6. 베이스라인
- `/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4`
- `/workspace/zap/ckpts/student_llava15_instruct_2000_lr1e4`

### 7. 예상 결과
학습 loss가 낮았던 checkpoint이므로 이전 `lr=1e-3` 계열보다 안정적인 결과를 기대한다.

### 8. 판단 기준 (Success Criteria)
전체 MileBench task에 대해 `keep05`와 `keep01` 결과 파일이 생성된다.

### 9. 이 실험으로 증명할 수 없는 것
학습 데이터가 GQA 중심이므로 전체 MileBench 일반화와 학습 loss 사이의 인과를 단독으로 증명할 수 없다.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090, GPU0
- 예상 시간: 수십 분 이상
- 디스크: prediction/eval JSON 파일
