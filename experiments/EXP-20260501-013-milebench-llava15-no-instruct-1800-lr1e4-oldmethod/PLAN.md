## Experiment Plan

**ID**: EXP-20260501-013-milebench-llava15-no-instruct-1800-lr1e4-oldmethod
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
최근 학습한 LLaVA-1.5 no-instruct 3-dataset 1800-sample student checkpoint를 과거 MileBench 평가 방식에 가깝게 재평가한다.

### 2. 가설 (Hypothesis)
`combined_1_images` 입력과 `VisualUtilityStudentPress + model.generate()` 경로를 사용하면 단일-image manual decode 평가보다 과거 kvpress 평가 조건에 더 가깝다.

### 3. 독립변수 (What we change)
- Student checkpoint: `/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4`
- Keep ratios: `total_keep_ratio=0.5`, `0.1`
- Evaluation backend: `press`

### 4. 종속변수 (What we measure)
- MileBench task별 `pred.json`
- MileBench task별 `eval.json`
- token keep statistics

### 5. 고정 조건 (What stays the same)
- Base model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Dataset: all MileBench tasks
- Image input field: `combined_1_images`
- Generation: greedy, `max_new_tokens=32`
- Hardware: GPU0

### 6. 베이스라인
- 과거 v2 `student_v2_A_gqa_lr1e4` MileBench result CSV
- 기존 단일-image/manual decode LLaVA15 student outputs

### 7. 예상 결과
ALFRED/MMCoQA 같은 text-history 중심 task에서 단일-image/manual decode 결과와 다른 점수가 나올 수 있다.

### 8. 판단 기준 (Success Criteria)
전체 MileBench task에 대해 `keep05`, `keep01` 결과 파일이 생성된다.

### 9. 이 실험으로 증명할 수 없는 것
이 실험만으로 학습 데이터 구성의 일반화 효과와 평가 backend 차이를 완전히 분리할 수는 없다.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090, GPU0
- 예상 시간: 수십 분 이상
- 디스크: prediction/eval JSON 파일
