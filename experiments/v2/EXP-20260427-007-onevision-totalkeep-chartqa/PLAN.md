## Experiment Plan

**ID**: EXP-20260427-007-onevision-totalkeep-chartqa
**Author**: Codex
**Date**: 2026-04-27
**Status**: [ ] Planned  [ ] Running  [x] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
LLaVA-OneVision student pruning 평가에서 `keep_ratio`가 전체 토큰 기준이 아니라 이미지 토큰 기준으로 적용되어 있었다. 논문 비교와 맞추기 위해 전체 prompt token budget 기준으로 ChartQA 성능을 다시 확인한다.

### 2. 가설 (Hypothesis)
텍스트 토큰을 항상 유지하는 image-only pruning에서는 전체 토큰 기준 keep ratio가 이미지 토큰 기준보다 더 강한 압축이 된다. 따라서 동일 숫자 ratio의 ChartQA 정확도는 이전 image-token-basis 결과보다 낮아질 가능성이 크다.

### 3. 독립변수 (What we change)
- `keep_ratio`: 0.5, 0.1, 0.05
- keep ratio basis: image-token basis에서 total-token basis로 변경

### 4. 종속변수 (What we measure)
- 주요 metric: `ChartQA_TEST` overall accuracy
- 보조 metric: human / augmented split accuracy, first-sample effective image token keep budget

### 5. 고정 조건 (What stays the same)
- 데이터셋: `ChartQA_TEST`
- 모델: `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf`
- student: `/workspace/zap/ckpts/student_onevision_A_lr1e4_20ep`
- decoding: greedy manual decode, `max_new_tokens=32`
- 하드웨어: GPU 0/1/2에 ratio별 병렬 실행

### 6. 베이스라인
- 이전 OneVision student ChartQA image-token-basis runs: 50%, 10%, 1%
- Full cache OneVision ChartQA 결과가 있으면 별도 비교

### 7. 예상 결과
같은 숫자 ratio에서 total-token-basis는 더 적은 이미지 토큰을 유지하므로 0.5 > 0.1 > 0.05 순으로 정확도가 나올 가능성이 높다.

### 8. 판단 기준 (Success Criteria)
세 ratio 모두 실패 없이 `ChartQA_TEST_acc.csv`가 생성되고, 로그에 `keep_ratio_basis=total` 및 실제 이미지 token keep 수가 기록된다.

### 9. 이 실험으로 증명할 수 없는 것
image-only pruning은 텍스트 토큰을 항상 유지하므로 global cache compression과 동일한 압축 scope가 아니다. 전체 토큰 budget을 맞추더라도 텍스트 토큰 수가 budget을 넘는 샘플은 이미지 토큰이 0개 남을 수 있다.

### 10. 예상 런타임 / 리소스
- GPU: 0, 1, 2
- 예상 시간: ChartQA full split 기준 수십 분 이상
- 디스크: eval results 및 로그 수십 MB
