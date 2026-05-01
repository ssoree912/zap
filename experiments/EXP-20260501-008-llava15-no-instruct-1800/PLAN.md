## Experiment Plan

**ID**: EXP-20260501-008-llava15-no-instruct-1800
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [ ] Running  [ ] Done  [x] Abandoned

---

### 1. 동기 (Motivation)
LLaVA-Instruct를 섞은 LLaVA-1.5 visual-utility student가 validation loss를 충분히 낮추지 못했다. instruction 데이터의 시각 토큰 중요도 label noise / distribution mismatch 가능성을 분리해서 확인한다.

### 2. 가설 (Hypothesis)
LLaVA-Instruct를 제외하고 TextVQA, ScienceQA, GQA만 각 600 sample씩 사용하면 teacher label 분포가 더 일관되어 student train/validation loss가 더 안정적으로 낮아질 것이다.

### 3. 독립변수 (What we change)
- 훈련 데이터셋: `llava_instruct` 제거
- 샘플 수: `textvqa`, `scienceqa`, `gqa` 각 600개, 총 1800개

### 4. 종속변수 (What we measure)
- 주요 metric: `train_log.jsonl`의 `val_loss`
- 보조 metric: `train_loss`, `val_mse`, `val_rank`

### 5. 고정 조건 (What stays the same)
- 모델: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- teacher root: `/workspace/zap/data/train/teacher_llava15`
- teacher collection: `max_new_tokens=64`, `trajectory_m=1`, `seed=0`
- student trainer: `foresight.train.llava_15`
- optimizer: AdamW, `lr=1e-3`, `weight_decay=0.0`
- epochs: 20
- validation ratio: 0.1
- hardware: GPU0

### 6. 베이스라인
- `/workspace/zap/ckpts/student_llava15_mmvet_instruct`
- `/workspace/zap/ckpts/student_llava15_v2`

### 7. 예상 결과
`llava_instruct` 포함 run보다 validation loss가 낮거나, 최소한 epoch 진행 중 악화 폭이 줄어들 것이다.

### 8. 판단 기준 (Success Criteria)
20 epoch 내 best `val_loss`가 기존 LLaVA-Instruct 포함 run의 best validation loss보다 낮고, loss curve가 후반부에 크게 악화되지 않으면 성공으로 본다.

### 9. 이 실험으로 증명할 수 없는 것
이 실험만으로 eviction downstream accuracy 개선을 보장할 수 없다. LLaVA-Instruct 제거 효과와 1500→1800 sample 증가 효과도 완전히 분리되지 않는다.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090, GPU0
- 예상 시간: teacher 보충 수 분 + student train 약 1시간 내외
- 디스크: teacher 추가 약 수십 MB, checkpoint 약 수백 MB
