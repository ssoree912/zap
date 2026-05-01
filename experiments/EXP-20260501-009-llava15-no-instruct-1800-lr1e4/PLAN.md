## Experiment Plan

**ID**: EXP-20260501-009-llava15-no-instruct-1800-lr1e4
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
기존 성능이 좋았던 `student_v2_A_gqa_lr1e4` 계열과 learning rate를 맞춰 LLaVA-Instruct 제거 실험을 다시 수행한다. 직전 `lr=1e-3` no-instruct run은 중단하고 비교 가능한 `lr=1e-4` 조건으로 재시작한다.

### 2. 가설 (Hypothesis)
`lr=1e-4`는 `lr=1e-3`보다 visual-utility student의 validation loss를 더 안정적으로 낮추고, 기존 좋은 checkpoint 계열과 더 가까운 학습 동작을 보일 것이다.

### 3. 독립변수 (What we change)
- Learning rate: `1e-3` -> `1e-4`
- Checkpoint path: `/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4`

### 4. 종속변수 (What we measure)
- 주요 metric: `train_log.jsonl`의 best `val_loss`
- 보조 metric: `train_loss`, `val_mse`, `val_rank`

### 5. 고정 조건 (What stays the same)
- 모델: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- 데이터셋: `textvqa`, `scienceqa`, `gqa`
- 샘플 수: 각 600개, 총 1800개
- teacher root: `/workspace/zap/data/train/teacher_llava15`
- teacher collection: `max_new_tokens=64`, `trajectory_m=1`, `seed=0`
- student trainer: `foresight.train.llava_15`
- epochs: 20
- validation ratio: 0.1
- hardware: GPU0

### 6. 베이스라인
- `/workspace/zap/ckpts/student_v2_A_gqa_lr1e4`
- `/workspace/zap/ckpts/student_llava15_no_instruct_1800` (`lr=1e-3`, 중단)
- `/workspace/zap/ckpts/student_llava15_mmvet_instruct`

### 7. 예상 결과
`lr=1e-3`보다 후반부 validation loss 악화가 작고, best validation loss가 낮아질 것이다.

### 8. 판단 기준 (Success Criteria)
20 epoch 내 best `val_loss`가 `lr=1e-3` no-instruct partial run의 best `0.15842`보다 낮거나, 유사하더라도 loss curve가 더 안정적이면 다음 inference 후보로 사용한다.

### 9. 이 실험으로 증명할 수 없는 것
Learning rate 효과와 dataset 구성 효과를 동시에 본다. Downstream MileBench 정확도 개선은 별도 eval로 확인해야 한다.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090, GPU0
- 예상 시간: teacher check 후 student train 약 1.5시간
- 디스크: checkpoint 약 수백 MB
