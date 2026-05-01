## Experiment Plan

**ID**: EXP-20260501-010-llava15-instruct-2000-lr1e4
**Author**: Codex
**Date**: 2026-05-01
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
LLaVA-Instruct 포함 조건에서 성능이 낮았던 원인이 데이터 자체인지, `lr=1e-3`가 너무 컸기 때문인지 분리한다.

### 2. 가설 (Hypothesis)
동일한 4-dataset 2000 sample 구성에서도 `lr=1e-4`를 쓰면 `lr=1e-3` 대비 MSE/rank/validation loss가 안정적으로 낮아질 것이다.

### 3. 독립변수 (What we change)
- Learning rate: `1e-4`
- 데이터셋: `llava_instruct`, `textvqa`, `scienceqa`, `gqa`
- 샘플 수: 각 500개, 총 2000개

### 4. 종속변수 (What we measure)
- 주요 metric: best `val_loss`
- 보조 metric: `train_loss`, `train_mse`, `train_rank`, `val_mse`, `val_rank`

### 5. 고정 조건 (What stays the same)
- 모델: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- teacher root: `/workspace/zap/data/train/teacher_llava15`
- student trainer: `foresight.train.llava_15`
- epochs: 20
- validation ratio: 0.1
- seed: 0
- physical GPU: GPU2

### 6. 베이스라인
- `/workspace/zap/ckpts/student_llava15_mmvet_instruct`
- `/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4`

### 7. 예상 결과
`lr=1e-3` LLaVA-Instruct 포함 run보다 loss가 낮아진다. 다만 no-instruct 1800보다 높으면 LLaVA-Instruct 자체의 label noise/distribution mismatch 영향이 남아 있는 것으로 본다.

### 8. 판단 기준 (Success Criteria)
best validation loss가 기존 LLaVA-Instruct 포함 run의 `~0.18`대보다 충분히 낮아지고, no-instruct lr1e4 run과 비교 가능한 범위인지 확인한다.

### 9. 이 실험으로 증명할 수 없는 것
Loss 개선이 곧 downstream MileBench 정확도 개선을 보장하지 않는다. checkpoint 완성 후 별도 inference/eval이 필요하다.

### 10. 예상 런타임 / 리소스
- GPU: 1x RTX 4090, physical GPU2
- 예상 시간: 약 1.5시간
- 디스크: checkpoint 약 수백 MB
