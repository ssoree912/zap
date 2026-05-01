## Experiment Plan

**ID**: EXP-20260427-004
**Author**: Codex
**Date**: 2026-04-27
**Status**: [ ] Planned  [ ] Running  [x] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
MMLongBench-Doc에서 수집한 LLaVA-1.5 `teacher_v2` 500개 record로 VisualUtilityStudent를 학습해 문서 QA 계열 데이터에 맞춘 student checkpoint를 만든다.

### 2. 가설 (Hypothesis)
단일 evidence page 기반 teacher record 500개만으로도 기존 `scope=A` student 구조가 teacher attention distribution을 재현하도록 학습될 수 있다.

### 3. 독립변수 (What we change)
- Dataset: `mmlongbench_doc`
- Epochs: 20
- Learning rate: 1e-4

### 4. 종속변수 (What we measure)
- Train loss / MSE / ranking loss
- Validation loss / MSE / ranking loss
- Best checkpoint saved by validation loss

### 5. 고정 조건 (What stays the same)
- Model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Teacher root: `/workspace/zap/data/teacher_v2`
- Student scope: `A` (all 32 layers)
- Seed: 0
- Validation ratio: 0.1
- GPU: 0

### 6. 베이스라인
- Existing LLaVA-1.5 students: `/workspace/zap/ckpts/student_v2_A_gqa_lr1e4`, `/workspace/zap/ckpts/student_v2_A_gqa_lr1e4_40ep`

### 7. 예상 결과
20 epoch 동안 train loss가 감소하고 validation loss 기준 best checkpoint가 `/workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep`에 저장된다.

### 8. 판단 기준 (Success Criteria)
훈련이 20 epoch까지 완료되고 output checkpoint에 `config.json`, `pytorch_model.bin`, `train_log.jsonl`이 생성된다.

### 9. 이 실험으로 증명할 수 없는 것
이 훈련만으로 MileBench/ChartQA/PPL 성능 향상을 보장하지 않는다. 단일 page rendering teacher의 한계와 document multi-page context 손실은 별도 평가가 필요하다.

### 10. 예상 런타임 / 리소스
- GPU: 1 x RTX 4090 on CUDA_VISIBLE_DEVICES=0
- 예상 시간: 약 1-2시간
- 디스크: student checkpoint + JSONL log
