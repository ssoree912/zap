## Experiment Plan

**ID**: EXP-20260427-005
**Author**: Codex
**Date**: 2026-04-27
**Status**: [ ] Planned  [ ] Running  [x] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
MMLongBench-Doc로 새로 학습한 LLaVA-1.5 `VisualUtilityStudent` checkpoint가 기존에 크게 낮았던 MileBench `TextNeedleInAHaystack` 성능을 회복하는지 확인한다.

### 2. 가설 (Hypothesis)
문서/텍스트 중심 teacher로 학습한 student는 `TextNeedleInAHaystack`에서 기존 `student_v2_A_gqa_lr1e4`보다 높은 ROUGE-L을 낼 가능성이 있다.

### 3. 독립변수 (What we change)
- Student checkpoint: `/workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep`
- Dataset: `TextNeedleInAHaystack`

### 4. 종속변수 (What we measure)
- MileBench `ROUGE-L`
- `exact_match_accuracy`
- failures/prediction count

### 5. 고정 조건 (What stays the same)
- Base model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Evaluation mode: `visual_utility_student`
- `total_keep_ratio`: 0.2
- `max_new_tokens`: 32
- `--truncate`
- Prompt style: `look_milebench`

### 6. 베이스라인
- Previous `student_v2_A_gqa_lr1e4` TextNeedleInAHaystack at total_keep=0.2: ROUGE-L 0.0587900060
- Previous `student_v2_traj` TextNeedleInAHaystack at total_keep=0.2: ROUGE-L 0.0587900060

### 7. 예상 결과
기존보다 높은 ROUGE-L이면 MMLongBench-Doc teacher가 TextNeedle 약점 보완에 도움이 된 것으로 본다.

### 8. 판단 기준 (Success Criteria)
320 samples 전체 평가가 완료되고 `metrics.json`에 `n_failures=0` 또는 허용 가능한 실패 수와 `look_eval.ROUGE-L`이 기록된다.

### 9. 이 실험으로 증명할 수 없는 것
단일 데이터셋 결과이므로 전체 MileBench 성능 개선이나 다른 task 일반화는 증명할 수 없다.

### 10. 예상 런타임 / 리소스
- GPU: 1 x RTX 4090
- 예상 시간: 수십 분 이하
- 디스크: predictions/eval files + logs
