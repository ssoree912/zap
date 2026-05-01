## Experiment Plan

**ID**: EXP-20260427-006
**Author**: Codex
**Date**: 2026-04-27
**Status**: [ ] Planned  [ ] Running  [x] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
`TextNeedleInAHaystack`에서 pruning student 결과가 낮았기 때문에, 같은 LLaVA-1.5와 같은 MileBench 평가 파이프라인에서 full-cache baseline을 측정한다.

### 2. 가설 (Hypothesis)
Full-cache LLaVA는 image-token pruning student보다 높은 TextNeedleInAHaystack ROUGE-L을 보일 수 있다.

### 3. 독립변수 (What we change)
- KV cache retention: full cache (`total_keep_ratio=1.0`)
- Dataset: `TextNeedleInAHaystack`

### 4. 종속변수 (What we measure)
- MileBench `ROUGE-L`
- `exact_match_accuracy`
- failures/prediction count

### 5. 고정 조건 (What stays the same)
- Base model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Dataset path: `/workspace/zap/data/MileBench/TextNeedleInAHaystack/TextNeedleInAHaystack.json`
- `max_new_tokens`: 32
- `--truncate`
- Prompt style: `look_milebench`

### 6. 베이스라인
- MMLongBench student at total_keep=0.2: ROUGE-L 0.0587900060
- `student_v2_A_gqa_lr1e4` at total_keep=0.2: ROUGE-L 0.0587900060

### 7. 예상 결과
Full-cache result가 pruning 결과와 다르면 cache pruning이 TextNeedle 성능 저하에 영향을 준 것으로 본다. 같으면 평가/출력 포맷 또는 LLaVA 자체의 task behavior 이슈로 본다.

### 8. 판단 기준 (Success Criteria)
320 samples 전체 평가가 완료되고 `metrics.json`에 `look_eval.ROUGE-L`이 기록된다.

### 9. 이 실험으로 증명할 수 없는 것
Full-cache가 낮게 나와도 더 큰 모델, 다른 prompt, answer formatting 개선 가능성은 배제하지 못한다.

### 10. 예상 런타임 / 리소스
- GPU: 1 x RTX 4090
- 예상 시간: 5-10분
- 디스크: predictions/eval files + logs
