## Experiment Result

**ID**: EXP-20260427-005
**Completed**: 2026-04-27
**Actual Runtime**: 318 seconds

---

### 결과 요약
| Method | Dataset | Metric | Value | vs Baseline |
|--------|---------|--------|-------|-------------|
| MMLongBench student, total_keep=0.2 | TextNeedleInAHaystack | ROUGE-L | 0.0587900060 | same as `A_gqa_lr1e4` |
| MMLongBench student, total_keep=0.2 | TextNeedleInAHaystack | exact_match_accuracy | 0.0 | same as `A_gqa_lr1e4` |
| MMLongBench student, total_keep=0.2 | TextNeedleInAHaystack | n_predictions | 320 | n/a |
| MMLongBench student, total_keep=0.2 | TextNeedleInAHaystack | n_failures | 0 | n/a |
| MMLongBench student, total_keep=0.2 | TextNeedleInAHaystack | prediction contains gold substring | 79/320 = 0.246875 | same as `A_gqa_lr1e4` |

### 가설 검증
- [ ] 가설이 맞았다
- [x] 가설이 틀렸다
- [ ] 불확실

### 예상과 달랐던 점
새 checkpoint의 320개 predictions가 기존 `/workspace/zap/ckpts/student_v2_A_gqa_lr1e4` run과 모두 동일했다.

### 결론
MMLongBench-Doc teacher로 학습한 student는 `TextNeedleInAHaystack`의 기존 낮은 ROUGE-L을 개선하지 못했다. 이 태스크는 정답 needle이 텍스트 context 안에 있고 현재 방법은 image-token KV만 pruning하므로, student checkpoint 변경이 출력에 영향을 거의 주지 않는 것으로 보인다.

### 다음 실험 제안
TextNeedleInAHaystack 성능을 개선하려면 image-only student가 아니라 text token retention / all-token pruning 쪽을 따로 봐야 한다. 또한 일부 prediction은 정답 숫자를 포함하지만 문장형 답변 때문에 exact/ROUGE가 낮으므로, output formatting을 숫자만 내도록 제한하는 비교도 필요하다.

### 재현 커맨드
```bash
cd /workspace/zap
CUDA_VISIBLE_DEVICES=0 bash experiments/EXP-20260427-005-textneedle-mmlong-eval/run.sh
```

### 체크포인트 / 아티팩트 위치
- checkpoint: `/workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep`
- output dir: `/workspace/zap/artifacts/EXP-20260427-005-textneedle-mmlong-eval/milebench_mmlong_total0.2/TextNeedleInAHaystack`
- metrics: `/workspace/zap/artifacts/EXP-20260427-005-textneedle-mmlong-eval/milebench_mmlong_total0.2/TextNeedleInAHaystack/metrics.json`
- run log: `/workspace/zap/experiments/EXP-20260427-005-textneedle-mmlong-eval/outputs/run.log`
