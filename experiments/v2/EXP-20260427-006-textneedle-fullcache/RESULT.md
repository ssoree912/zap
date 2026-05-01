## Experiment Result

**ID**: EXP-20260427-006
**Completed**: 2026-04-27
**Actual Runtime**: 328 seconds

---

### 결과 요약
| Method | Dataset | Metric | Value | vs Baseline |
|--------|---------|--------|-------|-------------|
| LLaVA-1.5 full-cache | TextNeedleInAHaystack | ROUGE-L | 0.0592995169 | +0.0005095109 vs MMLong student k=0.2 |
| LLaVA-1.5 full-cache | TextNeedleInAHaystack | exact_match_accuracy | 0.0 | same |
| LLaVA-1.5 full-cache | TextNeedleInAHaystack | n_predictions | 320 | n/a |
| LLaVA-1.5 full-cache | TextNeedleInAHaystack | n_failures | 0 | n/a |
| LLaVA-1.5 full-cache | TextNeedleInAHaystack | prediction contains gold substring | 77/320 = 0.240625 | -2 samples vs MMLong student k=0.2 |

### 가설 검증
- [ ] 가설이 맞았다
- [x] 가설이 틀렸다
- [ ] 불확실

### 예상과 달랐던 점
Full-cache도 `ROUGE-L=0.0593`으로 낮았다. MMLong student pruning run (`ROUGE-L=0.05879`)과 차이가 거의 없다.

### 결론
`TextNeedleInAHaystack`의 낮은 점수는 image-token pruning student만의 문제가 아니다. 같은 LLaVA-1.5 full-cache 조건에서도 점수가 낮게 나오므로, 현 평가에서는 LLaVA-1.5 자체의 answer formatting / extraction / task following 문제가 더 크다.

### 다음 실험 제안
정답 숫자만 생성하도록 prompt를 바꾸거나, postprocess로 숫자만 추출해 metric을 다시 계산한다. Full-cache에서도 exact match가 0인 것은 문장형 답변이 정답 문자열을 포함해도 현재 exact metric이 false로 처리되기 때문이다.

### 재현 커맨드
```bash
cd /workspace/zap
CUDA_VISIBLE_DEVICES=0 bash experiments/EXP-20260427-006-textneedle-fullcache/run.sh
```

### 체크포인트 / 아티팩트 위치
- output dir: `/workspace/zap/artifacts/EXP-20260427-006-textneedle-fullcache/milebench_fullcache/TextNeedleInAHaystack`
- metrics: `/workspace/zap/artifacts/EXP-20260427-006-textneedle-fullcache/milebench_fullcache/TextNeedleInAHaystack/metrics.json`
- run log: `/workspace/zap/experiments/EXP-20260427-006-textneedle-fullcache/outputs/run.log`
