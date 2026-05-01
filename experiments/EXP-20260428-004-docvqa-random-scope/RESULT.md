## Experiment Result

**ID**: EXP-20260428-004-docvqa-random-scope
**Completed**: 2026-04-28

### 결과 요약: PPL GT / ROUGE full-cache reference

| keep | random_image_only PPL ↓ | random_all_token PPL ↓ | random_image_only ROUGE-L ↑ | random_all_token ROUGE-L ↑ |
|------|--------------------------|------------------------|------------------------------|----------------------------|
| 0.1 | 1.4603 | 255.7058 | 0.9487 | 0.1547 |
| 0.2 | 1.4599 | 134.0328 | 0.9652 | 0.1793 |
| 0.3 | 1.4630 | 44.5658 | 0.9695 | 0.2133 |
| 0.4 | 1.4633 | 9.4267 | 0.9688 | 0.3097 |
| 0.5 | 1.4659 | 2.9346 | 0.9763 | 0.4647 |
| 0.6 | 1.4667 | 1.8109 | 0.9772 | 0.7081 |
| 0.7 | 1.4665 | 1.5598 | 0.9899 | 0.8106 |
| 0.8 | 1.4673 | 1.4865 | 0.9908 | 0.8879 |
| 0.9 | 1.4677 | 1.4857 | 0.9894 | 0.9411 |

### GT 기준 원점수 그래프

Full-cache 기준선을 제거하고, 두 random eviction scope만 원래 점수로 비교한 그래프를 추가했다. PPL은 기존 `ppl_gt`를 사용하고, ROUGE-L은 DocVQA GT answer 기준으로 새로 생성한 `rouge_gt`를 사용한다.

| keep | image-only PPL ↓ | all-token PPL ↓ | image-only ROUGE-L GT ↑ | all-token ROUGE-L GT ↑ |
|------|------------------|-----------------|--------------------------|------------------------|
| 0.9 | 1.4677 | 1.4857 | 0.3559 | 0.3595 |
| 0.8 | 1.4673 | 1.4865 | 0.3579 | 0.3488 |
| 0.7 | 1.4665 | 1.5598 | 0.3564 | 0.3350 |
| 0.6 | 1.4667 | 1.8109 | 0.3563 | 0.2875 |
| 0.5 | 1.4659 | 2.9346 | 0.3655 | 0.2048 |
| 0.4 | 1.4633 | 9.4267 | 0.3637 | 0.0938 |
| 0.3 | 1.4630 | 44.5658 | 0.3633 | 0.0669 |
| 0.2 | 1.4599 | 134.0328 | 0.3648 | 0.0595 |
| 0.1 | 1.4603 | 255.7058 | 0.3549 | 0.0488 |

### 체크포인트 / 아티팩트 위치
- CSV: `/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope/outputs/summary.csv`
- GT 원점수 CSV: `/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope/outputs/summary_original_gt.csv`
- GT PPL plot: `/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope/outputs/docvqa_random_scope_ppl_gt_nofull.png`
- GT ROUGE-L plot: `/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope/outputs/docvqa_random_scope_rouge_gt_nofull.png`
- Outputs: `/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope/outputs`
- Logs: `/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope/logs`

### Caveat
- `random_image_only`는 text token을 항상 보존하므로 낮은 keep ratio에서 effective prompt keep ratio가 nominal ratio보다 커질 수 있다.
- `summary.csv`의 ROUGE-L은 full-cache LLaVA generation 기준이고, `summary_original_gt.csv`의 ROUGE-L은 DocVQA GT answer 기준이다.
