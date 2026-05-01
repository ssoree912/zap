## Experiment Result

**ID**: EXP-20260427-007-onevision-totalkeep-chartqa
**Completed**: 2026-04-27
**Actual Runtime**: 34-36 minutes inference per ChartQA run, ratio runs executed in parallel on GPU 0/1/2

---

### 결과 요약

| Method | Dataset | Metric | Value | vs Baseline |
|--------|---------|--------|-------|-------------|
| OneVision student, total keep 0.50 | ChartQA_TEST | Overall | 79.76 | n/a |
| OneVision student, total keep 0.10 | ChartQA_TEST | Overall | 78.84 | n/a |
| OneVision student, total keep 0.05 | ChartQA_TEST | Overall | 76.80 | n/a |

Split metrics:

| keep_ratio | test_human | test_augmented | Overall | failures |
|------------|------------|----------------|---------|----------|
| 0.50 | 67.04 | 92.48 | 79.76 | 0/2500 |
| 0.10 | 65.76 | 91.92 | 78.84 | 0/2500 |
| 0.05 | 63.12 | 90.48 | 76.80 | 0/2500 |

First-sample budget logs confirmed total-token-basis conversion:

| keep_ratio | prompt_len | non_image_tokens | image_tokens | image_tokens_kept |
|------------|------------|------------------|--------------|-------------------|
| 0.50 | 4970 | 29 | 4941 | 2456 |
| 0.10 | 4970 | 29 | 4941 | 468 |
| 0.05 | 4970 | 29 | 4941 | 220 |

### 가설 검증

- [x] 가설이 맞았다
- [ ] 가설이 틀렸다
- [ ] 불확실

동일 숫자 ratio에서 total-token-basis는 image-token-basis보다 더 적은 이미지 토큰을 남긴다. 성능은 0.50 > 0.10 > 0.05 순으로 나왔다.

### 예상과 달랐던 점

Total keep 0.10도 Overall 78.84로 이전 image-token-basis 0.10 결과와 거의 같은 수준이었다. ChartQA prompt의 non-image token 수가 첫 샘플 기준 29개로 매우 작아, image-token-basis와 total-token-basis의 차이가 크지 않은 샘플이 많은 것으로 보인다.

### 결론

LLaVA-OneVision ChartQA student 평가의 `keep_ratio`는 total-token-basis로 수정되었고, 세 ratio 모두 실패 없이 완료됐다.

### 다음 실험 제안

Text 토큰 비중이 큰 데이터셋에서는 total-token-basis 전환의 영향이 더 커질 수 있으므로 TextNeedle / MMLongBench-Doc 계열에서 같은 확인이 필요하다.

### 재현 커맨드

```bash
cd /workspace/zap
bash experiments/EXP-20260427-007-onevision-totalkeep-chartqa/run.sh total50
bash experiments/EXP-20260427-007-onevision-totalkeep-chartqa/run.sh total10
bash experiments/EXP-20260427-007-onevision-totalkeep-chartqa/run.sh total05
```

### 체크포인트 / 아티팩트 위치

- student: `/workspace/zap/ckpts/student_onevision_A_lr1e4_20ep`
- config 0.50: `/workspace/zap/eval_configs/onevision_student_total50_chartqa.json`
- config 0.10: `/workspace/zap/eval_configs/onevision_student_total10_chartqa.json`
- config 0.05: `/workspace/zap/eval_configs/onevision_student_total05_chartqa.json`
- result 0.50: `/workspace/zap/eval_results/student_total50_chartqa/ov7b_student_tot50/T20260427-164632/ov7b_student_tot50_ChartQA_TEST_acc.csv`
- result 0.10: `/workspace/zap/eval_results/student_total10_chartqa/ov7b_student_tot10/T20260427-164632/ov7b_student_tot10_ChartQA_TEST_acc.csv`
- result 0.05: `/workspace/zap/eval_results/student_total05_chartqa/ov7b_student_tot05/T20260427-164632/ov7b_student_tot05_ChartQA_TEST_acc.csv`
- logs: `/workspace/zap/eval_results/student_total_logs/chartqa_total50.log`, `/workspace/zap/eval_results/student_total_logs/chartqa_total10.log`, `/workspace/zap/eval_results/student_total_logs/chartqa_total05.log`
