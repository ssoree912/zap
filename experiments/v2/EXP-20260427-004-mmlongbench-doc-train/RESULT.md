## Experiment Result

**ID**: EXP-20260427-004
**Completed**: 2026-04-27
**Actual Runtime**: 1237.13 seconds

---

### 결과 요약
| Method | Dataset | Metric | Value | vs Baseline |
|--------|---------|--------|-------|-------------|
| VisualUtilityStudent scope=A | MMLongBench-Doc | epochs | 20 | n/a |
| VisualUtilityStudent scope=A | MMLongBench-Doc | best val_loss | 0.1348286858 | n/a |
| VisualUtilityStudent scope=A | MMLongBench-Doc | best epoch | 15 | n/a |
| VisualUtilityStudent scope=A | MMLongBench-Doc | final train_loss | 0.1340087352 | n/a |
| VisualUtilityStudent scope=A | MMLongBench-Doc | final val_loss | 0.1348412773 | n/a |

### 가설 검증
- [x] 가설이 맞았다
- [ ] 가설이 틀렸다
- [ ] 불확실

### 예상과 달랐던 점
예상보다 빨리 완료됐다. 500 samples, train 450 / val 50, all 32 layers scope=A 기준 약 20.6분이 걸렸다.

### 결론
MMLongBench-Doc teacher_v2 500개로 LLaVA-1.5 VisualUtilityStudent `scope=A` checkpoint를 20 epoch 학습 완료했다. Best checkpoint는 validation loss 기준 epoch 15에서 저장됐다.

### 다음 실험 제안
이 checkpoint로 ChartQA/MileBench/Doc 계열 evaluation을 기존 `student_v2_A_gqa_lr1e4`와 같은 keep ratio에서 비교한다.

### 재현 커맨드
```bash
cd /workspace/zap
CUDA_VISIBLE_DEVICES=0 bash experiments/EXP-20260427-004-mmlongbench-doc-train/run.sh
```

### 체크포인트 / 아티팩트 위치
- checkpoint: `/workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep`
- train log jsonl: `/workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep/train_log.jsonl`
- run log: `/workspace/zap/experiments/EXP-20260427-004-mmlongbench-doc-train/outputs/run.log`
