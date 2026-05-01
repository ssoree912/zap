## Experiment Result

**ID**: EXP-20260427-003
**Completed**: 2026-04-27
**Actual Runtime**: 400.96 seconds

---

### 결과 요약
| Method | Dataset | Metric | Value | vs Baseline |
|--------|---------|--------|-------|-------------|
| teacher_v2 | MMLongBench-Doc | saved records | 500 | n/a |
| teacher_v2 | MMLongBench-Doc | skipped records | 0 | n/a |
| teacher_v2 | MMLongBench-Doc | mean decode length T | 30.508 | n/a |

### 가설 검증
- [x] 가설이 맞았다
- [ ] 가설이 틀렸다
- [ ] 불확실

### 예상과 달랐던 점
없음. PDF page rendering, LLaVA-1.5 processor input, and v2 teacher record saving all completed without skipped samples.

### 결론
MMLongBench-Doc train split에서 seed 0으로 500개 QA row를 샘플링하고, 각 row의 첫 evidence page를 단일 이미지로 렌더링해 기존 `collect_future_teacher_v2.py`의 v2 Future teacher schema로 저장했다.

### 다음 실험 제안
이 teacher set을 probe/student training input으로 연결하기 전에, multi-page evidence가 필요한 문항을 별도로 표시하거나 page policy를 확장할지 결정한다.

### 재현 커맨드
```bash
cd /workspace/zap
CUDA_VISIBLE_DEVICES=1 bash experiments/EXP-20260427-003-mmlongbench-doc-v2/run.sh
```

### 체크포인트 / 아티팩트 위치
- teacher records: `/workspace/zap/data/teacher_v2/mmlongbench_doc`
- summary: `/workspace/zap/data/teacher_v2/mmlongbench_doc/_summary.json`
- rendered pages: `/workspace/zap/data/MMLongBench-Doc/rendered_pages_v2`
- log: `/workspace/zap/data/teacher_v2/mmlongbench_doc_collect.log`
