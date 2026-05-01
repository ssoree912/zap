## Experiment Plan

**ID**: EXP-20260427-003
**Author**: Codex
**Date**: 2026-04-27
**Status**: [ ] Planned  [ ] Running  [x] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
MMLongBench-Doc를 LLaVA-1.5 probe/teacher 학습 데이터로 쓰기 위해 v2 Future teacher records를 500개 수집한다.

### 2. 가설 (Hypothesis)
각 QA의 evidence page를 단일 이미지로 렌더링하면 기존 `collect_future_teacher_v2.py`와 동일한 teacher record schema를 만들 수 있다.

### 3. 독립변수 (What we change)
- Dataset: MMLongBench-Doc
- Sample count: 500
- PDF page rendering: first evidence page, fallback page 1

### 4. 종속변수 (What we measure)
- Saved teacher record count
- Skipped sample count
- Mean generated decode length `T`

### 5. 고정 조건 (What stays the same)
- Model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Teacher collector: v2 Future teacher path from `collect_future_teacher_v2.py`
- Max new tokens: 64
- Seed: 0

### 6. 베이스라인
- Existing v2 teacher datasets under `/workspace/zap/data/teacher_v2`

### 7. 예상 결과
500 requested samples 중 PDF rendering or model OOM failures를 제외하고 대부분 teacher records가 저장될 것이다.

### 8. 판단 기준 (Success Criteria)
`/workspace/zap/data/teacher_v2/mmlongbench_doc` 아래에 500개 내외 `.pt` record와 `_summary.json`이 생성된다.

### 9. 이 실험으로 증명할 수 없는 것
단일 evidence page 렌더링은 multi-page document QA 전체 컨텍스트를 보장하지 않는다. Evidence가 비어 있는 not-answerable 샘플은 1페이지 fallback을 사용한다.

### 10. 예상 런타임 / 리소스
- GPU: 1 x RTX 4090
- 예상 시간: 수십 분
- 디스크: rendered page cache + 500 teacher records
