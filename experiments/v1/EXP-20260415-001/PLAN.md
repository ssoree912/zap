## Experiment Plan

**ID**: EXP-20260415-001  
**Author**: ZAP Team  
**Date**: 2026-04-15  
**Status**: [x] Done

---

### 1. 동기 (Motivation)

EXP-20260410~20260412 시리즈에서 수행한 ablation / 전체 비교 / 효율성 측정을 마무리하고,
다음 두 가지를 추가로 완성한다.

1. **Image token eviction 시각화 파이프라인 구현**  
   - 어떤 image patch가 제거되는지를 직접 보여줘야 논문 figure로 사용 가능
   - probe vs oracle score 차이를 per-image heatmap으로 시각화

2. **전체 실험 결과 최종 정리**  
   - EXP-20260410~20260412 결과를 하나의 문서로 통합
   - 공정한 효율성 비교 (r_eff_prompt=0.20 기준 통일, oracle 2-pass 포함)

---

### 2. 가설 (Hypothesis)

- Image eviction 시각화 결과, probe score heatmap이 oracle score heatmap과 유사한 공간적 패턴을 보일 것 (probe distillation 성공의 시각적 증거)
- task-relevant region(질문과 관련된 영역)에 높은 score가 집중될 것

---

### 3. 독립변수 (What we change)

- 없음 (새 학습 없음) — 기존 checkpoint + 새 시각화 코드 추가

---

### 4. 종속변수 (What we measure)

- 시각화 품질: per-image 24×24 heatmap의 공간 패턴
- probe vs oracle Pearson r (score correlation)

---

### 5. 고정 조건 (What stays the same)

- 모델: LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf)
- 평가: MileBench (공정 비교 common 69 샘플)
- keep ratio: total_keep_ratio=0.20

---

### 6. 베이스라인

- 기존 ablation_report.md의 수치 (EXP-20260410~20260412 결과)

---

### 7. 예상 결과

- probe와 oracle의 heatmap이 질문 관련 영역에서 일치할 것
- 두 score의 Pearson r > 0.5 (공간적 유사도 의미 있는 수준)

---

### 8. 판단 기준 (Success Criteria)

- VizCapture가 pruning semantics를 변경하지 않음 (동일 성능 확인)
- 시각화 스크립트가 에러 없이 실행됨
- per-sample .png 및 aggregate .png 생성 완료

---

### 9. 이 실험으로 증명할 수 없는 것

- 시각화만으로 probe가 "올바른" 이유를 선택한다는 것을 증명하기 어려움
- heatmap 유사도가 downstream 성능과 직접 연결되지는 않음

---

### 10. 예상 런타임 / 리소스

- GPU: A100 80GB (cuda:0)
- 시각화 데이터 생성: 샘플당 ~2초 (probe inference 포함)
- 렌더링: 샘플당 ~0.5초 (matplotlib, CPU)
- 디스크: 샘플당 ~100KB (.npz) + ~300KB (.png)
