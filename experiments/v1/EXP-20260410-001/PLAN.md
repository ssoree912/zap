## Experiment Plan

**ID**: EXP-20260410-001
**Author**: ZAP team
**Date**: 2026-04-10
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
우리 방법(att_only_postvision + image-only eviction)이 LOOK-M보다 성능이 좋은 이유를 두 가지 요인으로 분리한다.
- Factor 1 (Scoring): att_only_postvision vs H2O 누적 attention
- Factor 2 (Eviction scope): image token only vs 전체 token

### 2. 가설 (Hypothesis)
- **Scoring이 핵심**: att_only_postvision은 post-vision query에서의 attention만 쓰므로 task-relevant image token을 더 잘 선택할 것이다. H2O는 모든 query position을 누적하므로 덜 선택적일 것이다.
- **Eviction scope는 부차적**: image-only eviction은 text token을 항상 보존하므로 정보 손실이 적다. all-token eviction은 중요한 text token을 버릴 위험이 있다.
- 예상 순서: 우리 방법 ≥ Ablation B (oracle+all) ≥ Ablation A (H2O+image) ≥ LOOK-M

### 3. 독립변수 (What we change)
- Scoring signal: `att_only_postvision` (oracle teacher) vs `H2O` (accumulated attention)
- Eviction scope: image-only vs all-token

### 4. 종속변수 (What we measure)
- 주요 metric: LOOK-M evaluation score (ROUGE-L / accuracy per dataset)
- 보조 metric: r_eff_prompt (= total_keep_ratio, 모든 방법에서 동일해야 함)

### 5. 고정 조건 (What stays the same)
- 데이터셋: DocVQA, Spot-the-Diff, CLEVR-Change, IEdit
- 모델: LLaVA-1.5 (llava-hf/llava-1.5-7b-hf)
- total_keep_ratio: 0.10, 0.20, 0.30 sweep
- max_new_tokens: 32
- prompt_style: look_milebench
- limit: None (전체)
- Hardware: GPU_INDEX=0

### 6. 베이스라인
- 우리 방법 (att_only_postvision + image-only): 기존 실험 결과
- LOOK-M (H2O + all-token): 기존 실험 결과

### 7. 예상 결과
- Ablation A (H2O + image-only) < 우리 방법: scoring이 약해서 성능 하락
- Ablation B (oracle + all-token) ≈ 우리 방법 or 약간 낮음: text eviction으로 일부 손실
- 즉, scoring signal의 기여 > eviction scope의 기여

### 8. 판단 기준 (Success Criteria)
- Ablation A vs 우리 방법의 gap이 Ablation B vs 우리 방법의 gap보다 크면 scoring이 핵심 요인
- 모든 방법의 r_eff_prompt = total_keep_ratio로 통일됨 (공정 비교 확인)

### 9. 이 실험으로 증명할 수 없는 것
- probe score vs oracle score의 gap (probe는 별도 실험)
- 다른 모델 아키텍처에서의 일반화
- 긴 응답이 필요한 태스크에서의 TBT 효율 (eager attention 때문에 ablation은 TBT 비교 제외)

### 10. 예상 런타임 / 리소스
- GPU: 1x (GPU_INDEX=0)
- 실험 수: 2 modes × 3 ratios × 4 datasets = 24 runs
- 아티팩트: /workspace/hd/artifacts/ablation/
- 로그: /workspace/hd/artifacts/ablation/sweep.log

### 재현 커맨드
```bash
cd /workspace/zap
MODES="h2o_image_only oracle_all_token" \
TOTAL_KEEP_RATIOS="0.10 0.20 0.30" \
DATASETS="DocVQA Spot-the-Diff CLEVR-Change IEdit" \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/ablation \
bash scripts/run_ablation_sweep.sh 2>&1 | tee /workspace/hd/artifacts/ablation/sweep.log
```
