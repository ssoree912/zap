## Experiment Plan

**ID**: EXP-20260427-001
**Author**: Codex
**Date**: 2026-04-27
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)
OneVision ChartQA 평가에서 우리 full-cache baseline이 비교 논문 표의 80.3보다 낮게 관측되면, student pruning 결과도 같은 기준에서 낮아 보인다. 비교 논문 코드는 없으므로 동일 평가 파이프라인 안에서 seed 변화에 따른 full-cache baseline 상한을 확인한다.

### 2. 가설 (Hypothesis)
VLMEvalKit + LLaVA-OneVision-HF full-cache ChartQA 결과는 seed 또는 CUDA nondeterminism에 의해 소폭 변할 수 있으며, 일부 seed에서 80.3에 근접하거나 초과할 수 있다.

### 3. 독립변수 (What we change)
- Seed: Python, NumPy, Torch, CUDA seed

### 4. 종속변수 (What we measure)
- 주요 metric: ChartQA_TEST Overall accuracy
- 보조 metric: test_human, test_augmented

### 5. 고정 조건 (What stays the same)
- 데이터셋: ChartQA_TEST
- 모델: `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf`
- 평가 코드: VLMEvalKit via `/workspace/zap/scripts/run_vlmeval_student.py` equivalent wrapper
- Config: `/workspace/zap/eval_configs/onevision_baseline_chartqa.json`
- 하드웨어: local RTX 4090 GPUs

### 6. 베이스라인
- Existing full-cache result: 79.96 Overall from `/workspace/zap/eval_results/baseline_chartqa`
- Comparison paper table target: 80.3 Overall

### 7. 예상 결과
Greedy decoding path일 가능성이 높아 seed 효과가 작거나 없을 수 있다. 다만 평가/커널 nondeterminism 또는 generation edge case로 인해 ±0.1~0.4%p 변동 가능성을 확인한다.

### 8. 판단 기준 (Success Criteria)
하나 이상의 seed에서 ChartQA_TEST Overall >= 80.3, 또는 80.3에 매우 근접한 최고 seed를 기록한다.

### 9. 이 실험으로 증명할 수 없는 것
이 실험은 비교 논문과 동일한 데이터 전처리, 모델 checkpoint, prompt, decoding 설정을 사용했다는 것을 증명하지 않는다. 비교 논문 코드가 없으므로 재현성 차이는 seed 외 요인일 수 있다.

### 10. 예상 런타임 / 리소스
- GPU: up to 3 x RTX 4090, one seed per GPU
- 예상 시간: seed당 ChartQA full evaluation runtime
- 디스크: `eval_results/baseline_chartqa_seed_sweep/seed_<seed>` per seed
