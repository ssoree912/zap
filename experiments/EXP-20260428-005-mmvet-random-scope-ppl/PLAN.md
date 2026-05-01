## Experiment Plan

**ID**: EXP-20260428-005-mmvet-random-scope-ppl  
**Author**: ssoree912  
**Date**: 2026-04-28  
**Status**: [x] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation

Show on mm-vet that randomly evicting text+image tokens is much worse than
restricting eviction to image tokens, and that random image-token eviction still
raises PPL above full cache.

### 2. Hypothesis

At the same total keep ratio, `random_image_only` should stay close to full
cache, while `random_all_token` should sharply increase PPL, especially at low
keep ratios.

### 3. What We Change

- Eviction scope: `random_image_only` vs `random_all_token`
- Total keep ratio: `0.1, 0.2, ..., 0.9`

### 4. What We Measure

- PPL against original mm-vet ground-truth answers.

### 5. Fixed Conditions

- Model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Dataset: `/workspace/data/mm-vet/mm-vet.json`
- Images: `/workspace/data/mm-vet`
- Samples: 218
- Random seed: 0
- Attention implementation: `sdpa`

### 6. Baselines

- Full cache PPL on mm-vet.

### 7. Caveats

- `random_image_only` preserves text tokens, so its effective prompt retention
  can be higher than the nominal total keep ratio when the text budget alone is
  large.
- PPL is teacher-forced against GT answers and may move slightly below full
  cache for isolated ratios due to small logit changes, but the expected trend
  should be degradation relative to full cache.

### 8. Artifacts

- Outputs: `experiments/EXP-20260428-005-mmvet-random-scope-ppl/outputs`
- Logs: `experiments/EXP-20260428-005-mmvet-random-scope-ppl/logs`
