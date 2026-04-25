# EXP-20260422-001 — PPL + ROUGE ratio sweep (PrefixKV datasets)

**Date:** 2026-04-22 / 2026-04-23 (multi-ratio extension)  
**Probe:** `/workspace/zap/ckpts/future_probe_allL_limit100` (all-layer Future, 100-sample × 3 datasets, 10 epochs, all-token targets)  
**Method:** `future` (image-only eviction, `selected_layer_indices=0..31`)  
**Model:** `/workspace/zap/ckpts/llava-1.5-7b-hf`  
**Datasets:** mm-vet (218), detail_1k (1000)  
**Status:** 36/36 runs complete, 0 failures (k ∈ {0.1,...,0.9} × {mm-vet, detail_1k} × {PPL, ROUGE})

## PPL (teacher-forcing, lower ↓ is better)

| image_keep_ratio | mm-vet (n=218) | detail_1k (n=1000) |
|---|---|---|
| 0.1 | 5.1009 | 3.0935 |
| 0.2 | 5.0875 | 3.0391 |
| 0.3 | 5.0794 | 3.0055 |
| 0.4 | 5.0842 | 2.9840 |
| 0.5 | 5.0996 | 2.9702 |
| 0.6 | 5.1139 | 2.9611 |
| 0.7 | 5.1246 | 2.9555 |
| 0.8 | 5.1342 | 2.9528 |
| 0.9 | 5.1382 | 2.9525 |

- **mm-vet best:** r=0.3 → PPL 5.0794  (U-shape, non-monotonic)
- **detail_1k best:** r=0.9 → PPL 2.9525  (monotonic ↓ with r)

## ROUGE-L (free-generation, higher ↑ is better)

| image_keep_ratio | mm-vet (n=218) | detail_1k (n=1000) |
|---|---|---|
| 0.1 | 0.0952 | 0.3837 |
| 0.2 | 0.0998 | 0.3923 |
| 0.3 | 0.1029 | 0.3962 |
| 0.4 | 0.1056 | 0.3992 |
| 0.5 | 0.1078 | 0.4019 |
| 0.6 | 0.1085 | 0.4024 |
| 0.7 | 0.1033 | 0.4042 |
| 0.8 | 0.0989 | 0.4063 |
| 0.9 | 0.1015 | 0.4066 |

- **mm-vet best:** r=0.6 → ROUGE-L 0.1085  (peak at mid/high ratio)
- **detail_1k best:** r=0.9 → ROUGE-L 0.4066  (monotonic ↑ with r)

## Observations

- **detail_1k** is monotonic in both metrics: PPL ↓ and ROUGE-L ↑ as keep-ratio rises toward full-cache.
  Long-form caption generation (~150 tokens/answer) benefits from more preserved image KV, as expected.
- **mm-vet PPL** is **non-monotonic (U-shape)**: best at r=0.3 (5.0794), worse at both ends.
  Short answers (numeric / OCR / multi-choice) are insensitive to large KV; at r≥0.5 extra image KV slightly raises PPL — likely low-utility tokens introducing distractor signal in teacher-forced scoring.
- **mm-vet ROUGE** peaks at r=0.6 (0.1085) and degrades at r=0.8/0.9.
  Free generation on short answers suffers slightly from over-preserved cache — probe pruning acts as soft regularizer.
- Dynamic range is narrow on mm-vet (PPL span ≈ 0.06, ROUGE span ≈ 0.013) — the dataset has ~200 short samples, so per-ratio noise is non-negligible.

## Artifacts

- Per-run results: `/workspace/zap/artifacts/EXP-20260422-001/{ppl,rouge}/{mm-vet,detail_1k}/future_k{0p1..0p9}/result.json`
- Sweep driver: `/workspace/zap/artifacts/EXP-20260422-001/run_ppl_rouge.sh`
- Scripts: `/workspace/zap/eval_ppl.py`, `/workspace/zap/eval_rouge.py`
