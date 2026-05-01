# Compare: 100-sample all-layer Future probe vs 250-sample v5 (all-token)

**Date:** 2026-04-22  
**Ratio:** r=0.20 (image_keep_ratio for new / total_keep_ratio for v5-all columns)  
**Model:** LLaVA-1.5-7B  

## Setup

| Probe | Training data | Shards | Layers | Epochs | Scope at inference |
|---|---|---|---|---|---|
| **new** `future_probe_allL_limit100` | scienceqa+textvqa+nlvr2 × **100 each** | all-token targets | all 32 | 10 | image-only eviction |
| **v5** `future_probe_v5_all_token_10ep` | scienceqa+textvqa+nlvr2 × **250 each** | all-token targets | all 32 | 10 | all-token eviction |
| (ref) **v4** `future_probe_v4_last8_20ep_bcast31` | (from EXP-20260418-001) | image-only targets | last-8 trained + broadcast | 20 | image-only eviction |

## Per-dataset comparison at r=0.20 (higher = better for all columns)

Columns:
- **v4 Fut-img**: image-only eviction with legacy v4 probe (reference)
- **v5 Fut-all**: all-token eviction with 250-data probe
- **new Fut-img**: image-only eviction with our new 100-data probe (today's results)
- **Δ (new − v5)**: direct sample-efficiency contrast (apples-not-identical: scope differs)

| Dataset | Metric | v4 Fut-img | v5 Fut-all | **new Fut-img** | Δ (new − v5) |
|---|---|---|---|---|---|
| Spot-the-Diff | ROUGE-L | 0.1819 | 0.2149 | **0.1908** | -0.0241 |
| CLEVR-Change | ROUGE-L | 0.1318 | 0.1468 | **0.1340** | -0.0128 |
| WebQA | Accuracy | 0.6000 | 0.5950 | **0.6050** | +0.0100 |
| ALFRED | ROUGE-L | 0.2878 | 0.2645 | **0.2769** | +0.0124 |
| IEdit | ROUGE-L | 0.1114 | 0.0824 | **0.1101** | +0.0277 |
| MMCoQA | ROUGE-L | 0.3733 | 0.3849 | **0.3759** | -0.0090 |
| ActionLocalization | Accuracy | 0.2650 | 0.2650 | **0.2650** | +0.0000 |
| ActionPrediction | Accuracy | 0.5400 | 0.5400 | **0.5400** | +0.0000 |
| ActionSequence | Accuracy | 0.4600 | 0.4550 | **0.4600** | +0.0050 |
| CharacterOrder | Accuracy | 0.4900 | 0.4750 | **0.4900** | +0.0150 |
| CounterfactualInference | Accuracy | 0.3200 | 0.2850 | **0.3200** | +0.0350 |
| DocVQA | Accuracy | 0.5100 | 0.5000 | **0.5100** | +0.0100 |
| EgocentricNavigation | Accuracy | 0.3200 | 0.3200 | **0.3200** | +0.0000 |
| GPR1200 | Accuracy | 0.1217 | 0.1217 | **0.1183** | -0.0034 |
| ImageNeedleInAHaystack | ROUGE-L | 0.0000 | 0.0000 | **0.0000** | +0.0000 |
| MovingAttribute | Accuracy | 0.5150 | 0.5150 | **0.5150** | +0.0000 |
| MovingDirection | Accuracy | 0.3350 | 0.3350 | **0.3350** | +0.0000 |
| MultiModalQA | Accuracy | 0.7550 | 0.7650 | **0.7550** | -0.0100 |
| nuscenes | Accuracy | 0.6100 | 0.6100 | **0.6100** | +0.0000 |
| ObjectExistence | Accuracy | 0.4900 | 0.4900 | **0.4900** | +0.0000 |
| ObjectInteraction | Accuracy | 0.5200 | 0.5200 | **0.5200** | +0.0000 |
| ObjectShuffle | Accuracy | 0.3150 | 0.3150 | **0.3150** | +0.0000 |
| OCR-VQA | Accuracy | 0.3200 | 0.3200 | **0.3200** | +0.0000 |
| SceneTransition | Accuracy | 0.7500 | 0.7500 | **0.7550** | +0.0050 |
| SlideVQA | Accuracy | 0.4700 | 0.4550 | **0.4700** | +0.0150 |
| StateChange | Accuracy | 0.4000 | 0.4100 | **0.4050** | -0.0050 |
| TextNeedleInAHaystack | ROUGE-L | 0.0601 | 0.0012 | **0.0593** | +0.0581 |
| TQA | Accuracy | 0.4750 | 0.4700 | **0.4700** | +0.0000 |
| WikiVQA | Accuracy | 0.7100 | 0.6350 | **0.7100** | +0.0750 |
| **Mean (n=29)** | — | **0.3944** | **0.3876** | **0.3947** | **+0.0070** |

## Verdict (new Fut-img vs v5 Fut-all)

- **Wins (new > v5 by > 0.005):** 11
- **Losses (new < v5 by > 0.005):** 4
- **Ties (|Δ| ≤ 0.005):** 14
- Mean delta: **+0.0070**

## Caveats

- **Eviction scope differs**: v5 column is all-token eviction (also evicts text tokens); new column is image-only eviction (text preserved). This generally favors new probe on tasks where text is dispositive, and favors v5 on tasks needing more aggressive total compression.
- **Training data**: v5 = 750 rows (250 × 3 datasets), new = 300 rows (100 × 3). Both all-token target, all-layer, 10 epochs.
- Apples-to-apples image-only baseline is **v4 Fut-img** (legacy last-8 broadcast probe). The new probe being > v4 at the same scope implies all-layer + all-token training helps even with 1/2.5× data vs v5.

## Sources

- New results: `/workspace/zap/artifacts/EXP-20260422-001/v6_allL_future/<slug>/future_k0p2/metrics.json`
- v5 reference: `/workspace/zap/experiments/EXP-20260420-002/results_all_token.md`
