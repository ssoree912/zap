# EXP-20260420-002 — 3-axis Main Table (keep_ratio = 0.2)

**H2O** = prefill accumulated attention (observable at prefill).
**Future** = MLP probe, decode→image attention prediction (unobservable at prefill).
**Hybrid** = α · softmax(H2O) + (1−α) · softmax(Future),  α=0.5 at every layer.

Metric automatically picked per dataset: `ROUGE-L` (generative) or `Accuracy` (choice).

| Dataset                | metric   |   H2O   | Future  | Hybrid  | Δ(Hyb−H2O) | Δ(Hyb−Fut) |
|------------------------|----------|---------|---------|---------|------------|------------|
| Spot-the-Diff          | ROUGE-L  | 0.1744 | 0.1819 | 0.1695 | -0.0050 | -0.0124 |
| CLEVR-Change           | ROUGE-L  | 0.1252 | 0.1318 | 0.1298 | +0.0046 | -0.0020 |
| WebQA                  | Accuracy | 0.6150 | 0.6000 | 0.6000 | -0.0150 | 0.0000 |
| ALFRED                 | ROUGE-L  | 0.2568 | 0.2878 | 0.2586 | +0.0018 | -0.0292 |
| IEdit                  | ROUGE-L  | 0.1007 | 0.1114 | 0.1073 | +0.0065 | -0.0041 |
| MMCoQA                 | ROUGE-L  | 0.3778 | 0.3733 | 0.3680 | -0.0097 | -0.0053 |
| ActionLocalization     | Accuracy | 0.2650 | 0.2650 | 0.2650 | 0.0000 | 0.0000 |
| ActionPrediction       | Accuracy | 0.5400 | 0.5400 | 0.5400 | 0.0000 | 0.0000 |
| ActionSequence         | Accuracy | 0.4600 | 0.4600 | 0.4600 | 0.0000 | 0.0000 |
| CharacterOrder         | Accuracy | 0.4900 | 0.4900 | 0.4900 | 0.0000 | 0.0000 |
| CounterfactualInference | Accuracy | 0.3200 | 0.3200 | 0.3200 | 0.0000 | 0.0000 |
| DocVQA                 | Accuracy | 0.5100 | 0.5100 | 0.5100 | 0.0000 | 0.0000 |
| EgocentricNavigation   | Accuracy | 0.3200 | 0.3200 | 0.3200 | 0.0000 | 0.0000 |
| GPR1200                | Accuracy | 0.1217 | 0.1217 | 0.1217 | 0.0000 | 0.0000 |
| ImageNeedleInAHaystack | ROUGE-L  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MovingAttribute        | Accuracy | 0.5150 | 0.5150 | 0.5150 | 0.0000 | 0.0000 |
| MovingDirection        | Accuracy | 0.3350 | 0.3350 | 0.3350 | 0.0000 | 0.0000 |
| MultiModalQA           | Accuracy | 0.7650 | 0.7550 | 0.7550 | -0.0100 | 0.0000 |
| nuscenes               | Accuracy | 0.6100 | 0.6100 | 0.6100 | 0.0000 | 0.0000 |
| ObjectExistence        | Accuracy | 0.4900 | 0.4900 | 0.4900 | 0.0000 | 0.0000 |
| ObjectInteraction      | Accuracy | 0.5200 | 0.5200 | 0.5200 | 0.0000 | 0.0000 |
| ObjectShuffle          | Accuracy | 0.3150 | 0.3150 | 0.3150 | 0.0000 | 0.0000 |
| OCR-VQA                | Accuracy | 0.3200 | 0.3200 | 0.3200 | 0.0000 | 0.0000 |
| SceneTransition        | Accuracy | 0.7750 | 0.7500 | 0.7650 | -0.0100 | +0.0150 |
| SlideVQA               | Accuracy | 0.4750 | 0.4700 | 0.4750 | 0.0000 | +0.0050 |
| StateChange            | Accuracy | 0.4100 | 0.4000 | 0.4050 | -0.0050 | +0.0050 |
| TextNeedleInAHaystack  | ROUGE-L  | 0.0588 | 0.0601 | 0.0601 | +0.0013 | 0.0000 |
| TQA                    | Accuracy | 0.4700 | 0.4750 | 0.4700 | 0.0000 | -0.0050 |
| WikiVQA                | Accuracy | 0.7100 | 0.7100 | 0.7100 | 0.0000 | 0.0000 |
|------------------------|----------|---------|---------|---------|------------|------------|
| **Average (29 ds)**        |    —     | 0.3947 | 0.3944 | 0.3933 | -0.0014 | -0.0011 |

## Verdict counts

- Hybrid > H2O: **4** / ties 19 / losses 6
- Hybrid > Future: **3** / ties 20 / losses 6
- Future > H2O: **6** / ties 17 / losses 6

- Completed datasets (at least one method): **29** / 29
- Datasets with all three methods scored: **29** / 29

## Notes

- Hybrid α=0.5 is blended at every one of the 32 layers (no per-layer gating).
- Future probe weights are trained on layers 24-31; layers 0-23 use broadcast-31 (broadcast noise acknowledged, not dominant per EXP-20260420-001).
- Missing cells (`—`) indicate the sweep is still in progress or the run failed.
