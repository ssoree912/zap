# EXP-20260420-002 — All-token vs Image-only (keep_ratio = 0.2)

Comparing **image-only eviction** (text preserved) vs **all-token eviction** (20% of ALL tokens kept).
Core question: *does pure Future probe beat H2O when scoring domain extends to text tokens?*

**image-only** columns: evict only image tokens (keep_ratio of image_tokens).
**all-token** columns: evict any token (total_keep_ratio of all_tokens).

| Dataset                |  H2O-img | Fut-img  | Hyb-img  |  H2O-all | Fut-all  | Hyb-all  | Δ(Fut-all − H2O-all) |
|------------------------|----------|----------|----------|----------|----------|----------|----------------------|
| Spot-the-Diff          | 0.1744 | 0.1819 | 0.1695 | 0.1035 | 0.2149 | 0.1812 |            +0.1114 |
| CLEVR-Change           | 0.1252 | 0.1318 | 0.1298 | 0.1889 | 0.1468 | 0.1417 |            -0.0421 |
| WebQA                  | 0.6150 | 0.6000 | 0.6000 | 0.5050 | 0.5950 | 0.6100 |            +0.0900 |
| ALFRED                 | 0.2568 | 0.2878 | 0.2586 | 0.0392 | 0.2645 | 0.2382 |            +0.2253 |
| IEdit                  | 0.1007 | 0.1114 | 0.1073 | 0.0248 | 0.0824 | 0.0565 |            +0.0576 |
| MMCoQA                 | 0.3778 | 0.3733 | 0.3680 | 0.2880 | 0.3849 | 0.3725 |            +0.0969 |
| ActionLocalization     | 0.2650 | 0.2650 | 0.2650 | 0.2550 | 0.2650 | 0.2650 |            +0.0100 |
| ActionPrediction       | 0.5400 | 0.5400 | 0.5400 | 0.3550 | 0.5400 | 0.5400 |            +0.1850 |
| ActionSequence         | 0.4600 | 0.4600 | 0.4600 | 0.4050 | 0.4550 | 0.4600 |            +0.0500 |
| CharacterOrder         | 0.4900 | 0.4900 | 0.4900 | 0.3900 | 0.4750 | 0.4850 |            +0.0850 |
| CounterfactualInference | 0.3200 | 0.3200 | 0.3200 | 0.2350 | 0.2850 | 0.3200 |            +0.0500 |
| DocVQA                 | 0.5100 | 0.5100 | 0.5100 | 0.4000 | 0.5000 | 0.5100 |            +0.1000 |
| EgocentricNavigation   | 0.3200 | 0.3200 | 0.3200 | 0.3200 | 0.3200 | 0.3200 |            +0.0000 |
| GPR1200                | 0.1217 | 0.1217 | 0.1217 | 0.1183 | 0.1217 | 0.1217 |            +0.0033 |
| ImageNeedleInAHaystack | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |            +0.0000 |
| MovingAttribute        | 0.5150 | 0.5150 | 0.5150 | 0.4950 | 0.5150 | 0.5150 |            +0.0200 |
| MovingDirection        | 0.3350 | 0.3350 | 0.3350 | 0.2350 | 0.3350 | 0.3350 |            +0.1000 |
| MultiModalQA           | 0.7650 | 0.7550 | 0.7550 | 0.6950 | 0.7650 | 0.7700 |            +0.0700 |
| nuscenes               | 0.6100 | 0.6100 | 0.6100 | 0.6100 | 0.6100 | 0.6100 |            +0.0000 |
| ObjectExistence        | 0.4900 | 0.4900 | 0.4900 | 0.4200 | 0.4900 | 0.4900 |            +0.0700 |
| ObjectInteraction      | 0.5200 | 0.5200 | 0.5200 | 0.5100 | 0.5200 | 0.5200 |            +0.0100 |
| ObjectShuffle          | 0.3150 | 0.3150 | 0.3150 | 0.1750 | 0.3150 | 0.3150 |            +0.1400 |
| OCR-VQA                | 0.3200 | 0.3200 | 0.3200 | 0.3000 | 0.3200 | 0.3200 |            +0.0200 |
| SceneTransition        | 0.7750 | 0.7500 | 0.7650 | 0.7400 | 0.7500 | 0.7750 |            +0.0100 |
| SlideVQA               | 0.4750 | 0.4700 | 0.4750 | 0.4050 | 0.4550 | 0.4750 |            +0.0500 |
| StateChange            | 0.4100 | 0.4000 | 0.4050 | 0.3800 | 0.4100 | 0.4100 |            +0.0300 |
| TextNeedleInAHaystack  | 0.0588 | 0.0601 | 0.0601 | 0.0000 | 0.0012 | 0.0052 |            +0.0012 |
| TQA                    | 0.4700 | 0.4750 | 0.4700 | 0.4300 | 0.4700 | 0.4700 |            +0.0400 |
| WikiVQA                | 0.7100 | 0.7100 | 0.7100 | 0.2800 | 0.6350 | 0.6050 |            +0.3550 |
|------------------------|----------|----------|----------|----------|----------|----------|----------------------|
| **Average (29 ds)**       | 0.3947 | 0.3944 | 0.3933 | 0.3208 | 0.3876 | 0.3875 | +0.0668 |

## Verdict counts (all-token)

- Future-all > H2O-all: **25** / ties 3 / losses 1
- H2O-all > H2O-img: **1** / ties 3 / losses 25

## Notes

- Future probe used for all-token = `future_probe_v5_all_token_10ep` (trained on 750 samples across textvqa+scienceqa+nlvr2 with all-position targets).
- Image-only Hybrid uses `future_probe_v4_last8_20ep_bcast31` (image-only trained, last-8 + broadcast).
- Image-only Future uses `future_probe_v4_last8_20ep_bcast31` (EXP-20260418-001).
