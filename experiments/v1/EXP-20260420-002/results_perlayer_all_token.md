# EXP-20260420-002 — All-token per-layer Hybrid (α=0.5, keep_ratio=0.2)

Comparing flat vs layer-gated Hybrid(H2O + Future-all):
- **Hyb-flat**: α=0.5 at every one of 32 layers.
- **Hyb-late**: α=0.5 at layers 24–31 only (H2O-only at 0–23).
- **Hyb-early**: α=0.5 at layers 0–23 only (H2O-only at 24–31).

| Dataset                |  H2O   | Future | Hyb-flat | Hyb-late | Hyb-early | Δ(late − flat) | Δ(early − flat) |
|------------------------|--------|--------|----------|----------|-----------|----------------|-----------------|
| Spot-the-Diff          | 0.1035 | 0.2149 | 0.1812 | 0.1025 | 0.1639 |       -0.0788 |        -0.0174 |
| CLEVR-Change           | 0.1889 | 0.1468 | 0.1417 | 0.1688 | 0.1288 |       +0.0271 |        -0.0129 |
| WebQA                  | 0.5050 | 0.5950 | 0.6100 | 0.0750 | 0.5400 |       -0.5350 |        -0.0700 |
| ALFRED                 | 0.0392 | 0.2645 | 0.2382 | 0.0028 | 0.2436 |       -0.2354 |        +0.0054 |
| IEdit                  | 0.0248 | 0.0824 | 0.0565 | 0.0235 | 0.0447 |       -0.0330 |        -0.0118 |
| MMCoQA                 | 0.2880 | 0.3849 | 0.3725 | 0.0332 | 0.3370 |       -0.3393 |        -0.0355 |
| ActionLocalization     | 0.2550 | 0.2650 | 0.2650 | 0.0000 | 0.2600 |       -0.2650 |        -0.0050 |
| ActionPrediction       | 0.3550 | 0.5400 | 0.5400 | 0.0200 | 0.5400 |       -0.5200 |        +0.0000 |
| ActionSequence         | 0.4050 | 0.4550 | 0.4600 | 0.0050 | 0.4400 |       -0.4550 |        -0.0200 |
| CharacterOrder         | 0.3900 | 0.4750 | 0.4850 | 0.1650 | 0.4250 |       -0.3200 |        -0.0600 |
| CounterfactualInference | 0.2350 | 0.2850 | 0.3200 | 0.1350 | 0.3200 |       -0.1850 |        +0.0000 |
| DocVQA                 | 0.4000 | 0.5000 | 0.5100 | 0.0600 | 0.5050 |       -0.4500 |        -0.0050 |
| EgocentricNavigation   | 0.3200 | 0.3200 | 0.3200 | 0.0200 | 0.3200 |       -0.3000 |        +0.0000 |
| GPR1200                | 0.1183 | 0.1217 | 0.1217 | 0.0083 | 0.1083 |       -0.1133 |        -0.0133 |
| ImageNeedleInAHaystack | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |       +0.0000 |        +0.0000 |
| MovingAttribute        | 0.4950 | 0.5150 | 0.5150 | 0.2200 | 0.5150 |       -0.2950 |        +0.0000 |
| MovingDirection        | 0.2350 | 0.3350 | 0.3350 | 0.1600 | 0.3350 |       -0.1750 |        +0.0000 |
| MultiModalQA           | 0.6950 | 0.7650 | 0.7700 | 0.1300 | 0.7450 |       -0.6400 |        -0.0250 |
| nuscenes               | 0.6100 | 0.6100 | 0.6100 | 0.1850 | 0.5300 |       -0.4250 |        -0.0800 |
| ObjectExistence        | 0.4200 | 0.4900 | 0.4900 | 0.2300 | 0.4900 |       -0.2600 |        +0.0000 |
| ObjectInteraction      | 0.5100 | 0.5200 | 0.5200 | 0.0200 | 0.4550 |       -0.5000 |        -0.0650 |
| ObjectShuffle          | 0.1750 | 0.3150 | 0.3150 | 0.0000 | 0.3150 |       -0.3150 |        +0.0000 |
| OCR-VQA                | 0.3000 | 0.3200 | 0.3200 | 0.0700 | 0.3200 |       -0.2500 |        +0.0000 |
| SceneTransition        | 0.7400 | 0.7500 | 0.7750 | 0.0500 | 0.7050 |       -0.7250 |        -0.0700 |
| SlideVQA               | 0.4050 | 0.4550 | 0.4750 | 0.0650 | 0.4600 |       -0.4100 |        -0.0150 |
| StateChange            | 0.3800 | 0.4100 | 0.4100 | 0.0250 | 0.4100 |       -0.3850 |        +0.0000 |
| TextNeedleInAHaystack  | 0.0000 | 0.0012 | 0.0052 | 0.0000 | 0.0037 |       -0.0052 |        -0.0016 |
| TQA                    | 0.4300 | 0.4700 | 0.4700 | 0.1600 | 0.4750 |       -0.3100 |        +0.0050 |
| WikiVQA                | 0.2800 | 0.6350 | 0.6050 | 0.1900 | 0.0800 |       -0.4150 |        -0.5250 |
|------------------------|--------|--------|--------|--------|--------|----------------|-----------------|
| **Average (29 ds)**       | 0.3208 | 0.3876 | 0.3875 | 0.0801 | 0.3522 | -0.3073     | -0.0352      |

## Best-per-dataset counts

- H2O wins:       4 / 29
- Future wins:    15 / 29
- Hyb-flat wins:  9 / 29
- Hyb-late wins:  0 / 29
- Hyb-early wins: 1 / 29

## Per-layer vs flat deltas

- Hyb-late > Hyb-flat: 1/29
- Hyb-early > Hyb-flat: 2/29

## Category-averaged results (MileBench taxonomy, keep_ratio=0.2)

Category codes follow the MileBench paper. Values are means across the
datasets listed in the "Datasets" column. LOOK-M is the paper-style
all-token baseline. Image-only columns (prefill_eviction-img / Future-img /
Hyb-0.5-img) evict image tokens only (text/system KV always kept); all-token
columns (prefill_eviction / Future / Hyb-0.5 / Hyb-late / Hyb-early) evict
over the whole prompt. Future probe = `future_probe_v5_all_token_10ep`.

| Category | Datasets (n) | LOOK-M | prefill_eviction-img | Future-img | Hyb-0.5-img | prefill_eviction | Future | Hyb-0.5 | Hyb-late | Hyb-early |
|----------|--------------|--------|----------------------|------------|-------------|------------------|--------|---------|----------|-----------|
| T-1      | ActionLocalization, ActionPrediction, ActionSequence (3)                   | 0.3967 | 0.4217 | 0.4217 | 0.4217 | 0.3383 | 0.4200 | 0.4217 | 0.0083 | 0.4133 |
| T-2      | MovingAttribute, ObjectExistence, ObjectInteraction, ObjectShuffle (4)     | 0.4575 | 0.4600 | 0.4600 | 0.4600 | 0.4000 | 0.4600 | 0.4600 | 0.1175 | 0.4438 |
| T-3      | EgocentricNavigation, MovingDirection (2)                                  | 0.3050 | 0.3275 | 0.3275 | 0.3275 | 0.2775 | 0.3275 | 0.3275 | 0.0900 | 0.3275 |
| T-4      | CharacterOrder, CounterfactualInference, SceneTransition, StateChange (4)  | 0.4000 | 0.4988 | 0.4900 | 0.4950 | 0.4363 | 0.4800 | 0.4975 | 0.0938 | 0.4650 |
| S-1      | WebQA, MultiModalQA, TQA, WikiVQA (4)                                      | 0.5625 | 0.6400 | 0.6350 | 0.6338 | 0.4775 | 0.6163 | 0.6138 | 0.1388 | 0.4600 |
| S-2      | DocVQA, OCR-VQA, SlideVQA (3)                                              | 0.3400 | 0.4350 | 0.4333 | 0.4350 | 0.3683 | 0.4250 | 0.4350 | 0.0650 | 0.4283 |
| S-3      | Spot-the-Diff, CLEVR-Change, IEdit (3)                                     | 0.1263 | 0.1334 | 0.1417 | 0.1355 | 0.1057 | 0.1480 | 0.1265 | 0.0983 | 0.1125 |
| S-4      | ALFRED, MMCoQA (2)                                                         | 0.2475 | 0.3173 | 0.3306 | 0.3133 | 0.1636 | 0.3247 | 0.3054 | 0.0180 | 0.2903 |
| S-5      | nuscenes (1)                                                               | 0.6150 | 0.6100 | 0.6100 | 0.6100 | 0.6100 | 0.6100 | 0.6100 | 0.1850 | 0.5300 |
| N-1      | TextNeedleInAHaystack (1)                                                  | 0.1000 | 0.0588 | 0.0601 | 0.0601 | 0.0000 | 0.0012 | 0.0052 | 0.0000 | 0.0037 |
| N-2      | ImageNeedleInAHaystack (1)                                                 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| I-1      | GPR1200 (1)                                                                | 0.0417 | 0.1217 | 0.1217 | 0.1217 | 0.1183 | 0.1217 | 0.1217 | 0.0083 | 0.1083 |
|----------|----------------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Overall (29 ds)** |                                                                 | 0.3493 | 0.3947 | 0.3944 | 0.3933 | 0.3208 | 0.3876 | 0.3875 | 0.0801 | 0.3522 |
