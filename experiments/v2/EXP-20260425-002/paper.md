# Experimental Results — EXP-20260425-002

## MileBench @ keep\_ratio = 0.2

| Category | Datasets (n) | LOOK-M | Future | prefill\_eviction |
|----------|-------------|--------|--------|------------------|
| T-1 | ActionLocalization, ActionPrediction, ActionSequence (3) | 0.3733 | 0.4200 | 0.4200 |
| T-2 | MovingAttribute, ObjectExistence, ObjectInteraction, ObjectShuffle (4) | 0.4513 | 0.4600 | — |
| T-3 | EgocentricNavigation, MovingDirection (2) | 0.3150 | 0.3275 | 0.3275 |
| T-4 | CharacterOrder, CounterfactualInference, SceneTransition, StateChange (4) | 0.4237 | 0.4950 | 0.4050 |
| S-1 | WebQA, MultiModalQA, TQA, WikiVQA (4) | 0.6062 | 0.6350 | 0.7550 |
| S-2 | DocVQA, OCR-VQA, SlideVQA (3) | 0.3333 | 0.4350 | 0.4150 |
| S-3 | Spot-the-Diff, CLEVR-Change, IEdit (3) | 0.1149 | 0.1485 | 0.1246 |
| S-4 | ALFRED, MMCoQA (2) | 0.2648 | 0.3217 | 0.3214 |
| S-5 | nuscenes (1) | 0.6450 | — | — |
| N-1 | TextNeedleInAHaystack (1) | 0.2062 | 0.0588 | — |
| N-2 | ImageNeedleInAHaystack (1) | 0.0000 | 0.0000 | 0.0000 |
| I-1 | GPR1200 (1) | 0.0283 | 0.1133 | 0.1117 |
| **Overall (29 ds)** | | **0.3596** | **0.3872** | — |

- Future = `A_gqa_lr1e4` (all-token, keep\_ratio=0.2)
- prefill\_eviction = `A` (all-token, keep\_ratio=0.2); ObjectExistence / nuscenes / TextNeedleInAHaystack 누락

---

## PPL by keep\_ratio



### mm-vet

| method | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | full |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| image_only | 5.1009 | 5.0875 | 5.0794 | 5.0842 | 5.0996 | 5.1139 | 5.1246 | 5.1342 | 5.1382 | 5.1362 |
| PrefixKV | 7.3750 | 5.9688 | 5.7188 | 5.5313 | 5.5000 | 5.4063 | 5.3750 | 5.2813 | 5.2813 | — |

### detail\_1k (PrefixKV: coco)

| method | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | full |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| image_only | 3.0935 | 3.0391 | 3.0055 | 2.9840 | 2.9702 | 2.9611 | 2.9555 | 2.9528 | 2.9525 | 2.9567 |
| PrefixKV (coco) | 4.4063 | 3.6875 | 3.4844 | 3.4063 | 3.4063 | 3.4063 | 3.2500 | 3.2031 | 3.2031 | — |


## ROUGE-L vs Full Output by keep\_ratio


### mm-vet

| method | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | full |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| image_only | 0.6263 | 0.7384 | 0.7842 | 0.8184 | 0.8504 | 0.8529 | 0.8635 | 0.8887 | 0.8979 | 0.9994 |
| PrefixKV | 0.3832 | 0.4075 | 0.4620 | 0.4677 | 0.4834 | 0.4911 | 0.5966 | 0.7379 | 0.7675 | — |

### detail\_1k (PrefixKV: coco)

| method | 0.1 | 0.2 | 0.3 | 0.4 | full |
|--------|-----|-----|-----|-----|------|
| image_only | 0.5314 | 0.6218 | 0.6786 | 0.7213 | 1.0000 |
| PrefixKV  | 0.4306 | 0.5110 | 0.5505 | 0.5689 | — |
