# Summary — PLAN Success Criteria Verdict

## H1 — per-layer Hybrid ≥ global Hybrid (average over 6 datasets)

- k=0.2: perL_avg=0.2870  global_avg=0.2884  Δ=-0.0014  → **FAIL**
- k=0.1: perL_avg=0.2836  global_avg=0.2875  Δ=-0.0040  → **FAIL**

## H2 — per-layer α* shifted toward Future (lower α) vs global

| dataset       |  k   | perL α* | glob α* | diff (perL − glob) |
|---------------|------|---------|---------|---------------------|
| spot_the_diff | 0.2  | 0.75    | 1.0     | -0.25 |
| spot_the_diff | 0.1  | 0.75    | 0.75    | +0.00 |
| clevr_change  | 0.2  | 0.0     | 0.25    | -0.25 |
| clevr_change  | 0.1  | 0.0     | 0.0     | +0.00 |
| webqa         | 0.2  | 0.5     | 0.75    | -0.25 |
| webqa         | 0.1  | 0.5     | 1.0     | -0.50 |
| alfred        | 0.2  | 0.5     | 1.0     | -0.50 |
| alfred        | 0.1  | 0.0     | 0.0     | +0.00 |
| iedit         | 0.2  | 0.0     | 1.0     | -1.00 |
| iedit         | 0.1  | 0.5     | 1.0     | -0.50 |
| mmcoqa        | 0.2  | 0.25    | 0.75    | -0.50 |
| mmcoqa        | 0.1  | 0.75    | 0.25    | +0.50 |

**Toward-Future count**: 8/12 cells have perL_α* < global_α*.

## H3 — on Future-dominant datasets, per-layer Hybrid ≥ Future-only (α=0.0 global)

- clevr_change k=0.2: perL(α=0.0)=0.1328  vs Future-only=0.1318  Δ=+0.0010  → **PASS**
- clevr_change k=0.1: perL(α=0.0)=0.1374  vs Future-only=0.1442  Δ=-0.0068  → **FAIL**
- alfred k=0.2: perL(α=0.5)=0.2911  vs Future-only=0.2878  Δ=+0.0033  → **PASS**
- alfred k=0.1: perL(α=0.0)=0.2808  vs Future-only=0.2924  Δ=-0.0116  → **FAIL**

**H3 score**: 2/4 ≥ (broadcast-noise removal benefit).

## Overall verdict

- H1 (average gain): 0/2 keep ratios pass.
- H2 (shift toward Future): 8/12 cells.
- H3 (Future-dominant datasets): 2/4 cells.
