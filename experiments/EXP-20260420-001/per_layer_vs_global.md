# Per-layer vs Global: Best-α Comparison

For each (dataset, keep_ratio) cell we pick:
  - `perL_best` = max over α ∈ {0.0, 0.25, 0.5, 0.75} of per-layer Hybrid score.
  - `glob_best` = max over α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} of global baseline.
  - `Δ = perL_best - glob_best`.  Positive = per-layer wins.

| dataset       |  k   | perL α* | perL score | glob α* | glob score |   Δ    |
|---------------|------|---------|------------|---------|------------|--------|
| spot_the_diff | 0.2  | 0.75    | 0.1940    | 1.0     | 0.1949    | -0.0009 - |
| spot_the_diff | 0.1  | 0.75    | 0.1835    | 0.75    | 0.1832    | +0.0002 + |
| clevr_change  | 0.2  | 0.0     | 0.1328    | 0.25    | 0.1329    | -0.0001 - |
| clevr_change  | 0.1  | 0.0     | 0.1374    | 0.0     | 0.1442    | -0.0068 - |
| webqa         | 0.2  | 0.5     | 0.6150    | 0.75    | 0.6150    | +0.0000 = |
| webqa         | 0.1  | 0.5     | 0.6100    | 1.0     | 0.6150    | -0.0050 - |
| alfred        | 0.2  | 0.5     | 0.2911    | 1.0     | 0.2903    | +0.0008 + |
| alfred        | 0.1  | 0.0     | 0.2808    | 0.0     | 0.2924    | -0.0116 - |
| iedit         | 0.2  | 0.0     | 0.1167    | 1.0     | 0.1145    | +0.0022 + |
| iedit         | 0.1  | 0.5     | 0.1156    | 1.0     | 0.1148    | +0.0008 + |
| mmcoqa        | 0.2  | 0.25    | 0.3728    | 0.75    | 0.3830    | -0.0102 - |
| mmcoqa        | 0.1  | 0.75    | 0.3741    | 0.25    | 0.3756    | -0.0015 - |

**Score**: per-layer wins/ties/losses = 4/1/7 of 12.
**Significant wins (Δ ≥ 0.005)**: 0/12.
