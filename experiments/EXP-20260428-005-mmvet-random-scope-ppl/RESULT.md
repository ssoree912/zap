## Experiment Result

**ID**: EXP-20260428-005-mmvet-random-scope-ppl
**Status**: done

### Summary

- Full-cache PPL: `5.253553`

| keep | random_image_only PPL ↓ | Δ vs full | random_all_token PPL ↓ | Δ vs full |
|------|--------------------------|-----------|------------------------|-----------|
| 0.1 | 5.2925 | +0.0390 | 37.7349 | +32.4813 |
| 0.2 | 5.2281 | -0.0255 | 75.8915 | +70.6380 |
| 0.3 | 5.1994 | -0.0542 | 54.7105 | +49.4569 |
| 0.4 | 5.1971 | -0.0564 | 27.7294 | +22.4758 |
| 0.5 | 5.2162 | -0.0374 | 14.5735 | +9.3199 |
| 0.6 | 5.2419 | -0.0116 | 8.4311 | +3.1775 |
| 0.7 | 5.2490 | -0.0046 | 6.3673 | +1.1137 |
| 0.8 | 5.2492 | -0.0044 | 5.6967 | +0.4432 |
| 0.9 | 5.2510 | -0.0026 | 5.3944 | +0.1408 |

### Artifacts

- CSV: `/workspace/zap/experiments/EXP-20260428-005-mmvet-random-scope-ppl/outputs/summary.csv`
- PPL plot: `/workspace/zap/experiments/EXP-20260428-005-mmvet-random-scope-ppl/outputs/mmvet_random_scope_ppl.png`
- PPL zoom plot: `/workspace/zap/experiments/EXP-20260428-005-mmvet-random-scope-ppl/outputs/mmvet_random_scope_ppl_zoom.png`
- Logs: `/workspace/zap/experiments/EXP-20260428-005-mmvet-random-scope-ppl/logs` and per-run `run.log` under each output directory
