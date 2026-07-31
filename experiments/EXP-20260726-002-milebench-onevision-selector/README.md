# MileBench OneVision selector comparison

This experiment compares Full Cache, question-to-visual Prefill top-K,
1D-smoothed Prefill top-K, Q-ViK, and Future Oracle on four mixed/open-ended
MileBench tasks. The eviction budget is 20% of total prompt tokens, matching
the paper wrapper formula while retaining every text token.

Prefill saliency is computed from actual user-message query positions.
The implementation reconstructs only those query rows, so it does not allocate
the full OneVision prefill attention matrix. Future utility is the mean
answer-query attention to visual keys along the full-cache greedy trajectory.

Run all four GPUs:

```bash
bash experiments/EXP-20260726-002-milebench-onevision-selector/run_all_gpus.sh
```

Outputs are written to:

```text
artifacts/rebuttal_milebench_onevision_selector_total0p2/
```

After all four jobs finish:

```bash
/workspace/nips/.conda/envs/qvik/bin/python \
  experiments/EXP-20260726-002-milebench-onevision-selector/summarize_results.py
```
