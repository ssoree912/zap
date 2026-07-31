# LLaVA-1.5 visual-token signal agreement

This experiment answers whether the base Q-ViK scorer predicts future visual
utility rather than merely reproducing prefill attention.

Inputs:

- original LLaVA-1.5-7B at `/workspace/nips/models/llava-v1.5-7b`;
- future-only teacher shards, 600 each for TextVQA, ScienceQA, and GQA;
- the `base` scorer trained from those shards.

For each sample and each of 32 layers, the script computes visual-token vectors
for actual-user-question prefill attention, its 3x3 smoothed version, last-user-
question attention, last-prompt attention, the saved future teacher, and the
Q-ViK prediction. It then measures Spearman, cosine, and Jaccard at 10%, 20%,
and 50%.

The primary report uses the exact 10% held-out split recovered from the base
checkpoint's training configuration. The full 1,800-sample report is written
separately because it includes training samples.

Run:

```bash
bash experiments/EXP-20260726-001-signal-agreement/run_base_n600.sh
```

Results are written under:

```text
artifacts/rebuttal_signal_agreement_llava15_base_n600/
  samples/
  logs/
  summary/
```

The runner also evaluates Full Cache, Prefill, Smoothed Prefill, Q-ViK, and a
live Future Oracle on the exact held-out split at a total-prompt KV keep ratio
of 20%. All text KV is preserved and visual KV fills the remaining budget.
Those results are written under `intervention_total_keep_0p2/`. The earlier
visual-token-20% diagnostic is retained under `intervention/`.

The correlation target is the saved future teacher used to train the student.
The intervention's Future Oracle is deliberately recomputed from the current
full-cache answer trajectory, so it remains a functional oracle even if
generation code changes.
