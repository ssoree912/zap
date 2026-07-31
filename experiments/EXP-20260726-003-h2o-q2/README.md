# Corrected H2O-style Q2 analysis

This experiment separates two prefill baselines:

- Primary: H2O-style attention accumulated from every causal prefill query.
- Secondary: attention from user-question queries only.

H2O Top-K is selected independently for each of LLaVA-1.5's 32 attention/KV
heads. The runner saves packed H2O, Q-ViK, and Full-Future keep masks and
reports exact-budget token recall, retained visual Future mass, useful token
swaps, and downstream intervention scores.

The run is tied to the exact provenance of the n600 base artifact:

- held-out split recovered from the base student's training configuration
  (180 samples: 60 TextVQA, 56 ScienceQA, 64 GQA);
- the original Q-ViK LLaVA backend and padded `process_images` preprocessing;
- the artifact trainer's post-layer `hidden_states[layer + 1]` convention;
- Future captured from the full-cache `generate()` attention blocks
  `[last prompt, y_1, ..., y_(T-1)]`, identical to the stored teacher target;
- total-prompt keep ratio 0.2 with the exact H2O budget
  `ceil(0.2 * prompt_len) - n_text`, shared by both selectors.

`Full Cache` is standard greedy `generate()`. A no-eviction manual-decode
control is also reported to detect any decode-path confound.

```bash
bash experiments/EXP-20260726-003-h2o-q2/run_all_gpus.sh

/workspace/nips/.conda/envs/qvik/bin/python \
  experiments/EXP-20260726-003-h2o-q2/run_llava15_h2o_q2.py \
  --summarize-only
```
