# Eval-7200 Q2 experiment

This experiment extends the audited seven-dataset LLaVA-1.5 runner with all
29 locally available MileBench tasks. Dataset names are namespaced as
`milebench__<original-name>`, so standard `docvqa` and MileBench
`milebench__DocVQA` cannot collide.

The official MileBench suite contains 28 tasks. The local `nuscenes` task is
included because this run covers every locally available evaluation task, but
its macro is reported separately as a local 29-task extension.

## Budget

Standard VQA/caption tasks retain 20% of the total prompt cache while preserving
all text KVs. That definition is often infeasible for text-heavy MileBench
samples with one 576-token combined image: text alone can exceed 20% of the
prompt. MileBench therefore defaults to visual-token retention:

\[
K_v=\left\lceil 0.2N_v\right\rceil=116.
\]

The runner explicitly supports `--milebench-budget-mode visual|total`. In
`total` mode it uses

\[
K_v=\left\lceil0.2N_{\rm prompt}\right\rceil-N_{\rm text},
\]

and fails with the sample ID and token counts when this is non-positive. It
never silently clamps an infeasible total budget.

## Long prompts and attention storage

Prompt construction uses `combined_1_images`. The task instruction is kept,
and task-context tokens are removed from the left until the expanded Vicuna
prompt—including the 576 visual tokens—is at most 4096 tokens.

Full quadratic attention is still computed one layer at a time, but is not
retained across layers. A forward hook losslessly replaces the first prefill
output with two rows,
`[question_mean, all_prefill_sum - question_mean]`, and the generation prefill
with its final query row. Thus the established runner recovers exactly the
H2O cumulative score, semantic-question score, and Future trajectory while
avoiding storage of 32 complete attention matrices.

For MileBench, the diagnostic question rows are the complete retained
**semantic user-prompt** span. This is distinct from H2O, which aggregates all
prefill query rows.

## Validation

```bash
cd /workspace/nips/zap
PYTHONPATH=experiments/EXP-20260727-002-eval7200-q2:/workspace/nips/Q-ViK:/workspace/nips/zap \
  /workspace/nips/.conda/envs/qvik/bin/python -m pytest -q \
  experiments/EXP-20260727-002-eval7200-q2/test_attention_compaction.py \
  experiments/EXP-20260727-002-eval7200-q2/test_eval7200_data.py
```

Manifest-only validation for all 29 local tasks:

```bash
PYTHONPATH=/workspace/nips/Q-ViK:/workspace/nips/zap \
  /workspace/nips/.conda/envs/qvik/bin/python \
  experiments/EXP-20260727-002-eval7200-q2/run_eval7200_q2.py \
  --datasets milebench --n-samples 200 --manifest-only \
  --output-dir /workspace/nips/zap/artifacts/rebuttal_eval7200_q2_manifest
```

## Execution

GPU 3 is occupied by OneVision training, so the launcher uses GPUs 0–2:

```bash
bash experiments/EXP-20260727-002-eval7200-q2/run_milebench_3gpu.sh
```

Each GPU loads the model once and processes its assigned datasets sequentially.
The launcher forces a fresh artifact directory and runs all MileBench tasks
with `max_new_tokens=128`. A direct worker resume is accepted only when its
stored run-config fingerprint matches exactly.

If a sample fails (including CUDA OOM), its JSON/mask pair is removed and GPU
memory is reclaimed before continuing. Summarize partial runs explicitly:

After extraction completes, summarize the MileBench portion with:

```bash
PYTHONPATH=/workspace/nips/Q-ViK:/workspace/nips/zap \
  /workspace/nips/.conda/envs/qvik/bin/python \
  experiments/EXP-20260727-002-eval7200-q2/run_eval7200_q2.py \
  --datasets milebench --n-samples 200 --summarize-only \
  --allow-partial \
  --output-dir /workspace/nips/zap/artifacts/rebuttal_eval7200_q2_llava15_keep0p2
```

If the separately running standard 1,400 records are linked or copied into
the same artifact tree, pass `--datasets all`. Standard total-cache 20% and
MileBench visual-token 20% macros remain strictly separate; the report never
combines them into one cross-budget macro.
