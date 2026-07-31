# Paired P+A versus Q+A full-caption evaluation

This experiment evaluates two strictly paired LLaVA-1.5 visual-KV
students on the complete local COCO Caption 2017, NoCaps, and TextCaps
evaluation sets. All non-caption tasks are excluded.

The compared teacher targets are

\[
Y_{\mathrm{PA}}^\ell
=\mathcal N(0.5D_P^\ell+0.5D_A^\ell),\qquad
Y_{\mathrm{QA}}^\ell
=\mathcal N(0.5D_Q^\ell+0.5D_A^\ell),
\]

where \(P=[0,N_p)\) contains all expanded multimodal prefill queries,
\(Q=[\max(V)+1,N_p)\) is the post-image prompt tail, and
\(A=[N_p,N_p+T)\) contains generated-answer queries
\([y_1,\ldots,y_T]\). The paired teachers share the same serialized
answer component, sample manifest, answer trajectory, student
conditioning indices, split, initialization, architecture, and optimizer
configuration. Only \(P\) versus \(Q\) changes.

## Checkpoints

- `prefill_answer_50_50`:
  `/workspace/nips/zap/artifacts/original_llava_teacher/student_llava15_prefill_answer50_paired_1800_e15_seed0`
- `question_answer_50_50`:
  `/workspace/nips/zap/artifacts/original_llava_teacher/student_llava15_question_answer50_paired_1800_e15_seed0`

The harness can be tested before these directories exist, but an actual
launch fails before creating a run directory unless both checkpoints are
complete. At launch, it dynamically hashes `pytorch_model.bin`,
`config.json`, `train_config.json`, and `train_log.jsonl`, validates all 15
training epochs, and freezes those hashes and the relevant evaluation and
paired-training source hashes into `run_manifest.json`. The finalizer
rehashes them before accepting results.

The lineage is additionally pinned to the extraction verification artifact
`artifacts/paired_prefill_question_answer/runs/20260727_065959/teacher_verification.json`
and the paired-training verification artifact
`artifacts/paired_prefill_question_answer/training_runs/20260727_070541/training_verification.json`.
Their contents and SHA256 hashes are validated at run preparation and again
after evaluation. The frozen source set includes
`recompose_saved_targets.py` and `verify_paired_training.py` from EXP-006.

## Evaluation contract

- Total-cache keep ratio is exactly 20%:
  \(K_{\rm total}=\lceil0.2N_{\rm prompt}\rceil\).
- All text tokens remain, and
  \(K_{\rm visual}=\operatorname{clamp}(K_{\rm total}-N_{\rm text},0,N_{\rm visual})\).
- Student features use `hidden_states[layer_idx + 1]`.
- Each method creates its own layer-wise visual Top-K mask; that mask is
  broadcast unchanged across all KV heads.
- `student_failure_policy=raise`; no full-cache fallback is accepted.
- Both methods must contain the complete expected task count and exactly
  matching sample identities and per-sample budget traces.
- Reported metrics are the official local LMMS-eval ROUGE-L aggregates:
  `coco_ROUGE_L,none`, `nocaps_ROUGE_L,none`, and
  `textcaps_ROUGE_L,none`. Their arithmetic mean is the three-caption
  macro.

Full counts are COCO Caption 2017 5,000, NoCaps 4,500, and TextCaps
3,166, for 12,666 paired rows per student.

## Four-GPU assignment

- physical GPU 0: P+A COCO Caption 2017
- physical GPU 1: Q+A COCO Caption 2017
- physical GPU 2: P+A NoCaps, then TextCaps
- physical GPU 3: Q+A NoCaps, then TextCaps

The launcher refuses to start if any allowed GPU has more than 512 MiB of
active compute memory.

```bash
bash experiments/EXP-20260727-008-full-caption-pa-qa-paired/run_4gpu.sh
```

To launch automatically only after the fixed paired-training run has passed:

```bash
bash experiments/EXP-20260727-008-full-caption-pa-qa-paired/launch_after_training.sh
```

`launch_after_training.sh` defaults `TRAINING_RUN_ROOT` to
`artifacts/paired_prefill_question_answer/training_runs/20260727_070541`,
aborts on any training `*.failed` marker, and records launcher stdout and
the resulting evaluation `run_root`.

Every launch creates a new timestamped directory under
`artifacts/rebuttal_full_caption_pa_qa_paired_llava15_total0p2/runs/`.
No summary is written unless all six jobs and every validation check
pass. Successful runs contain:

- `summary/paired_summary.json`
- `summary/paired_metrics.csv`
- `summary/paired_samples.jsonl`

Run CPU-only contract tests with:

```bash
/workspace/nips/.conda/envs/qvik/bin/python -m pytest -q \
  experiments/EXP-20260727-008-full-caption-pa-qa-paired/test_run_contract.py
```
