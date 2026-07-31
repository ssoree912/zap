# Full-VQA shared-mask student pair

This experiment reruns the four full local VQA benchmarks—GQA, TextVQA,
DocVQA, and ChartQA—for two trained LLaVA-1.5 students. MileBench is
intentionally excluded.

## Compared checkpoints

- `last_prompt_plus_answer_steps`:
  `artifacts/rebuttal_tradeoff_llava15_zap_teacher_n600/checkpoints/base`
- `question_answer_50_50`:
  `artifacts/original_llava_teacher/student_llava15_qa50_1800_e15_gpu0_rerun`

The first label is deliberately explicit. Its teacher target is the
full-cache generation trajectory
`[last prompt query, y_1, ..., y_(T-1)]`; the last prompt query remains in
the target. It is not an oracle and it is not an all-prefill/answer 50:50
mixture.

The Q+A checkpoint separately normalizes and mixes question-prompt-tail
attention and full-pass generated-answer attention at 0.5/0.5. The
question block includes every causal prompt-tail row after the final image
token, including template/instruction tokens. Its answer rows are
`[y_1, ..., y_T]`.

Both checkpoints use the same 600 GQA, 600 TextVQA, and 600 ScienceQA
training sample identities/prompts, seed 0, 15 epochs, and student
architecture. Two lineage confounds remain and are recorded instead of
being hidden:

1. Their answer-side query rows are offset by one step:
   `[last prompt, y_1, ..., y_(T-1)]` versus `[y_1, ..., y_T]`.
2. The base teachers used `max_new_tokens=32`; Q+A teachers used
   `max_new_tokens=64`.

## Controlled evaluation contract

- Exact total prompt-cache budget:
  `K_total = ceil(0.2 * N_prompt)`.
- Every text token is retained:
  `K_visual = clamp(K_total - N_text, 0, N_visual)`.
- Each student produces one visual Top-K mask per layer. The same 1-D mask
  is broadcast unchanged to every KV head.
- Student features use post-block
  `hidden_states[layer_idx + 1]`, matching both checkpoint trainers.
- Pruning failure policy is `raise`. A failed pruning sample is never
  silently replaced by a full-cache answer.
- Local dataset routing is pinned to `/workspace/nips/data/eval`, including
  the separately loaded GQA image dataset.
- Official LMMS-eval task aggregates are the reported metrics. In
  particular, this runner does not replace official DocVQA scoring with
  the separate eval700 continuous-ANLS implementation.

Earlier full Q+A VQA outputs used legacy Python `round` budgeting. They are
not aggregate regression references for this exact-`ceil` run. Three known
sample-level parity checks from exact-ceil development are:

- GQA row 8548 / id `201595808`: base `Cloudy`, Q+A `Cloudless`
- TextVQA row 3936 / id `38538`: base `11/04/2015`, Q+A `10/04/2015`
- ChartQA row 847: base `50`, Q+A `30`

These strings are diagnostic examples only; the validator trusts the new
full-run outputs and official metric files.

## Failure and pairing policy

The accepted run must contain the complete expected sample count for both
students and exactly matching `doc_id`, document, prompt, target, and input
identities. The validator also checks the exact budget on every sample,
checkpoint/model paths, complete LMMS model arguments, primary metric
keys, and frozen source hashes.

The four tasks/checkpoints have already completed under the older budget
without OOM, so this run is strict and fail-closed. If an unexpected OOM or
per-sample failure occurs, the current run aborts and produces no paired
summary. A subsequent exclusion run must first form the union of failed
sample identities and apply that one common exclusion manifest to both
students. Independent exclusions or an intersection-only post-hoc
denominator are not allowed.

## Launch

The launcher refuses GPUs with more than 512 MiB of active compute memory
and uses only physical GPUs 0–2:

```bash
bash experiments/EXP-20260727-005-full-vqa-shared-student-pair/run_3gpu.sh
```

Historical per-job runtimes give this allocation:

- GPU 0: base GQA + base DocVQA, about 103 minutes
- GPU 1: Q+A GQA + Q+A DocVQA, about 103 minutes
- GPU 2: both TextVQA + both ChartQA, about 90 minutes

Each launch gets a new timestamped directory under
`artifacts/rebuttal_full_vqa_shared_student_pair_llava15_total0p2/runs/`.
The finalizer writes:

- `summary/paired_summary.json`
- `summary/paired_metrics.csv`
- `summary/paired_samples.jsonl`

No summary is written unless all eight jobs and every paired validation
check pass. The main summary also reports an arithmetic four-task macro:
GQA exact match, TextVQA exact match, DocVQA official ANLS, and ChartQA
relaxed overall. It includes each student's macro and Q+A minus baseline.
