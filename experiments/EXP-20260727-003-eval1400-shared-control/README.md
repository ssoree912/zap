# Eval-1400 shared-mask control

This is a new, isolated rerun over the seven standard benchmarks only:

- GQA, TextVQA, DocVQA, ChartQA
- COCO Caption, NoCaps, TextCaps
- 200 deterministic examples per task (`seed=42`), 1,400 total
- MileBench is excluded
- exact total prompt-cache keep ratio 0.20

Every method preserves all text KVs. For each sample,

\[
K_v
=
\max\left(
0,
\left\lceil 0.2N_{\mathrm{prompt}}\right\rceil-N_{\mathrm{text}}
\right).
\]

The controlled all-prefill score is

\[
P_i^\ell
=
\frac{1}{H}
\sum_h\sum_{q=1}^{N_{\mathrm{prompt}}}
A_{\mathrm{pre}}^{\ell,h}[q,i].
\]

The semantic-question score is

\[
Q_i^\ell
=
\frac{1}{H|Q|}
\sum_h\sum_{q\in Q}
A_{\mathrm{pre}}^{\ell,h}[q,i].
\]

The raw Full-cache Future reference is

\[
F_i^\ell
=
\frac{1}{HT}
\sum_h\sum_{t=1}^{T}
A_{\mathrm{full},t}^{\ell,h}[i].
\]

Its causal trajectory remains
`[last prompt query, y_1, ..., y_(T-1)]`: the final prompt query predicts the
first answer token, and subsequent generated-token queries predict later
answer tokens.

All three attention-derived scores are therefore `[L,N_visual]`. Q-ViK is
already `[L,N_visual]`. All-prefill shared, semantic-question shared, and
Q-ViK shared each select one layer-wise Top-K mask, and that exact same
one-dimensional prompt mask is applied to every KV head during decoding.
Packed artifacts store those exact applied masks and their fingerprints.

`future_shared` is only the raw Full-cache first-pass reference used for
overlap, Jaccard, and retained Future mass. This experiment does not run or
compare an Oracle second-pass decode.

As in the existing manual-cache intervention protocol, every pruned selector
uses the first answer token predicted by the unpruned prefill and applies its
eviction before predicting subsequent tokens. The raw `generate()` response
remains the primary response-fidelity reference.

## Launch

The launcher is fixed to physical GPUs 0, 1, and 2. It checks their occupancy
in one `nvidia-smi` snapshot, refuses a pre-existing output directory, launches
three disjoint workers with `--n-samples 200 --extract-only`, and runs one
`--summarize-only` process after all three workers succeed:

```bash
cd /workspace/nips/zap
bash experiments/EXP-20260727-003-eval1400-shared-control/run_3gpu.sh
```

GPU 3 is never targeted. The default fresh artifact is:

```text
/workspace/nips/zap/artifacts/rebuttal_eval1400_shared_control_llava15_total0p2
```

The balanced assignment is:

- GPU 0: `coco_caption docvqa`
- GPU 1: `textcaps chartqa`
- GPU 2: `nocaps gqa textvqa`

Resume skips only complete JSON+NPZ pairs. Rerun the failed worker's exact
command through `run_worker.sh`; manifests and run fingerprints must match, or
the runner refuses the existing artifact. Summary likewise consumes only
complete manifest-addressed JSON+NPZ pairs. A sample with an explicit
`failures/<dataset>/<sample>.txt` marker is excluded in common from every
selector; an unexplained missing pair or a one-sided JSON/NPZ pair is a hard
summary error.

The primary response-fidelity table compares each selector with raw
`full_cache_generate`. Matched manual Full Cache is reported only as a
diagnostic. Pairwise exact-match deltas, exact-K mask agreement,
head-averaged score agreement, downstream task scores, and layer-wise Future
agreement are written under `summary/`.

Scientific paired deltas are written separately:

- `future_agreement_paired.csv`: per-sample layer-mean Q-ViK minus
  all-prefill/question differences for Top-K recall, Jaccard, and retained
  Future mass.
- `vector_agreement_paired.csv`: per-sample layer-mean differences between
  Q-ViK↔Future agreement and all-prefill/question↔Future agreement for every
  vector metric.

Both files report sample-bootstrap 95% confidence intervals.

## Completed-run native comparison

After all 1,400 shared samples are complete, run the CPU-only comparator:

```bash
/workspace/nips/.conda/envs/qvik/bin/python \
  experiments/EXP-20260727-003-eval1400-shared-control/compare_native_shared.py
```

It requires byte-identical manifest SHA-256 values and an exact 1,400-sample
join on `(dataset, row_index, sample_id)`. It writes only:

- `summary/invariance_audit.json`
- `summary/native_vs_shared_prefill.csv`

The CSV compares native per-head H2O with head-averaged shared all-prefill
using downstream task score and exact response preservation against raw Full
Cache (with matched manual Full Cache as a secondary diagnostic). Full Cache,
Question, and Q-ViK predictions, scores, and applicable masks are strict
invariants. Native head-wise Future masks/metrics are never compared with the
new shared Future masks/metrics because their Top-K definitions differ.
