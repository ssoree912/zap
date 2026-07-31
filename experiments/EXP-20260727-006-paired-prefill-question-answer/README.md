# Strictly paired P+A versus Q+A

This experiment isolates the first half of a 50:50 teacher target:

\[
D_X^\ell(i)=
\frac{
\frac{1}{H|X|}\sum_h\sum_{r\in X}A_{h,r,i}^{\ell}
}{
\sum_{j\in V}
\frac{1}{H|X|}\sum_h\sum_{r\in X}A_{h,r,j}^{\ell}
},
\]

\[
Y_{\mathrm{PA}}^\ell=\mathcal N(0.5D_P^\ell+0.5D_A^\ell),
\qquad
Y_{\mathrm{QA}}^\ell=\mathcal N(0.5D_Q^\ell+0.5D_A^\ell).
\]

- \(P=\{0,\ldots,N_p-1\}\): every expanded multimodal prefill query.
- \(Q=\{\max(V)+1,\ldots,N_p-1\}\): the post-image prompt tail used by
  the existing Q+A experiment.
- \(A=\{N_p,\ldots,N_p+T-1\}\): generated answer queries
  \([y_1,\ldots,y_T]\).

Both targets reuse the same saved fp16 answer tensor from
`zap_llava15_qa50_n600_seed0`. Both keep the same `question_token_indices` as
the student-conditioning input. The 1,800 sample identities, answer
trajectory, maximum generation length, train/validation split, initialization,
architecture, and optimization are therefore paired.

After prompt-attention extraction, all saved fp16 first and answer components
are converted to float32 and independently L1-normalized again before the
50:50 targets are recomposed. The original saved answer component remains
byte-identical in both roots.

Run teacher derivation:

```bash
bash experiments/EXP-20260727-006-paired-prefill-question-answer/run_extract_4gpu.sh
```

After `verification.done`, run paired training:

```bash
TEACHER_RUN_ROOT=/path/from/the/previous/command \
bash experiments/EXP-20260727-006-paired-prefill-question-answer/run_train_2gpu.sh
```
