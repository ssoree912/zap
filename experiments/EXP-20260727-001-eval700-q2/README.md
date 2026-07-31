# Eval-700 Q2 experiment

This experiment evaluates 100 deterministic held-out samples from each local
VQA/caption benchmark and excludes MileBench:

- GQA, TextVQA, DocVQA, ChartQA
- COCO Caption, NoCaps, TextCaps

For layer \(\ell\), KV head \(h\), and visual token \(i\), native H2O prefill
importance is

\[
P_i^{\ell,h}
=
\sum_{q=1}^{N_{\mathrm{prompt}}}
A_{\mathrm{pre}}^{\ell,h}[q,i].
\]

The semantic-question-only diagnostic is

\[
Q_i^\ell
=
\frac{1}{H|Q_{\mathrm{semantic}}|}
\sum_h\sum_{q\in Q_{\mathrm{semantic}}}
A_{\mathrm{pre}}^{\ell,h}[q,i].
\]

The full-cache future target is

\[
F_i^{\ell,h}
=
\frac{1}{T}\sum_{t=1}^{T}
A_{\mathrm{dec},t}^{\ell,h}[i],
\]

using the same generation-block convention as the trained student:
`[last prompt query, y_1, ..., y_(T-1)]`.

Every intervention preserves all text KVs.  With requested total-cache ratio
\(r=0.2\), the visual-token budget is

\[
K_v
=
\max\left(0,\left\lceil rN_{\mathrm{prompt}}\right\rceil-N_{\mathrm{text}}\right).
\]

H2O applies this Top-K independently per KV head. Question-only, Q-ViK, and
Future Oracle use a shared layer-wise visual-token mask. The exact selected
masks, per-layer agreement, predictions, task scores, and sample-bootstrap
confidence intervals are saved.
