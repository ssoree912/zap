# EXP-20260420-003 — PPL Results on PrefixKV protocol

Teacher-forcing perplexity on **mm-vet (218)** and **detail_1k / LLaVA-Description
(1000)** with LLaVA-1.5-7B. All methods reported on a unified `keep =
fraction-of-KV-retained` axis (PrefixKV `--ratio r` is mapped to `keep = 1 − r`).

## Methods

| Method | Scope | Knob | Scoring signal | Notes |
|---|---|---|---|---|
| `zap_future_v5_image` | IMAGE only | `--image-keep-ratio` | Future probe v5 (all-layer MLP) | text/system KV always kept |
| `zap_future_v5_total` | ALL tokens | `--total-keep-ratio` | Future probe v5 | scope-matched to PrefixKV |
| `zap_h2o_all_token` | ALL tokens | `--total-keep-ratio` | prefill H2O heavy-hitter | eager attention, prefill-only hook |
| `zap_hybrid_a050` | ALL tokens | `--total-keep-ratio` | 0.5·softmax(H2O) + 0.5·softmax(Future) | flat α=0.5 at every layer |
| `PrefixKV` | ALL tokens | `--ratio` = REMOVE fraction | layer-wise adaptive prefix | external baseline |

- **Probe**: `future_probe_v5_all_token_10ep` (all 32 layers, 10 ep, best val
  Spearman 0.232 @ ep9). Used by `zap_future_v5_*` and `zap_hybrid_a050`.
- **Runs collected**: PrefixKV=12, zap_h2o_all_token=5, zap_hybrid_a050=5, zap_future_v5_total=10, zap_future_v5_image=12.

## mm-vet

| keep | PrefixKV | prefill_evict | hybrid_0.5 | future | future_img_only_eviction | Δ(PrefixKV − future) |
|---|---|---|---|---|---|---|
| 0.1 | 7.3750 | 6.1135 | 5.0803 | 5.1452 | 5.1079 | +2.2298 |
| 0.3 | 5.7188 | 5.8781 | 5.0740 | 5.0833 | 5.0806 | +0.6354 |
| 0.5 | 5.5000 | 5.4913 | 5.0827 | 5.1027 | 5.1047 | +0.3973 |
| 0.7 | 5.3750 | 5.3397 | 5.1096 | 5.1264 | 5.1290 | +0.2486 |
| 0.9 | 5.2812 | 5.1992 | 5.1390 | 5.1377 | 5.1373 | +0.1435 |

## detail_1k (LLaVA-Description)

| keep | PrefixKV | future | future_img_only_eviction | Δ(PrefixKV − future) |
|---|---|---|---|---|
| 0.1 | 4.4062 | 3.1286 | 3.0907 | +1.2776 |
| 0.3 | 3.4844 | 3.0073 | 3.0001 | +0.4771 |
| 0.5 | 3.4062 | 2.9687 | 2.9659 | +0.4375 |
| 0.7 | 3.2500 | 2.9549 | 2.9545 | +0.2951 |
| 0.9 | 3.2031 | 2.9526 | 2.9529 | +0.2505 |

## Observations (scope-matched, all-token methods)

### zap Future v5 vs PrefixKV
- Future-v5 (total) beats PrefixKV at every keep on both datasets.
- Gap is largest at aggressive eviction: detail_1k keep=0.1 is +1.3 PPL in
  zap's favor; mm-vet keep=0.1 is +2.2. Converges as keep → 0.9.

### Hybrid (H2O + Future, α=0.5) vs H2O-only
- On mm-vet, `hybrid_a050` improves on `h2o_all_token` at every keep — the Future
  signal adds information that raw prefill attention misses.
- Against `zap_future_v5_total`, hybrid is either comparable or slightly worse
  at high keep, and slightly better at very low keep on mm-vet — weak
  complementarity. Not a clear win.

### Image-only scope (reference line)
- `zap_future_v5_image` is nearly flat across keep (mm-vet 5.08–5.14, detail_1k
  2.95–3.09). Image KV is highly redundant — the Future probe keeps the few
  positions that actually matter for decode.
- Comparing `_image` to `_total` at the same keep isolates the cost of
  evicting text: small at high keep, larger at keep=0.1 (where the total
  budget starts cutting into text too).

## Files

- CSV: `results.csv` (long form, one row per run)
- This report: `results.md`
- Zap raw artifacts: `/workspace/zap/artifacts/EXP-20260420-003/{mmvet,detail1k}/*/result.json`
- PrefixKV raw logs: `/workspace/PrefixKV/logs/{mmvet,detail1k}-gpu*/prefixkv/*/*.txt`
