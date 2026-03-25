# `s_i^+` 분포 분석

`/workspace/zap/analyze_splus_distributions.py` 는 LLaVA extractor가 저장한 `.pt` 레코드에서 다음 두 score를 계산합니다.

- attention-only: `max_j a_{ji}`
- output-aware teacher: `max_j a_{ji} * ||W_O v_i|| / ||h_j||`

그리고 prompt token을 image/text로 나눠 아래 산출물을 저장합니다.

- `all_scores_long.csv`
- `summary_by_modality.csv`
- `layerwise_fraction_below_global_median.csv`
- `failures.csv`
- `log_*_hist_by_modality.png`
- `heatmap_*.png`
- `scatter_att_vs_splus_*.png`

## 현재 파일럿 기준

현재 extractor 파일럿 조건은 아래입니다.

- report model: `liuhaotian/llava-v1.5-7b`
- implementation model: `llava-hf/llava-1.5-7b-hf`
- dataset: `DocVQA` 10 samples
- `max_new_tokens=32`
- `capture_vproj=True`
- `save_full_attentions=True`

이 설정으로 extractor는 `10/10` 성공, validation failure `0` 이었습니다.

## 필요한 입력 키

가장 이상적인 입력 레코드 구조:

- `sample_id`
- `attn_answer_to_prompt`: layer별 `[H, A, P]`
- `hidden_answer`: layer별 `[A, D]`
- `wov_norm_prompt`: layer별 `[H, P]`
- `is_image_pos_mm`: prompt 길이 이상의 bool mask

fallback으로 아래 조합도 지원합니다.

- `attentions` 또는 `full_attentions`
- `hidden_states`
- `prompt_pos_mm`, `answer_pos_mm`
- `W_O` + `vproj_outputs` 또는 `vproj_prompt`

정확한 `s_i^+` 계산에는 `||W_O v_i||`가 필요합니다. 현재 extractor 포맷은 `wov_norm_prompt`를 저장하므로 exact 계산이 됩니다.

## 실행

스크립트는 extractor output 루트 또는 그 아래 `records/` 디렉터리 둘 다 받을 수 있습니다.

repo 기준 권장 실행:

```bash
cd /workspace/zap
/opt/conda/envs/kv/bin/python /workspace/zap/analyze_splus_distributions.py \
  --input_dir /workspace/zap/artifacts/llava_docvqa_pilot \
  --out_dir /workspace/zap/artifacts/llava_docvqa_pilot_splus_analysis
```

`records/`를 직접 넘겨도 됩니다.

```bash
/opt/conda/envs/kv/bin/python /workspace/zap/analyze_splus_distributions.py \
  --input_dir /workspace/zap/artifacts/llava_docvqa_pilot/records \
  --glob '*.pt' \
  --out_dir /workspace/zap/artifacts/llava_docvqa_pilot_splus_analysis
```

샘플 수를 제한해서 빠르게 확인하려면:

```bash
/opt/conda/envs/kv/bin/python /workspace/zap/analyze_splus_distributions.py \
  --input_dir /workspace/zap/artifacts/llava_docvqa_pilot \
  --out_dir /workspace/zap/artifacts/llava_docvqa_pilot_splus_analysis_debug \
  --limit 1
```

## 먼저 볼 파일

우선순위는 아래 순서가 맞습니다.

1. `summary_by_modality.csv`
2. `log_splus_hist_by_modality.png`
3. `scatter_att_vs_splus_image.png`
4. `heatmap_log_splus_image.png`
5. `log_splus_fraction_below_median.png`

해석 포인트:

- image token의 `log_splus`가 attention-only보다 덜 눌려 있으면 output-aware teacher 쪽 신호가 더 강할 수 있습니다.
- `scatter_att_vs_splus_image.png` 에서 attention은 낮은데 `s_i^+`는 높은 visual token이 보이면, attention-only보다 output-aware teacher가 더 informative하다는 첫 근거가 됩니다.
- `heatmap_log_splus_image.png` 에서 특정 layer/head 집중이 보이면 후속 pruning 또는 teacher design에 바로 연결할 수 있습니다.

## 현재 파일럿 결과 예시

`/workspace/zap/artifacts/llava_docvqa_pilot_splus_analysis_run2/summary_by_modality.csv` 기준으로:

- image `log_splus` median: `-10.87`
- text `log_splus` median: `-8.42`
- image `wov_norm` median: `4.87`
- text `wov_norm` median: `4.94`

즉 현재 10샘플 파일럿에서는 text prompt token 쪽 score가 전반적으로 더 크지만, `wov_norm` 자체는 image/text가 비슷한 범위에 있습니다. 다음 확인 포인트는 image scatter에서 low-attention / high-`s_i^+` token이 실제로 존재하는지입니다.
