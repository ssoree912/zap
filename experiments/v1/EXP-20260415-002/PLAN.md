## Experiment Plan

**ID**: EXP-20260415-002  
**Author**: ZAP Team  
**Date**: 2026-04-15  
**Last Updated**: 2026-04-16  
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 목적

`image_probe_combined_v1`(ScienceQA+TextVQA+NLVR2 학습 완료)를 MileBench 전체에 적용해,
1) Probe 추론 결과를 안정적으로 수집하고,
2) 동일 truncation 조건에서 `oracle_onthefly` 결과를 전수 수집한다.

최종 산출물:
- Probe 결과: `/workspace/zap/artifacts/combine_prob/{dataset_slug}/...`
- Oracle 결과: `/workspace/zap/artifacts/oracle/{dataset_slug}/oracle_onthefly/keep_0p20`

---

### 2. 고정 실행 조건

- Conda env: `kv`
- Implementation model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Probe ckpt: `/workspace/zap/ckpts/image_probe_combined_v1/mlp`
- Data root: `/workspace/zap/data/MileBench`
- Prompt style: `look_milebench`
- Keep ratio:
  - Probe: `image_keep_ratio=0.20`
  - Oracle on-the-fly: `total_keep_ratio=0.20`
- Truncation: `--truncate_like_lookm`
- Truncation params:
  - `look_max_context_len=4096`
  - `look_n_tokens_per_image=576` (OOM 방지 핵심)

---

### 3. 현재까지 반영된 코드 변경

1. Prompt placeholder 보정 (multi-image)
- 파일: `/workspace/zap/kvzap/image_teacher_utils.py`
- `{image#n}`, `<ImageHere>`, `<image>`를 통합 정규화하고,
  placeholder 개수와 image 개수가 달라도 자동 보정되도록 수정.

2. Probe 결과 저장 경로 구조 통일
- 파일:
  - `/workspace/zap/scripts/run_milebench_probe_all.sh`
  - `/workspace/zap/scripts/run_docvqa_probe_teacher_sweep.sh`
- 저장 형식:
  - `/workspace/zap/artifacts/combine_prob/{dataset_slug}/probe_mlp/keep_0p20`

3. Oracle on-the-fly 전체 데이터셋 실행 스크립트 확장
- 파일: `/workspace/zap/scripts/run_oracle_onthefly_common69.py`
- 변경 내용:
  - common69 manifest 의존 제거 (기본: MileBench 전수 스캔)
  - 기본 경로를 `/workspace/zap/...`로 전환
  - 출력 루트를 `/workspace/zap/artifacts/oracle`로 전환
  - 저장 구조를 `{dataset_slug}/oracle_onthefly/keep_0p20` 형태로 통일
  - dataset별 `look_model_name` 분리
  - 실행 요약 저장:
    `/workspace/zap/artifacts/oracle/_run_summaries/oracle_onthefly_all_datasets_summary.json`

---

### 4. Probe 실행 현황 (2026-04-16 기준)

핵심 복구 완료:
- `actionlocalization`: `200/200`, failures `0`
- `mmcoqa`: `200/200`, failures `0`
- `multimodalqa`: `200/200`, failures `0`

정리:
- 초기 실패 원인은 placeholder mismatch + OOM 혼재였음.
- 현재는 placeholder 경로 패치 완료, OOM은 truncation 파라미터(`576`)와 allocator 설정으로 복구됨.

---

### 5. Oracle(on-the-fly) 실행 계획 (전체 데이터셋)

실행 스크립트:
- `/workspace/zap/scripts/run_oracle_onthefly_common69.py`

권장 실행 (GPU 1 물리카드 사용):
```bash
cd /workspace/zap
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n kv --no-capture-output \
python /workspace/zap/scripts/run_oracle_onthefly_common69.py \
  --milebench_root /workspace/zap/data/MileBench \
  --artifact_root /workspace/zap/artifacts/oracle \
  --model_name /workspace/zap/ckpts/llava-1.5-7b-hf \
  --device cuda:0 \
  --total_keep_ratio 0.20 \
  --truncate_like_lookm \
  --look_max_context_len 4096 \
  --look_n_tokens_per_image 576 \
  --skip_existing
```

출력 구조 예시:
- `/workspace/zap/artifacts/oracle/actionlocalization/oracle_onthefly/keep_0p20`
- `/workspace/zap/artifacts/oracle/mmcoqa/oracle_onthefly/keep_0p20`
- `/workspace/zap/artifacts/oracle/multimodalqa/oracle_onthefly/keep_0p20`

성공 기준:
- 각 데이터셋 `metrics.json`에서 `n_predictions == n_samples` and `n_failures == 0`

---

### 6. 실행/모니터링 규칙

- 장시간 실행은 `screen` 세션 사용.
- `SKIP_EXISTING` 기반으로 완료 데이터셋 자동 스킵.
- 실패 데이터셋은 `--datasets`로 부분 재실행.
- 완료 후 `_run_summaries/oracle_onthefly_all_datasets_summary.json`으로 전체 상태 집계.

---

### 7. 남은 작업 체크리스트

- [x] Probe 결과 경로 구조 통일 (`combine_prob/{dataset_slug}/...`)
- [x] multi-image placeholder mismatch 보정
- [x] Probe 주요 실패 데이터셋 복구 (ActionLocalization/MMCoQA/MultiModalQA)
- [ ] Oracle on-the-fly 전체 데이터셋 실행 (GPU 1)
- [ ] Oracle 실패 데이터셋 재실행 및 최종 `failures=0` 확인
- [ ] EXP-20260415-002 `RESULTS.md` 최종 반영

---

### 8. Oracle 실행 로그 (2026-04-16 시작)

- screen 세션: `oracle_all_gpu1`
- 로그 파일: `/workspace/zap/artifacts/oracle/_run_logs/oracle_all_gpu1_20260416T050607Z.log`
- 출력 루트: `/workspace/zap/artifacts/oracle/`
- 첫 실행 항목:
  - `ALFRED -> /workspace/zap/artifacts/oracle/alfred/oracle_onthefly/keep_0p20`
