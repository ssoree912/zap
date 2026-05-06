## Experiment Plan

**ID**: EXP-20260506-027
**Author**: ssoree912
**Date**: 2026-05-06
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기
LLaVA-OneVision 7B + ZAP student (future all-token)를 MileBench 전체 28개 데이터셋에 대해 1-combine 이미지 입력으로 keep_ratio=0.2 조건에서 평가.

### 2. 가설
1-combine 입력(여러 이미지를 하나로 합친 그리드) 환경에서 keep_ratio=0.2로 ZAP student eviction이 작동하며, 전체 28 데이터셋 평균 점수가 baseline 대비 허용 가능한 범위(≤5% drop)를 유지할 것.

### 3. 독립변수
- keep_ratio: 0.2
- combine_image: 1 (combined_1_images)
- 모델: LLaVA-OneVision 7B (LMMs-Lab format)

### 4. 종속변수
- 주요 metric: 28개 데이터셋 평균 accuracy
- 보조 metric: 각 데이터셋별 accuracy, avg image keep ratio, avg total keep ratio

### 5. 고정 조건
- 데이터셋: MileBench 전체 28개
- student: student_onevision_original_future_1800_lr1e4_15ep
- max_new_tokens: 64
- hardware: GPU 1대

### 6. 베이스라인
- fullcache (keep_ratio=1.0): 미실행, 비교는 이후 진행
- EXP-20260430-003-progressive keep020 결과 (multi-image 입력 방식)

### 7. 예상 결과
1-combine 입력으로 image token이 하나의 이미지로 합쳐지므로, multi-image 방식 대비 이미지 구조 정보 일부 손실이 있을 수 있으나 전반적 성능은 유사할 것.

### 8. 판단 기준
전체 28 데이터셋 _summary.json 생성 완료 및 평균 accuracy ≥ 50%.

### 9. 이 실험으로 증명할 수 없는 것
- fullcache 대비 정확한 drop 폭 (별도 fullcache 실행 필요)
- 1-combine vs multi-image 직접 비교 (동일 keep_ratio 조건 multi-image 실험 필요)

### 10. 예상 런타임 / 리소스
- GPU: 1 (dlpc server)
- 예상 시간: 28 datasets × ~30분 = ~14시간
- 디스크: ~500MB

### 재현 커맨드
```bash
cd /mnt/srv/home/dlpc.3842/zap
GPU=0 KEEP_RATIO=0.2 COMBINE_IMAGE=1 ./scripts/run_milebench_zap_student_onevision.sh
```
