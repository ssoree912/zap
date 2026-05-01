#!/bin/bash
# M=3 trajectory teacher 재수집 — teacher_v2_traj/ 에 저장.
# llava_instruct는 teacher_v2/llava_instruct/ 에서 복사.
# scienceqa / textvqa / docvqa 는 M=3으로 새로 수집.
# GPU 2 사용.
set -uo pipefail

DEVICE_ID=${DEVICE_ID:-2}
DEVICE=cuda:0  # CUDA_VISIBLE_DEVICES 로 격리하므로 항상 cuda:0
TRAJ_M=3
TRAJ_TEMP=0.7
TRAJ_TOP_P=0.9
N=200
MAX_NEW_TOKENS=64
SRC_ROOT=/workspace/zap/data/teacher_v2
OUT_ROOT=/workspace/zap/data/teacher_v2_traj
COLLECT=/workspace/zap/collect_future_teacher_v2.py

mkdir -p "$OUT_ROOT"

# ── llava_instruct: teacher_v2 → teacher_v2_traj 복사 ──────────────────────
echo "[step 0] llava_instruct 복사: $SRC_ROOT/llava_instruct -> $OUT_ROOT/llava_instruct"
mkdir -p "$OUT_ROOT/llava_instruct"
cp -n "$SRC_ROOT/llava_instruct/"*.pt "$OUT_ROOT/llava_instruct/" 2>/dev/null || true
cp -n "$SRC_ROOT/llava_instruct/_summary.json" "$OUT_ROOT/llava_instruct/" 2>/dev/null || true
N_COPIED=$(ls "$OUT_ROOT/llava_instruct/"*.pt 2>/dev/null | wc -l)
echo "[step 0] 복사 완료: $N_COPIED files"

# ── scienceqa M=3 재수집 ─────────────────────────────────────────────────────
echo "[step 1/3] scienceqa n=$N M=$TRAJ_M"
CUDA_VISIBLE_DEVICES=$DEVICE_ID python "$COLLECT" \
  --dataset scienceqa \
  --n-samples "$N" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUT_ROOT" \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0 \
  2>&1 | tee "$OUT_ROOT/scienceqa_collect.log"

# ── textvqa M=3 재수집 ───────────────────────────────────────────────────────
echo "[step 2/3] textvqa n=$N M=$TRAJ_M"
CUDA_VISIBLE_DEVICES=$DEVICE_ID python "$COLLECT" \
  --dataset textvqa \
  --n-samples "$N" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUT_ROOT" \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0 \
  2>&1 | tee "$OUT_ROOT/textvqa_collect.log"

# ── docvqa M=3 재수집 ────────────────────────────────────────────────────────
echo "[step 3/3] docvqa n=$N M=$TRAJ_M"
CUDA_VISIBLE_DEVICES=$DEVICE_ID python "$COLLECT" \
  --dataset docvqa \
  --n-samples "$N" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUT_ROOT" \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0 \
  2>&1 | tee "$OUT_ROOT/docvqa_collect.log"

echo ""
echo "[done] teacher_v2_traj 수집 완료"
for d in llava_instruct scienceqa textvqa docvqa; do
  n=$(ls "$OUT_ROOT/$d/"*.pt 2>/dev/null | wc -l)
  echo "  $d: $n samples"
done
echo "총 $(ls "$OUT_ROOT/"*/*.pt 2>/dev/null | wc -l) / 800 samples"
echo ""
echo "학습 실행 예시:"
echo "  python /workspace/zap/train_visual_utility_student.py \\"
echo "    --teacher-root $OUT_ROOT \\"
echo "    --datasets scienceqa textvqa docvqa llava_instruct \\"
echo "    --scope future_all_token \\"
echo "    --output-dir /workspace/zap/ckpts/student_v2_traj"
