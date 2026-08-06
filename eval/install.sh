#!/bin/bash
# Installs zap/eval's custom lmms-eval models/tasks into a target lmms-eval
# checkout, so evaluation can be driven entirely from zap's own copies
# (no need to reach into VFlowOpt_llava1.5/look_rebuttal for the model/task
# *definitions* -- an lmms-eval install, llava package, and matching
# transformers are still required as normal external dependencies, see
# README.md).
#
# Usage: eval/install.sh /path/to/lmms_eval_pkg_root
#   (the directory containing models/ and tasks/, e.g. .../lmms_eval-0.2.4/lmms_eval)
set -euo pipefail

TARGET="${1:?Usage: install.sh /path/to/lmms_eval_pkg_root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$TARGET/models" ] || [ ! -d "$TARGET/tasks" ]; then
  echo "error: $TARGET doesn't look like an lmms_eval package root (no models/ or tasks/)" >&2
  exit 1
fi

cp "$HERE/models/llava_onevision_student_delayed_replay.py" "$TARGET/models/"
cp "$HERE/models/llava_onevision_visionzip.py" "$TARGET/models/"
cp "$HERE/models/llava_onevision_training_free.py" "$TARGET/models/"
cp "$HERE/models/delayed_replay.py" "$TARGET/models/"
cp "$HERE/models/image_only_kv_cache.py" "$TARGET/models/"

mkdir -p "$TARGET/tasks/coco_cap"
cp "$HERE/tasks/coco_cap/coco2017_cap_val_chair500.yaml" "$TARGET/tasks/coco_cap/"
cp "$HERE/tasks/coco_cap/coco_cap_utils.py" "$TARGET/tasks/coco_cap/utils.py"

for benchmark in pope mme mmbench mmstar vizwiz_vqa; do
  mkdir -p "$TARGET/tasks/$benchmark"
  cp "$HERE/tasks/$benchmark/"* "$TARGET/tasks/$benchmark/"
done

mkdir -p "$TARGET/tasks/videomme_local" "$TARGET/tasks/seedbench_local"
cp "$HERE/tasks/videomme_local/"* "$TARGET/tasks/videomme_local/"
cp "$HERE/tasks/seedbench_local/"* "$TARGET/tasks/seedbench_local/"

python3 - "$TARGET/models/__init__.py" <<'PYEOF'
import re
import sys

path = sys.argv[1]
src = open(path).read()

entries = {
    "llava_onevision_student_delayed_replay": "Llava_OneVision_Student_DelayedReplay",
    "llava_onevision_visionzip": "Llava_OneVision_VisionZip",
    "llava_onevision_training_free": "Llava_OneVision_Training_Free",
}

changed = False
for model_name, cls_name in entries.items():
    if f'"{model_name}"' in src:
        continue
    m = re.search(r"AVAILABLE_MODELS\s*=\s*\{", src)
    if not m:
        raise SystemExit(f"Couldn't find AVAILABLE_MODELS dict in {path}")
    insert_at = m.end()
    line = f'\n    "{model_name}": "{cls_name}",'
    src = src[:insert_at] + line + src[insert_at:]
    changed = True
    print(f"registered {model_name}")

if changed:
    open(path, "w").write(src)
else:
    print("nothing to register, already present")
PYEOF

echo "done -- models/tasks copied into $TARGET"
