# OneVision video + hallucination eval -- setup for a fresh server

This branch (`rebuttal_a6000/video_hallucination`) holds the KV-eviction
video eval workers (`onevision/`), the lmms-eval model/task definitions
(`../eval/`), and the hallucination scorers (`../chair/`, `../amber/`).
All pure Python + small task-config files, but they depend on data,
checkpoints, and framework code that live outside this repo. This file
lists everything needed to actually run them somewhere new.

## What's in this repo (this branch)

- `onevision/generate_onevision.py` -- LOOK-M delayed-replay worker (Video-MME, SEED-Bench-video)
- `onevision/generate_onevision_student.py` -- visual-utility student-probe worker
- `onevision/generate_onevision_prefixkv.py` -- PrefixKV worker (needs `prefix_rebuttal`, see below)
- `onevision/build_video_datasets.py`, `onevision/extract_frames_parallel.py` -- data prep
- `onevision/delayed_replay.py`, `onevision/image_only_kv_cache.py` -- shared eviction/replay
  primitives, imported as siblings by every worker above (also duplicated at
  `../eval/models/` for the lmms-eval-driven side -- same file, two
  locations, since both expect same-directory imports)
- `onevision/lmms_tasks/{seedbench_local,videomme_local}/` -- same video task defs as `../eval/tasks/`, kept here too since the standalone workers above reference this path directly
- `../eval/` -- **the tested, standalone lmms-eval driver** (model classes +
  all task defs for POPE/MME/MMBench/MMStar/VizWiz-VQA/Video-MME/SEED-Bench-video/CHAIR).
  Has its own `install.sh` and a verified-working walkthrough in its README
  -- start there for anything lmms-eval-CLI-based (full-cache/VFlowOpt/VisionZip/student
  on any of those benchmarks).
- `../chair/` -- CHAIR_s/CHAIR_i scorer (COCO captions)
- `../amber/` -- AMBER benchmark response generation + scoring adapter
- `../foresight/eval/keep_budget.py` -- shared keep-ratio/keep-budget utility

## Other repos needed

| Repo | Remote | What it provides |
|---|---|---|
| An lmms-eval install (e.g. VFlowOpt_llava1.5's `src/lmms_eval-0.2.4`, or a fresh `pip install lmms-eval==0.2.4`) | `github.com/ssoree912/VFlowOpt_llava1.5` (branch `rebuttal_a6000/video_hallucination` or `rebuttal/lmms-video`) for the vendored/patched copy | The lmms-eval framework itself, LLaVA-OneVision codebase (`src/LLaVA-OneVision/llava/`), vendored transformers-4.46.0 (has the VFlowOpt `forward_illava` patch to `modeling_qwen2.py`). Run `../eval/install.sh <that install>/lmms_eval` to drop this repo's model/task files into place -- see `../eval/README.md`, verified working end to end. |
| prefix_rebuttal | `github.com/ssoree912/prefix_rebuttal` (branch `master`) | `prefixkv.py` -- `generate_onevision_prefixkv.py` hardcodes `sys.path.insert(0, "/workspace/zap/prefix_rebuttal")`, so it must be cloned at that exact path (or edit the path) |
| VisionZip | `github.com/JIA-Lab-research/VisionZip` (origin) + `github.com/ssoree912/Visionzip_onevision` (remote `onevision`, has the OneVision-specific `visionzip/onevision_*.py`) | vision-tower-level token pruning; run with `PYTHONPATH=/workspace/VisionZip` set (its lmms-eval wrapper imports it lazily, not at module load) |
| AMBER | `github.com/junyangwang0410/AMBER` | `data/query/*.json`, `data/annotations.json`, `inference.py` (official scorer). Images are NOT in the repo -- download via `gdown` from the Google Drive link in the repo's README and extract to `image/image/AMBER_*.jpg` |

## External data / checkpoints (not in any repo)

- `llava-onevision-qwen2-7b-ov` checkpoint (HF hub `lmms-lab/llava-onevision-qwen2-7b-ov`, expected locally at `ckpts/llava-onevision-qwen2-7b-ov`)
- `student_onevision` checkpoint (visual-utility student probe weights, expected at `ckpts/student_onevision`)
- COCO val2014 captions (parquet, symlinked at `data/eval/COCO-Caption2017` -> wherever the actual parquet shards live -- this symlink does not survive every environment reset, recreate with `ln -sfn <coco_cap parquet dir> data/eval/COCO-Caption2017` if `coco2017_cap_val_chair500` fails with a dataset-generation error) + `instances_val2014.json` / `captions_val2014.json` from `images.cocodataset.org/annotations/annotations_trainval2014.zip` (needed by `../chair/`, path is hardcoded in `chair_metric.py`/`run_chair.py` as `data/coco_annotations/`)
- Video-MME / SEED-Bench-video source data (parquet or extracted frames, `build_video_datasets.py`/`extract_frames_parallel.py` build these)

## Python environment

conda env `kv`: `transformers==4.46.0` (vendored/patched copy from VFlowOpt_llava1.5, not pip), `torch==2.5.1+cu121`, `lmms_eval` (editable install from VFlowOpt_llava1.5's `src/lmms_eval-0.2.4`), plus for AMBER scoring specifically: `spacy` + `en_core_web_lg` model, `nltk` (`punkt_tab`, `averaged_perceptron_tagger`, `wordnet`), `gdown`. A system `java` runtime is also needed for pycocoevalcap's METEOR scorer if running the `coco2017_cap_val*` lmms-eval tasks (not needed for CHAIR scoring itself).
