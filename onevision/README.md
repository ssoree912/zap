# OneVision video + hallucination eval -- setup for a fresh server

This branch (`rebuttal_a6000/video_hallucination`) holds the KV-eviction
video eval workers (`onevision/`) and the hallucination scorers
(`../foresight/eval/{chair,amber}/`). Both are pure Python + a few small
task-config files, but they depend on data, checkpoints, and code that live
outside this repo. This file lists everything needed to actually run them
somewhere new.

## What's in this repo (this branch)

- `onevision/generate_onevision.py` -- LOOK-M delayed-replay worker (Video-MME, SEED-Bench-video)
- `onevision/generate_onevision_student.py` -- visual-utility student-probe worker
- `onevision/generate_onevision_prefixkv.py` -- PrefixKV worker (needs `prefix_rebuttal`, see below)
- `onevision/build_video_datasets.py`, `onevision/extract_frames_parallel.py` -- data prep
- `onevision/delayed_replay.py`, `onevision/image_only_kv_cache.py` -- shared eviction/replay
  primitives, imported as siblings by every worker above (also duplicated at
  `../foresight/eval/` for the hallucination-eval side -- same file, two
  locations, since the workers expect same-directory imports)
- `onevision/lmms_tasks/{seedbench_local,videomme_local}/` -- lmms-eval task defs for the two video benchmarks
- `../foresight/eval/chair/` -- CHAIR_s/CHAIR_i scorer (COCO captions)
- `../foresight/eval/amber/` -- AMBER benchmark response generation + scoring adapter
- `../foresight/eval/keep_budget.py` -- shared keep-ratio/keep-budget utility
- `../foresight/eval/lmms_eval_models/` -- reference copies of the lmms-eval
  model/task files (working copies live in the VFlowOpt_llava1.5 repo, see
  below); that folder's own README says exactly where each file needs to go

## Other repos needed

| Repo | Remote | What it provides |
|---|---|---|
| VFlowOpt_llava1.5 | `github.com/ssoree912/VFlowOpt_llava1.5` (branch `rebuttal_a6000/video_hallucination` or `rebuttal/lmms-video`) | LLaVA-OneVision codebase (`src/LLaVA-OneVision/llava/`), vendored transformers-4.46.0 (has the VFlowOpt `forward_illava` patch to `modeling_qwen2.py`), lmms-eval-0.2.4 -- clone this, then copy `foresight/eval/lmms_eval_models/*` into its `src/lmms_eval-0.2.4/lmms_eval/{models,tasks/coco_cap}/` and register the two new model names in `lmms_eval/models/__init__.py`'s `AVAILABLE_MODELS` |
| prefix_rebuttal | `github.com/ssoree912/prefix_rebuttal` (branch `master`) | `prefixkv.py` -- `generate_onevision_prefixkv.py` hardcodes `sys.path.insert(0, "/workspace/zap/prefix_rebuttal")`, so it must be cloned at that exact path (or edit the path) |
| VisionZip | `github.com/JIA-Lab-research/VisionZip` (origin) + `github.com/ssoree912/Visionzip_onevision` (remote `onevision`, has the OneVision-specific `visionzip/onevision_*.py`) | vision-tower-level token pruning; run with `PYTHONPATH=/workspace/VisionZip` set (its lmms-eval wrapper imports it lazily, not at module load) |
| AMBER | `github.com/junyangwang0410/AMBER` | `data/query/*.json`, `data/annotations.json`, `inference.py` (official scorer). Images are NOT in the repo -- download via `gdown` from the Google Drive link in the repo's README and extract to `image/image/AMBER_*.jpg` |

## External data / checkpoints (not in any repo)

- `llava-onevision-qwen2-7b-ov` checkpoint (HF hub `lmms-lab/llava-onevision-qwen2-7b-ov`, expected locally at `ckpts/llava-onevision-qwen2-7b-ov`)
- `student_onevision` checkpoint (visual-utility student probe weights, expected at `ckpts/student_onevision`)
- COCO val2014 captions (parquet, `data/eval/COCO-Caption2017` in earlier setup) + `instances_val2014.json` / `captions_val2014.json` from `images.cocodataset.org/annotations/annotations_trainval2014.zip` (needed by `foresight/eval/chair/`, path is hardcoded in `chair_metric.py`/`run_chair.py` as `data/coco_annotations/`)
- Video-MME / SEED-Bench-video source data (parquet or extracted frames, `build_video_datasets.py`/`extract_frames_parallel.py` build these)

## Python environment

conda env `kv`: `transformers==4.46.0` (vendored/patched copy from VFlowOpt_llava1.5, not pip), `torch==2.5.1+cu121`, `lmms_eval` (editable install from VFlowOpt_llava1.5's `src/lmms_eval-0.2.4`), plus for AMBER scoring specifically: `spacy` + `en_core_web_lg` model, `nltk` (`punkt_tab`, `averaged_perceptron_tagger`, `wordnet`), `gdown`. A system `java` runtime is also needed for pycocoevalcap's METEOR scorer if running the `coco2017_cap_val*` lmms-eval tasks (not needed for CHAIR scoring itself).
