# Standalone lmms-eval driver for zap

Everything under `models/` and `tasks/` here is enough, by itself, to drive
the lmms-eval side of this project's evaluation, on both image and video
tasks, for all four methods (full-cache/VFlowOpt/VisionZip/student) -- no
need to reach into VFlowOpt_llava1.5 or look_rebuttal for the model or task
*definitions*. See the compatibility table below for what's been tested at
what depth; student's video path (added 2026-08-06) is smoke-tested only,
not yet run at full scale. An actual lmms-eval installation (the
framework itself: `lmms_eval/__main__.py`, `evaluator.py`, `api/`, the
stock `llava_onevision.py` base class, etc.), the `llava` package, and a
transformers build with the VFlowOpt `forward_illava` patch are still
required as external dependencies -- this directory holds the
project-specific code, not a reimplementation of lmms-eval itself.

## Setup

```
eval/install.sh /path/to/lmms_eval_pkg_root
```

where the target is the directory containing `models/` and `tasks/` inside
an lmms-eval checkout (e.g. `.../lmms_eval-0.2.4/lmms_eval`). This copies
the model classes and task defs into place and registers the three custom
models in the target's `AVAILABLE_MODELS` dict (idempotent -- safe to
re-run). It does NOT install lmms-eval/llava/transformers themselves.

Then, whatever process runs `python -m lmms_eval`, make sure zap's repo
root is also on `PYTHONPATH` (the model classes import `kvpress` and
`foresight.eval.keep_budget` from there):

```
export PYTHONPATH=/path/to/lmms_eval_pkg_root/..:/path/to/zap
```

## Verified 2026-08-06

Installed into a *fresh* lmms-eval-0.2.4 copy with all zap-specific
customizations stripped out first (no `llava_onevision_student_delayed_replay.py`,
no `llava_onevision_visionzip.py`, no `llava_onevision_training_free.py`, no
`process_docs_chair500`, nothing in `AVAILABLE_MODELS`) -- i.e. simulating a
genuinely clean machine with only vanilla lmms-eval present. After running
`install.sh` against it:

```
--model llava_onevision_student_delayed_replay --tasks coco2017_cap_val_chair500 --limit 5
```

produced `coco_ROUGE_L = 0.4594` and captions identical word-for-word to
the original run (against the "production" VFlowOpt_llava1.5 checkout) on
all 5 samples. Model registration, task loading, and generation all work
end to end from zap's own files.

**Not covered by this verification**: `llava_onevision_student_delayed_replay`
on video tasks -- at the time of this specific test, that combination did
not work at all (confirmed 2026-08-06 on a separate machine, GPU 3:
`ValueError: Unsupported visual type for delayed-replay: <class 'str'>`,
exactly per the code's then-current docstring, "video/multi-image handling
is dropped"). Video support was added the same day; see the next section.

## student video support, added 2026-08-06

`llava_onevision_student_delayed_replay` now decodes video
(`videomme_local`/`seedbench_local`'s raw `.mp4` paths, via
`self.load_video`/`read_video_pyav` same as the base class) and threads
`modalities=["video"]` + explicit per-frame `image_sizes` through the
delayed-replay prefill call -- `image_sizes` has to be explicit here
(unlike `Llava_OneVision.generate_until`'s video branch, which omits it)
because this goes through a direct `self.model(...)` call rather than
`.generate()`; omitting it only works through the `.generate()` call
chain. Same fix `generate_onevision_student.py` needed on the standalone
video path.

Smoke-tested on 3 real `videomme_local` samples end to end (video decode
-> delayed-replay prefill/evict/replay -> lmms-eval's own
`videomme_percetion_score` scoring): all 3 answers matched target exactly.
**Not yet run at full scale or cross-checked in aggregate** against
`../onevision/generate_onevision_student.py`'s numbers on the same
samples -- that's the next step before trusting this for a real
comparison table entry.

## Layout

- `models/` -- `llava_onevision_student_delayed_replay.py` (delayed-replay,
  needs `delayed_replay.py`/`image_only_kv_cache.py`, both here as
  siblings), `llava_onevision_visionzip.py`, `llava_onevision_training_free.py`
  (standard `.generate()`, no delayed-replay needed for either)
- `tasks/coco_cap/` -- `coco2017_cap_val_chair500.yaml` +
  `coco_cap_utils.py` (stock lmms-eval coco_cap/utils.py plus
  `process_docs_chair500`, the 500-image fixed-seed subset; install as
  `utils.py`, not `coco_cap_utils.py` -- the yaml resolves `!function
  utils.X` relative to its own directory)
- `tasks/{pope,mme,mmbench,mmstar,vizwiz_vqa}/` -- the `*_local.yaml`
  variants used throughout, reading from local dataset paths rather than
  the HF hub (see each task's own README note in the repo root
  `image_tasks/`... consolidated here instead as of this commit)
- `tasks/{videomme_local,seedbench_local}/` -- video task defs, shared with
  the standalone workers in `../onevision/`

## Model x task compatibility

| `--model` | image tasks (pope_local / mme_local / mmbench_en_dev_local / mmstar_local / vizwiz_vqa_val_local / coco2017_cap_val_chair500) | video tasks (videomme_local / seedbench_local) |
|---|---|---|
| `llava_onevision` (full-cache, stock lmms-eval, not copied here) | verified against a comparison table, matches to ~2 decimal places | run for the video comparison table earlier this project |
| `llava_onevision_training_free` (VFlowOpt) | verified against a comparison table, matches to 2 decimal places | run for the video comparison table earlier this project |
| `llava_onevision_visionzip` (VisionZip) | verified against a comparison table, matches to 2 decimal places | run for the video comparison table earlier this project |
| `llava_onevision_student_delayed_replay` (student) | verified against a comparison table, matches to 2 decimal places | smoke-tested only (3 samples, all correct) -- not yet run at scale or cross-checked against `../onevision/generate_onevision_student.py`'s numbers |

For CHAIR scoring itself (turning generated captions into CHAIR_s/CHAIR_i),
see `../chair/`. For AMBER, see `../amber/`. For the standalone
(non-lmms-eval) video workers -- LOOK-M, student, PrefixKV, which need
delayed-replay but don't go through lmms-eval at all -- see `../onevision/`.
