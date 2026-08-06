lmms-eval task defs for the 5 image benchmarks used in the full-cache /
student / VFlowOpt / VisionZip comparison (POPE, MME, MMBench-EN-dev,
MMStar, VizWiz-VQA). Working copies live in VFlowOpt_llava1.5's
`src/lmms_eval-0.2.4/lmms_eval/tasks/{pope,mme,mmbench,mmstar,vizwiz_vqa}/`
-- to run, place each pair back at that path (e.g.
`pope/pope_local.yaml` + `pope/utils.py` -> `lmms_eval/tasks/pope/`).

Each `*_local.yaml` reads from a local dataset path rather than the HF
hub (`dataset_path: /workspace/zap/data/eval/<Benchmark>`), which in this
container are symlinks into `/workspace/data/<Benchmark>` -- the actual
benchmark data isn't in any repo and needs to be present at that path
separately (POPE/MME/MMBench/MMStar/VizWiz-VQA, standard downloads for
each).

Model side: run via the standard lmms-eval CLI, no delayed-replay needed
for full-cache/VFlowOpt/VisionZip (see `../foresight/eval/lmms_eval_models/`
for those model classes); student needs
`llava_onevision_student_delayed_replay` (same directory) instead of
plain `llava_onevision`, since eviction has to happen between prefill and
the first answer token.

    --model llava_onevision --tasks pope_local
    --model llava_onevision_training_free --tasks pope_local
    --model llava_onevision_visionzip --tasks pope_local
    --model llava_onevision_student_delayed_replay --tasks pope_local
    (swap pope_local for mme_local / mmbench_en_dev_local / mmstar_local / vizwiz_vqa_val_local)
