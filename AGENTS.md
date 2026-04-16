# AGENTS.md

## Scope

This repository is no longer the generic `kvpress` project. Treat it as a focused experimental codebase for:
- ScienceQA teacher shard collection
- linear / MLP probe training
- MileBench oracle / probe pruning evaluation
- efficiency measurement against full cache and LOOK-M

## Primary code paths

- `collect_scienceqa_teacher_xy.py`
- `train_image_teacher_probe.py`
- `train_image_teacher_probe_shards.py`
- `build_docvqa_teacher4.py`
- `evaluate_image_teacher_pruning.py`
- `scripts/measure_milebench_efficiency.py`

Helpers live in:
- `kvzap/image_teacher_utils.py`
- `kvzap/llava_extractor.py`
- `kvzap/milebench_look_metrics.py`
- `kvpress/presses/base_press.py`
- `kvpress/presses/image_token_press.py`
- `kvpress/presses/kvzap_press.py`

## What to avoid reintroducing

- generic `kvpress` pipeline APIs
- legacy press implementations
- notebook-first workflows
- the deleted `evaluation/` package
- stale docs that describe the old generic compression library

## Style

- Keep SPDX headers in Python files.
- Prefer direct imports over broad package-level exports.
- Keep the repository scoped to the retained workflow; do not add back generic benchmark abstractions unless they are required by the current ScienceQA/MileBench pipeline.
- For long-running jobs, use `screen` and save logs.
