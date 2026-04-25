# AGENTS.md

## Scope

This repository is a focused experimental codebase for:
- Unified teacher shard collection (PV + Future labels in a single forward pass)
- Per-layer MLP probe training (PostVision / Future / Hybrid)
- MileBench oracle / probe pruning evaluation
- PPL evaluation on PrefixKV datasets (detail_1k, mm-vet)
- Efficiency measurement against full cache and LOOK-M

## Primary code paths

- `collect_unified_teacher_shards.py`
- `train_unified_probe_onepass.py`
- `evaluate_image_teacher_pruning.py`
- `eval_ppl.py`
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
- legacy PV-only collection / training scripts (removed)
- legacy v1/v2 Future probe scripts (replaced by unified)
- notebook-first workflows
- the deleted `evaluation/` package

## Style

- Keep SPDX headers in Python files.
- Prefer direct imports over broad package-level exports.
- For long-running jobs, use `screen` (or `nohup`) and save logs.
