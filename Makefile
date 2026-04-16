SHELL := /bin/bash
UV ?= $(shell which uv)
PYTHON ?= python

RETAINED_PY := \
	build_docvqa_teacher4.py \
	build_scienceqa_manifest.py \
	collect_scienceqa_teacher_xy.py \
	evaluate_image_teacher_pruning.py \
	train_image_teacher_probe.py \
	train_image_teacher_probe_shards.py \
	kvpress/__init__.py \
	kvpress/utils.py \
	kvpress/presses/__init__.py \
	kvpress/presses/base_press.py \
	kvpress/presses/image_token_press.py \
	kvpress/presses/kvzap_press.py \
	kvzap/__init__.py \
	kvzap/image_teacher_utils.py \
	kvzap/llava_extractor.py \
	kvzap/milebench_look_metrics.py \
	scripts/export_efficiency_comparison_csv.py \
	scripts/export_probe_lookm_performance_csv.py \
	scripts/export_scienceqa_probe_csv_summaries.py \
	scripts/measure_milebench_efficiency.py

.PHONY: format
format:
	$(UV) run isort .
	$(UV) run black .

.PHONY: lint
lint:
	$(UV) run flake8 $(RETAINED_PY)
	$(UV) run mypy $(RETAINED_PY)

.PHONY: compile
compile:
	$(PYTHON) -m py_compile $(RETAINED_PY)
