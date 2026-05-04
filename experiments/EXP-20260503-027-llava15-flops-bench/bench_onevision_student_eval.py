#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaVA-OneVision-Qwen2-7B + trained student inference bench (latency + accuracy).

Wires `VisualUtilityStudentOneVisionPress` to the model via kvpress forward
hooks. For each MileBench sample and each keep ratio:
  - Run prefill with press(model) context active. Hooks fire per layer to
    score image tokens with the trained student and prune image KV.
  - Generate up to `max_new_tokens` tokens (manual decode loop with timing).
  - Record latency, peak memory, KV cache size, generated text.

After all runs, score predictions vs MileBench ground truth using:
  - exact_match (case-folded, whitespace-collapsed, after answer normalization)
  - contains    (ground-truth substring present in prediction)
  - rougel      (token-level ROUGE-L F1)

Usage:
  python bench_onevision_student_eval.py \\
      --output-dir outputs/onevision_student_eval \\
      --student-path /workspace/zap/artifacts/original_onevision_teacher/student_onevision_original_future_1800_lr1e4_15ep \\
      --sample-size 50 --ratios 1.0 0.2 0.05
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VFLOWOPT_LLAVA_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from kvpress.presses.image_token_press import VisualUtilityStudentOneVisionPress  # noqa: E402

GIB = 1024 ** 3


def patch_siglip_loader(local_siglip_path: str) -> None:
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel.from_pretrained(
            local_siglip_path, device_map=device_map
        )
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


@dataclass(frozen=True)
class PooledSample:
    dataset: str
    sample_id: str
    question: str
    image_paths: tuple[str, ...]
    answer: Optional[str]
    choice_list: tuple[str, ...] = ()

    @property
    def sample_key(self) -> str:
        return f"{self.dataset}:{self.sample_id}"


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------
# MileBench sample loading (single image, combined_1_images)
# ----------------------------------------------------------------------


def _resolve_task_instruction(task_instructions: Any, task_instruction_id: Any) -> str:
    if isinstance(task_instructions, list):
        try:
            idx = int(task_instruction_id)
            if 0 <= idx < len(task_instructions):
                return str(task_instructions[idx])
        except Exception:
            pass
    if isinstance(task_instructions, dict):
        key = str(task_instruction_id)
        if task_instruction_id in task_instructions:
            return str(task_instructions[task_instruction_id])
        if key in task_instructions:
            return str(task_instructions[key])
    return ""


def _choice_label(index: int) -> str:
    if index < 26:
        return chr(65 + index)
    if index < 52:
        return "A" + chr(65 + index - 26)
    return "B" + chr(65 + index - 52)


def _build_choice_block(choice_list: Any, dataset_name: str) -> str:
    if not isinstance(choice_list, list) or not choice_list:
        return ""
    lines = []
    for idx, choice in enumerate(choice_list):
        text = str(choice)
        lines.append(text if dataset_name == "GPR1200" else f"{_choice_label(idx)}. {text}")
    return "\nChoice list: \n" + "\n".join(lines) + "\nYour answer is:"


def _build_milebench_question(record: dict[str, Any], meta: dict[str, Any], dataset_name: str) -> str:
    task = record.get("task_instance", {})
    context = str(task.get("context", "")).strip()
    instruction = _resolve_task_instruction(meta.get("task_instruction"), record.get("task_instruction_id", 0))
    choice_block = _build_choice_block(task.get("choice_list"), dataset_name)
    return f"{instruction}\n{context}{choice_block}".strip()


def _resolve_answer(record: dict[str, Any]) -> Optional[str]:
    for key in ("answer", "response", "target", "label"):
        if record.get(key) is not None:
            return str(record[key])
    task = record.get("task_instance")
    if isinstance(task, dict):
        for key in ("answer", "answers", "label"):
            value = task.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                return str(value[0]) if value else None
            return str(value)
    return None


def _sanitize_id(value: Any, fallback: int) -> str:
    text = str(value) if value is not None else f"sample-{fallback:06d}"
    text = text.strip() or f"sample-{fallback:06d}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:128]


def build_qwen_prompt(question: str, conv_template: str = "qwen_1_5") -> str:
    import copy
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"<image>\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def sample_manifest(
    *,
    milebench_root: Path,
    datasets: Optional[list[str]],
    tokenizer: Any,
    sample_size: int,
    seed: int,
    max_raw_prompt_tokens: int,
    conv_template: str,
) -> list[PooledSample]:
    if datasets is None:
        datasets = []
        for child in sorted(milebench_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / f"{child.name}.json").is_file():
                datasets.append(child.name)
    candidates: list[PooledSample] = []
    for dataset_name in datasets:
        ddir = milebench_root / dataset_name
        jpath = ddir / f"{dataset_name}.json"
        if not jpath.is_file():
            continue
        payload = json.loads(jpath.read_text())
        meta = payload.get("meta_data", {}) if isinstance(payload, dict) else {}
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            continue
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            task = rec.get("task_instance", {})
            if not isinstance(task, dict):
                continue
            image_values = task.get("combined_1_images")
            image_root = ddir / "combined_1_images"
            if not image_values:
                continue
            if isinstance(image_values, (list, tuple)):
                raw_paths = [str(v) for v in image_values if str(v)]
            else:
                raw_paths = [str(image_values)] if image_values else []
            if not raw_paths:
                continue
            paths: list[str] = []
            for raw_path in raw_paths:
                p = Path(raw_path)
                if not p.is_absolute():
                    p = image_root / p
                if not p.is_file():
                    paths = []
                    break
                paths.append(str(p.resolve()))
            if not paths:
                continue
            paths = paths[:1]

            question = _build_milebench_question(rec, meta, dataset_name)
            prompt = build_qwen_prompt(question, conv_template)
            try:
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            except Exception:
                continue
            if int(input_ids.shape[1]) > max_raw_prompt_tokens:
                continue
            choice_list = task.get("choice_list") if isinstance(task, dict) else None
            cl = tuple(str(c) for c in choice_list) if isinstance(choice_list, (list, tuple)) else ()
            candidates.append(
                PooledSample(
                    dataset=dataset_name,
                    sample_id=_sanitize_id(rec.get("sample_id"), idx),
                    question=question,
                    image_paths=tuple(paths),
                    answer=_resolve_answer(rec),
                    choice_list=cl,
                )
            )
    if len(candidates) < sample_size:
        raise RuntimeError(f"Only found {len(candidates)} candidates, need {sample_size}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:sample_size]


# ----------------------------------------------------------------------
# Per-sample inference (with optional press) — measure latency + capture text
# ----------------------------------------------------------------------


@torch.no_grad()
def run_one(
    *,
    sample: PooledSample,
    method: str,
    ratio: float,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    press: Optional[VisualUtilityStudentOneVisionPress],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with Image.open(sample.image_paths[0]) as img:
        img = img.convert("RGB")
        image_size = img.size
        image_tensor = process_images([img], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    prompt = build_qwen_prompt(sample.question, args.conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    raw_ids = input_ids[0].detach().cpu().tolist()
    placeholders = [i for i, t in enumerate(raw_ids) if int(t) == IMAGE_TOKEN_INDEX]
    if len(placeholders) != 1:
        raise ValueError(f"Expected 1 image placeholder, found {len(placeholders)}")

    if press is not None:
        # The press needs to know image positions and question positions in the
        # POST-EXPANSION sequence. Since we don't have prepare_inputs_labels
        # output beforehand, we run prepare_inputs_labels to compute these.
        with torch.no_grad():
            (
                _new_input_ids,
                _,
                _,
                _,
                new_input_embeds,
                _,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                images=image_tensor,
                modalities=["image"],
                image_sizes=[image_size],
            )
        L_p = int(new_input_embeds.shape[1])
        image_start = int(placeholders[0])
        image_feature_len = L_p - (input_ids.shape[1] - 1)
        image_positions = torch.arange(image_start, image_start + image_feature_len, dtype=torch.long, device=device)
        last_img = int(image_positions.max().item())
        if last_img + 1 < L_p:
            question_positions = torch.arange(last_img + 1, L_p, dtype=torch.long, device=device)
        else:
            question_positions = torch.empty(0, dtype=torch.long, device=device)
        press.set_image_positions(image_positions)
        press.set_question_positions(question_positions)
        press.reset_student_timing()

    empty_cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if press is not None:
        ctx = press(model)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    with ctx:
        out = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            modalities=["image"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    end_to_end_ms = (time.perf_counter() - t0) * 1000.0
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / GIB

    sequences = out.sequences
    seq_len = int(sequences.shape[1])
    input_len = int(input_ids.shape[1])
    n_generated = seq_len - input_len if seq_len > input_len else seq_len
    n_generated = max(0, n_generated)
    decoded = tokenizer.decode(sequences[0, -n_generated:].tolist(), skip_special_tokens=True).strip() if n_generated > 0 else ""

    student_ms = float(getattr(press, "student_score_total_ms", 0.0)) if press is not None else None
    if press is not None:
        press.clear_sample_context()

    return {
        "method": method,
        "ratio": ratio,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "sample_key": sample.sample_key,
        "answer": sample.answer,
        "choice_list": list(sample.choice_list),
        "prediction": decoded,
        "n_generated": n_generated,
        "end_to_end_ms": end_to_end_ms,
        "student_forward_ms": student_ms,
        "peak_gpu_memory_gib": peak_alloc_gib,
        "peak_gpu_reserved_gib": peak_reserved_gib,
    }


# ----------------------------------------------------------------------
# Accuracy metrics
# ----------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_LETTER_PAT = re.compile(r"^\s*([A-Z])\s*[\.\):]?\s*$", re.IGNORECASE)


def map_letter_prediction(pred: str, choice_list: list[str]) -> str:
    """If pred is a single letter, map to choice_list[idx]. Else return pred."""
    if not choice_list:
        return pred
    m = _LETTER_PAT.match(pred.strip())
    if not m:
        return pred
    idx = ord(m.group(1).upper()) - ord("A")
    if 0 <= idx < len(choice_list):
        return choice_list[idx]
    return pred


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_text(pred) == normalize_text(gt))


def contains_match(pred: str, gt: str) -> int:
    return int(normalize_text(gt) in normalize_text(pred))


def rouge_l(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()
    if not p or not g:
        return 0.0
    # LCS length
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if p[i] == g[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    out: list[dict[str, Any]] = []
    full = by_method.get("full_cache_100", [])
    full_by_key = {r["sample_key"]: r for r in full}
    for method, mrows in by_method.items():
        item: dict[str, Any] = {
            "method": method,
            "ratio": mrows[0]["ratio"],
            "n_samples": len(mrows),
        }
        for f in [
            "end_to_end_ms", "student_forward_ms", "peak_gpu_memory_gib",
            "peak_gpu_reserved_gib", "n_generated",
        ]:
            vals = [r[f] for r in mrows if r.get(f) is not None]
            item[f"{f}_mean"] = float(mean(vals)) if vals else None
            item[f"{f}_std"] = float(stdev(vals)) if len(vals) > 1 else 0.0
        # accuracy metrics (with letter-to-choice-text mapping)
        em_list, contains_list, rouge_list, letter_em_list = [], [], [], []
        for r in mrows:
            if not r.get("answer"):
                continue
            choices = r.get("choice_list") or []
            pred_text = map_letter_prediction(r["prediction"], choices) if choices else r["prediction"]
            em_list.append(exact_match(pred_text, r["answer"]))
            contains_list.append(contains_match(pred_text, r["answer"]))
            rouge_list.append(rouge_l(pred_text, r["answer"]))
            # Pure letter-EM: was the chosen letter the correct option?
            if choices:
                m = _LETTER_PAT.match(r["prediction"].strip())
                gt_idx = None
                for i, c in enumerate(choices):
                    if normalize_text(c) == normalize_text(r["answer"]):
                        gt_idx = i
                        break
                if m and gt_idx is not None:
                    pred_idx = ord(m.group(1).upper()) - ord("A")
                    letter_em_list.append(int(pred_idx == gt_idx))
        item["n_with_answer"] = len(em_list)
        item["exact_match"] = float(mean(em_list)) if em_list else None
        item["contains"] = float(mean(contains_list)) if contains_list else None
        item["rouge_l"] = float(mean(rouge_list)) if rouge_list else None
        item["letter_accuracy"] = float(mean(letter_em_list)) if letter_em_list else None
        item["n_letter_eval"] = len(letter_em_list)

        if method != "full_cache_100" and full_by_key:
            paired = [(r, full_by_key[r["sample_key"]]) for r in mrows if r["sample_key"] in full_by_key]
            if paired:
                item["paired_n_vs_full"] = len(paired)
                lat = [r["end_to_end_ms"] / b["end_to_end_ms"] for r, b in paired if b["end_to_end_ms"] > 0]
                mem = [r["peak_gpu_memory_gib"] / b["peak_gpu_memory_gib"] for r, b in paired if b["peak_gpu_memory_gib"] > 0]
                item["end_to_end_pct_of_full"] = float(100.0 * mean(lat)) if lat else None
                item["peak_gpu_memory_pct_of_full"] = float(100.0 * mean(mem)) if mem else None
        out.append(item)
    return sorted(out, key=lambda x: float(x["ratio"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    extras = sorted({k for r in rows for k in r.keys()} - set(fieldnames))
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OneVision student inference (latency + accuracy) bench.")
    p.add_argument("--milebench-root", default="/workspace/zap/data/MileBench")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    p.add_argument("--model-name", default="llava_qwen")
    p.add_argument("--conv-template", default="qwen_1_5")
    p.add_argument("--student-path", default="/workspace/zap/artifacts/original_onevision_teacher/student_onevision_original_future_1800_lr1e4_15ep")
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.2, 0.05])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-raw-prompt-tokens", type=int, default=2000)
    p.add_argument("--warmup-samples", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    patch_siglip_loader("/workspace/zap/ckpts/siglip-so400m-patch14-384")

    print(f"[load] model={args.model_path} attn={args.attn_implementation}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        multimodal=True,
    )
    model.eval()
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"hidden={model.config.hidden_size} n_kv={model.config.num_key_value_heads}",
        flush=True,
    )

    samples = sample_manifest(
        milebench_root=Path(args.milebench_root).resolve(),
        datasets=args.datasets,
        tokenizer=tokenizer,
        sample_size=args.sample_size,
        seed=args.seed,
        max_raw_prompt_tokens=args.max_raw_prompt_tokens,
        conv_template=args.conv_template,
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    (out_dir / "sample_manifest.json").write_text(json.dumps({
        "seed": args.seed, "sample_size": len(samples),
        "samples": [{"dataset": s.dataset, "sample_id": s.sample_id, "question": s.question,
                      "image_paths": list(s.image_paths), "answer": s.answer} for s in samples]
    }, ensure_ascii=False, indent=2))
    print(f"[data] {len(samples)} samples", flush=True)

    ratios = sorted(set(float(r) for r in args.ratios), reverse=True)
    method_specs: list[tuple[str, float]] = []
    for r in ratios:
        if math.isclose(r, 1.0):
            method_specs.append(("full_cache_100", 1.0))
        else:
            method_specs.append((f"student_keep_{int(round(r * 100)):03d}", r))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for method, r in method_specs:
        press: Optional[VisualUtilityStudentOneVisionPress] = None
        if r < 1.0:
            press = VisualUtilityStudentOneVisionPress(
                total_keep_ratio=r,
                head_reduce="amax",
                student_model_name=args.student_path,
            )
            press.post_init_from_model(model)
            if press._student is not None:
                press._student = press._student.to(device=device, dtype=torch.float16).eval()
            empty_cuda()

        if args.warmup_samples > 0:
            for s in samples[: args.warmup_samples]:
                try:
                    run_one(sample=s, method="warmup", ratio=r, model=model, tokenizer=tokenizer,
                            image_processor=image_processor, press=press, device=device, args=args)
                except Exception:
                    pass
                empty_cuda()

        for s in tqdm(samples, desc=f"Run {method}"):
            try:
                row = run_one(
                    sample=s, method=method, ratio=r, model=model, tokenizer=tokenizer,
                    image_processor=image_processor, press=press, device=device, args=args,
                )
                rows.append(row)
                write_csv(out_dir / "per_sample.csv", rows)
            except torch.cuda.OutOfMemoryError as exc:
                failures.append({"sample_key": s.sample_key, "method": method, "error": f"CUDA OOM: {exc}"})
                empty_cuda()
                if not args.continue_on_error:
                    raise
            except Exception as exc:
                failures.append({"sample_key": s.sample_key, "method": method, "error": repr(exc)})
                empty_cuda()
                if not args.continue_on_error:
                    raise
            finally:
                if press is not None:
                    press.clear_sample_context()

        if press is not None:
            if press._student is not None:
                press._student = press._student.to("cpu")
            del press
            empty_cuda()

    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
