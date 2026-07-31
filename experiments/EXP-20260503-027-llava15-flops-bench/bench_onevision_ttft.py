#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TTFT (time-to-first-token) bench: full cache vs trained student press.

The student prunes image KV *after* a full prefill, so TTFT cannot improve;
this bench quantifies the exact overhead the scorer+prune adds to TTFT:

    TTFT_full    = prefill + 1st-token forward
    TTFT_student = prefill (with per-layer scoring/prune hooks) + 1st-token

Prompt length is swept with synthetic images (pixel content is irrelevant for
latency): single-image anyres at several resolutions, plus uniform multi-image
prompts for k-scale L_p. Each config is measured `--repeats` times after
`--warmup` untimed runs; medians are reported.

Usage (vflowopt env, GPU 0):
  python bench_onevision_ttft.py --output-dir outputs/ttft_gpu0 \\
      --ratios 0.2 0.05 --repeats 3 --warmup 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Optional

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _cand in ("/workspace/VFlowOpt/src/LLaVA-OneVision", "/workspace/nips/VFlowOpt/src/LLaVA-OneVision"):
    if Path(_cand).is_dir():
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from kvpress.presses.image_token_press import VisualUtilityStudentOneVisionPress  # noqa: E402


def patch_siglip_loader_to_local_init() -> None:
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):  # noqa: ANN001, ARG001
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


def build_qwen_prompt(question: str, n_images: int, conv_template: str) -> str:
    import copy

    conv = copy.deepcopy(conv_templates[conv_template])
    image_block = "\n".join(["<image>"] * max(1, n_images))
    conv.append_message(conv.roles[0], f"{image_block}\n{question.strip()}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def make_synthetic_image(size: int, seed: int) -> Image.Image:
    g = torch.Generator().manual_seed(seed)
    arr = torch.randint(0, 256, (size, size, 3), generator=g, dtype=torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def infer_image_blocks(
    raw_ids: list[int], expanded_len: int, n_images: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image / question positions in the post-expansion sequence.

    Assumes all placeholders expand to the same length (identical synthetic
    images), which the integer division asserts.
    """
    placeholders = [i for i, t in enumerate(raw_ids) if int(t) == IMAGE_TOKEN_INDEX]
    if len(placeholders) != n_images:
        raise ValueError(f"Expected {n_images} placeholders, found {len(placeholders)}")
    total_expansion = expanded_len - len(raw_ids) + n_images
    if total_expansion % n_images != 0:
        raise ValueError(
            f"Non-uniform image expansion: expanded={expanded_len} raw={len(raw_ids)} n={n_images}"
        )
    per_image = total_expansion // n_images
    blocks = []
    offset = 0
    for p in placeholders:
        start = p + offset
        blocks.append(torch.arange(start, start + per_image, dtype=torch.long))
        offset += per_image - 1
    image_positions = torch.cat(blocks)
    last_img = int(image_positions.max().item())
    if last_img + 1 < expanded_len:
        question_positions = torch.arange(last_img + 1, expanded_len, dtype=torch.long)
    else:
        question_positions = torch.empty(0, dtype=torch.long)
    return image_positions, question_positions


@torch.no_grad()
def measure_ttft(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    image_tensor: Any,
    image_sizes: list[tuple[int, int]],
    n_images: int,
    press: Optional[VisualUtilityStudentOneVisionPress],
) -> tuple[float, float]:
    """One timed prefill + first-token forward. Returns (ttft_ms, student_ms).

    Manual path instead of generate(max_new_tokens=1): lm_head is applied to
    the LAST position only. generate() in transformers 4.46 materializes
    full-sequence vocab logits (L_p x 152k fp16 ~ 3.4 GiB at L_p=12k), which
    OOMs a 24 GiB card and is pure waste for TTFT in both arms anyway.
    """
    from contextlib import nullcontext

    if press is not None:
        press.reset_student_timing()
    empty_cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    (
        _ids,
        _pos,
        _mask,
        _pkv,
        inputs_embeds,
        _labels,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None,
        images=image_tensor, modalities=["image"] * n_images, image_sizes=image_sizes,
    )
    ctx = press(model) if press is not None else nullcontext()
    with ctx:
        out = model.model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    logits.argmax(dim=-1)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    student_ms = float(getattr(press, "student_score_total_ms", 0.0)) if press is not None else 0.0
    del out, logits, inputs_embeds
    return ttft_ms, student_ms


def main() -> None:
    p = argparse.ArgumentParser(description="OneVision TTFT bench: full cache vs student press.")
    p.add_argument("--model-path", default=str(REPO_ROOT / "ckpts/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--model-name", default="llava_qwen")
    p.add_argument(
        "--student-path",
        default=str(
            REPO_ROOT
            / "artifacts/original_onevision_teacher/student_onevision_zap_future_answer_agnostic_1800_lr1e4_15ep_2gpu"
        ),
    )
    p.add_argument("--conv-template", default="qwen_1_5")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.05])
    p.add_argument("--single-image-sizes", type=int, nargs="+", default=[336, 672, 1008])
    p.add_argument("--multi-image-counts", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--multi-image-size", type=int, default=384)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "outputs/ttft_gpu0"))
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_siglip_loader_to_local_init()
    print(f"[load] model={args.model_path} attn={args.attn_implementation}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        multimodal=True,
    )
    model.eval()
    print(f"[load-ok] class={model.__class__.__name__}", flush=True)

    question = "Describe the visual content of the image(s) in detail."

    configs: list[dict[str, Any]] = []
    for size in args.single_image_sizes:
        configs.append({"tag": f"1img_{size}px", "n_images": 1, "size": size})
    for n in args.multi_image_counts:
        configs.append({"tag": f"{n}img_{args.multi_image_size}px", "n_images": n, "size": args.multi_image_size})

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        n_images = cfg["n_images"]
        pil_images = [make_synthetic_image(cfg["size"], seed=100 + i) for i in range(n_images)]
        image_sizes = [img.size for img in pil_images]
        image_tensor = process_images(pil_images, image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(device=device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(device=device, dtype=torch.float16)

        prompt = build_qwen_prompt(question, n_images, args.conv_template)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )
        raw_ids = input_ids[0].detach().cpu().tolist()

        (
            _ids,
            _pos,
            _mask,
            _labels,
            new_input_embeds,
            _lab,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None,
            images=image_tensor, modalities=["image"] * n_images, image_sizes=image_sizes,
        )
        L_p = int(new_input_embeds.shape[1])
        del new_input_embeds
        image_positions, question_positions = infer_image_blocks(raw_ids, L_p, n_images)
        n_img = int(image_positions.numel())
        print(f"[cfg] {cfg['tag']} L_p={L_p} n_img={n_img}", flush=True)

        arms: list[tuple[str, float, Optional[VisualUtilityStudentOneVisionPress]]] = [
            ("full", 1.0, None)
        ]
        for r in args.ratios:
            press = VisualUtilityStudentOneVisionPress(
                total_keep_ratio=r,
                head_reduce="amax",
                student_model_name=args.student_path,
            )
            press.post_init_from_model(model)
            if press._student is not None:
                press._student = press._student.to(device=device, dtype=torch.float16).eval()
            arms.append((f"student_keep{r:g}", r, press))

        for method, ratio, press in arms:
            if press is not None:
                press.set_image_positions(image_positions.to(device))
                press.set_question_positions(question_positions.to(device))
            ttfts, students = [], []
            try:
                for _ in range(args.warmup):
                    measure_ttft(
                        model=model, tokenizer=tokenizer, input_ids=input_ids,
                        image_tensor=image_tensor, image_sizes=image_sizes,
                        n_images=n_images, press=press,
                    )
                for _ in range(args.repeats):
                    t, s = measure_ttft(
                        model=model, tokenizer=tokenizer, input_ids=input_ids,
                        image_tensor=image_tensor, image_sizes=image_sizes,
                        n_images=n_images, press=press,
                    )
                    ttfts.append(t)
                    students.append(s)
            except torch.OutOfMemoryError:
                print(f"[oom] {cfg['tag']} {method} — skipping arm", flush=True)
                empty_cuda()
                if press is not None:
                    press.clear_sample_context()
                continue
            if press is not None:
                press.clear_sample_context()
            row = {
                "config": cfg["tag"],
                "n_images": n_images,
                "image_px": cfg["size"],
                "L_p": L_p,
                "n_img_tokens": n_img,
                "method": method,
                "ratio": ratio,
                "ttft_ms_median": round(median(ttfts), 2),
                "ttft_ms_all": [round(t, 2) for t in ttfts],
                "student_ms_median": round(median(students), 2),
            }
            rows.append(row)
            print(
                f"[ttft] {cfg['tag']:14s} {method:16s} ttft={row['ttft_ms_median']:8.1f}ms "
                f"student={row['student_ms_median']:6.1f}ms",
                flush=True,
            )

    # Attach overhead vs the full-cache arm of the same config.
    full_by_cfg = {r["config"]: r["ttft_ms_median"] for r in rows if r["method"] == "full"}
    for r in rows:
        base = full_by_cfg.get(r["config"])
        if base:
            r["overhead_ms"] = round(r["ttft_ms_median"] - base, 2)
            r["overhead_pct"] = round(100.0 * (r["ttft_ms_median"] - base) / base, 2)

    (out_dir / "ttft_results.json").write_text(json.dumps(rows, indent=2))
    fields = [
        "config", "n_images", "image_px", "L_p", "n_img_tokens", "method", "ratio",
        "ttft_ms_median", "student_ms_median", "overhead_ms", "overhead_pct",
    ]
    with (out_dir / "ttft_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote {out_dir}/ttft_results.json and .csv", flush=True)


if __name__ == "__main__":
    main()
