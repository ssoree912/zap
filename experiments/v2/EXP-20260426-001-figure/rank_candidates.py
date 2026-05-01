#!/usr/bin/env python3
"""Rank Figure 1 hero candidates by composite mismatch + student validity score.

Metrics per sample:
  prefill_future_spearman   : Spearman(s_prefill, s_future)         ↓ want low
  overlap_pf_20             : TopK overlap(prefill, future) @20%     ↓ want low
  overlap_sf_20             : TopK overlap(student, future) @20%     ↑ want high
  mismatch_score            : (1 - overlap_pf_20) + (1 - spearman)  ↑ want high
  composite_score           : mismatch_score + overlap_sf_20        ↑ sort by this

Usage:
  CUDA_VISIBLE_DEVICES=2 python rank_candidates.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, "/workspace/zap")

DUMP_ROOT  = "/workspace/zap/artifacts/EXP-20260426-001-figure/attn_dump"
OUT_ROOT   = "/workspace/zap/artifacts/EXP-20260426-001-figure"
CKPT       = "/workspace/zap/ckpts/student_v2_A_gqa_lr1e4"
MODEL_PATH = "/workspace/zap/ckpts/llava-1.5-7b-hf"
GRID_H, GRID_W = 24, 24


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return len(top_a & top_b) / max(k, 1)


def avg_layers(t: torch.Tensor) -> np.ndarray:
    return t.float().mean(dim=0).numpy()


def run_student_scores(pt_files: list[Path], device: torch.device) -> dict[str, np.ndarray]:
    """Returns {pt_path_str: [N_I] student score averaged over layers}."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from kvpress.presses.visual_utility_student import VisualUtilityStudent
    from foresight.llava_extractor import infer_llava_image_positions_no_forward
    from PIL import Image

    print(f"[load] student ckpt {CKPT}")
    student = VisualUtilityStudent.from_pretrained(CKPT).to(device).eval()

    print(f"[load] LLaVA model")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=str(device),
        attn_implementation="eager",
    )
    model.eval()

    results: dict[str, np.ndarray] = {}
    for pt_path in tqdm(pt_files, desc="student forward"):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            image = Image.open(data["image_path"]).convert("RGB")
            prompt = f"USER: <image>\n{data['question']}\nASSISTANT:"
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
                prompt_inputs=inputs, model_config=model.config, num_images=1,
            )
            img_idx = image_positions.to(device)
            last_img = int(image_positions.max().item())
            q_idx = (
                torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long, device=device)
                if last_img + 1 < prompt_len_mm
                else torch.empty(0, dtype=torch.long, device=device)
            )

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)

            hidden_states = out.hidden_states
            layer_scores = []
            for li in student.layer_indices:
                H_l = hidden_states[li + 1].to(device=device, dtype=torch.float16)
                layer_mod = student.layers[str(li)].to(device=device, dtype=torch.float16).eval()
                score = layer_mod(H_l, img_idx, q_idx, GRID_H, GRID_W).squeeze(0).float().cpu()
                layer_scores.append(score)

            stacked = torch.stack(layer_scores)  # [n_layers, N_I]
            results[str(pt_path)] = stacked.mean(0).detach().numpy()

            del out, hidden_states
        except Exception as e:
            print(f"[WARN] {pt_path.name}: {e}")
            results[str(pt_path)] = None

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump-root", default=DUMP_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--datasets", nargs="+",
                    default=["DocVQA", "OCR-VQA", "SlideVQA", "mmvet"])
    ap.add_argument("--keep-ratio", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device(args.device)
    dump_root = Path(args.dump_root)

    all_pt = []
    for ds in args.datasets:
        all_pt.extend(sorted((dump_root / ds).glob("*.pt")))
    print(f"[info] {len(all_pt)} samples across {args.datasets}")

    student_scores = run_student_scores(all_pt, device)

    rows = []
    for pt_path in all_pt:
        ds = pt_path.parent.name
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        prefill = data["prefill"].float()  # [L, N_I]
        future  = data["future"].float()   # [L, N_I]
        pref_avg = avg_layers(prefill)
        fut_avg  = avg_layers(future)
        N_I = len(pref_avg)
        k = max(1, int(N_I * args.keep_ratio))

        rho, _ = spearmanr(pref_avg, fut_avg)
        ov_pf = topk_overlap(pref_avg, fut_avg, k)

        stud_avg = student_scores.get(str(pt_path))
        if stud_avg is not None:
            ov_sf = topk_overlap(stud_avg, fut_avg, k)
            rho_sf, _ = spearmanr(stud_avg, fut_avg)
        else:
            ov_sf = float("nan")
            rho_sf = float("nan")

        mismatch = (1 - ov_pf) + (1 - float(rho))
        composite = mismatch + ov_sf  # high mismatch + high student validity

        rows.append({
            "dataset":       ds,
            "sample_id":     data.get("sample_id", pt_path.stem),
            "question":      data.get("question", "")[:100],
            "answer":        data.get("generated_answer", data.get("answer", ""))[:60],
            "N_I":           N_I,
            "spearman_pf":   round(float(rho), 4),
            "overlap_pf_20": round(ov_pf, 4),
            "spearman_sf":   round(float(rho_sf), 4),
            "overlap_sf_20": round(ov_sf, 4),
            "mismatch_score":  round(mismatch, 4),
            "composite_score": round(composite, 4),
            "pt_path":       str(pt_path),
        })

    # sort by composite_score descending
    rows.sort(key=lambda x: x["composite_score"], reverse=True)

    out_path = Path(args.out_root) / "figure1_candidates_ranked.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Rank':>4}  {'DS':10}  {'ID':8}  {'rho_pf':>7}  {'ov_pf':>6}  {'ov_sf':>6}  {'composite':>9}  Q")
    print("-" * 100)
    for i, r in enumerate(rows[:20]):
        print(f"{i+1:4d}  {r['dataset']:10}  {r['sample_id']:8}  "
              f"{r['spearman_pf']:7.3f}  {r['overlap_pf_20']:6.3f}  "
              f"{r['overlap_sf_20']:6.3f}  {r['composite_score']:9.4f}  "
              f"{r['question'][:55]}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
