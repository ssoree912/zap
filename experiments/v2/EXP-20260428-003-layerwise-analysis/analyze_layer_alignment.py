"""Sub-exp B: Per-layer student-teacher alignment on ChartQA.

For each sample:
  1. Run LLaVA-OneVision prefill with output_hidden_states=True
  2. Pass hidden states through the student probe (one forward per layer)
  3. Compare student scores vs teacher_norm using:
     - Spearman rank correlation
     - Overlap@K  (K = 20% of N_I image tokens)

Outputs:
  outputs/layer_alignment.csv       — per-sample, per-layer metrics
  outputs/layer_alignment_agg.csv   — mean/std aggregated per layer
  outputs/layer_alignment_plot.png  — Spearman + Overlap@K per layer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parents[2]))
from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision
from foresight.onevision_extractor import configure_onevision_processor

SHARD_DIR = Path(__file__).parents[2] / "data" / "teacher_v2_onevision" / "chartqa"
LLAVA_PATH = Path("/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf")
STUDENT_PATH = Path("/workspace/zap/ckpts/student_onevision_A_lr1e4_20ep")
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP_K_RATIO = 0.2  # Overlap@(20% of N_I)


def _build_prompt(processor, question: str) -> str:
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def overlap_at_k(score_a: np.ndarray, score_b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(score_a)[-k:])
    top_b = set(np.argsort(score_b)[-k:])
    return len(top_a & top_b) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=50, help="Number of shards to evaluate")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"[load] LLaVA-OneVision from {LLAVA_PATH}", flush=True)
    lvlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for p in lvlm.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(str(LLAVA_PATH))
    processor = configure_onevision_processor(processor, lvlm.config)

    print(f"[load] Student probe from {STUDENT_PATH}", flush=True)
    student = VisualUtilityStudentOneVision.from_pretrained(str(STUDENT_PATH), map_location=str(device)).to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)

    shard_paths = sorted(SHARD_DIR.glob("*.pt"))[: args.n_samples]
    print(f"[data] Evaluating {len(shard_paths)} samples", flush=True)

    records = []

    for i, path in enumerate(shard_paths):
        rec = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = rec["sample_id"]

        # Build processed inputs
        question = rec.get("question_text") or rec.get("prompt_text") or ""
        img_path = rec["image_path"]
        if isinstance(img_path, (list, tuple)):
            images = [Image.open(p).convert("RGB") for p in img_path]
        else:
            images = Image.open(img_path).convert("RGB")
        prompt_text = _build_prompt(processor, question)
        inputs = processor(images=images, text=prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_idx = rec["image_token_indices"].to(device, dtype=torch.long)
        q_idx = rec["question_token_indices"].to(device, dtype=torch.long)
        teacher_norm = rec["teacher_norm"].float()  # [28, N_I] on CPU

        n_img = int(image_idx.numel())
        k_overlap = max(1, int(n_img * OVERLAP_K_RATIO))

        # Prefill with hidden states
        with torch.no_grad():
            out = lvlm(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        H_all = out.hidden_states  # tuple: (embed, layer0_out, ..., layer27_out)

        for li in student.layer_indices:
            H_l = H_all[li + 1].float()  # [1, N_prefill, D]

            with torch.no_grad():
                s_pred = student.forward_layer(li, H_l, image_idx, q_idx)  # [1, N_I]
            s_pred = s_pred[0].cpu().numpy()  # [N_I]

            t_score = teacher_norm[li].numpy()  # [N_I]

            rho, _ = spearmanr(s_pred, t_score)
            ovlp = overlap_at_k(s_pred, t_score, k_overlap)

            records.append({
                "sample_id": sample_id,
                "layer": li,
                "spearman": float(rho),
                "overlap_at_k": float(ovlp),
                "n_img": n_img,
                "k_overlap": k_overlap,
            })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(shard_paths)}] done", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "layer_alignment.csv", index=False)
    print(f"\nSaved per-sample alignment → {OUT_DIR / 'layer_alignment.csv'}")

    agg = (
        df.groupby("layer")[["spearman", "overlap_at_k"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["layer", "spearman_mean", "spearman_std", "overlap_mean", "overlap_std"]
    agg.to_csv(OUT_DIR / "layer_alignment_agg.csv", index=False)
    print(f"Saved aggregated alignment → {OUT_DIR / 'layer_alignment_agg.csv'}")
    print()
    print(agg.to_string(index=False, float_format="{:.4f}".format))

    _plot(agg)


def _plot(agg: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = agg["layer"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, metric, label, color in [
        (axes[0], "spearman", "Spearman ρ", "steelblue"),
        (axes[1], "overlap", f"Overlap@K  (K=20% of N_I)", "darkorange"),
    ]:
        mean = agg[f"{metric}_mean"].values
        std = agg[f"{metric}_std"].values
        ax.plot(layers, mean, marker="o", markersize=4, linewidth=1.5, color=color)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.25, color=color)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(label)
        ax.set_title(f"Per-layer student-teacher {label} — ChartQA")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "layer_alignment_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
