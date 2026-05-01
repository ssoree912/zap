"""Sub-exp C: Single-layer pruning sensitivity on ChartQA.

For each layer l in {0 .. 27}:
  - Keep only 10% of image tokens in layer l (student-scored top-k)
  - All other layers: full KV cache (keep_ratio = 1.0)
  - Evaluate ChartQA accuracy on N_SAMPLES samples

Also evaluates:
  - full_cache baseline  (keep_ratio = 1.0 for all layers)

Outputs:
  outputs/sensitivity_raw.csv       — per-sample, per-layer correctness
  outputs/sensitivity_agg.csv       — accuracy + delta vs full_cache per layer
  outputs/sensitivity_plot.png      — bar chart of accuracy delta per layer
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
from foresight.eval.vlmeval_onevision_student import LLaVA_OneVision_HF_Student

CHARTQA_TSV = Path("/workspace/zap/data/eval_LMU/ChartQA_TEST.tsv")
IMG_DIR = Path("/workspace/zap/data/eval_LMU/images/ChartQA_TEST")
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_RATIO_PRUNE = 0.1   # aggressive prune for the target layer
N_LAYERS = 28


def _relaxed_correct(pred: str, gt: str) -> bool:
    """ChartQA relaxed accuracy: numeric within 5%, else exact string match."""
    pred = pred.strip().lower()
    gt = str(gt).strip().lower()
    try:
        p_val = float(pred.replace(",", "").replace("%", ""))
        g_val = float(gt.replace(",", "").replace("%", ""))
        return abs(p_val - g_val) <= 0.05 * abs(g_val) + 1e-6
    except ValueError:
        return pred == gt


def load_chartqa(n_samples: int, seed: int) -> list[dict]:
    df = pd.read_csv(CHARTQA_TSV, sep="\t")
    df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    records = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        img_path = IMG_DIR / f"{idx}.png"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            # fallback: decode base64 from TSV
            img_bytes = base64.b64decode(str(row["image"]))
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        records.append({"index": idx, "question": str(row["question"]), "answer": str(row["answer"]), "image": img})
    return records


def run_eval(wrapper: LLaVA_OneVision_HF_Student, samples: list[dict]) -> list[bool]:
    results = []
    for s in samples:
        message = [
            {"type": "image", "value": s["image"]},
            {"type": "text", "value": s["question"]},
        ]
        pred = wrapper._run_single(s["image"], s["question"])
        results.append(_relaxed_correct(pred, s["answer"]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--start-layer", type=int, default=0, help="Resume from this layer")
    args = ap.parse_args()

    print(f"[load] Loading ChartQA ({args.n_samples} samples)", flush=True)
    samples = load_chartqa(args.n_samples, args.seed)
    print(f"[load] Loaded {len(samples)} samples", flush=True)

    model_path = "/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf"
    student_path = "/workspace/zap/ckpts/student_onevision_A_lr1e4_20ep"

    print(f"[load] Building model wrapper on {args.device}", flush=True)
    wrapper = LLaVA_OneVision_HF_Student(
        model_path=model_path,
        student_path=student_path,
        keep_ratio=1.0,  # default full cache; overridden per layer
        max_new_tokens=32,
    )
    # Move to target device
    wrapper.model = wrapper.model.to(args.device)
    wrapper.student = wrapper.student.to(args.device)

    raw_records = []
    agg_records = []

    # ── Full cache baseline ────────────────────────────────────────────────
    if args.start_layer == 0:
        print("\n[eval] Full cache baseline ...", flush=True)
        wrapper.keep_ratio = 1.0
        wrapper.per_layer_keep_ratios = None
        wrapper._reported_keep_budget = False
        correct = _eval_all(wrapper, samples, args.device)
        acc = sum(correct) / len(correct)
        print(f"  full_cache acc={acc:.4f}", flush=True)
        for i, c in enumerate(correct):
            raw_records.append({"layer": "full_cache", "sample_idx": i, "correct": int(c)})
        agg_records.append({"layer": "full_cache", "accuracy": acc, "delta": 0.0})
        _save(raw_records, agg_records)
        full_cache_acc = acc
    else:
        # Load previously saved results to get full_cache_acc
        prev = pd.read_csv(OUT_DIR / "sensitivity_agg.csv")
        full_cache_acc = float(prev.loc[prev["layer"] == "full_cache", "accuracy"].iloc[0])
        prev_raw = pd.read_csv(OUT_DIR / "sensitivity_raw.csv")
        raw_records = prev_raw.to_dict("records")
        agg_records = prev.to_dict("records")

    # ── Per-layer pruning ──────────────────────────────────────────────────
    for l in range(args.start_layer, N_LAYERS):
        print(f"\n[eval] Layer {l}/{N_LAYERS-1}: keep={KEEP_RATIO_PRUNE} only this layer ...", flush=True)
        wrapper.keep_ratio = 1.0
        wrapper.per_layer_keep_ratios = {l: KEEP_RATIO_PRUNE}
        wrapper._reported_keep_budget = False
        correct = _eval_all(wrapper, samples, args.device)
        acc = sum(correct) / len(correct)
        delta = acc - full_cache_acc
        print(f"  layer={l} acc={acc:.4f} delta={delta:+.4f}", flush=True)
        for i, c in enumerate(correct):
            raw_records.append({"layer": l, "sample_idx": i, "correct": int(c)})
        agg_records.append({"layer": l, "accuracy": acc, "delta": delta})
        _save(raw_records, agg_records)

    # Final plot
    agg_df = pd.read_csv(OUT_DIR / "sensitivity_agg.csv")
    _plot(agg_df)
    print(f"\nDone. Results in {OUT_DIR}", flush=True)


def _eval_all(wrapper, samples: list[dict], device: str) -> list[bool]:
    correct = []
    for s in tqdm(samples, leave=False):
        pred = _infer(wrapper, s["image"], s["question"], device)
        correct.append(_relaxed_correct(pred, s["answer"]))
    return correct


@torch.no_grad()
def _infer(wrapper: LLaVA_OneVision_HF_Student, image: Image.Image, question: str, device: str) -> str:
    from foresight.onevision_extractor import infer_onevision_image_positions_no_forward
    from foresight.eval.vlmeval_onevision_student import _greedy_decode_with_kv, _resolve_eos_token_id

    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = wrapper.processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = wrapper.processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    prefill = wrapper.model(**inputs, use_cache=True, output_hidden_states=True, output_attentions=False, return_dict=True)
    H_all = prefill.hidden_states
    past_kv = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    eos_token_id = _resolve_eos_token_id(wrapper.processor, wrapper.model.config)

    try:
        image_positions, prompt_len = infer_onevision_image_positions_no_forward(
            prompt_inputs={"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
            model_config=wrapper.model.config,
            num_images=1,
        )
    except ValueError:
        answer_ids = _greedy_decode_with_kv(wrapper.model, past_kv, next_token,
                                             prompt_len=int(inputs["input_ids"].shape[1]),
                                             eos_token_id=eos_token_id, max_new_tokens=wrapper.max_new_tokens)
        return wrapper.processor.decode(answer_ids.tolist(), skip_special_tokens=True)

    n_img = int(image_positions.numel())
    image_idx_dev = image_positions.to(device)
    last_img = int(image_positions.max().item())
    q_idx_dev = torch.arange(last_img + 1, prompt_len, dtype=torch.long, device=device) if last_img + 1 < prompt_len else torch.empty(0, dtype=torch.long, device=device)

    keep_masks: dict[int, torch.Tensor] = {}
    for li in wrapper.student.layer_indices:
        if wrapper.per_layer_keep_ratios is not None and li in wrapper.per_layer_keep_ratios:
            li_ratio = wrapper.per_layer_keep_ratios[li]
        else:
            li_ratio = wrapper.keep_ratio
        li_n_keep = max(1, int(round(n_img * li_ratio)))
        if li_n_keep >= n_img:
            continue
        H_l = H_all[li + 1]
        student_layer = wrapper.student.layers[str(li)]
        scores = student_layer(H_l, image_idx_dev, q_idx_dev).squeeze(0)
        top = torch.topk(scores, k=li_n_keep, largest=True).indices
        mask = torch.ones(prompt_len, dtype=torch.bool)
        image_keep = torch.zeros(n_img, dtype=torch.bool)
        image_keep[top.cpu()] = True
        mask[image_positions.cpu()] = image_keep
        keep_masks[li] = mask

    if keep_masks:
        from foresight.eval.vlmeval_onevision_student import _trim_kv_cache_per_layer
        past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks)

    answer_ids = _greedy_decode_with_kv(wrapper.model, past_kv, next_token,
                                         prompt_len=int(prompt_len), eos_token_id=eos_token_id,
                                         max_new_tokens=wrapper.max_new_tokens)
    text = wrapper.processor.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    return text


def _save(raw_records: list, agg_records: list) -> None:
    pd.DataFrame(raw_records).to_csv(OUT_DIR / "sensitivity_raw.csv", index=False)
    pd.DataFrame(agg_records).to_csv(OUT_DIR / "sensitivity_agg.csv", index=False)


def _plot(agg_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_df = agg_df[agg_df["layer"] != "full_cache"].copy()
    layer_df["layer"] = layer_df["layer"].astype(int)
    layer_df = layer_df.sort_values("layer")

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["crimson" if d < -0.02 else "steelblue" for d in layer_df["delta"]]
    ax.bar(layer_df["layer"], layer_df["delta"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Accuracy delta vs full cache")
    ax.set_title(f"Single-layer pruning sensitivity (keep={KEEP_RATIO_PRUNE}) — ChartQA")
    ax.set_xticks(layer_df["layer"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "sensitivity_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    main()
