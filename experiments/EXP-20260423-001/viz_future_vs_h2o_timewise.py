#!/usr/bin/env python3
"""Per-layer time-resolved score heatmaps: Future-probe vs H2O (accumulated attention).

For a single WikiVQA sample we run the full prefill + a short decode loop and
record, at six target layers, (a) the head-mean attention matrix A[t, i] over
all query / kv positions and (b) the pre-attention hidden state used by the
Future probe. From this we build two heatmaps per layer:

  - H2O score:      cumulative_sum_{s<=t}  A[s, i]      (what H2O actually ranks on)
  - Future score:   MLP(h_i)                            (broadcast along t>=i)

Axes:
  x = past token index i
  y = current time t (= prefill position; then decode steps appended)

The resulting 6x2 grid of heatmaps is written next to this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoProcessor, LlavaForConditionalGeneration

from kvpress.presses.kvzap_press import KVzapModel
from kvzap.image_teacher_utils import build_prompt, load_vlm_samples
from kvzap.llava_extractor import (
    _get_model_device,
    _get_model_float_dtype,
    _move_batch_to_device,
    configure_llava_processor,
    infer_llava_image_positions_no_forward,
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", default="/workspace/zap/data/MileBench/WikiVQA/WikiVQA.json")
    parser.add_argument(
        "--image_root",
        default="/workspace/zap/data/MileBench/WikiVQA/combined_1_images",
    )
    parser.add_argument("--image_column", default="combined_1_images")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index into the dataset (0-based).")
    parser.add_argument("--model_name", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument(
        "--future_probe_name",
        default="/workspace/zap/ckpts/future_probe_v5_all_token_10ep",
        help="All-token future probe (operates on every KV token, not just image).",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 14, 17, 26, 29],
                        help="Model layer indices to visualize (default: 2 bottom, 2 middle, 2 top).")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048,
                        help="Cap the *text* side of the prompt to avoid OOM during eager prefill.")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "viz_out"))
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Capture hook
# ──────────────────────────────────────────────────────────────────────────────


class LayerCapture:
    """Per-layer capture of attention weights (head-mean) and pre-attn hidden states.

    Registered on ``self_attn`` modules of the target layers. The hook is
    invoked once per forward pass (prefill + each decode step). Captured
    tensors are moved to CPU/fp32 immediately to keep GPU memory flat.
    """

    def __init__(self, target_layers: List[int]):
        self.target_layers = set(int(x) for x in target_layers)
        # For each layer index, we store a list of (q_slice, kv_len, attn_row,
        # hidden_row) tuples in call order.
        self.attn_steps: Dict[int, List[torch.Tensor]] = {li: [] for li in self.target_layers}
        # Pre-attention hidden states per step (cpu/fp32, shape [q_len, D]).
        self.hidden_steps: Dict[int, List[torch.Tensor]] = {li: [] for li in self.target_layers}
        # Keys are kv_len sizes observed per step (same across layers).
        self.kv_lens: List[int] = []
        self.q_lens: List[int] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # --- hook ----------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, args, kwargs, output):
            # eager attention returns (attn_output, attn_weights, past_kv?)
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
            if attn_weights is None:
                raise RuntimeError(
                    "attn_weights was None — make sure model is loaded with "
                    "attn_implementation='eager'."
                )
            # attn_weights: [1, H, q_len, kv_len]
            aw = attn_weights.detach()
            head_mean = aw.mean(dim=1)[0].to(torch.float32).cpu()  # [q_len, kv_len]
            self.attn_steps[layer_idx].append(head_mean)

            hidden = kwargs.get("hidden_states")
            if hidden is None and len(args) >= 1 and torch.is_tensor(args[0]):
                hidden = args[0]
            if hidden is None:
                raise RuntimeError("Could not find hidden_states in self_attn forward kwargs")
            hidden_cpu = hidden.detach()[0].to(torch.float32).cpu()  # [q_len, D]
            self.hidden_steps[layer_idx].append(hidden_cpu)

            # Record q/kv sizes on the first captured layer of each step.
            if layer_idx == min(self.target_layers):
                self.q_lens.append(int(head_mean.shape[0]))
                self.kv_lens.append(int(head_mean.shape[1]))

        return hook

    def register(self, model) -> None:
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for li in sorted(self.target_layers):
            layer = language_model.layers[li]
            h = layer.self_attn.register_forward_hook(self._make_hook(li), with_kwargs=True)
            self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# ──────────────────────────────────────────────────────────────────────────────
# Build full lower-triangular attention + per-position hidden from captures
# ──────────────────────────────────────────────────────────────────────────────


def assemble_per_layer(cap: LayerCapture) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return dicts {layer_idx: attn_LxL, layer_idx: hidden_LxD} in step order.

    Merges prefill (q_len == kv_len) with per-step decode rows (q_len == 1).
    Missing upper-triangular entries are zero.
    """
    if not cap.q_lens:
        raise RuntimeError("No captures recorded")

    total_q = sum(cap.q_lens)
    total_kv = cap.kv_lens[-1]
    assert total_q == total_kv, (
        f"Sum of q_lens ({total_q}) must equal final kv_len ({total_kv}) — "
        "each decoded token extends the kv cache by 1."
    )
    T = total_kv

    per_layer_attn: Dict[int, np.ndarray] = {}
    per_layer_hidden: Dict[int, np.ndarray] = {}

    for li in sorted(cap.target_layers):
        attn_full = np.zeros((T, T), dtype=np.float32)
        hidden_list: List[torch.Tensor] = []

        t_cursor = 0
        for step_idx, (q_len, kv_len) in enumerate(zip(cap.q_lens, cap.kv_lens)):
            aw = cap.attn_steps[li][step_idx].numpy()  # (q_len, kv_len)
            h = cap.hidden_steps[li][step_idx]  # (q_len, D)
            # rows [t_cursor : t_cursor + q_len], cols [0 : kv_len]
            attn_full[t_cursor : t_cursor + q_len, :kv_len] = aw
            hidden_list.append(h)
            t_cursor += q_len
        per_layer_attn[li] = attn_full
        per_layer_hidden[li] = torch.cat(hidden_list, dim=0).numpy()  # (T, D)

    return per_layer_attn, per_layer_hidden


# ──────────────────────────────────────────────────────────────────────────────
# Future probe scoring
# ──────────────────────────────────────────────────────────────────────────────


def future_score_per_layer(
    probe: KVzapModel,
    per_layer_hidden: Dict[int, np.ndarray],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """Apply each layer's Future MLP to its captured hidden states.

    Returns a dict {layer_idx: (T,) float32}.
    """
    out: Dict[int, np.ndarray] = {}
    for li, hidden in per_layer_hidden.items():
        mlp = probe.layers[li].to(device, dtype=torch.float32).eval()
        with torch.no_grad():
            x = torch.from_numpy(hidden).to(device=device, dtype=torch.float32)  # (T, D)
            s = mlp(x).squeeze(-1).cpu().numpy()  # (T,)
        out[li] = s.astype(np.float32)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap plotting
# ──────────────────────────────────────────────────────────────────────────────


def _tri_mask(T: int) -> np.ndarray:
    # True where t >= i (lower triangular including diagonal)
    ii, tt = np.meshgrid(np.arange(T), np.arange(T))
    return tt >= ii


def build_h2o_heatmap(attn: np.ndarray) -> np.ndarray:
    """Cumulative attention along the query (time) axis.

    attn[t, i] is head-mean attention from query t to kv position i. Below the
    diagonal (t >= i), the cumulative sum_{s=i..t} attn[s, i] reproduces the
    H2O heavy-hitter score at time t. Above the diagonal we leave NaN so
    imshow shows it masked.
    """
    T = attn.shape[0]
    cum = np.cumsum(attn, axis=0)
    mask = _tri_mask(T)
    out = np.where(mask, cum, np.nan).astype(np.float32)
    return out


def build_future_heatmap(future_scores: np.ndarray, T: int) -> np.ndarray:
    """Broadcast softmax(s_future)[i] along rows t >= i.

    The raw probe logits can be negative, so we softmax across past tokens
    (matching HybridH2OFuturePress) to put Future on the same positive
    probability scale as (row-normalized) H2O.
    """
    logits = future_scores.astype(np.float64)
    logits = logits - logits.max()
    prob = np.exp(logits)
    prob = prob / prob.sum()
    mask = _tri_mask(T)
    heat = np.broadcast_to(prob[None, :], (T, T)).copy()
    heat = np.where(mask, heat, np.nan)
    return heat.astype(np.float32)


def _global_range(*heatmap_dicts: Dict[int, np.ndarray]) -> Tuple[float, float]:
    chunks = []
    for d in heatmap_dicts:
        for h in d.values():
            chunks.append(h[np.isfinite(h)].ravel())
    if not chunks:
        return 1e-6, 1.0
    all_vals = np.concatenate(chunks)
    pos = all_vals[all_vals > 0]
    if pos.size == 0:
        return 1e-6, 1.0
    vmin = max(float(np.quantile(pos, 0.02)), 1e-8)
    vmax = float(np.quantile(pos, 0.999))
    return vmin, max(vmax, vmin * 10)


def plot_grid(
    layers: List[int],
    heat_h2o: Dict[int, np.ndarray],
    heat_future: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: Path,
    sample_meta: dict,
) -> None:
    n_rows = len(layers)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 3.0 * n_rows),
        squeeze=False,
    )

    # Row-normalize H2O (each row → probability distribution over past tokens)
    # so it lives on the same [0, 1] scale as softmax(Future). Then both
    # methods share one LogNorm range and colormap.
    heat_h2o_norm: Dict[int, np.ndarray] = {}
    for li, h in heat_h2o.items():
        row_sum = np.nansum(h, axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        heat_h2o_norm[li] = (h / row_sum).astype(np.float32)

    vmin, vmax = _global_range(heat_h2o_norm, heat_future)
    shared_norm = LogNorm(vmin=vmin, vmax=vmax)

    for row, li in enumerate(layers):
        h2o = heat_h2o_norm[li]
        fu = heat_future[li]

        ax_h2o = axes[row][0]
        ax_fu = axes[row][1]

        im0 = ax_h2o.imshow(
            h2o,
            cmap="magma",
            aspect="auto",
            origin="upper",
            norm=shared_norm,
            interpolation="nearest",
        )
        ax_h2o.axvline(prompt_len - 0.5, color="cyan", lw=0.6, alpha=0.7)
        ax_h2o.axhline(prompt_len - 0.5, color="cyan", lw=0.6, alpha=0.7)
        ax_h2o.set_title(f"Layer {li} · H2O row-normalized prob")
        ax_h2o.set_xlabel("past token index i")
        ax_h2o.set_ylabel("current time t")
        plt.colorbar(im0, ax=ax_h2o, fraction=0.035, pad=0.02)

        im1 = ax_fu.imshow(
            fu,
            cmap="magma",
            aspect="auto",
            origin="upper",
            norm=shared_norm,
            interpolation="nearest",
        )
        ax_fu.axvline(prompt_len - 0.5, color="cyan", lw=0.6, alpha=0.7)
        ax_fu.axhline(prompt_len - 0.5, color="cyan", lw=0.6, alpha=0.7)
        ax_fu.set_title(f"Layer {li} · Future softmax prob")
        ax_fu.set_xlabel("past token index i")
        ax_fu.set_ylabel("current time t")
        plt.colorbar(im1, ax=ax_fu, fraction=0.035, pad=0.02)

    fig.suptitle(
        f"WikiVQA sample_id={sample_meta.get('sample_id')} "
        f"(prompt_len={prompt_len}, decode={sample_meta.get('n_gen')}) "
        f"head-mean across 32 heads",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- sample ------------------------------------------------------------
    samples = load_vlm_samples(
        dataset_path=args.dataset_path,
        image_root=args.image_root,
        image_column=args.image_column,
        limit=args.sample_index + 1,
    )
    sample = samples[args.sample_index]
    print(f"[sample] id={sample['sample_id']} n_images={len(sample['image_paths'])}")
    print(f"[sample] answer={sample.get('answer')}")

    # ---- model / processor ------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": getattr(torch, args.torch_dtype),
        "device_map": None,
    }
    model = LlavaForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    configure_llava_processor(processor, model.config)
    model = model.to(torch.device(args.device)).eval()
    device = _get_model_device(model)
    dtype = _get_model_float_dtype(model)

    # ---- build prompt / inputs --------------------------------------------
    image_paths = sample["image_paths"]
    prompt_text = build_prompt(sample["question"], image_count=len(image_paths))

    # Optional text truncation to stay within GPU memory.
    tok = processor.tokenizer
    tokenized = tok(prompt_text, return_tensors=None, add_special_tokens=True)
    text_len = len(tokenized["input_ids"])
    if args.max_prompt_tokens and text_len > args.max_prompt_tokens:
        ids = tokenized["input_ids"][: args.max_prompt_tokens]
        prompt_text = tok.decode(ids, skip_special_tokens=False)
        print(f"[sample] truncated text prompt {text_len} -> {len(ids)} tokens")

    images = [Image.open(p).convert("RGB") for p in image_paths]
    images = images[0] if len(images) == 1 else images
    prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    prompt_inputs = _move_batch_to_device(prompt_inputs, device, dtype)
    prompt_len_text = int(prompt_inputs["input_ids"].shape[1])

    image_positions, prompt_len_mm = infer_llava_image_positions_no_forward(
        prompt_inputs=prompt_inputs,
        model_config=model.config,
        num_images=len(image_paths),
    )
    print(f"[sample] prompt_len_text={prompt_len_text}  prompt_len_mm={prompt_len_mm}  "
          f"n_image_tokens={image_positions.numel()}")

    # ---- capture hooks ----------------------------------------------------
    cap = LayerCapture(args.layers)
    cap.register(model)

    try:
        with torch.no_grad():
            gen_out = model.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
    finally:
        cap.remove()

    answer_ids = gen_out.sequences[0, prompt_len_text:].tolist()
    answer_text = tok.decode(answer_ids, skip_special_tokens=True).strip()
    print(f"[sample] generated answer = {answer_text!r}")

    # ---- assemble ---------------------------------------------------------
    per_layer_attn, per_layer_hidden = assemble_per_layer(cap)
    T = next(iter(per_layer_attn.values())).shape[0]
    n_gen = len(cap.q_lens) - 1  # first capture is prefill
    print(f"[capture] total time steps T={T}  n_gen={n_gen}")

    # Free GPU before running probe + plotting.
    del model
    torch.cuda.empty_cache()

    # ---- future probe ------------------------------------------------------
    probe = KVzapModel.from_pretrained(args.future_probe_name)
    future_scores = future_score_per_layer(probe, per_layer_hidden, device)

    # ---- build heatmaps ---------------------------------------------------
    heat_h2o: Dict[int, np.ndarray] = {}
    heat_future: Dict[int, np.ndarray] = {}
    for li in args.layers:
        heat_h2o[li] = build_h2o_heatmap(per_layer_attn[li])
        heat_future[li] = build_future_heatmap(future_scores[li], T)

    # ---- plot -------------------------------------------------------------
    out_png = out_dir / f"wikivqa_sample{sample['sample_id']}_future_vs_h2o.png"
    plot_grid(
        args.layers,
        heat_h2o,
        heat_future,
        prompt_len=prompt_len_mm,
        out_path=out_png,
        sample_meta={"sample_id": sample["sample_id"], "n_gen": n_gen},
    )
    print(f"[viz] wrote {out_png}")

    # Also save raw arrays so the user can re-plot / slice later.
    npz_path = out_dir / f"wikivqa_sample{sample['sample_id']}_raw.npz"
    np.savez_compressed(
        npz_path,
        layers=np.array(args.layers, dtype=np.int32),
        prompt_len_mm=np.array([prompt_len_mm], dtype=np.int32),
        n_gen=np.array([n_gen], dtype=np.int32),
        image_positions=image_positions.numpy().astype(np.int32),
        **{f"attn_L{li}": per_layer_attn[li] for li in args.layers},
        **{f"future_L{li}": future_scores[li] for li in args.layers},
    )
    print(f"[viz] wrote {npz_path}")


if __name__ == "__main__":
    main()
