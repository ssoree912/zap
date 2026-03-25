#!/usr/bin/env python3
"""
Analyze VLM teacher-score distributions from LLaVA extractor outputs.

Expected per-sample record structure (best case):
{
  "sample_id": str,
  "attn_answer_to_prompt": list[tensor[H, A, P]],
  "hidden_answer": list[tensor[A, D]],
  "wov_norm_prompt": list[tensor[H, P]],  # preferred
  "is_image_pos_mm": tensor[P] bool,       # prompt-space modality mask
}

Supported fallback structures:
- full attentions under key "attentions" or "full_attentions"
- prompt/answer positions under keys "prompt_pos_mm", "answer_pos_mm"
- full hidden states under key "hidden_states"
- full W_O + vproj outputs under keys "W_O" and "vproj_outputs" / "vproj_prompt"

This script computes:
1) attention-only score: max_j a_{ji}
2) output-aware teacher: max_j a_{ji} * ||W_O v_i|| / ||h_j||

And generates summary CSVs + plots.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def load_record(path: Path) -> Dict[str, Any]:
    if path.suffix in {".pt", ".pth", ".bin"}:
        rec = torch.load(path, map_location="cpu")
        if not isinstance(rec, dict):
            raise ValueError(f"Expected dict in {path}, got {type(rec)}")
        return rec
    raise ValueError(f"Unsupported file format: {path}")


def tensor_list(rec: Dict[str, Any], key: str) -> Optional[List[torch.Tensor]]:
    if key not in rec:
        return None
    val = rec[key]
    if isinstance(val, (list, tuple)):
        return [_to_cpu(v) for v in val]
    return None


def maybe_tensor(rec: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    if key not in rec:
        return None
    val = rec[key]
    return _to_cpu(val) if isinstance(val, torch.Tensor) else None


def get_prompt_mask(rec: Dict[str, Any], prompt_len: int) -> torch.Tensor:
    # Best case: already prompt-space bool mask.
    mask = maybe_tensor(rec, "is_image_pos_mm")
    if mask is not None:
        mask = mask.bool()
        if mask.numel() >= prompt_len:
            return mask[:prompt_len]

    mask = maybe_tensor(rec, "is_image_prompt")
    if mask is not None and mask.numel() == prompt_len:
        return mask.bool()

    raise KeyError(
        "Could not find prompt-space image mask. Expected `is_image_pos_mm` or `is_image_prompt`."
    )


def slice_answer_to_prompt_from_full_attn(
    full_attn: List[torch.Tensor],
    prompt_pos: torch.Tensor,
    answer_pos: torch.Tensor,
) -> List[torch.Tensor]:
    blocks: List[torch.Tensor] = []
    for a in full_attn:
        # Accept [1,H,S,S] or [H,S,S]
        if a.dim() == 4:
            a = a[0]
        if a.dim() != 3:
            raise ValueError(f"Unexpected attention shape {tuple(a.shape)}")
        block = a[:, answer_pos, :][:, :, prompt_pos]  # [H, A, P]
        blocks.append(block)
    return blocks


def infer_attn_answer_to_prompt(rec: Dict[str, Any]) -> List[torch.Tensor]:
    ready = tensor_list(rec, "attn_answer_to_prompt")
    if ready is not None:
        return ready

    full_attn = tensor_list(rec, "full_attentions")
    if full_attn is None:
        full_attn = tensor_list(rec, "attentions")
    if full_attn is None:
        raise KeyError("Need `attn_answer_to_prompt` or full attentions under `attentions/full_attentions`.")

    prompt_pos = maybe_tensor(rec, "prompt_pos_mm")
    answer_pos = maybe_tensor(rec, "answer_pos_mm")
    if prompt_pos is None or answer_pos is None:
        raise KeyError("Need `prompt_pos_mm` and `answer_pos_mm` to slice full attentions.")

    return slice_answer_to_prompt_from_full_attn(full_attn, prompt_pos.long(), answer_pos.long())



def infer_hidden_answer(rec: Dict[str, Any]) -> List[torch.Tensor]:
    ready = tensor_list(rec, "hidden_answer")
    if ready is not None:
        return ready

    hidden = tensor_list(rec, "hidden_states")
    if hidden is None:
        raise KeyError("Need `hidden_answer` or full `hidden_states`.")

    answer_pos = maybe_tensor(rec, "answer_pos_mm")
    if answer_pos is None:
        raise KeyError("Need `answer_pos_mm` to slice full hidden states.")
    answer_pos = answer_pos.long()

    out: List[torch.Tensor] = []
    for h in hidden:
        # Accept [1,S,D] or [S,D]
        if h.dim() == 3:
            h = h[0]
        if h.dim() != 2:
            raise ValueError(f"Unexpected hidden shape {tuple(h.shape)}")
        out.append(h[answer_pos])
    return out



def compute_wov_norm_from_vproj_and_wo(rec: Dict[str, Any], prompt_len: int) -> Optional[List[torch.Tensor]]:
    W_O = tensor_list(rec, "W_O")
    if W_O is None:
        return None

    vproj = tensor_list(rec, "vproj_outputs")
    if vproj is None:
        vproj = tensor_list(rec, "vproj_prompt")
    if vproj is None:
        return None

    norms: List[torch.Tensor] = []
    for Wo, Vcat in zip(W_O, vproj):
        # Wo: [d_model, H*Dh] or [out_dim, in_dim]
        # Vcat: [1,S,H*Dh] or [S,H*Dh]
        if Vcat.dim() == 3:
            Vcat = Vcat[0]
        if Vcat.dim() != 2 or Wo.dim() != 2:
            return None

        hidden_in = Wo.shape[1]
        if Vcat.shape[1] != hidden_in:
            # incompatible shapes
            return None

        # Try to infer num_heads from saved attention block if available.
        attn = infer_attn_answer_to_prompt(rec)
        num_heads = attn[0].shape[0]
        if hidden_in % num_heads != 0:
            return None
        head_dim = hidden_in // num_heads

        V = Vcat[:prompt_len].view(prompt_len, num_heads, head_dim)
        layer_norms = []
        for h in range(num_heads):
            Wo_h = Wo[:, h * head_dim : (h + 1) * head_dim]  # [d_model, head_dim]
            proj = V[:, h, :] @ Wo_h.T                       # [P, d_model]
            layer_norms.append(torch.norm(proj, dim=-1))
        norms.append(torch.stack(layer_norms, dim=0))       # [H, P]
    return norms



def infer_wov_norm_prompt(rec: Dict[str, Any], prompt_len: int) -> List[torch.Tensor]:
    ready = tensor_list(rec, "wov_norm_prompt")
    if ready is not None:
        out = []
        for x in ready:
            if x.dim() == 2 and x.shape[-1] >= prompt_len:
                out.append(x[:, :prompt_len])
            else:
                raise ValueError(f"Unexpected wov_norm_prompt shape {tuple(x.shape)}")
        return out

    computed = compute_wov_norm_from_vproj_and_wo(rec, prompt_len)
    if computed is not None:
        return computed

    raise KeyError(
        "Could not infer `||W_O v_i||`. Need `wov_norm_prompt` or (`W_O` and `vproj_outputs/vproj_prompt`)."
    )



def compute_scores_for_record(rec: Dict[str, Any], eps: float = 1e-8) -> pd.DataFrame:
    attn_blocks = infer_attn_answer_to_prompt(rec)
    hidden_answer = infer_hidden_answer(rec)

    prompt_len = attn_blocks[0].shape[-1]
    is_image = get_prompt_mask(rec, prompt_len)
    wov_norm = infer_wov_norm_prompt(rec, prompt_len)

    rows = []
    sample_id = rec.get("sample_id", "unknown")

    # Hidden states often include embedding output at index 0, attentions don't.
    # Align by taking the last N hidden tensors if needed.
    if len(hidden_answer) == len(attn_blocks) + 1:
        hidden_answer = hidden_answer[1:]
    elif len(hidden_answer) != len(attn_blocks):
        raise ValueError(
            f"Hidden/attention length mismatch for sample {sample_id}: "
            f"hidden={len(hidden_answer)}, attn={len(attn_blocks)}"
        )

    if len(wov_norm) != len(attn_blocks):
        raise ValueError(
            f"wov/attention length mismatch for sample {sample_id}: "
            f"wov={len(wov_norm)}, attn={len(attn_blocks)}"
        )

    for layer_idx, (a, h_ans, wnorm) in enumerate(zip(attn_blocks, hidden_answer, wov_norm)):
        # a: [H, A, P], h_ans: [A, D], wnorm: [H, P]
        if a.dim() != 3:
            raise ValueError(f"Unexpected a shape {tuple(a.shape)}")
        h_norm = torch.norm(h_ans, dim=-1).clamp_min(eps)  # [A]
        att_only = a.max(dim=1).values                      # [H, P]
        weighted = a * (1.0 / h_norm.view(1, -1, 1))       # [H, A, P]
        splus = (weighted * wnorm.view(a.shape[0], 1, a.shape[2])).max(dim=1).values

        for head_idx in range(a.shape[0]):
            for tok_idx in range(a.shape[2]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "prompt_idx": int(tok_idx),
                        "modality": "image" if bool(is_image[tok_idx]) else "text",
                        "att_only": float(att_only[head_idx, tok_idx].item()),
                        "splus": float(splus[head_idx, tok_idx].item()),
                        "log_att_only": float(math.log(max(att_only[head_idx, tok_idx].item(), eps))),
                        "log_splus": float(math.log(max(splus[head_idx, tok_idx].item(), eps))),
                        "wov_norm": float(wnorm[head_idx, tok_idx].item()),
                    }
                )
    return pd.DataFrame(rows)



def plot_hist_by_modality(df: pd.DataFrame, outdir: Path) -> None:
    for col in ["log_att_only", "log_splus"]:
        plt.figure(figsize=(7, 4.5))
        for modality in ["image", "text"]:
            vals = df.loc[df["modality"] == modality, col].values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=80, alpha=0.5, density=True, label=modality)
        plt.xlabel(col)
        plt.ylabel("density")
        plt.title(f"{col} distribution by modality")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{col}_hist_by_modality.png", dpi=180)
        plt.close()


def plot_layerwise_median_fraction(df: pd.DataFrame, outdir: Path) -> None:
    records = []
    for col in ["log_att_only", "log_splus"]:
        global_median = df[col].median()
        tmp = df.assign(is_below=(df[col] <= global_median)).groupby(["layer", "modality"])["is_below"].mean().reset_index()
        tmp["score_type"] = col
        records.append(tmp)
    plot_df = pd.concat(records, ignore_index=True)
    plot_df.to_csv(outdir / "layerwise_fraction_below_global_median.csv", index=False)

    for score_type in ["log_att_only", "log_splus"]:
        plt.figure(figsize=(7, 4.5))
        sub = plot_df[plot_df["score_type"] == score_type]
        for modality in ["image", "text"]:
            s = sub[sub["modality"] == modality]
            plt.plot(s["layer"], s["is_below"], marker="o", label=modality)
        plt.xlabel("layer")
        plt.ylabel("fraction <= global median")
        plt.title(score_type)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{score_type}_fraction_below_median.png", dpi=180)
        plt.close()


def plot_layer_head_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    for modality in ["image", "text"]:
        for col in ["log_att_only", "log_splus"]:
            piv = (
                df[df["modality"] == modality]
                .groupby(["layer", "head"])[col]
                .mean()
                .reset_index()
                .pivot(index="layer", columns="head", values=col)
            )
            plt.figure(figsize=(9, 5))
            plt.imshow(piv.values, aspect="auto")
            plt.colorbar(label=f"mean {col}")
            plt.xlabel("head")
            plt.ylabel("layer")
            plt.title(f"{col} heatmap ({modality})")
            plt.tight_layout()
            plt.savefig(outdir / f"heatmap_{col}_{modality}.png", dpi=180)
            plt.close()


def plot_scatter_att_vs_splus(df: pd.DataFrame, outdir: Path, sample_n: int = 15000) -> None:
    rng = np.random.default_rng(0)
    for modality in ["image", "text"]:
        sub = df[df["modality"] == modality]
        if len(sub) > sample_n:
            idx = rng.choice(len(sub), size=sample_n, replace=False)
            sub = sub.iloc[idx]
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(sub["log_att_only"], sub["log_splus"], s=4, alpha=0.25)
        plt.xlabel("log_att_only")
        plt.ylabel("log_splus")
        plt.title(f"att-only vs splus ({modality})")
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_att_vs_splus_{modality}.png", dpi=180)
        plt.close()


def summarize(df: pd.DataFrame, outdir: Path) -> None:
    summary = []
    for modality in ["image", "text"]:
        sub = df[df["modality"] == modality]
        for col in ["att_only", "splus", "log_att_only", "log_splus", "wov_norm"]:
            summary.append(
                {
                    "modality": modality,
                    "metric": col,
                    "mean": float(sub[col].mean()),
                    "std": float(sub[col].std()),
                    "median": float(sub[col].median()),
                    "p05": float(sub[col].quantile(0.05)),
                    "p95": float(sub[col].quantile(0.95)),
                }
            )
    pd.DataFrame(summary).to_csv(outdir / "summary_by_modality.csv", index=False)


def resolve_record_files(input_dir: Path, pattern: str) -> tuple[Path, list[Path]]:
    candidates = [input_dir]
    records_dir = input_dir / "records"
    if records_dir.is_dir():
        candidates.append(records_dir)

    tried = []
    for candidate in candidates:
        tried.append(candidate)
        files = sorted(candidate.glob(pattern))
        if files:
            return candidate, files

    tried_str = ", ".join(str(path) for path in tried)
    raise FileNotFoundError(f"No files matched {pattern} in {input_dir}. Tried: {tried_str}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with per-sample .pt records")
    parser.add_argument("--glob", type=str, default="*.pt", help="Glob pattern for record files")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_dir, files = resolve_record_files(input_dir, args.glob)
    if args.limit is not None:
        files = files[: args.limit]

    all_df = []
    failures = []
    for fp in files:
        try:
            rec = load_record(fp)
            df = compute_scores_for_record(rec)
            df["record_file"] = fp.name
            all_df.append(df)
        except Exception as e:  # noqa: BLE001
            failures.append({"file": fp.name, "error": repr(e)})

    if not all_df:
        raise RuntimeError(f"All files failed. Failures: {failures}")

    df = pd.concat(all_df, ignore_index=True)
    df.to_csv(out_dir / "all_scores_long.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    summarize(df, out_dir)
    plot_hist_by_modality(df, out_dir)
    plot_layerwise_median_fraction(df, out_dir)
    plot_layer_head_heatmap(df, out_dir)
    plot_scatter_att_vs_splus(df, out_dir)

    print(f"Resolved record dir: {record_dir}")
    print(f"Saved outputs to {out_dir}")
    print(f"Processed {df['sample_id'].nunique()} samples from {len(files)} files")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
