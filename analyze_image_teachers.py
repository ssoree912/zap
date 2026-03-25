#!/usr/bin/env python3
"""
Analyze image-only VLM teachers from LLaVA extractor outputs.

This script focuses on the next-step teacher design after the global text/image
score gap becomes clear. It computes image-token-only scores for three teachers:

- att_only_answer:     max_{j in answer} a_{ji}
- splus_answer:        max_{j in answer} a_{ji} * ||W_O v_i|| / ||h_j||
- splus_postvision:    max_{j in postvision prompt text} a_{ji} * ||W_O v_i|| / ||h_j||

The script saves compact per-sample teacher tensors and aggregates the following:
- image-only histograms
- layerwise mean / approximate median plots
- layerwise top-k overlap between teachers

The current v1 post-vision query set is defined as all non-image prompt positions
that appear after the image span in multimodal prompt space.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


EPS = 1e-8
DEFAULT_KEEP_RATIOS = (0.05, 0.10, 0.20, 0.40)
SCORE_NAMES = ("att_only_answer", "splus_answer", "splus_postvision")
COMPARE_PAIRS = (
    ("att_only_answer", "splus_answer"),
    ("att_only_answer", "splus_postvision"),
    ("splus_answer", "splus_postvision"),
)


def _to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x



def load_record(path: Path) -> Dict[str, Any]:
    rec = torch.load(path, map_location="cpu")
    if not isinstance(rec, dict):
        raise ValueError(f"Expected dict in {path}, got {type(rec)}")
    return rec



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



def align_hidden_layers(hidden_list: List[torch.Tensor], n_layers: int, key: str) -> List[torch.Tensor]:
    if len(hidden_list) == n_layers + 1:
        return hidden_list[1:]
    if len(hidden_list) == n_layers:
        return hidden_list
    raise ValueError(f"{key} length mismatch: got {len(hidden_list)}, expected {n_layers} or {n_layers + 1}")



def get_image_positions(rec: Dict[str, Any]) -> torch.Tensor:
    pos = maybe_tensor(rec, "image_pos_mm")
    if pos is not None:
        return pos.long().flatten()

    mask = maybe_tensor(rec, "is_image_pos_mm")
    if mask is None:
        raise KeyError("Need `image_pos_mm` or `is_image_pos_mm`.")
    pos = mask.bool().nonzero(as_tuple=False).flatten()
    if pos.numel() == 0:
        raise ValueError("No image positions found in record.")
    return pos.long()



def get_postvision_positions(rec: Dict[str, Any], image_pos: torch.Tensor) -> torch.Tensor:
    pos = maybe_tensor(rec, "postvision_text_pos_mm")
    if pos is not None:
        return pos.long().flatten()

    prompt_len_mm = int(rec["prompt_len_mm"])
    text_mask = maybe_tensor(rec, "is_text_prompt_pos_mm")
    if text_mask is None:
        image_mask = maybe_tensor(rec, "is_image_pos_mm")
        if image_mask is None:
            raise KeyError("Need `is_text_prompt_pos_mm` or `is_image_pos_mm` to infer post-vision positions.")
        text_mask = ~image_mask.bool()
    else:
        text_mask = text_mask.bool()

    last_image = int(image_pos.max().item())
    positions = torch.arange(prompt_len_mm, dtype=torch.long)
    postvision = positions[(positions > last_image) & text_mask]
    if postvision.numel() == 0:
        raise ValueError("No post-vision text positions found after the image span.")
    return postvision



def infer_attn_answer_to_prompt(rec: Dict[str, Any]) -> List[torch.Tensor]:
    ready = tensor_list(rec, "attn_answer_to_prompt")
    if ready is not None:
        return ready
    raise KeyError("Need `attn_answer_to_prompt` in extractor records.")



def infer_hidden_answer(rec: Dict[str, Any], n_layers: int) -> List[torch.Tensor]:
    hidden = tensor_list(rec, "hidden_answer")
    if hidden is None:
        raise KeyError("Need `hidden_answer` in extractor records.")
    return align_hidden_layers(hidden, n_layers, "hidden_answer")



def infer_hidden_prompt(rec: Dict[str, Any], n_layers: int) -> List[torch.Tensor]:
    hidden = tensor_list(rec, "hidden_prompt")
    if hidden is None:
        raise KeyError("Need `hidden_prompt` in extractor records.")
    return align_hidden_layers(hidden, n_layers, "hidden_prompt")



def infer_full_attentions(rec: Dict[str, Any], n_layers: int) -> List[torch.Tensor]:
    full_attn = tensor_list(rec, "full_attentions")
    if full_attn is None:
        full_attn = tensor_list(rec, "attentions")
    if full_attn is None:
        raise KeyError("Need `full_attentions` for post-vision teacher computation.")
    if len(full_attn) != n_layers:
        raise ValueError(f"full attentions length mismatch: got {len(full_attn)}, expected {n_layers}")
    return full_attn



def compute_wov_norm_from_vproj_and_wo(rec: Dict[str, Any], prompt_len: int) -> Optional[List[torch.Tensor]]:
    W_O = tensor_list(rec, "W_O")
    if W_O is None:
        return None

    vproj = tensor_list(rec, "vproj_outputs")
    if vproj is None:
        vproj = tensor_list(rec, "vproj_prompt")
    if vproj is None:
        return None

    attn = infer_attn_answer_to_prompt(rec)
    num_layers = len(attn)
    norms: List[torch.Tensor] = []
    for layer_idx, (Wo, Vcat) in enumerate(zip(W_O, vproj)):
        if layer_idx >= num_layers:
            break
        if Vcat.dim() == 3:
            Vcat = Vcat[0]
        if Vcat.dim() != 2 or Wo.dim() != 2:
            return None

        hidden_in = Wo.shape[1]
        if Vcat.shape[1] != hidden_in:
            return None

        num_heads = attn[layer_idx].shape[0]
        if hidden_in % num_heads != 0:
            return None
        head_dim = hidden_in // num_heads

        V = Vcat[:prompt_len].view(prompt_len, num_heads, head_dim)
        layer_norms = []
        for head_idx in range(num_heads):
            Wo_h = Wo[:, head_idx * head_dim : (head_idx + 1) * head_dim]
            proj = V[:, head_idx, :] @ Wo_h.T
            layer_norms.append(torch.norm(proj, dim=-1))
        norms.append(torch.stack(layer_norms, dim=0))
    return norms



def infer_wov_norm_prompt(rec: Dict[str, Any], prompt_len: int, n_layers: int) -> List[torch.Tensor]:
    ready = tensor_list(rec, "wov_norm_prompt")
    if ready is not None:
        out = []
        for x in ready:
            if x.dim() != 2 or x.shape[-1] < prompt_len:
                raise ValueError(f"Unexpected wov_norm_prompt shape {tuple(x.shape)}")
            out.append(x[:, :prompt_len])
        if len(out) != n_layers:
            raise ValueError(f"wov_norm_prompt length mismatch: got {len(out)}, expected {n_layers}")
        return out

    computed = compute_wov_norm_from_vproj_and_wo(rec, prompt_len)
    if computed is not None and len(computed) == n_layers:
        return computed

    raise KeyError("Need `wov_norm_prompt` or (`W_O` and `vproj_outputs/vproj_prompt`).")



def compute_image_teacher_tensors(rec: Dict[str, Any], eps: float = EPS) -> Dict[str, Any]:
    attn_answer_to_prompt = infer_attn_answer_to_prompt(rec)
    n_layers = len(attn_answer_to_prompt)
    image_pos = get_image_positions(rec)
    postvision_pos = get_postvision_positions(rec, image_pos)
    prompt_len_mm = int(rec["prompt_len_mm"])

    hidden_answer = infer_hidden_answer(rec, n_layers)
    hidden_prompt = infer_hidden_prompt(rec, n_layers)
    full_attentions = infer_full_attentions(rec, n_layers)
    wov_norm_prompt = infer_wov_norm_prompt(rec, prompt_len_mm, n_layers)

    att_only_answer = []
    splus_answer = []
    splus_postvision = []

    for layer_idx in range(n_layers):
        answer_block = attn_answer_to_prompt[layer_idx][:, :, image_pos]  # [H, A, I]
        if answer_block.dim() != 3:
            raise ValueError(f"Unexpected answer attention block shape {tuple(answer_block.shape)}")

        full_attn = full_attentions[layer_idx]
        if full_attn.dim() == 4:
            full_attn = full_attn[0]
        if full_attn.dim() != 3:
            raise ValueError(f"Unexpected full attention shape {tuple(full_attn.shape)}")
        postvision_block = full_attn[:, postvision_pos, :][:, :, image_pos]  # [H, Q, I]

        wnorm_image = wov_norm_prompt[layer_idx][:, image_pos]  # [H, I]
        answer_norm = torch.norm(hidden_answer[layer_idx], dim=-1).clamp_min(eps)  # [A]
        postvision_hidden = hidden_prompt[layer_idx][postvision_pos]
        postvision_norm = torch.norm(postvision_hidden, dim=-1).clamp_min(eps)  # [Q]

        att_only_answer.append(answer_block.max(dim=1).values)
        splus_answer.append((answer_block * (1.0 / answer_norm.view(1, -1, 1)) * wnorm_image.view(answer_block.shape[0], 1, -1)).max(dim=1).values)
        splus_postvision.append(
            (postvision_block * (1.0 / postvision_norm.view(1, -1, 1)) * wnorm_image.view(postvision_block.shape[0], 1, -1)).max(dim=1).values
        )

    return {
        "sample_id": rec.get("sample_id", "unknown"),
        "question": rec.get("question", ""),
        "answer_text": rec.get("answer_text", ""),
        "image_pos_mm": image_pos,
        "postvision_query_pos_mm": postvision_pos,
        "att_only_answer": torch.stack(att_only_answer, dim=0).float(),
        "splus_answer": torch.stack(splus_answer, dim=0).float(),
        "splus_postvision": torch.stack(splus_postvision, dim=0).float(),
    }



def parse_keep_ratios(values: Sequence[str] | None) -> tuple[float, ...]:
    if not values:
        return DEFAULT_KEEP_RATIOS
    ratios: list[float] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            ratio = float(chunk)
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"keep ratio must be in (0, 1], got {ratio}")
            ratios.append(ratio)
    return tuple(ratios)



def hist_quantile(counts: np.ndarray, bin_edges: np.ndarray, q: float) -> float:
    total = int(counts.sum())
    if total <= 0:
        return float("nan")
    threshold = q * total
    cdf = np.cumsum(counts)
    idx = int(np.searchsorted(cdf, threshold, side="left"))
    idx = min(max(idx, 0), len(bin_edges) - 2)
    return float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)



class TeacherAggregator:
    def __init__(self, keep_ratios: Sequence[float], log_min: float = -20.0, log_max: float = 5.0, n_bins: int = 320):
        self.keep_ratios = tuple(float(x) for x in keep_ratios)
        self.bin_edges = np.linspace(log_min, log_max, n_bins + 1, dtype=np.float64)
        self.n_layers: Optional[int] = None
        self.global_hist: dict[str, np.ndarray] = {}
        self.layer_hist: dict[str, np.ndarray] = {}
        self.global_sum_raw: dict[str, float] = {}
        self.global_sum_log: dict[str, float] = {}
        self.global_count: dict[str, int] = {}
        self.layer_sum_raw: dict[str, np.ndarray] = {}
        self.layer_sum_log: dict[str, np.ndarray] = {}
        self.layer_count: dict[str, np.ndarray] = {}
        self.overlap_sum: dict[tuple[str, str, float], np.ndarray] = {}
        self.overlap_count: dict[tuple[str, str, float], np.ndarray] = {}

    def _ensure_initialized(self, score_tensors: Dict[str, torch.Tensor]) -> None:
        if self.n_layers is not None:
            return
        first = next(iter(score_tensors.values()))
        self.n_layers = int(first.shape[0])
        for score_name in SCORE_NAMES:
            self.global_hist[score_name] = np.zeros(len(self.bin_edges) - 1, dtype=np.int64)
            self.layer_hist[score_name] = np.zeros((self.n_layers, len(self.bin_edges) - 1), dtype=np.int64)
            self.global_sum_raw[score_name] = 0.0
            self.global_sum_log[score_name] = 0.0
            self.global_count[score_name] = 0
            self.layer_sum_raw[score_name] = np.zeros(self.n_layers, dtype=np.float64)
            self.layer_sum_log[score_name] = np.zeros(self.n_layers, dtype=np.float64)
            self.layer_count[score_name] = np.zeros(self.n_layers, dtype=np.int64)
        for lhs, rhs in COMPARE_PAIRS:
            compare_name = f"{lhs}__vs__{rhs}"
            for ratio in self.keep_ratios:
                key = (lhs, rhs, ratio)
                self.overlap_sum[key] = np.zeros(self.n_layers, dtype=np.float64)
                self.overlap_count[key] = np.zeros(self.n_layers, dtype=np.int64)

    def update(self, score_tensors: Dict[str, torch.Tensor]) -> None:
        self._ensure_initialized(score_tensors)
        assert self.n_layers is not None

        for score_name, tensor in score_tensors.items():
            arr = tensor.detach().cpu().float().clamp_min(EPS).numpy()  # [L, H, I]
            log_arr = np.log(arr)
            self.global_hist[score_name] += np.histogram(log_arr.reshape(-1), bins=self.bin_edges)[0]
            self.global_sum_raw[score_name] += float(arr.sum())
            self.global_sum_log[score_name] += float(log_arr.sum())
            self.global_count[score_name] += int(arr.size)
            for layer_idx in range(self.n_layers):
                layer_vals = arr[layer_idx].reshape(-1)
                layer_log_vals = log_arr[layer_idx].reshape(-1)
                self.layer_hist[score_name][layer_idx] += np.histogram(layer_log_vals, bins=self.bin_edges)[0]
                self.layer_sum_raw[score_name][layer_idx] += float(layer_vals.sum())
                self.layer_sum_log[score_name][layer_idx] += float(layer_log_vals.sum())
                self.layer_count[score_name][layer_idx] += int(layer_vals.size)

        for lhs, rhs in COMPARE_PAIRS:
            lhs_tensor = score_tensors[lhs].detach().cpu().float()
            rhs_tensor = score_tensors[rhs].detach().cpu().float()
            if lhs_tensor.shape != rhs_tensor.shape:
                raise ValueError(f"Shape mismatch for overlap computation: {lhs} {tuple(lhs_tensor.shape)} vs {rhs} {tuple(rhs_tensor.shape)}")
            _, n_heads, n_tokens = lhs_tensor.shape
            for layer_idx in range(self.n_layers):
                lhs_layer = lhs_tensor[layer_idx]
                rhs_layer = rhs_tensor[layer_idx]
                for ratio in self.keep_ratios:
                    k = max(1, int(math.ceil(n_tokens * ratio)))
                    lhs_idx = torch.topk(lhs_layer, k=k, dim=-1).indices
                    rhs_idx = torch.topk(rhs_layer, k=k, dim=-1).indices
                    lhs_mask = torch.zeros_like(lhs_layer, dtype=torch.bool)
                    rhs_mask = torch.zeros_like(rhs_layer, dtype=torch.bool)
                    lhs_mask.scatter_(1, lhs_idx, True)
                    rhs_mask.scatter_(1, rhs_idx, True)
                    overlap = (lhs_mask & rhs_mask).float().sum(dim=-1) / float(k)
                    key = (lhs, rhs, ratio)
                    self.overlap_sum[key][layer_idx] += float(overlap.sum().item())
                    self.overlap_count[key][layer_idx] += int(n_heads)

    def build_summary_df(self) -> pd.DataFrame:
        rows = []
        for score_name in SCORE_NAMES:
            rows.append(
                {
                    "score_type": score_name,
                    "mean_score": self.global_sum_raw[score_name] / max(self.global_count[score_name], 1),
                    "mean_log_score": self.global_sum_log[score_name] / max(self.global_count[score_name], 1),
                    "approx_median_log_score": hist_quantile(self.global_hist[score_name], self.bin_edges, 0.5),
                    "approx_p05_log_score": hist_quantile(self.global_hist[score_name], self.bin_edges, 0.05),
                    "approx_p95_log_score": hist_quantile(self.global_hist[score_name], self.bin_edges, 0.95),
                }
            )
        df = pd.DataFrame(rows)
        df["approx_median_score"] = np.exp(df["approx_median_log_score"])
        df["approx_p05_score"] = np.exp(df["approx_p05_log_score"])
        df["approx_p95_score"] = np.exp(df["approx_p95_log_score"])
        return df

    def build_layerwise_stats_df(self) -> pd.DataFrame:
        assert self.n_layers is not None
        rows = []
        for score_name in SCORE_NAMES:
            for layer_idx in range(self.n_layers):
                count = max(int(self.layer_count[score_name][layer_idx]), 1)
                median_log = hist_quantile(self.layer_hist[score_name][layer_idx], self.bin_edges, 0.5)
                rows.append(
                    {
                        "layer": layer_idx,
                        "score_type": score_name,
                        "mean_score": self.layer_sum_raw[score_name][layer_idx] / count,
                        "mean_log_score": self.layer_sum_log[score_name][layer_idx] / count,
                        "approx_median_log_score": median_log,
                        "approx_median_score": math.exp(median_log),
                    }
                )
        return pd.DataFrame(rows)

    def build_overlap_df(self) -> pd.DataFrame:
        assert self.n_layers is not None
        rows = []
        for lhs, rhs in COMPARE_PAIRS:
            for ratio in self.keep_ratios:
                key = (lhs, rhs, ratio)
                for layer_idx in range(self.n_layers):
                    denom = max(int(self.overlap_count[key][layer_idx]), 1)
                    rows.append(
                        {
                            "lhs": lhs,
                            "rhs": rhs,
                            "keep_ratio": ratio,
                            "layer": layer_idx,
                            "mean_topk_overlap": self.overlap_sum[key][layer_idx] / denom,
                        }
                    )
        return pd.DataFrame(rows)



def plot_histogram(summary_counts: Dict[str, np.ndarray], bin_edges: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7.5, 4.8))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    for score_name in SCORE_NAMES:
        counts = summary_counts[score_name].astype(np.float64)
        density = counts / max(counts.sum(), 1.0) / widths
        plt.plot(centers, density, label=score_name)
    plt.xlabel("log score")
    plt.ylabel("density")
    plt.title("Image-only teacher distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_layerwise_stats(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.8))
    for score_name in SCORE_NAMES:
        sub = df[df["score_type"] == score_name]
        plt.plot(sub["layer"], sub[value_col], marker="o", label=score_name)
    plt.xlabel("layer")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_topk_overlap(df: pd.DataFrame, out_dir: Path) -> None:
    for lhs, rhs in COMPARE_PAIRS:
        sub = df[(df["lhs"] == lhs) & (df["rhs"] == rhs)]
        plt.figure(figsize=(8, 4.8))
        for ratio in sorted(sub["keep_ratio"].unique()):
            ratio_sub = sub[sub["keep_ratio"] == ratio]
            plt.plot(ratio_sub["layer"], ratio_sub["mean_topk_overlap"], marker="o", label=f"top-{int(round(ratio * 100))}%")
        plt.xlabel("layer")
        plt.ylabel("mean top-k overlap")
        plt.ylim(0.0, 1.0)
        plt.title(f"Top-k overlap: {lhs} vs {rhs}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"topk_overlap__{lhs}__vs__{rhs}.png", dpi=180)
        plt.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Extractor output root or records directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save image-only teacher analysis")
    parser.add_argument("--glob", type=str, default="*.pt", help="Glob pattern for record files")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep_ratios", nargs="*", default=None, help="Keep ratios such as 0.05 0.10 0.20 0.40")
    parser.add_argument("--save_teacher_records", action="store_true", help="Save per-sample teacher tensors")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = out_dir / "teacher_records"
    if args.save_teacher_records:
        teacher_dir.mkdir(parents=True, exist_ok=True)

    keep_ratios = parse_keep_ratios(args.keep_ratios)
    record_dir, files = resolve_record_files(input_dir, args.glob)
    if args.limit is not None:
        files = files[: args.limit]

    aggregator = TeacherAggregator(keep_ratios=keep_ratios)
    failures: list[dict[str, str]] = []
    processed = 0

    for fp in files:
        try:
            rec = load_record(fp)
            teacher = compute_image_teacher_tensors(rec)
            score_tensors = {score_name: teacher[score_name] for score_name in SCORE_NAMES}
            aggregator.update(score_tensors)
            if args.save_teacher_records:
                payload = {
                    "sample_id": teacher["sample_id"],
                    "question": teacher["question"],
                    "answer_text": teacher["answer_text"],
                    "image_pos_mm": teacher["image_pos_mm"],
                    "postvision_query_pos_mm": teacher["postvision_query_pos_mm"],
                    **score_tensors,
                }
                torch.save(payload, teacher_dir / f"{teacher['sample_id']}.pt")
            processed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": fp.name, "error": repr(exc)})

    if processed == 0:
        raise RuntimeError(f"All files failed. Failures: {failures}")

    summary_df = aggregator.build_summary_df()
    layerwise_df = aggregator.build_layerwise_stats_df()
    overlap_df = aggregator.build_overlap_df()

    summary_df.to_csv(out_dir / "summary_image_scores.csv", index=False)
    layerwise_df.to_csv(out_dir / "layerwise_image_score_stats.csv", index=False)
    overlap_df.to_csv(out_dir / "topk_overlap.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    plot_histogram(aggregator.global_hist, aggregator.bin_edges, out_dir / "image_only_log_score_hist.png")
    plot_layerwise_stats(layerwise_df, "mean_log_score", out_dir / "layerwise_mean_log_score.png", "Layerwise mean log score")
    plot_layerwise_stats(layerwise_df, "approx_median_log_score", out_dir / "layerwise_median_log_score.png", "Layerwise median log score")
    plot_topk_overlap(overlap_df, out_dir)

    config = {
        "input_dir": str(input_dir.resolve()),
        "resolved_record_dir": str(record_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "glob": args.glob,
        "limit": args.limit,
        "keep_ratios": list(keep_ratios),
        "score_names": list(SCORE_NAMES),
        "compare_pairs": [list(pair) for pair in COMPARE_PAIRS],
        "postvision_definition": "all non-image prompt positions after the image span",
        "save_teacher_records": bool(args.save_teacher_records),
        "n_files_seen": len(files),
        "n_files_processed": processed,
        "n_failures": len(failures),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Resolved record dir: {record_dir}")
    print(f"Saved outputs to {out_dir}")
    print(f"Processed {processed} files")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
