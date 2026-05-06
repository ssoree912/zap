#!/usr/bin/env python3
"""Per-sample, per-layer future-utility mismatch metrics.

Compares three image-token scoring methods against the future-decode oracle
(teacher) extracted by `collect_original_llava15_teacher.py`:

    1. Prefill saliency (H2O-style): column-sum of prefill attention to image keys
    2. Q-ViK student:                trained VisualUtilityStudent on prefill hidden
    3. Future oracle (teacher):      loaded from .pt dump (= reference)

Mismatch metrics per (sample, layer, method):
    - Cosine similarity (raw score vector)
    - Spearman rank correlation
    - Jaccard@K for K = ceil(keep_ratio * N_I)   for each requested keep_ratio

Output:  one JSON with config + per-sample/per-layer/per-method numbers.

Usage:
    python compute_mismatch.py \
        --teacher-dir /workspace/zap/artifacts/original_llava_teacher/future_decode_llava15_7b \
        --datasets textvqa docvqa gqa \
        --student-ckpt /workspace/zap/artifacts/student_llava15_original_future_1800_lr1e4_15ep \
        --n-samples 100 \
        --keep-ratios 0.1 0.2 0.5 \
        --output mismatch_textvqa_docvqa_gqa.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

# Reuse the teacher collector's LLaVA loading path. Override via env var
# VFLOWOPT_LLAVA_ROOT if not at the default workspace location.
import os as _os  # noqa: E402
VFLOWOPT_LLAVA_ROOT = Path(_os.environ.get(
    "VFLOWOPT_LLAVA_ROOT",
    "/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision",
))
if str(VFLOWOPT_LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(VFLOWOPT_LLAVA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

# Repo root for kvpress imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Mismatch metrics
# ──────────────────────────────────────────────────────────────────────────────

def _to_distribution(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-shift then sum-normalize to a probability-like vector ≥ 0."""
    x = x - x.min()             # non-negative
    s = x.sum()
    return x / s if s > eps else np.full_like(x, 1.0 / max(x.size, 1))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine on probability-normalized vectors (sign-stable across raw scales)."""
    pa = _to_distribution(a)
    pb = _to_distribution(b)
    na = float(np.linalg.norm(pa))
    nb = float(np.linalg.norm(pb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(pa, pb) / (na * nb))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if (rho is None or np.isnan(rho)) else float(rho)


def _jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = max(1, min(k, a.size))
    top_a = set(np.argpartition(-a, k - 1)[:k].tolist())
    top_b = set(np.argpartition(-b, k - 1)[:k].tolist())
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return float(inter / union) if union else 0.0


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, keep_ratios: list[float]) -> dict[str, float]:
    p = pred.detach().cpu().numpy().astype(np.float64).ravel()
    t = target.detach().cpu().numpy().astype(np.float64).ravel()
    n = p.size
    out: dict[str, float] = {
        "cos_sim": _cosine_sim(p, t),
        "spearman": _spearman(p, t),
    }
    for kr in keep_ratios:
        k = int(np.ceil(n * float(kr)))
        out[f"jaccard@{kr:g}"] = _jaccard_topk(p, t, k)
    return out


def aggregate_layers(per_layer: dict[int, dict[str, float]]) -> dict[str, float]:
    keys = next(iter(per_layer.values())).keys()
    return {k: float(np.mean([v[k] for v in per_layer.values()])) for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# Score extraction
# ──────────────────────────────────────────────────────────────────────────────

def h2o_score_per_layer(
    attentions: tuple[torch.Tensor, ...],
    image_indices: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """H2O column-sum at image positions, per layer.

    attentions[l]: [1, H, P, P] (eager attention, prefill forward).
    Returns {layer_idx: (N_I,) tensor on cpu/float32}.
    """
    out: dict[int, torch.Tensor] = {}
    img_idx = image_indices.long()
    for l, attn in enumerate(attentions):
        a = attn[0].float()                  # [H, P, P]
        col_sum = a.sum(dim=1).mean(dim=0)   # sum over Q, mean over heads → [P]
        out[l] = col_sum.index_select(dim=-1, index=img_idx.to(col_sum.device)).cpu()
    return out


def lookm_score_per_layer(
    attentions: tuple[torch.Tensor, ...],
    image_indices: torch.Tensor,
    keep_ratio: float,
) -> dict[int, torch.Tensor]:
    """LOOK-M: per-(layer, head) top-K selection on column-sum scores.

    Score per patch = fraction of heads that keep it in their per-head top-K.
    Uses K_per_head = ceil(keep_ratio * N_I). Captures LOOK-M's head-wise
    eviction (each head independently selects its top-K).
    """
    out: dict[int, torch.Tensor] = {}
    img_idx = image_indices.long()
    n_img = int(img_idx.numel())
    K = max(1, int(np.ceil(keep_ratio * n_img)))
    for l, attn in enumerate(attentions):
        a = attn[0].float()                                  # [H, P, P]
        head_score = a.sum(dim=1)                            # [H, P]
        head_score_img = head_score.index_select(            # [H, N_I]
            dim=-1, index=img_idx.to(head_score.device)).cpu()
        H = head_score_img.size(0)
        keep_count = torch.zeros(n_img)
        for h in range(H):
            top_idx = head_score_img[h].argsort(descending=True)[:K]
            keep_count[top_idx] += 1
        out[l] = keep_count / float(H)                       # [N_I] in [0, 1]
    return out


def vflowopt_score_per_layer(
    attentions: tuple[torch.Tensor, ...],
    image_indices: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """VFlowOpt: last query's attention to image keys, mean over heads."""
    out: dict[int, torch.Tensor] = {}
    img_idx = image_indices.long()
    for l, attn in enumerate(attentions):
        a = attn[0].float()                          # [H, P, P]
        last_q = a[:, -1, :].mean(dim=0)             # [P], mean over heads
        out[l] = last_q.index_select(dim=-1, index=img_idx.to(last_q.device)).cpu()
    return out


def prefixkv_score_per_layer(
    attentions: tuple[torch.Tensor, ...],
    image_indices: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """PrefixKV: head-mean column-sum, equivalent to H2O on image positions
    (prefix/recent position protection does not apply to image tokens)."""
    return h2o_score_per_layer(attentions, image_indices)


def student_score_per_layer(
    student: VisualUtilityStudent,
    hidden_states: tuple[torch.Tensor, ...],
    image_indices: torch.Tensor,
    question_indices: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run student per layer it covers. hidden_states[l+1] = output of LLM layer l."""
    out: dict[int, torch.Tensor] = {}
    img_idx_dev = image_indices.long().to(device)
    q_idx_dev = question_indices.long().to(device)
    for l in student.layer_indices:
        H_l = hidden_states[l + 1].to(device=device, dtype=torch.float16)
        layer_mod = student.layers[str(l)].to(device=device, dtype=torch.float16).eval()
        with torch.no_grad():
            score = layer_mod(H_l, img_idx_dev, q_idx_dev, student.grid_h, student.grid_w)
        out[l] = score.squeeze(0).float().cpu()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass over teacher .pt sample
# ──────────────────────────────────────────────────────────────────────────────

def _remap_path(p: str, remaps: list[tuple[str, str]]) -> str:
    for src, dst in remaps:
        if p.startswith(src):
            return dst + p[len(src):]
    return p


def run_prefill(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    rec: dict,
    device: torch.device,
    path_remaps: list[tuple[str, str]] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Re-run prefill forward to get (attentions, hidden_states).

    Uses prompt_text + image_path stored in the teacher .pt — exactly the same
    inputs used during teacher extraction.
    """
    # Prefer in-place image bytes (new collector); fall back to image_path (legacy).
    if rec.get("image_bytes"):
        import io as _io
        image = Image.open(_io.BytesIO(rec["image_bytes"])).convert("RGB")
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
    else:
        image_path = rec["image_path"]
        if path_remaps:
            image_path = _remap_path(image_path, path_remaps)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        rec["prompt_text"], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            modalities=["image"],
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    return out.attentions, out.hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-dir", required=True,
                   help="Root of teacher .pt dumps (subdirs per dataset).")
    p.add_argument("--datasets", nargs="+", default=["textvqa", "docvqa", "gqa"])
    p.add_argument("--n-samples", type=int, default=100, help="Per dataset.")
    p.add_argument("--student-ckpt", required=True,
                   help="VisualUtilityStudent checkpoint dir (CNN+MLP / variant).")
    p.add_argument("--keep-ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--model-path", default="/workspace/zap/ckpts/llava-v1.5-7b")
    p.add_argument("--model-name", default="llava-v1.5-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--save-raw-scores", action="store_true",
                   help="Also save raw per-layer score vectors (large; for debugging).")
    p.add_argument("--path-remap", action="append", default=[],
                   help="Remap stored image_path prefix, e.g. "
                        "'/workspace/zap=/mnt/srv/home/dlpc.3842/zap'. May repeat.")
    return p.parse_args()


def load_model(args: argparse.Namespace):
    print(f"[load] {args.model_path} attn=eager", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
        attn_implementation="eager",
        multimodal=True,
    )
    model.eval()
    return tokenizer, model, image_processor


def main() -> int:
    args = parse_args()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Parse path remaps: 'src=dst' tokens
    path_remaps: list[tuple[str, str]] = []
    for spec in args.path_remap:
        if "=" not in spec:
            raise SystemExit(f"--path-remap must be 'src=dst', got: {spec}")
        src, dst = spec.split("=", 1)
        path_remaps.append((src.rstrip("/"), dst.rstrip("/")))

    tokenizer, model, image_processor = load_model(args)
    student = VisualUtilityStudent.from_pretrained(args.student_ckpt)
    student = student.to(device).eval()
    print(f"[student] variant={student.variant} layers={len(student.layer_indices)}", flush=True)

    teacher_root = Path(args.teacher_dir)
    samples_out: list[dict[str, Any]] = []
    raw_out: dict[str, dict] = {} if args.save_raw_scores else {}

    for dataset in args.datasets:
        ds_dir = teacher_root / dataset
        if not ds_dir.exists():
            print(f"[skip-dataset] {dataset}: no teacher dir at {ds_dir}", flush=True)
            continue
        pts = sorted(ds_dir.glob("*.pt"))[: args.n_samples]
        print(f"[dataset] {dataset} loading {len(pts)} teacher .pt files", flush=True)

        t0 = time.time()
        for idx, tpath in enumerate(pts, 1):
            rec = torch.load(tpath, map_location="cpu", weights_only=False)
            try:
                attns, hids = run_prefill(
                    model, tokenizer, image_processor, rec, device, path_remaps)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {dataset}/{tpath.stem}: forward failed: {exc}", flush=True)
                continue

            image_indices = rec["image_token_indices"].long()
            question_indices = rec["question_token_indices"].long()
            teacher_norm = rec["teacher_norm"].float()  # [L, N_I]

            # Per-layer score vectors
            h2o_scores = h2o_score_per_layer(attns, image_indices)
            stu_scores = student_score_per_layer(student, hids, image_indices, question_indices, device)
            # LOOK-M needs a K_per_head; use the smallest keep_ratio as reference.
            lookm_scores = lookm_score_per_layer(
                attns, image_indices, keep_ratio=min(args.keep_ratios))
            vflowopt_scores = vflowopt_score_per_layer(attns, image_indices)
            prefixkv_scores = prefixkv_score_per_layer(attns, image_indices)

            # Free GPU before metric loop
            del attns, hids
            torch.cuda.empty_cache()

            n_layers = teacher_norm.shape[0]
            per_layer: dict[str, dict[int, dict[str, float]]] = {
                "h2o_vs_teacher": {}, "student_vs_teacher": {},
                "lookm_vs_teacher": {}, "vflowopt_vs_teacher": {},
                "prefixkv_vs_teacher": {},
            }
            for l in range(n_layers):
                tgt = teacher_norm[l]                             # (N_I,)
                if l in h2o_scores:
                    per_layer["h2o_vs_teacher"][l] = compute_metrics(
                        h2o_scores[l], tgt, args.keep_ratios)
                if l in stu_scores:
                    per_layer["student_vs_teacher"][l] = compute_metrics(
                        stu_scores[l], tgt, args.keep_ratios)
                if l in lookm_scores:
                    per_layer["lookm_vs_teacher"][l] = compute_metrics(
                        lookm_scores[l], tgt, args.keep_ratios)
                if l in vflowopt_scores:
                    per_layer["vflowopt_vs_teacher"][l] = compute_metrics(
                        vflowopt_scores[l], tgt, args.keep_ratios)
                if l in prefixkv_scores:
                    per_layer["prefixkv_vs_teacher"][l] = compute_metrics(
                        prefixkv_scores[l], tgt, args.keep_ratios)

            # Aggregate over layers (mean)
            mean_metrics = {
                k: aggregate_layers(per_layer[k]) if per_layer[k] else {}
                for k in per_layer
            }

            # Global comparison: layer-mean each score vector first, then compute
            # metrics once. This treats teacher and each method as a single global
            # keep ranking per sample (vs current per-layer averaging which can
            # over-credit methods whose per-layer signal aligns with teacher's
            # per-layer signal in correlated ways).
            def _layer_mean(d: dict[int, torch.Tensor]) -> torch.Tensor:
                vs = [v for _, v in sorted(d.items())]
                return torch.stack(vs, dim=0).mean(dim=0) if vs else torch.zeros(0)

            tgt_global = teacher_norm.mean(dim=0)
            global_metrics: dict[str, dict[str, float]] = {}
            score_pools = {
                "h2o_vs_teacher_global": h2o_scores,
                "student_vs_teacher_global": stu_scores,
                "lookm_vs_teacher_global": lookm_scores,
                "vflowopt_vs_teacher_global": vflowopt_scores,
                "prefixkv_vs_teacher_global": prefixkv_scores,
            }
            for key, pool in score_pools.items():
                if not pool:
                    continue
                global_metrics[key] = compute_metrics(
                    _layer_mean(pool), tgt_global, args.keep_ratios)
            mean_metrics.update(global_metrics)

            sample_entry = {
                "sample_id": str(rec["sample_id"]),
                "dataset": dataset,
                "n_img": int(rec["n_img"]),
                "n_layers": int(n_layers),
                "T": int(rec.get("T", 0)),
                "prompt_len_mm": int(rec.get("prompt_len_mm", 0)),
                "decoded": str(rec.get("decoded", "")),
                "answer": str(rec.get("answer", "")),
                "per_layer": {
                    k: {str(li): v for li, v in d.items()} for k, d in per_layer.items()
                },
                "mean": mean_metrics,
            }
            samples_out.append(sample_entry)

            if args.save_raw_scores:
                raw_out[f"{dataset}/{rec['sample_id']}"] = {
                    "h2o": {l: v.tolist() for l, v in h2o_scores.items()},
                    "student": {l: v.tolist() for l, v in stu_scores.items()},
                    "teacher": teacher_norm.numpy().tolist(),
                }

            if idx % 10 == 0 or idx == len(pts):
                rate = idx / max(time.time() - t0, 1e-6)
                print(f"[progress] {dataset} {idx}/{len(pts)}  {rate:.2f}/s", flush=True)

    # Final dataset-level aggregates
    by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for ds in {s["dataset"] for s in samples_out}:
        by_dataset[ds] = {}
        for method in ("h2o_vs_teacher", "student_vs_teacher",
                       "lookm_vs_teacher", "vflowopt_vs_teacher",
                       "prefixkv_vs_teacher",
                       "h2o_vs_teacher_global", "student_vs_teacher_global",
                       "lookm_vs_teacher_global", "vflowopt_vs_teacher_global",
                       "prefixkv_vs_teacher_global"):
            metric_lists: dict[str, list[float]] = {}
            for s in samples_out:
                if s["dataset"] != ds:
                    continue
                for k, v in s["mean"].get(method, {}).items():
                    metric_lists.setdefault(k, []).append(v)
            by_dataset[ds][method] = {
                k: float(np.mean(v)) for k, v in metric_lists.items()
            }

    payload = {
        "config": {
            "teacher_dir": str(teacher_root),
            "datasets": args.datasets,
            "n_samples_requested": args.n_samples,
            "student_ckpt": args.student_ckpt,
            "student_variant": student.variant,
            "keep_ratios": args.keep_ratios,
            "model_path": args.model_path,
            "n_samples_collected": len(samples_out),
        },
        "by_dataset_mean": by_dataset,
        "samples": samples_out,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out_path}  samples={len(samples_out)}", flush=True)

    if args.save_raw_scores:
        raw_path = out_path.with_suffix(".raw.json")
        with raw_path.open("w") as f:
            json.dump(raw_out, f)
        print(f"[saved-raw] {raw_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
