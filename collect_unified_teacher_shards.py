#!/usr/bin/env python3
"""Collect unified teacher shards (PostVision + Future) from ONE LLaVA forward pass.

Per-row shard schema (method B, per-sample softmax-compatible):
  x         [R, D]   float16   hidden at image position
  y_pv      [R]      float16   post-vision text → image attention (max over Q, mean over H)
  y_future  [R]      float16   decode → image attention (mean over H, mean over T)
  layer     [R]      uint8
  sample_id [R]      int32     unique per sample
  token_idx [R]      int16     image token index within sample (0..N_image-1)

Two teacher MLPs (PostVision vs Future) can be trained on the same shards by
choosing y_pv or y_future — identical loss formulation (per-sample softmax MSE)
and identical (x, layer, sample_id) grouping.

Usage:
  python collect_unified_teacher_shards.py \\
      --dataset textvqa --data_dir /workspace/zap/data/textvqa/train \\
      --out_dir /workspace/zap/artifacts/teacher/unified/textvqa \\
      --limit 500 --device cuda:0 --max_new_tokens 64
"""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from kvzap.llava_extractor import (
    LlavaAnalysisDataCollector,
    _parse_torch_dtype,
    configure_llava_processor,
)


# ── shard writer ────────────────────────────────────────────────────────────

@dataclass
class UnifiedShardWriter:
    output_dir: Path
    max_rows_per_shard: int = 50000
    input_dim: int | None = field(default=None, init=False)
    n_layers: int | None = field(default=None, init=False)
    _xs: list[torch.Tensor] = field(default_factory=list, init=False)
    _y_pv: list[torch.Tensor] = field(default_factory=list, init=False)
    _y_fu: list[torch.Tensor] = field(default_factory=list, init=False)
    _layer: list[torch.Tensor] = field(default_factory=list, init=False)
    _sid: list[torch.Tensor] = field(default_factory=list, init=False)
    _tok: list[torch.Tensor] = field(default_factory=list, init=False)
    _n_buf: int = field(default=0, init=False)
    _shard_idx: int = field(default=0, init=False)
    _total_rows: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        (self.output_dir / "shards").mkdir(parents=True, exist_ok=True)

    def add_sample(
        self,
        hidden_image: torch.Tensor,   # [L, N_image, D] float16
        y_pv: torch.Tensor,            # [L, N_image] float16
        y_future: torch.Tensor,        # [L, N_image] float16
        sample_id_int: int,
    ) -> None:
        L, N, D = hidden_image.shape
        if self.input_dim is None:
            self.input_dim = D
            self.n_layers = L
        assert L == self.n_layers and D == self.input_dim

        layer_ids = torch.arange(L, dtype=torch.uint8).view(L, 1).expand(L, N).reshape(-1)
        token_ids = torch.arange(N, dtype=torch.int16).view(1, N).expand(L, N).reshape(-1)
        sid = torch.full((L * N,), sample_id_int, dtype=torch.int32)

        self._xs.append(hidden_image.reshape(L * N, D).to(torch.float16).cpu())
        self._y_pv.append(y_pv.reshape(L * N).to(torch.float16).cpu())
        self._y_fu.append(y_future.reshape(L * N).to(torch.float16).cpu())
        self._layer.append(layer_ids)
        self._tok.append(token_ids)
        self._sid.append(sid)
        self._n_buf += L * N

        if self._n_buf >= self.max_rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if self._n_buf == 0:
            return
        payload = {
            "x": torch.cat(self._xs, dim=0),
            "y_pv": torch.cat(self._y_pv, dim=0),
            "y_future": torch.cat(self._y_fu, dim=0),
            "layer": torch.cat(self._layer, dim=0),
            "token_idx": torch.cat(self._tok, dim=0),
            "sample_id": torch.cat(self._sid, dim=0),
        }
        path = self.output_dir / "shards" / f"shard_{self._shard_idx:04d}.pt"
        torch.save(payload, path)
        self._shard_idx += 1
        self._total_rows += self._n_buf
        self._xs.clear(); self._y_pv.clear(); self._y_fu.clear()
        self._layer.clear(); self._tok.clear(); self._sid.clear()
        self._n_buf = 0


# ── label extraction ────────────────────────────────────────────────────────

def _resolve_image_indices(record: dict[str, Any]) -> torch.Tensor:
    if "image_indices_mm" in record:
        return record["image_indices_mm"].long().flatten()
    if "is_image_pos_mm" in record:
        return record["is_image_pos_mm"].nonzero(as_tuple=False).flatten().long()
    raise KeyError("Record missing image_indices_mm / is_image_pos_mm")


def _resolve_postvision_text_indices(record: dict[str, Any]) -> torch.Tensor:
    image_idx = _resolve_image_indices(record)
    last_image = int(image_idx.max().item())
    prompt_len = int(record["prompt_len_mm"])
    is_text = ~record["is_image_pos_mm"]
    all_pos = torch.arange(prompt_len, dtype=torch.long)
    return all_pos[(all_pos > last_image) & is_text]


def compute_unified_labels(
    record: dict[str, Any],
    full_attentions: list[torch.Tensor],
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (hidden_image [L, N_image, D], y_pv [L, N_image], y_future [L, N_image]).

    Requires FULL prefill-prefill attention saved in `full_attentions`
    (list[L] of [H, full_len, full_len]).  The existing Future record only has
    `attn_answer_to_prompt` (decode→prompt), so PV label requires the full matrix.
    """
    hidden_list = record["hidden_prompt"]
    attn_decode_list = record["attn_answer_to_prompt"]
    n_layers = len(attn_decode_list)
    assert len(full_attentions) == n_layers, "full_attentions length mismatch"

    image_idx = _resolve_image_indices(record)
    pv_text_idx = _resolve_postvision_text_indices(record)
    if image_idx.numel() == 0 or pv_text_idx.numel() == 0:
        raise ValueError("empty image or post-vision text index set")

    hidden_offset = 1 if len(hidden_list) == n_layers + 1 else 0

    hid_list, ypv_list, yfu_list = [], [], []
    for l in range(n_layers):
        # hidden at image positions
        h = hidden_list[l + hidden_offset].float()
        hid_list.append(h[image_idx].to(storage_dtype))

        # PV: prefill-attn[post_vision_text, image], max over Q then mean over H
        attn_l = full_attentions[l].float()  # [H, full_len, full_len]
        pv_block = attn_l[:, pv_text_idx, :][:, :, image_idx]  # [H, Q, N_image]
        y_pv = pv_block.max(dim=1).values.mean(dim=0)          # [N_image]
        ypv_list.append(y_pv.to(storage_dtype))

        # Future: decode-attn[T, image], mean over H and T
        attn_dec = attn_decode_list[l].float()                  # [H, T, prompt_len]
        fu_block = attn_dec[:, :, image_idx]                    # [H, T, N_image]
        y_fu = fu_block.mean(dim=0).mean(dim=0)                 # [N_image]
        yfu_list.append(y_fu.to(storage_dtype))

    return (
        torch.stack(hid_list, dim=0),   # [L, N_image, D]
        torch.stack(ypv_list, dim=0),   # [L, N_image]
        torch.stack(yfu_list, dim=0),   # [L, N_image]
    )


# ── augmented forward that captures full prefill attentions ─────────────────

class UnifiedCollector(LlavaAnalysisDataCollector):
    """Runs one full LLaVA pass and extracts BOTH PV + Future labels on-GPU.

    Avoids storing full [H, full_len, full_len] attention on CPU (can be 6-8GB/sample).
    """

    def collect_sample_unified(
        self,
        sample: dict[str, Any],
        prompt_template: str,
        max_new_tokens: int = 64,
        storage_dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        from kvzap.llava_extractor import (
            _build_prompt, _open_images, _move_batch_to_device, _trim_after_eos,
            _get_visual_inputs, _resolve_prompt_image_mask,
            enable_merge_trace, disable_merge_trace, register_analysis_hooks, remove_analysis_hooks,
        )

        images = _open_images(sample["image_paths"])
        prompt_text = _build_prompt(sample, prompt_template, image_count=len(sample["image_paths"]))
        prompt_inputs = self.processor(text=prompt_text, images=images, return_tensors="pt")
        prompt_inputs = _move_batch_to_device(prompt_inputs, self.device, self.float_dtype)
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_len_text = int(prompt_input_ids.shape[1])

        trace_ctx = enable_merge_trace(self.model)
        try:
            with torch.no_grad():
                prompt_out = self.model(
                    **prompt_inputs, use_cache=False,
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
        finally:
            disable_merge_trace(trace_ctx)

        prompt_len_mm = int(prompt_out.logits.shape[1])
        is_image_pos_mm = _resolve_prompt_image_mask(self.model, prompt_input_ids, prompt_len_mm, trace_ctx)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **prompt_inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            )
        full_ids_text = _trim_after_eos(generated_ids[0], self.eos_token_id)
        analysis_input_ids = full_ids_text.unsqueeze(0).to(self.device)
        analysis_inputs = {
            "input_ids": analysis_input_ids,
            "attention_mask": torch.ones_like(analysis_input_ids),
            **_get_visual_inputs(prompt_inputs),
        }

        hook_ctx = register_analysis_hooks(self.model, capture_vproj=False)
        try:
            with torch.no_grad():
                full_out = self.model(
                    **analysis_inputs, use_cache=False,
                    output_attentions=True, output_hidden_states=True, return_dict=True,
                )
        finally:
            remove_analysis_hooks(hook_ctx)

        full_len_mm = int(full_out.hidden_states[0].shape[1])

        image_idx = is_image_pos_mm.nonzero(as_tuple=False).flatten().long().to(self.device)
        is_text_prompt_pos_mm = ~is_image_pos_mm
        last_image = int(image_idx.max().item())
        all_pos = torch.arange(prompt_len_mm, dtype=torch.long, device=self.device)
        pv_text_idx = all_pos[(all_pos > last_image) & is_text_prompt_pos_mm.to(self.device)]
        if pv_text_idx.numel() == 0 or image_idx.numel() == 0:
            raise ValueError("no image or post-vision text tokens")

        n_layers = len(full_out.attentions)
        hid_layers, ypv_layers, yfu_layers = [], [], []

        decode_start = prompt_len_mm
        decode_end = full_len_mm
        hidden_offset = 1 if len(full_out.hidden_states) == n_layers + 1 else 0

        for l in range(n_layers):
            attn = full_out.attentions[l][0]       # [H, full_len, full_len]
            h_l = full_out.hidden_states[l + hidden_offset][0, :prompt_len_mm, :]  # [prompt_len, D]

            # hidden at image positions
            h_img = h_l.index_select(0, image_idx).to(storage_dtype)

            # PV: prefill-attn[post_vision_text → image], max over Q then mean over H
            pv_block = attn[:, pv_text_idx, :].index_select(2, image_idx)  # [H, Q, N_image]
            y_pv = pv_block.max(dim=1).values.mean(dim=0).to(storage_dtype)

            # Future: decode-attn[T → image], mean over H and T
            fu_block = attn[:, decode_start:decode_end, :].index_select(2, image_idx)  # [H, T, N_image]
            y_fu = fu_block.mean(dim=0).mean(dim=0).to(storage_dtype)

            hid_layers.append(h_img.cpu())
            ypv_layers.append(y_pv.cpu())
            yfu_layers.append(y_fu.cpu())

        del full_out  # release GPU memory
        torch.cuda.empty_cache()

        return {
            "hidden_image": torch.stack(hid_layers, dim=0),   # [L, N_image, D]
            "y_pv": torch.stack(ypv_layers, dim=0),            # [L, N_image]
            "y_future": torch.stack(yfu_layers, dim=0),        # [L, N_image]
            "sample_id": sample["sample_id"],
        }


# ── dataset loaders ─────────────────────────────────────────────────────────

TEXTVQA_TEMPLATE = "USER: <image>\nQuestion: {question}\nASSISTANT:"
SCIENCEQA_TEMPLATE = "USER: <image>\n{prompt_body}\nASSISTANT:"
NLVR2_TEMPLATE = (
    "USER: <image>\n<image>\n"
    "Statement: {sentence}\n"
    "Is this statement True or False?\n"
    "Options:\nA. True\nB. False\n"
    "ASSISTANT:"
)
DATASET_TEMPLATES = {"textvqa": TEXTVQA_TEMPLATE, "nlvr2": NLVR2_TEMPLATE, "scienceqa": SCIENCEQA_TEMPLATE}


def _load_json_samples(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    records = json.load((data_dir / "data.json").open())
    if limit:
        records = records[:limit]
    img_root = data_dir.parent.parent
    out = []
    for rec in records:
        sid = str(rec.get("question_id", rec.get("sample_id", rec.get("identifier", len(out)))))
        question = str(rec.get("question", rec.get("sentence", ""))).strip()
        raw_paths = rec.get("image_paths") or [rec["image_path"]]
        resolved = [str(Path(p) if Path(p).is_absolute() else (img_root / p)) for p in raw_paths]
        out.append({
            "sample_id": sid, "question": question, "sentence": question,
            "prompt_body": question, "image_paths": resolved,
        })
    return out


def _load_scienceqa_samples(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    from build_scienceqa_manifest import iter_scienceqa_samples
    OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]
    out = []
    for raw in iter_scienceqa_samples(base_dir=data_dir, split="train", limit=limit):
        hint, question, choices = raw.get("hint", ""), raw.get("question", ""), raw.get("choices", [])
        lines = ([f"Context: {hint}"] if hint else []) + [f"Question: {question}", "Options:"]
        for i, c in enumerate(choices):
            lines.append(f"{OPTION_LETTERS[i]}. {c}")
        lines.append("Select the best answer based on the image and text.")
        out.append({
            "sample_id": raw["sample_id"], "question": question, "sentence": question,
            "prompt_body": "\n".join(lines), "image_paths": [raw["image_path"]],
        })
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASET_TEMPLATES), required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        import shutil
        if (out_dir / "shards").exists():
            shutil.rmtree(out_dir / "shards")

    dtype = _parse_torch_dtype(args.torch_dtype)
    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map=args.device, attn_implementation="eager",
    )
    configure_llava_processor(processor, model.config)
    model.eval()

    data_dir = Path(args.data_dir).resolve()
    if args.dataset == "scienceqa":
        samples = _load_scienceqa_samples(data_dir, args.limit)
    else:
        samples = _load_json_samples(data_dir, args.limit)
    prompt_template = DATASET_TEMPLATES[args.dataset]

    print(f"Samples: {len(samples)}, T_max: {args.max_new_tokens}")

    collector = UnifiedCollector(model, processor)
    writer = UnifiedShardWriter(output_dir=out_dir, max_rows_per_shard=args.shard_size)

    stats = {"saved": 0, "failed": 0}
    for sample_int_id, sample in enumerate(tqdm(samples, desc=f"{args.dataset} unified collect")):
        try:
            record = collector.collect_sample_unified(
                sample=sample, prompt_template=prompt_template, max_new_tokens=args.max_new_tokens,
            )
            writer.add_sample(record["hidden_image"], record["y_pv"], record["y_future"], sample_int_id)
            stats["saved"] += 1
        except Exception as exc:
            print(f"[WARN] sample {sample.get('sample_id')} failed: {exc}")
            traceback.print_exc()
            stats["failed"] += 1

    writer.flush()
    summary = {
        "dataset": args.dataset, "data_dir": str(data_dir),
        "n_samples_requested": len(samples), **stats,
        "n_shards_written": writer._shard_idx,
        "n_rows_written": writer._total_rows,
        "input_dim": writer.input_dim, "n_layers": writer.n_layers,
    }
    (out_dir / "validation_report.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
