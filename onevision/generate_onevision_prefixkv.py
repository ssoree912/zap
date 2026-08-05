"""Run LLaVA-OneVision on a look_rebuttal-format video annotation.json,
applying PrefixKV's image-only KV-cache eviction (image_keep_ratio) via the
same delayed-replay decode path used for LOOK-M and the student probe, and
write pred.json in the format generate.py/evaluate.py expect.

PrefixKV (/workspace/zap/prefix_rebuttal/prefixkv.py) differs from LOOK-M in
two ways that matter for the writeup, not in anything that required a code
change here:

1. Per-layer budget allocation. PrefixKV's actual contribution is a
   profiled, non-uniform forget-ratio per layer (`layer_forget_ratios`,
   loaded from confs/prefixkv_<model>_<ratio>.json). No such profile exists
   for llava-onevision-qwen2-7b-ov (only llava-v1.5-{7b,13b} are profiled),
   so this run passes `layer_forget_ratios=np.ones(num_layers)` explicitly --
   a uniform per-layer budget, i.e. the profiled contribution is inactive.
   Passing it explicitly (rather than leaving it None) also skips
   PrefixKV's own conf-file lookup entirely, so there's no silent
   dependence on whether a file happens to exist.
2. Scoring window and head handling, even with uniform per-layer budget:
   - PrefixKv ranks image keys by attention from TEXT queries only
     (attentions[layer, :, ~image_mask, :][..., image_indices]); LOOK-M's
     heavy-hitter score sums attention from ALL queries, including
     image->image.
   - PrefixKV keeps one shared kept-index set per layer (attention averaged
     over all query heads first). LOOK-M's ImageOnlyKVCache_LayerWise keeps
     a separate kept-index set per KV head (hh_score has a head dimension,
     topk is per-head). This is a structural difference from LOOK-M's
     per-head eviction, not a rounding artifact.

Exact per-layer keep counts, not just an average ratio, are required here:
PrefixKV's `_compress_image_tokens` calls `_allocate_integer_budget` across
all layers for a *global* image-token keep budget
(image_count * image_keep_ratio * num_layers total kept KVs), then splits
that budget across layers by `layer_forget_ratios`. With uniform ratios and
image_keep_ratio picked without care, the largest-remainder allocation
inside `_allocate_integer_budget` hands out the +1 remainder tokens to only
some layers (whichever the stable sort favors), so different layers end up
keeping different numbers of image tokens -- e.g. 22 layers keep 157, 6
keep 156 image tokens for a representative 8-frame prompt at ratio 0.1.
DynamicCache.get_seq_length() only reads layer 0's length, so the replay
step's causal mask would be sized for 157+text tokens while some layers
actually hold 156+text -- attn_weights/mask shapes wouldn't broadcast.
milebench_onevision_prefixkv.py hits exactly this and "fixes" it by
truncating every layer's cache to the shortest layer's length, which
chops tokens off the *end* of each layer's sorted keep-list -- for an
image-only cache the tail is prompt text ("...Your answer is: "), so that
truncation silently drops text tokens and would corrupt the very last
token before the delayed-replay step reads it. Not applicable to
image-only mode, and not used here.

The actual fix (see `image_keep_ratio` computation in `generate()` below):
round the image keep-count to an exact integer first
(`keep_per_layer = round(image_count * image_keep_ratio)`), then derive
`image_keep_ratio` from that integer
(`keep_per_layer / image_count`) before constructing PrefixKV. This makes
the global keep budget (`image_count * image_keep_ratio * num_layers`) an
exact multiple of num_layers, so `_allocate_integer_budget`'s
largest-remainder pass is a no-op and every layer keeps exactly
`keep_per_layer` image tokens. Verified layer-length-equality is asserted
after every compression call, not just assumed.

See generate_onevision.py's module docstring for why this is a separate
entry point from look_rebuttal's LLaVA-1.5/InternVL/MobileVLM harness
(transformers version conflict via the vendored `llava` package), and
delayed_replay.py for why the pruned decode steps bypass
LlavaQwenForCausalLM.generate()/.forward().
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from image_only_kv_cache import infer_image_token_spans  # noqa: E402
from delayed_replay import forward_one_token_with_kv, greedy_decode_after_prompt_replay  # noqa: E402

sys.path.insert(0, "/workspace/zap/prefix_rebuttal")
from prefixkv import PrefixKV  # noqa: E402

ONEVISION_ROOT = Path(
    os.environ.get("VFLOWOPT_LLAVA_ROOT", "/workspace/VFlowOpt_llava1.5/src/LLaVA-OneVision")
).resolve()
TRANSFORMERS_ROOT = Path(
    os.environ.get("VFLOWOPT_TRANSFORMERS_ROOT", "/workspace/VFlowOpt_llava1.5/src/transformers-4.46.0/src")
).resolve()
for _p in (str(TRANSFORMERS_ROOT), str(ONEVISION_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import DynamicCache  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


class PrefixKVAttentionHooks:
    """Captures per-layer attention, reduced to PrefixKV's required shape,
    via forward hooks -- same rationale as generate_onevision.py's
    AttentionScoreHooks (outputs.attentions holding all 28 layers' full
    (bsz, num_attn_heads, q_len, kv_len) tensors at once is ~100GB+ at 32
    video frames; a hook sees each layer's tensor the instant it's produced
    and reduces it before the next layer even runs).

    Unlike AttentionScoreHooks (which reduces straight to a heavy-hitter
    score vector for ImageOnlyKVCache_LayerWise), PrefixKV's
    `_compress_image_tokens` needs the full (seq_len, seq_len) attention
    matrix per layer -- it does its own text-query-row / image-key-column
    slicing internally. So this hook only reduces the head dimension
    (mean over all query heads, matching what `_compress_image_tokens`
    itself does via `attention.mean(dim=1)`), keeping a size-1 head dim
    (not squeezed) so that `_compress_image_tokens`'s own `.mean(dim=1)`
    call downstream is a no-op on an already-reduced tensor rather than
    incorrectly averaging over the sequence dimension. No GQA head-group
    reduction is needed here (unlike LOOK-M's port): attention weights come
    out of the model in query-head space (28 for this model) regardless of
    the 4 KV heads, and PrefixKV never indexes per-KV-head -- it keeps one
    shared kept-index set per layer -- so a plain mean over all 28 query
    heads is exactly what the unmodified downstream code expects.
    """

    def __init__(self, model):
        self.scores = {}
        self._handles = []
        for layer_idx, layer in enumerate(model.get_model().layers):
            handle = layer.self_attn.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            attn_output, attn_weights, past_key_value = output
            if attn_weights is not None:
                # float() before mean: fp32 accumulation, not bf16-then-cast.
                # keepdim=True: stays rank-4 so downstream .mean(dim=1) no-ops.
                # .cpu(): let the multi-GB per-head tensor free before the
                # next layer runs instead of accumulating 28 of them on GPU.
                reduced = attn_weights.detach().float().mean(dim=1, keepdim=True)
                self.scores[layer_idx] = reduced.cpu()
            return (attn_output, None, past_key_value)

        return hook

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def build_prompt(question, conv_template):
    import copy

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class OneVisionPrefixKVWorker:
    def __init__(
        self,
        pretrained,
        device="cuda:0",
        attn_implementation="eager",
        image_keep_ratio=0.1,
        max_new_tokens=64,
        conv_template="qwen_1_5",
    ):
        self.device = torch.device(device)
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained,
            None,
            "llava_qwen",
            device_map=device,
            attn_implementation=attn_implementation,
        )
        if attn_implementation == "eager":
            # See generate_onevision.py: eager attention's raw QK^T matmul
            # overflows fp16 to NaN by the late layers on this model; bf16
            # avoids it and SDPA can't return real attention weights anyway.
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.image_keep_ratio = image_keep_ratio
        self.max_new_tokens = max_new_tokens
        self.conv_template = conv_template
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id
        # Only used for PrefixKV's conf-file lookup, which this run bypasses
        # by passing layer_forget_ratios explicitly -- kept descriptive for
        # any diagnostic printing PrefixKV does, not functionally load-bearing.
        self.model_name_for_conf = "llava-onevision-qwen2-7b-ov"

    def _prepare_video(self, frame_paths):
        # See generate_onevision.py: deliberately NOT process_images().
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        image_sizes = [img.size for img in images]
        video_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(dtype=next(self.model.parameters()).dtype, device=self.device)
        return [video_tensor], image_sizes

    @torch.no_grad()
    def generate(self, question, frame_paths, no_pruning=False):
        prompt = build_prompt(question, self.conv_template)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        images, image_sizes = self._prepare_video(frame_paths)
        modalities = ["video"]

        if no_pruning or self.image_keep_ratio >= 1.0:
            out = self.model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            gen_ids = out[0][input_ids.shape[1] :] if out.shape[1] > input_ids.shape[1] else out[0]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return text, {"pruned": False, "path": "stock_generate"}

        prefill_ids = input_ids[:, :-1].contiguous()
        replay_ids = input_ids[:, -1:].contiguous()

        with PrefixKVAttentionHooks(self.model) as hooks:
            prefill = self.model(
                input_ids=prefill_ids,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
                past_key_values=DynamicCache(),
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
            layer_scores = hooks.scores
        past_kv = prefill.past_key_values
        prompt_len = past_kv.get_seq_length()

        stats = {"pruned": False, "path": "delayed_replay", "prompt_len": int(prompt_len)}
        image_positions_in_raw = torch.where(prefill_ids[0] == IMAGE_TOKEN_INDEX)[0]
        if image_positions_in_raw.numel() > 0:
            spans = infer_image_token_spans(
                prefill_ids, expanded_seq_len=prompt_len, image_token_index=IMAGE_TOKEN_INDEX
            )
            image_token_ranges = [(int(s), int(e)) for s, e in spans.tolist()]
            image_count = sum(e - s for s, e in image_token_ranges)

            # Round the keep-count to an exact integer *before* building the
            # ratio PrefixKV sees, so image_count * ratio * num_layers is an
            # exact multiple of num_layers and every layer keeps the same
            # number of image tokens. See module docstring.
            keep_per_layer = round(image_count * self.image_keep_ratio)
            effective_ratio = keep_per_layer / image_count

            kv_list = [(past_kv.key_cache[i], past_kv.value_cache[i]) for i in range(self.num_layers)]
            attn_list = [layer_scores[i] for i in range(self.num_layers)]

            cache_obj = PrefixKV(
                model_name=self.model_name_for_conf,
                layer_num=self.num_layers,
                batch_size=1,
                image_keep_ratio=effective_ratio,
                layer_forget_ratios=np.ones(self.num_layers),
            )
            compressed = cache_obj(kv_list, attentions=attn_list, image_token_ranges=image_token_ranges)

            layer_lens = set()
            for layer_idx, (key, value) in enumerate(compressed):
                past_kv.key_cache[layer_idx] = key
                past_kv.value_cache[layer_idx] = value
                layer_lens.add(key.shape[2])
            if len(layer_lens) != 1:
                raise RuntimeError(
                    f"PrefixKV produced unequal per-layer KV lengths: {sorted(layer_lens)}"
                )

            new_len = past_kv.get_seq_length()
            stats.update(
                {
                    "pruned": True,
                    "image_tokens_before": image_count,
                    "image_tokens_kept": cache_obj.last_stats["image_tokens_kept_per_layer"][0],
                    "effective_image_keep_ratio": cache_obj.last_stats["effective_image_keep_ratio"],
                    "pruned_prompt_len": int(new_len),
                }
            )

        del prefill, layer_scores
        gc.collect()
        torch.cuda.empty_cache()

        replay_logits, replay_kv = forward_one_token_with_kv(
            self.model, replay_ids, past_kv, absolute_position=prompt_len
        )
        first_token = replay_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out_tokens = greedy_decode_after_prompt_replay(
            self.model,
            replay_kv,
            first_token,
            prompt_len=prompt_len + 1,
            eos_token_id=self.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        text = self.tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        torch.cuda.empty_cache()
        return text, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, help="path to <dataset>.json (core annotation)")
    parser.add_argument("--output_dir", required=True, help="e.g. outputs/onevision_prefixkv_0.1/Video-MME")
    parser.add_argument("--pretrained", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--image_keep_ratio", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--conv_template", default="qwen_1_5")
    parser.add_argument("--no_pruning", action="store_true", help="stock model.generate(), no eviction (smoke test)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    with open(args.annotation) as f:
        core = json.load(f)
    samples = core["data"]
    if args.limit:
        samples = samples[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    worker = OneVisionPrefixKVWorker(
        pretrained=args.pretrained,
        device=args.device,
        attn_implementation=args.attn_implementation,
        image_keep_ratio=args.image_keep_ratio,
        max_new_tokens=args.max_new_tokens,
        conv_template=args.conv_template,
    )

    preds = []
    keep_ratios_seen = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        ti = sample["task_instance"]
        question = ti["context"]
        choice_str = "\nChoice list:\n" + "\n".join(
            f"{chr(65+j)}. {c}" for j, c in enumerate(ti["choice_list"])
        ) + "\nYour answer is: "
        full_question = question + choice_str
        try:
            pred_text, stats = worker.generate(
                full_question, ti["images_path"], no_pruning=args.no_pruning
            )
        except Exception as exc:  # keep going; a single bad sample shouldn't kill a multi-hour run
            print(f"[generate_onevision_prefixkv] ERROR sample_id={sample['sample_id']}: {exc}", file=sys.stderr, flush=True)
            pred_text = ""
            stats = {"error": str(exc)}

        preds.append(
            {
                "sample_id": sample["sample_id"],
                "pred_response": pred_text,
                "gt_response": sample["response"],
            }
        )
        if stats.get("pruned") and stats.get("image_tokens_before"):
            keep_ratios_seen.append(stats["effective_image_keep_ratio"])

        if (i + 1) % args.log_every == 0 or i == 0:
            elapsed = time.time() - t0
            avg_kr = sum(keep_ratios_seen) / len(keep_ratios_seen) if keep_ratios_seen else float("nan")
            print(
                f"[generate_onevision_prefixkv] {i+1}/{len(samples)} elapsed={elapsed:.1f}s "
                f"avg_image_keep_ratio={avg_kr:.4f} last_stats={stats}",
                flush=True,
            )

    with open(os.path.join(args.output_dir, "pred.json"), "w") as f:
        json.dump(preds, f)
    print(f"[generate_onevision_prefixkv] wrote {len(preds)} predictions -> {args.output_dir}/pred.json")


if __name__ == "__main__":
    main()
