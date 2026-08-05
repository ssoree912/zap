"""Run LLaVA-OneVision on a look_rebuttal-format video annotation.json,
applying LOOK-M's image-only KV-cache eviction (image_keep_ratio, pivot
merge) via the delayed-replay decode path, and write pred.json in the same
format generate.py/evaluate.py expect for the rest of the harness.

Deliberately a separate entry point from generate.py/workers/model_workers.py
(look_rebuttal's existing LLaVA-1.5/InternVL/MobileVLM harness): that harness
vendors its own `llava` package pinned to transformers==4.33.1
(LLaVA-mix_merge_v1), while OneVision needs VFlowOpt's LLaVA-OneVision
checkout + transformers==4.46.0. Both define a top-level `llava` package, so
they cannot coexist on sys.path in the same process -- this script only ever
imports the OneVision one.

Not used: LlavaQwenForCausalLM.generate()/`.forward()` for the pruned decode
steps (see delayed_replay.py for why -- it drops `cache_position`). The
prefill call *does* go through the ordinary wrapped forward(), since that
call's auto-derived cache_position (0-based, no prior cache) is correct.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from image_only_kv_cache import ImageOnlyKVCache_LayerWise, infer_image_token_spans  # noqa: E402
from delayed_replay import forward_one_token_with_kv, greedy_decode_after_prompt_replay  # noqa: E402

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


def reduce_attn_heads_to_kv_heads(attn, num_kv_heads):
    """(bsz, num_attn_heads, q_len, kv_len) -> (bsz, num_kv_heads, q_len, kv_len).

    GQA models (Qwen2: 28 attention heads, 4 KV heads) group attn_heads //
    kv_heads consecutive attention heads per KV head (the inverse of
    `repeat_kv`). ImageOnlyKVCache_LayerWise indexes/gathers per KV head, so
    scores must live in KV-head space before it ever sees them -- LLaVA-1.5
    was MHA (32/32) so this reduction was a no-op there and never existed.
    """
    bsz, num_attn_heads, q_len, kv_len = attn.shape
    assert num_attn_heads % num_kv_heads == 0, (num_attn_heads, num_kv_heads)
    group = num_attn_heads // num_kv_heads
    return attn.view(bsz, num_kv_heads, group, q_len, kv_len).sum(dim=2)


def _reduce_to_hh_score_4d(attn, num_kv_heads):
    """(bsz, num_attn_heads, q_len, kv_len) -> (1, num_kv_heads, 1, kv_len).

    Same head-group reduction as reduce_attn_heads_to_kv_heads, plus the
    batch/query-dim reduction ImageOnlyKVCache_LayerWise._update_hh_score
    does internally (`.sum(0).sum(1)`), done here as keepdim sums instead of
    dim-removal so the result is still rank-4. Feeding *that* into the
    unmodified `_update_hh_score` reproduces its real-attention output
    exactly (summing a size-1 dim is a no-op on values) while only ever
    holding one layer's full (num_attn_heads, q_len, kv_len) tensor at a
    time -- the thing this function is called from a forward hook for.
    """
    reduced = reduce_attn_heads_to_kv_heads(attn, num_kv_heads)
    return reduced.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)


class AttentionScoreHooks:
    """Captures per-layer heavy-hitter scores via forward hooks instead of
    `output_attentions=True` + `outputs.attentions`.

    `outputs.attentions` accumulates every layer's full (bsz, num_attn_heads,
    q_len, kv_len) tensor simultaneously before returning -- at 8 video
    frames (~1.6K tokens) that's a few GB across 28 layers, but at 32 frames
    (~6.3K tokens) it's over 100GB (quadratic in sequence length) and simply
    doesn't fit. A forward hook sees the same per-layer tensor the instant
    it's produced, reduces it to a (1, num_kv_heads, 1, kv_len) score
    immediately, and lets the original multi-GB tensor get freed before the
    next layer even runs -- peak memory is one layer's worth, not 28.

    The hook also overwrites the attention-weights slot in the module's
    return value with None, so `outputs.attentions` (which the model still
    populates when output_attentions=True) only ever holds cheap Nones.
    """

    def __init__(self, model, num_kv_heads):
        self.scores = {}
        self._handles = []
        for layer_idx, layer in enumerate(model.get_model().layers):
            handle = layer.self_attn.register_forward_hook(self._make_hook(layer_idx, num_kv_heads))
            self._handles.append(handle)

    def _make_hook(self, layer_idx, num_kv_heads):
        def hook(module, inputs, output):
            attn_output, attn_weights, past_key_value = output
            if attn_weights is not None:
                self.scores[layer_idx] = _reduce_to_hh_score_4d(attn_weights.detach(), num_kv_heads)
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


class OneVisionLookMWorker:
    def __init__(
        self,
        pretrained,
        device="cuda:0",
        attn_implementation="eager",
        image_keep_ratio=0.1,
        merge_strategy="pivot",
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
            # load_pretrained_model hardcodes torch_dtype=float16. Eager attention
            # here does a raw QK^T matmul without SDPA's internal stabilization;
            # this model's hidden-state norms exceed 400 by the last layers, and
            # squaring that in fp16 overflows to NaN in the final RMSNorm. bf16's
            # wider dynamic range avoids it. (SDPA doesn't hit this because its
            # fused kernel never materializes the raw fp16 product this way, but
            # SDPA also can't return real attention weights, which is what we need.)
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.config = self.model.config
        self.num_attn_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        assert self.num_attn_heads % self.num_kv_heads == 0
        self.num_layers = self.config.num_hidden_layers
        self.image_keep_ratio = image_keep_ratio
        self.merge_strategy = merge_strategy
        self.max_new_tokens = max_new_tokens
        self.conv_template = conv_template
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def _prepare_video(self, frame_paths):
        # Deliberately NOT process_images(): that dispatches on
        # config.image_aspect_ratio ("anyres_max_9" here), which splits each
        # frame into a base image + grid sub-patches meant for single-image
        # anyres mode. Video frames go through the vision tower directly at
        # its native resolution instead, matching the official OneVision
        # video_demo.py (`image_processor.preprocess(video)["pixel_values"]`)
        # -- prepare_inputs_labels_for_multimodal's video path (get_2dPool)
        # expects one flat feature map per frame, not an anyres patch stack.
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

        with AttentionScoreHooks(self.model, self.num_kv_heads) as hooks:
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
            for layer_idx in range(self.num_layers):
                key = past_kv.key_cache[layer_idx]
                value = past_kv.value_cache[layer_idx]
                attn_kv = layer_scores[layer_idx]
                cache_obj = ImageOnlyKVCache_LayerWise(
                    image_keep_ratio=self.image_keep_ratio, merge_strategy=self.merge_strategy
                )
                cache_obj.image_token_spans = spans
                pruned_key, pruned_value = cache_obj((key, value), attn_kv)
                past_kv.key_cache[layer_idx] = pruned_key
                past_kv.value_cache[layer_idx] = pruned_value
                if layer_idx == 0:
                    stats["image_tokens_before"] = cache_obj.last_image_token_count
                    stats["image_tokens_kept"] = cache_obj.last_image_keep_count
                    stats["text_tokens"] = cache_obj.last_text_token_count
                    stats["pruned"] = True
            new_len = past_kv.get_seq_length()
            stats["pruned_prompt_len"] = int(new_len)

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
    parser.add_argument("--output_dir", required=True, help="e.g. outputs/text_prior_pivot_merge_0.1/Video-MME")
    parser.add_argument("--pretrained", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--image_keep_ratio", type=float, default=0.1)
    parser.add_argument("--merge_strategy", default="pivot", choices=["none", "avg", "weighted", "pivot"])
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
    worker = OneVisionLookMWorker(
        pretrained=args.pretrained,
        device=args.device,
        attn_implementation=args.attn_implementation,
        image_keep_ratio=args.image_keep_ratio,
        merge_strategy=args.merge_strategy,
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
            print(f"[generate_onevision] ERROR sample_id={sample['sample_id']}: {exc}", file=sys.stderr, flush=True)
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
            keep_ratios_seen.append(stats["image_tokens_kept"] / stats["image_tokens_before"])

        if (i + 1) % args.log_every == 0 or i == 0:
            elapsed = time.time() - t0
            avg_kr = sum(keep_ratios_seen) / len(keep_ratios_seen) if keep_ratios_seen else float("nan")
            print(
                f"[generate_onevision] {i+1}/{len(samples)} elapsed={elapsed:.1f}s "
                f"avg_image_keep_ratio={avg_kr:.4f} last_stats={stats}",
                flush=True,
            )

    with open(os.path.join(args.output_dir, "pred.json"), "w") as f:
        json.dump(preds, f)
    print(f"[generate_onevision] wrote {len(preds)} predictions -> {args.output_dir}/pred.json")


if __name__ == "__main__":
    main()
