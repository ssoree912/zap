"""Run LLaVA-OneVision on video using zap's own trained visual-utility
student probe (image-token importance predicted from hidden states, not
attention) for KV eviction, via the same delayed-replay decode path as
generate_onevision.py: full prefill (scoring) -> image KV eviction -> replay
the held-out last prompt token -> first answer token -> continue decode.

Mirrors zap/foresight/eval/lmms_onevision_original_student.py's
delayed_replay=True path, but:
- talks to our own video frame prep / dataset / evaluate.py-compatible
  output instead of going through lmms-eval, and
- fixes the same two bugs generate_onevision.py needed for real video input
  (process_images()'s anyres split is wrong for pooled video frames; a bare
  ["image"]*N modalities list skips OneVision's 2D pooling) -- the original
  zap script has both, since it was written for MileBench multi-image, not
  true video.

The student scores at the hidden-state level (one score per token, shared
across heads), not per-attention-head like LOOK-M's H2O ranking, so no GQA
head-reduction and no per-head keep_indices: one boolean mask per layer,
applied identically to every KV head. No SDPA/eager conflict either --
hidden_states don't need output_attentions, so this loads fp16+sdpa (fast,
no bf16 workaround needed).
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

from image_only_kv_cache import infer_image_token_spans  # noqa: E402
from delayed_replay import forward_one_token_with_kv, greedy_decode_after_prompt_replay  # noqa: E402

ZAP_ROOT = Path(os.environ.get("ZAP_REPO_ROOT", "/workspace/zap")).resolve()
ONEVISION_ROOT = Path(
    os.environ.get("VFLOWOPT_LLAVA_ROOT", "/workspace/VFlowOpt_llava1.5/src/LLaVA-OneVision")
).resolve()
TRANSFORMERS_ROOT = Path(
    os.environ.get("VFLOWOPT_TRANSFORMERS_ROOT", "/workspace/VFlowOpt_llava1.5/src/transformers-4.46.0/src")
).resolve()
for _p in (str(TRANSFORMERS_ROOT), str(ONEVISION_ROOT), str(ZAP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import DynamicCache  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision  # noqa: E402
from foresight.eval.keep_budget import image_keep_budget, normalize_keep_ratio_basis  # noqa: E402


def build_prompt(question, conv_template):
    import copy

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class OneVisionStudentWorker:
    def __init__(
        self,
        pretrained,
        student_path,
        device="cuda:0",
        attn_implementation="sdpa",
        keep_ratio=0.1,
        keep_ratio_basis="image",
        max_new_tokens=32,
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
        self.model.eval()
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers

        self.student = VisualUtilityStudentOneVision.from_pretrained(student_path, map_location="cpu")
        self.student = self.student.to(device=self.device, dtype=torch.float16).eval()

        self.keep_ratio = float(keep_ratio)
        self.keep_ratio_basis = normalize_keep_ratio_basis(keep_ratio_basis)
        self.max_new_tokens = max_new_tokens
        self.conv_template = conv_template
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def _prepare_video(self, frame_paths):
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

        if no_pruning or self.keep_ratio >= 1.0:
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

        prefill = self.model(
            input_ids=prefill_ids,
            images=images,
            image_sizes=image_sizes,
            modalities=modalities,
            past_key_values=DynamicCache(),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv = prefill.past_key_values
        prompt_len = past_kv.get_seq_length()
        H_all = prefill.hidden_states  # len = num_layers + 1 (embed output first)

        stats = {"pruned": False, "path": "delayed_replay", "prompt_len": int(prompt_len)}
        image_positions_in_raw = torch.where(prefill_ids[0] == IMAGE_TOKEN_INDEX)[0]
        if image_positions_in_raw.numel() > 0:
            spans = infer_image_token_spans(
                prefill_ids, expanded_seq_len=prompt_len, image_token_index=IMAGE_TOKEN_INDEX
            )
            start, end = int(spans[0, 0]), int(spans[0, 1])
            image_positions = torch.arange(start, end, dtype=torch.long, device=self.device)
            question_positions = torch.arange(end, int(prompt_len), dtype=torch.long, device=self.device)
            n_img = image_positions.numel()
            n_keep = image_keep_budget(
                n_img=n_img, prompt_len=int(prompt_len), keep_ratio=self.keep_ratio, basis=self.keep_ratio_basis
            )

            if n_keep < n_img:
                for layer_idx in self.student.layer_indices:
                    H_l = H_all[layer_idx + 1]
                    scores = self.student.forward_layer(layer_idx, H_l, image_positions, question_positions).squeeze(0)
                    top = torch.topk(scores, k=n_keep, largest=True).indices
                    mask = torch.ones(int(prompt_len), dtype=torch.bool, device=self.device)
                    image_keep = torch.zeros(n_img, dtype=torch.bool, device=self.device)
                    image_keep[top] = True
                    mask[image_positions] = image_keep

                    past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, mask, :].contiguous()
                    past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, mask, :].contiguous()

                stats["pruned"] = True
                stats["image_tokens_before"] = n_img
                stats["image_tokens_kept"] = n_keep
                stats["text_tokens"] = int(prompt_len) - n_img
                stats["pruned_prompt_len"] = int(past_kv.get_seq_length())

        del prefill, H_all
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
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--student_path", default="/workspace/zap/ckpts/student_onevision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--keep_ratio", type=float, default=0.1)
    parser.add_argument("--keep_ratio_basis", default="image", choices=["image", "total"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--conv_template", default="qwen_1_5")
    parser.add_argument("--no_pruning", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    with open(args.annotation) as f:
        core = json.load(f)
    samples = core["data"]
    if args.limit:
        samples = samples[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    worker = OneVisionStudentWorker(
        pretrained=args.pretrained,
        student_path=args.student_path,
        device=args.device,
        attn_implementation=args.attn_implementation,
        keep_ratio=args.keep_ratio,
        keep_ratio_basis=args.keep_ratio_basis,
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
            pred_text, stats = worker.generate(full_question, ti["images_path"], no_pruning=args.no_pruning)
        except Exception as exc:
            print(f"[generate_onevision_student] ERROR sample_id={sample['sample_id']}: {exc}", file=sys.stderr, flush=True)
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
                f"[generate_onevision_student] {i+1}/{len(samples)} elapsed={elapsed:.1f}s "
                f"avg_image_keep_ratio={avg_kr:.4f} last_stats={stats}",
                flush=True,
            )

    with open(os.path.join(args.output_dir, "pred.json"), "w") as f:
        json.dump(preds, f)
    print(f"[generate_onevision_student] wrote {len(preds)} predictions -> {args.output_dir}/pred.json")


if __name__ == "__main__":
    main()
