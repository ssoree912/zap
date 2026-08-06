"""lmms-eval adapter: LLaVA-OneVision + zap's visual-utility student probe
KV eviction, via delayed-replay decode (full prefill -> student scoring on
hidden states -> evict image KV -> replay the held-out last prompt token ->
first answer token -> continue decode).

Subclasses `Llava_OneVision` and copies its `generate_until` (single-image
anyres path, plus a video path added 2026-08-06 -- multi-image is still
dropped since none of the tasks this runs use it) so request batching,
dataset plumbing, and prompt building are reused unchanged. Only the final
`self.model.generate(...)` call is replaced with the delayed-replay body,
which is otherwise identical to `generate_onevision_student.py`'s
`OneVisionStudentWorker.generate` -- already validated on 6457 video
samples (Video-MME + SEED-Bench-video), via a *different*, non-lmms-eval
code path. This module's own video path reuses that worker's fix (explicit
per-frame `image_sizes` passed into the raw `self.model(...)` prefill call,
not omitted the way `Llava_OneVision.generate_until`'s video branch does
it -- that omission only works through `.generate()`, not a direct
`self.model(...)` call) but drives it through lmms-eval's standard
`videomme_local`/`seedbench_local` tasks and scoring instead of the
standalone script + MileBench-style `evaluate.py`, so results are
comparable to full-cache/VFlowOpt/VisionZip's video numbers (all three
already run via the standard lmms-eval CLI) on equal harness footing.

Smoke-tested 2026-08-06 on 3 real videomme_local samples end to end
(video decode -> delayed-replay prefill/evict/replay -> lmms-eval's own
`videomme_percetion_score` scoring): all 3 answers matched target exactly
(C/A/D). Not yet run at full scale or cross-checked in aggregate against
the standalone worker's numbers on the same samples -- that comparison,
and whatever discrepancy shows up between the two independently-coded
video paths, is still open.

Why this exists: the pre-existing student-probe sweep at
/workspace/zap/logs/local_all_onevision_* shows near-flat accuracy across
full-cache/keep0.5/keep0.1 on POPE/MME/MMStar despite
keep_ratio_stats.json confirming real ~10% image-token pruning occurred --
consistent with (though, per the tiny 0.5-0.8% per-doc answer-mismatch
rate across all three, not conclusively proving) the same
first-token-from-unpruned-cache issue LOOK-M's video eval had. Either way,
a verified delayed-replay implementation is the correct next step: it
settles the question empirically and gives a trustworthy keep_ratio=0.1
number regardless of what the old sweep's pruning mechanism actually was.

Two things the video path never exercised, worth the smoke-test check
called out in this module's tests/callers:
- anyres expands one <image> placeholder into base tile + grid tiles +
  image_newline tokens. `infer_image_token_spans` still returns one
  contiguous span (num_images=1 so its divisibility check is trivial), but
  that span includes the newline tokens as eviction candidates -- LOOK-M's
  video path never had newlines inside the image span.
- VizWiz-VQA is free-form generation, not first-token-determined -- expect
  a smaller keep-ratio-driven delta there than on the other four tasks.
"""

import copy
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.registry import register_model
from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.models.model_utils.load_video import read_video_pyav

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from transformers import DynamicCache

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ZAP_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _ZAP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from image_only_kv_cache import infer_image_token_spans  # noqa: E402
from delayed_replay import forward_one_token_with_kv, greedy_decode_after_prompt_replay  # noqa: E402
from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision  # noqa: E402
from foresight.eval.keep_budget import image_keep_budget, normalize_keep_ratio_basis  # noqa: E402


@register_model("llava_onevision_student_delayed_replay")
class Llava_OneVision_Student_DelayedReplay(Llava_OneVision):
    def __init__(
        self,
        *args,
        student_path: str = "/workspace/zap/ckpts/student_onevision",
        image_keep_ratio: float = 0.1,
        keep_ratio_basis: str = "image",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.student = VisualUtilityStudentOneVision.from_pretrained(student_path, map_location="cpu")
        self.student = self.student.to(device=self.device, dtype=torch.float16).eval()
        self.image_keep_ratio = float(image_keep_ratio)
        self.keep_ratio_basis = normalize_keep_ratio_basis(keep_ratio_basis)
        self.num_layers = self._config.num_hidden_layers

    @torch.no_grad()
    def _generate_one_delayed_replay(self, input_ids, image_tensor, image_sizes, gen_kwargs, modalities=None):
        prompt_len_raw = input_ids.shape[1]
        prefill_ids = input_ids[:, :-1].contiguous()
        replay_ids = input_ids[:, -1:].contiguous()
        max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)

        extra_kwargs = {}
        if modalities is not None:
            extra_kwargs["modalities"] = modalities
            # Belt-and-suspenders, matching Llava_OneVision.generate_until's video
            # branch: mm_spatial_pool_stride/mode are set from these same __init__
            # kwargs already, but re-assert right before generation in case
            # anything else touched self._config in between calls.
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        prefill = self.model(
            input_ids=prefill_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            past_key_values=DynamicCache(),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **extra_kwargs,
        )
        past_kv = prefill.past_key_values
        prompt_len = past_kv.get_seq_length()
        H_all = prefill.hidden_states

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
                n_img=n_img, prompt_len=int(prompt_len), keep_ratio=self.image_keep_ratio, basis=self.keep_ratio_basis
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

            if os.environ.get("DELAYED_REPLAY_DEBUG"):
                print(
                    f"[delayed_replay] prompt_len={int(prompt_len)} image_span=({start},{end}) "
                    f"n_img={n_img} n_keep={n_keep} ratio={n_keep/n_img:.4f}",
                    flush=True,
                )

        del prefill, H_all
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
            eos_token_id=self.eot_token_id,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.empty_cache()
        return self.tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

    def generate_until(self, requests):
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1, "delayed-replay only supports batch_size=1"

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            visual, context = batched_visuals[0], batched_contexts[0]
            modalities = None
            if visual is None or visual == []:
                image_tensor = None
                task_type = "text"
            elif isinstance(visual[0], Image.Image):
                # Matches Llava_OneVision.generate_until: multi-image/multi-frame
                # inputs (e.g. seedbench_local's 8 pre-decoded PIL frames) use
                # "pad" aspect ratio rather than the checkpoint's default
                # anyres_max_9, which would otherwise split each frame into up
                # to 9 tiles. Without this override, delayed-replay processes a
                # structurally different (much larger, tiled) visual input than
                # full_cache/VisionZip/VFlowOpt do for the same samples -- single
                # image tasks are unaffected since len(visual) == 1 never
                # triggers it either way.
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                image_tensor = process_images(visual, self._image_processor, self._config)
                if isinstance(image_tensor, list):
                    image_tensor = [_img.to(dtype=torch.float16, device=self.device) for _img in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                task_type = "image"
            elif isinstance(visual[0], str):
                # Video: decode to frames here (raw .mp4/.mkv path(s) in `visual`,
                # same as Llava_OneVision.generate_until's video branch), but --
                # unlike that branch -- also compute explicit per-frame
                # image_sizes, since this goes into a direct self.model(...)
                # prefill call below rather than .generate(); omitting image_sizes
                # only works through the .generate() call chain. Matches
                # generate_onevision_student.py's _prepare_video, which hit this
                # exact bug first on the standalone (non-lmms-eval) video path.
                if self.video_decode_backend == "decord":
                    frames = self.load_video(visual, self.max_frames_num)
                elif self.video_decode_backend == "pyav":
                    frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                else:
                    raise ValueError(f"Unsupported video_decode_backend: {self.video_decode_backend}")
                num_frames = frames.shape[0]
                frame_h, frame_w = frames.shape[1], frames.shape[2]
                video_image_sizes = [(frame_w, frame_h)] * num_frames
                frames_processed = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
                image_tensor = [frames_processed.to(dtype=torch.float16, device=self.device)]
                task_type = "video"
                modalities = ["video"]
            else:
                raise ValueError(f"Unsupported visual type for delayed-replay: {type(visual[0])}")

            if image_tensor is not None and DEFAULT_IMAGE_TOKEN not in context:
                # Matches Llava_OneVision.generate_until: multi-image/multi-frame
                # inputs get one <image> placeholder per frame (placeholder_count
                # = len(visual)), not one placeholder for the whole batch. For
                # seedbench_local's 8-frame samples this means 8 tokens, not 1 --
                # collapsing them to 1 changes how prepare_inputs_labels_for_multimodal
                # lays out image features relative to text, which is exactly the
                # kind of mismatch that's invisible on single-image tasks
                # (placeholder_count == 1 either way) and material on video ones.
                placeholder_count = len(visual) if task_type == "image" and isinstance(visual, list) else 1
                question = " ".join([DEFAULT_IMAGE_TOKEN] * placeholder_count) + "\n" + context
            else:
                question = context

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            if utils.is_json(question):
                question = json.loads(question)
                for idx, item in enumerate(question):
                    role = conv.roles[idx % 2]
                    conv.append_message(role, item["value"])
                assert len(conv.messages) % 2 == 1
                conv.append_message(conv.roles[1], None)
            else:
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("do_sample", False)

            input_ids = (
                tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(self.device)
            )
            if task_type == "image":
                image_sizes = [visual[idx].size for idx in range(len(visual))]
            elif task_type == "video":
                image_sizes = video_image_sizes
            else:
                image_sizes = None

            text_output = self._generate_one_delayed_replay(input_ids, image_tensor, image_sizes, gen_kwargs, modalities=modalities)
            res.append(text_output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
