"""lmms-eval adapter: LLaVA-OneVision + zap's visual-utility student probe
KV eviction, via delayed-replay decode (full prefill -> student scoring on
hidden states -> evict image KV -> replay the held-out last prompt token ->
first answer token -> continue decode).

Subclasses `Llava_OneVision` and copies its `generate_until` (single-image
anyres path only -- video/multi-image handling is dropped since none of
MME/MMBench/MMStar/POPE/VizWiz-VQA use them) so request batching, dataset
plumbing, and prompt building are reused unchanged. Only the final
`self.model.generate(...)` call is replaced with the delayed-replay body,
which is otherwise identical to `generate_onevision_student.py`'s
`OneVisionStudentWorker.generate` -- already validated on 6457 video
samples (Video-MME + SEED-Bench-video).

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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from transformers import DynamicCache

for _p in ("/workspace/zap/look_rebuttal/onevision", "/workspace/zap"):
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
    def _generate_one_delayed_replay(self, input_ids, image_tensor, image_sizes, gen_kwargs):
        prompt_len_raw = input_ids.shape[1]
        prefill_ids = input_ids[:, :-1].contiguous()
        replay_ids = input_ids[:, -1:].contiguous()
        max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)

        prefill = self.model(
            input_ids=prefill_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            past_key_values=DynamicCache(),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
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
            if visual is None or visual == []:
                image_tensor = None
                task_type = "text"
            elif isinstance(visual[0], Image.Image):
                image_tensor = process_images(visual, self._image_processor, self._config)
                if isinstance(image_tensor, list):
                    image_tensor = [_img.to(dtype=torch.float16, device=self.device) for _img in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                task_type = "image"
            else:
                raise ValueError(f"Unsupported visual type for delayed-replay: {type(visual[0])}")

            if image_tensor is not None and DEFAULT_IMAGE_TOKEN not in context:
                question = f"{DEFAULT_IMAGE_TOKEN}\n{context}"
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
            image_sizes = [visual[idx].size for idx in range(len(visual))] if task_type == "image" else None

            text_output = self._generate_one_delayed_replay(input_ids, image_tensor, image_sizes, gen_kwargs)
            res.append(text_output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
