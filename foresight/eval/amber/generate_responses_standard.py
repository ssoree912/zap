"""Generate AMBER responses for methods that call the model's own standard
.generate() (no delayed-replay two-pass decode needed):
- full_cache:  stock Llava_OneVision, no pruning at all (baseline)
- vflowopt:    Llava_OneVision_Training_Free, prunes inline in forward_illava
- visionzip:   Llava_OneVision_VisionZip, prunes at the SigLIP vision tower

Each backend's single-image .generate() call is copied from that class's
own generate_until (llava_onevision.py L538 / llava_onevision_training_free.py
L627) minus lmms-eval's request-batching plumbing, since AMBER isn't an
lmms-eval task. Companion to generate_responses.py, which instead handles
the delayed-replay methods (student/LOOK-M/PrefixKV) that must evict KV
before the first answer token is produced.
"""

import argparse
import importlib
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, "/workspace/VFlowOpt_llava1.5/src/lmms_eval-0.2.4")

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402

IMAGE_DIR = "/workspace/AMBER/image/image"
QUERY_ALL = "/workspace/AMBER/data/query/query_all.json"
DISCRIMINATIVE_SUFFIX = "\nAnswer the question using a single word or phrase."

BACKENDS = {
    "full_cache": dict(
        module="lmms_eval.models.llava_onevision",
        cls="Llava_OneVision",
        model_args=dict(
            pretrained="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov",
            conv_template="qwen_1_5",
            model_name="llava_qwen",
            device_map="cuda:0",
            attn_implementation="sdpa",
        ),
    ),
    "vflowopt": dict(
        module="lmms_eval.models.llava_onevision_training_free",
        cls="Llava_OneVision_Training_Free",
        model_args=dict(
            pretrained="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov",
            conv_template="qwen_1_5",
            model_name="llava_qwen_training_free",
            device_map="cuda:0",
            attn_implementation="sdpa",
            enable_illava_vit=True,
            illava_vit_k=25,
            enable_illava_llm=True,
            illava_llm_k="9-18",
            illava_keep_ratio=0.10,
        ),
    ),
    "visionzip": dict(
        module="lmms_eval.models.llava_onevision_visionzip",
        cls="Llava_OneVision_VisionZip",
        model_args=dict(
            pretrained="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov",
            conv_template="qwen_1_5",
            model_name="llava_qwen",
            device_map="cuda:0",
            attn_implementation="sdpa",
            visionzip_keep_ratio=0.1,
            visionzip_contextual_ratio=0.05,
        ),
    ),
}


def normalize_yesno(text):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    return text.strip()


def load_model(backend):
    spec = BACKENDS[backend]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["cls"])
    return cls(**spec["model_args"])


def generate_one(model, backend, image, query_text, max_new_tokens):
    image_tensor = process_images([image], model._image_processor, model._config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(dtype=torch.float16, device=model.device) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)

    conv = conv_templates[model.conv_template].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{query_text}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    )
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    attention_mask = input_ids.ne(pad_token_id)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, temperature=0, do_sample=False, top_p=None, num_beams=1,
        image_sizes=[image.size],
    )
    extra = {}
    if backend == "vflowopt":
        extra = dict(illava_config=model.illava_config, questions_only=None, raw_frames=[[image]])

    with torch.inference_mode():
        cont = model.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            images=image_tensor,
            use_cache=model.use_cache,
            **extra,
            **gen_kwargs,
        )
    text = model.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text


def load_done_ids(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=list(BACKENDS))
    parser.add_argument("--split", choices=["generative", "discriminative", "all"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    done_ids = load_done_ids(args.output)
    print(f"{len(done_ids)} ids already done, resuming from {args.output}", flush=True)

    queries = json.load(open(QUERY_ALL))
    if args.split == "generative":
        queries = [q for q in queries if q["id"] <= 1004]
    elif args.split == "discriminative":
        queries = [q for q in queries if q["id"] >= 1005]
    queries = [q for q in queries if q["id"] not in done_ids]
    if args.limit is not None:
        queries = queries[: args.limit]
    print(f"{len(queries)} remaining queries to run (backend={args.backend})", flush=True)
    if not queries:
        print("DONE (nothing left to do)", flush=True)
        return

    model = load_model(args.backend)

    last_image_name, last_image = None, None
    out_f = open(args.output, "a")
    for i, q in enumerate(queries):
        if q["image"] != last_image_name:
            last_image = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")
            last_image_name = q["image"]

        is_generative = q["id"] <= 1004
        if is_generative:
            query_text = q["query"]
            max_new_tokens = 512
        else:
            query_text = q["query"] + DISCRIMINATIVE_SUFFIX
            max_new_tokens = 8

        text = generate_one(model, args.backend, last_image, query_text, max_new_tokens)
        response = text if is_generative else normalize_yesno(text)

        out_f.write(json.dumps({"id": q["id"], "response": response}) + "\n")
        out_f.flush()

        if (i + 1) % args.log_every == 0:
            print(f"[{i+1}/{len(queries)}] id={q['id']} response={response!r}", flush=True)

    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
