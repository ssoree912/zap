"""Generate AMBER benchmark responses using the same verified delayed-replay
OneVision student wrapper (llava_onevision_student_delayed_replay) already
used for POPE/MME/MMStar/MMBench_EN/VizWiz-VQA and the CHAIR COCO-caption
run -- same checkpoint, same eviction mechanism, same image_keep_ratio=0.1,
so this AMBER number sits in the same trustworthy-provenance bucket as the
rest of the table instead of a one-off script.

Runs directly against the model class (bypassing lmms-eval's CLI/task
plumbing, since AMBER isn't an lmms-eval task) for the generative task
(id 1-1004, free-form "Describe this image.") and the discriminative task
(id 1005-15220, yes/no) against the 1004 AMBER images. Appends each result
to a jsonl immediately and skips ids already present on restart, since the
full run is long enough that surviving an interruption matters.
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, "/workspace/VFlowOpt_llava1.5/src/lmms_eval-0.2.4")
from lmms_eval.models.llava_onevision_student_delayed_replay import Llava_OneVision_Student_DelayedReplay  # noqa: E402

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402

IMAGE_DIR = "/workspace/AMBER/image/image"
QUERY_ALL = "/workspace/AMBER/data/query/query_all.json"
DISCRIMINATIVE_SUFFIX = "\nAnswer the question using a single word or phrase."


def normalize_yesno(text):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    return text.strip()


def build_input(model, image, query_text):
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
    return input_ids, image_tensor, [image.size]


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
    parser.add_argument("--split", choices=["generative", "discriminative", "all"], default="all")
    parser.add_argument("--output", default="/workspace/zap/look_rebuttal/outputs/amber_student_dr_0.1/responses.jsonl")
    parser.add_argument("--image_keep_ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
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
    print(f"{len(queries)} remaining queries to run", flush=True)
    if not queries:
        print("DONE (nothing left to do)", flush=True)
        return

    model = Llava_OneVision_Student_DelayedReplay(
        pretrained="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov",
        conv_template="qwen_1_5",
        model_name="llava_qwen",
        device_map=args.device,
        attn_implementation="sdpa",
        student_path="/workspace/zap/ckpts/student_onevision",
        image_keep_ratio=args.image_keep_ratio,
        keep_ratio_basis="image",
    )

    last_image_name, last_image = None, None
    out_f = open(args.output, "a")
    for i, q in enumerate(queries):
        if q["image"] != last_image_name:
            last_image = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")
            last_image_name = q["image"]

        is_generative = q["id"] <= 1004
        if is_generative:
            query_text = q["query"]
            gen_kwargs = {"max_new_tokens": 512, "temperature": 0, "do_sample": False}
        else:
            query_text = q["query"] + DISCRIMINATIVE_SUFFIX
            gen_kwargs = {"max_new_tokens": 8, "temperature": 0, "do_sample": False}

        input_ids, image_tensor, image_sizes = build_input(model, last_image, query_text)
        text = model._generate_one_delayed_replay(input_ids, image_tensor, image_sizes, gen_kwargs)
        response = text if is_generative else normalize_yesno(text)

        out_f.write(json.dumps({"id": q["id"], "response": response}) + "\n")
        out_f.flush()

        if (i + 1) % args.log_every == 0:
            print(f"[{i+1}/{len(queries)}] id={q['id']} response={response!r}", flush=True)

    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
