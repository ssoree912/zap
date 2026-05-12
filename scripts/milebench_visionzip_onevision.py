#!/usr/bin/env python3
"""MileBench evaluation with VisionZip dominant+contextual token eviction on LLaVA-OneVision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


ZAP_ROOT = Path("/workspace/zap")
VFLOWOPT_ROOT = Path("/workspace/VFlowOpt")
VISIONZIP_ROOT = Path("/workspace/Visionzip_onevision")
MILEBENCH_ROOT = ZAP_ROOT / "data/MileBench"
MODEL_PATH = ZAP_ROOT / "ckpts/llava-onevision-qwen2-7b-ov"
DATASETS_4 = ["CLEVR-Change", "IEdit", "Spot-the-Diff", "ALFRED"]
DATASETS_FULL = [
    "ALFRED", "ActionLocalization", "ActionPrediction", "ActionSequence",
    "CLEVR-Change", "CharacterOrder", "CounterfactualInference", "DocVQA",
    "EgocentricNavigation", "GPR1200", "IEdit", "ImageNeedleInAHaystack",
    "MMCoQA", "MovingAttribute", "MovingDirection", "MultiModalQA",
    "OCR-VQA", "ObjectExistence", "ObjectInteraction", "ObjectShuffle",
    "SceneTransition", "SlideVQA", "Spot-the-Diff", "StateChange",
    "TQA", "TextNeedleInAHaystack", "WebQA", "WikiVQA",
]
MAX_CONTEXT_LEN = 4096
N_TOKENS_PER_IMAGE = 200
DEFAULT_MAX_NEW_TOKENS = 64


def setup_paths() -> None:
    paths = [
        VFLOWOPT_ROOT / "src/lmms_eval-0.2.4",
        VFLOWOPT_ROOT / "src/LLaVA-OneVision",
        VFLOWOPT_ROOT / "src/transformers-4.46.0/src",
        VISIONZIP_ROOT,
        Path("/workspace/look-m"),
        ZAP_ROOT,
    ]
    for path in paths:
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _fix_accelerate_old_forwards(model) -> None:
    """accelerate hooks capture _old_forward before our class-level patches.

    Walk all submodules: for any that have _old_forward, replace it with the
    current class-level forward so our VisionZip patches actually run.
    """
    import types

    for module in model.modules():
        if hasattr(module, "_old_forward"):
            cls_forward = type(module).forward
            module._old_forward = types.MethodType(cls_forward, module)


def load_visionzip_model(*, keep_ratio: float, contextual_ratio: float, device: str, model_path: str):
    setup_paths()
    from lmms_eval.models.llava_onevision import Llava_OneVision  # noqa: PLC0415
    from visionzip import visionzip_onevision  # noqa: PLC0415

    wrapper = Llava_OneVision(
        pretrained=model_path,
        conv_template="qwen_1_5",
        model_name="llava_qwen",
        device=device,
        device_map="auto",
    )

    if keep_ratio < 0.999:
        visionzip_onevision(wrapper.model, keep_ratio=keep_ratio, contextual_ratio=contextual_ratio)
        _fix_accelerate_old_forwards(wrapper.model)

    wrapper.milebench_method = "visionzip_onevision"
    wrapper.milebench_keep_ratio = keep_ratio
    wrapper.illava_config = None
    return wrapper


@torch.inference_mode()
def generate_answer(
    *,
    wrapper,
    question: str,
    image_paths: list[str],
    max_new_tokens: int,
) -> str:
    setup_paths()
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: PLC0415
    from llava.conversation import conv_templates  # noqa: PLC0415
    from llava.mm_utils import process_images, tokenizer_image_token  # noqa: PLC0415

    visuals = [Image.open(path).convert("RGB") for path in image_paths]
    prompt_text = question.replace("<ImageHere>", DEFAULT_IMAGE_TOKEN)

    if visuals and DEFAULT_IMAGE_TOKEN not in prompt_text:
        prompt_text = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"

    if visuals:
        wrapper._config.image_aspect_ratio = "pad"
        image_tensor = process_images(visuals, wrapper._image_processor, wrapper._config)
        if isinstance(image_tensor, list):
            image_tensor = [x.to(dtype=torch.float16, device=wrapper.device) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.float16, device=wrapper.device)
    else:
        image_tensor = None

    conv = conv_templates[wrapper.conv_template].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        wrapper.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(wrapper.device)
    pad_token_id = (
        wrapper.tokenizer.pad_token_id
        if wrapper.tokenizer.pad_token_id is not None
        else wrapper.tokenizer.eos_token_id
    )
    attention_mask = input_ids.ne(pad_token_id).to(wrapper.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        "do_sample": False,
        "top_p": None,
        "num_beams": 1,
        "attention_mask": attention_mask,
        "pad_token_id": pad_token_id,
        "images": image_tensor,
        "use_cache": wrapper.use_cache,
    }
    if visuals:
        gen_kwargs["image_sizes"] = [image.size for image in visuals]

    output_ids = wrapper.model.generate(input_ids, **gen_kwargs)
    text = wrapper.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text


def parse_datasets(value: str) -> list[str]:
    if value == "all":
        return DATASETS_4
    if value == "full":
        return DATASETS_FULL
    return [part.strip() for part in value.split(",") if part.strip()]


def run_dataset(
    *,
    dataset: str,
    wrapper,
    output_dir: Path,
    limit: int | None,
    overwrite: bool,
    max_new_tokens: int,
    combine_image: int,
) -> None:
    setup_paths()
    from utils import MileBenchDataset  # noqa: PLC0415

    task_out = output_dir / dataset
    task_out.mkdir(parents=True, exist_ok=True)
    pred_path = task_out / "pred.json"
    if pred_path.exists() and not overwrite:
        print(f"[skip] {dataset}: pred.json exists", flush=True)
        return

    data_path = MILEBENCH_ROOT / dataset / f"{dataset}.json"
    img_dir = str(MILEBENCH_ROOT / dataset / "images")
    core = json.loads(data_path.read_text())
    samples_raw = core["data"]
    if limit is not None:
        samples_raw = samples_raw[:limit]

    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples_raw:
        by_n[combine_image].append(sample)

    predictions: list[dict[str, Any]] = []
    for n_img in sorted(by_n):
        ds = MileBenchDataset(
            annotation=by_n[n_img],
            task_instructions=core["meta_data"]["task_instruction"],
            img_dir=img_dir,
            max_context_len=MAX_CONTEXT_LEN,
            n_tokens_per_image=N_TOKENS_PER_IMAGE,
            tokenizer=wrapper.tokenizer,
            dataset_name=dataset,
            combine_image=combine_image,
        )
        for idx in tqdm(range(len(ds)), desc=f"{dataset}(n={n_img})"):
            item = ds[idx]
            try:
                answer = generate_answer(
                    wrapper=wrapper,
                    question=item["context"],
                    image_paths=item["raw_img_list"],
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc(file=sys.stderr)
                print(f"[warn] {dataset} sample {item['sample_id']} failed: {exc}", file=sys.stderr)
                answer = ""
            predictions.append(
                {
                    "sample_id": item["sample_id"],
                    "question": item["context"],
                    "image": item["raw_img_list"],
                    "pred_response": answer,
                    "gt_response": item["response"],
                    "task_instance": item.get("task_instance", {}),
                    "meta": {
                        "method": wrapper.milebench_method,
                        "keep_ratio": wrapper.milebench_keep_ratio,
                        "max_new_tokens": max_new_tokens,
                    },
                }
            )

    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"[{dataset}] saved {len(predictions)} -> {pred_path}", flush=True)


def score_dataset(dataset: str, output_dir: Path) -> None:
    import subprocess
    subprocess.run(
        [
            sys.executable,
            "/workspace/look-m/evaluate.py",
            "--data-dir", str(MILEBENCH_ROOT),
            "--dataset", dataset,
            "--result-dir", str(output_dir),
        ],
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS_4))
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--contextual-ratio", type=float, default=0.05)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--combine-image", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_datasets(args.datasets)
    print(
        f"[init] method=visionzip_onevision keep_ratio={args.keep_ratio} "
        f"contextual_ratio={args.contextual_ratio} max_new_tokens={args.max_new_tokens} "
        f"combine_image={args.combine_image} datasets={datasets}",
        flush=True,
    )
    wrapper = load_visionzip_model(
        keep_ratio=args.keep_ratio,
        contextual_ratio=args.contextual_ratio,
        device=args.device,
        model_path=args.model_path,
    )
    for dataset in datasets:
        run_dataset(
            dataset=dataset,
            wrapper=wrapper,
            output_dir=output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            max_new_tokens=args.max_new_tokens,
            combine_image=args.combine_image,
        )
        if not args.no_score:
            score_dataset(dataset, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
