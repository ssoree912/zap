# KVzap

KVzap is a training-based KV cache compression method for visual language models.
A lightweight student MLP is trained to predict which image tokens are important for future decoding,
and used at inference time to selectively prune the KV cache — without any changes to the base model.

## Environment

`environment.yml` lists all package versions. The LLaVA-1.5 and
LLaVA-OneVision model code is vendored under `qvik/llava15/` and
`qvik/llava_onevision/`, so no external model repo is needed.

```bash
conda env create -f environment.yml
conda activate qvik
```

## Models

| Model | HuggingFace | Local path |
|---|---|---|
| LLaVA-OneVision-Qwen2-7B | [lmms-lab/llava-onevision-qwen2-7b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) | `model/llava-onevision-qwen2-7b-ov` |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | `model/llava-v1.5-7b` |

## Datasets

### Training (teacher extraction)

| Dataset | Link |
|---|---|
| TextVQA | https://textvqa.org/ |
| GQA | https://cs.stanford.edu/people/dorarad/gqa/ |
| ScienceQA | https://scienceqa.github.io/ |

### Evaluation

| Dataset | Link |
|---|---|
| TextVQA | https://textvqa.org/ |
| GQA | https://cs.stanford.edu/people/dorarad/gqa/ |
| ChartQA | https://github.com/vis-nlp/ChartQA |
| DocVQA | https://www.docvqa.org/ |
| NoCaps | https://nocaps.org/ |
| TextCaps | https://textvqa.org/textcaps/ |
| MileBench | https://milebench.github.io/ |

## Data Structure

```
data/
├── train/
│   ├── textvqa/          # TextVQA train images & annotations
│   ├── gqa/              # GQA train images & questions
│   ├── scienceqa/        # ScienceQA images & problems.json
│   └── teacher/
│       ├── llava15/      # extracted teacher scores for LLaVA-1.5
│       └── llava_onevision/  # extracted teacher scores for OneVision
└── eval/
    ├── TextVQA/
    ├── GQA/
    ├── ChartQA/
    ├── DocVQA/
    ├── NoCaps/
    ├── TextCaps/
    └── MileBench/
```

## Pipeline

### 1. Teacher extraction

Each script extracts one dataset per run; loop over the three datasets.

```bash
# OneVision
for ds in textvqa gqa scienceqa; do
  python qvik/teacher/extract_llava_onevision.py \
    --model model/llava-onevision-qwen2-7b-ov \
    --dataset $ds --n-samples 600 \
    --output-root data/train/teacher/llava_onevision
done

# LLaVA-1.5
for ds in textvqa gqa scienceqa; do
  python qvik/teacher/extract_llava15.py \
    --model model/llava-v1.5-7b \
    --dataset $ds --n-samples 600 \
    --output-root data/train/teacher/llava15
done
```

### 2. Student training

The best checkpoint (lowest val_loss) is saved to `<output-dir>/pytorch_model.bin`;
`last_checkpoint.pt` holds the final epoch + optimizer state for resuming.

```bash
# OneVision
python qvik/train/llava_onevision.py \
  --teacher-root data/train/teacher/llava_onevision \
  --llava-path model/llava-onevision-qwen2-7b-ov \
  --epochs 20 \
  --output-dir ckpts/student_onevision

# LLaVA-1.5
python qvik/train/llava15.py \
  --teacher-root data/train/teacher/llava15 \
  --epochs 20 \
  --output-dir ckpts/student_llava15
```

### 3. Evaluation

Results are written to a consistent layout: `results/<model_tag>/<task>/`.

Available `_local` tasks (auto-generated at runtime, route to `data/eval/`):
`textvqa_val_local`, `chartqa_local`, `docvqa_val_local`, `gqa_local`,
`coco2017_cap_val_local`, `nocaps_val_local`, `textcaps_val_local`

**lmms-eval (LLaVA-1.5)** → `results/llava15_lmms/<task>/`:
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_llava15_student \
  --model_args pretrained=model/llava-v1.5-7b,student_path=ckpts/student_llava15,keep_ratio=0.5,device=cuda:0 \
  --tasks textvqa_val_local,chartqa_local,docvqa_val_local,gqa_local,coco2017_cap_val_local,nocaps_val_local,textcaps_val_local \
  --batch_size 1 \
  --output_path results
```

**lmms-eval (OneVision)** → `results/onevision_lmms/<task>/`:
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_onevision_student \
  --model_args pretrained=model/llava-onevision-qwen2-7b-ov,student_path=ckpts/student_onevision,keep_ratio=0.5,device=cuda:0 \
  --tasks textvqa_val_local,chartqa_local,docvqa_val_local,gqa_local,coco2017_cap_val_local,nocaps_val_local,textcaps_val_local \
  --batch_size 1 \
  --output_path results
```

**MileBench (OneVision)** → `results/onevision_milebench/<dataset>/`:
```bash
for ds in ALFRED CLEVR-Change IEdit Spot-the-Diff; do
  python qvik/eval/milebench_onevision_student.py \
    --dataset $ds \
    --pretrained model/llava-onevision-qwen2-7b-ov \
    --student_path ckpts/student_onevision \
    --keep_ratio 0.5
done
```
