# KVzap

KVzap is a training-based KV cache compression method for visual language models.
A lightweight student MLP is trained to predict which image tokens are important for future decoding,
and used at inference time to selectively prune the KV cache — without any changes to the base model.

## Environment

```bash
conda env create -f environment.yml
conda activate qvik
```

## Models

| Model | Checkpoint |
|---|---|
| LLaVA-OneVision-Qwen2-7B | [lmms-lab/llava-onevision-qwen2-7b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |

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

```bash
# OneVision
python qvik/teacher/extract_llava_onevision.py \
  --model-path ckpts/llava-onevision-qwen2-7b-ov \
  --datasets textvqa gqa scienceqa \
  --n-samples 300 \
  --output-root data/train/teacher/llava_onevision

# LLaVA-1.5
python qvik/teacher/extract_llava15.py \
  --datasets textvqa gqa scienceqa \
  --n-samples 300 \
  --output-root data/train/teacher/llava15
```

### 2. Student training

```bash
# OneVision
python qvik/train/llava_onevision.py \
  --teacher-root data/train/teacher/llava_onevision \
  --model-path ckpts/llava-onevision-qwen2-7b-ov \
  --epochs 15 \
  --output-dir ckpts/student_onevision

# LLaVA-1.5
python qvik/train/llava15.py \
  --teacher-root data/train/teacher/llava15 \
  --epochs 15 \
  --output-dir ckpts/student_llava15
```

### 3. Evaluation

**lmms-eval (OneVision):**
```bash
python qvik/eval/lmms_onevision_student.py \
  --pretrained ckpts/llava-onevision-qwen2-7b-ov \
  --student ckpts/student_onevision \
  --keep-ratio 0.25 \
  --tasks textvqa_val chartqa docvqa_val
```

**lmms-eval (LLaVA-1.5):**
```bash
python qvik/eval/lmms_llava15_original_student.py \
  --pretrained ckpts/llava-v1.5-7b \
  --student ckpts/student_llava15 \
  --keep-ratio 0.25 \
  --tasks textvqa_val chartqa
```

**MileBench (OneVision):**
```bash
python qvik/eval/milebench_onevision_student.py \
  --pretrained ckpts/llava-onevision-qwen2-7b-ov \
  --student ckpts/student_onevision \
  --keep-ratio 0.25
```
