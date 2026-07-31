# Post-eviction decode benchmark

Paired full-cache and Q-ViK cached-decode latency on single NVIDIA RTX 3090
GPUs. The benchmark uses 25 samples from each of ChartQA, DocVQA, GQA, and
TextVQA (100 total), batch size 1, and 32 fixed decode forward calls per
sample.

- LLaVA-1.5-7B: 20% total prompt-KV retention.
- LLaVA-OneVision-Qwen2-7B: 10% image-KV retention.

The timed CUDA region begins after multimodal prefill and, for Q-ViK, after
student scoring and per-layer KV eviction. Prefill and eviction latency are not
included in reported decode latency. Full and compressed measurements are
paired by sample and their execution order alternates to reduce ordering bias.

## Long-decode benchmark

The additional MM-Vet run uses 50 deterministic test samples and the official
lmms-eval reasoning pre-prompt. It executes 256 fixed cached-decode forward
calls per sample. This keeps the full/Q-ViK work count equal while measuring a
substantially longer decoding interval than the 32-step VQA run. MM-Vet's GPT
judge is not needed because this run measures latency and does not score
answers.
