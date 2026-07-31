# LLaVA-1.5 adaptation efficiency

This experiment fills rebuttal Tables R.F2 and R.F3 without warm-starting.
Every scorer is randomly initialized and the LLaVA-1.5-7B backbone remains
frozen.

## R.F2 conditions

| Tag | Teacher samples | Samples per dataset | Epochs |
|---|---:|---:|---:|
| `vicuna_generated_n900_e15` | 900 | 300 | 15 |
| `vicuna_generated_n300_e15` | 300 | 100 | 15 |
| `vicuna_generated_n1800_e3` | 1,800 | 600 | 3 |
| `vicuna_generated_n1800_e7` | 1,800 | 600 | 7 |
| `vicuna_generated_n1800_e10` | 1,800 | 600 | 10 |

The controlled full-setting reference is
`vicuna_generated_n1800_e15`. All four generated-answer settings use the same
1,800-record answer-only teacher cache, so the Vicuna conversation prompt,
generated-answer construction, attention implementation, trainer, and seed are
identical. Teacher collection costs for the 900- and 300-sample rows are
attributed linearly from the measured 1,800-sample run; student training time
is measured directly. The earlier hard-coded May baseline also used
generated-answer decode-step attention, not the question/answer mixture that
was added in July. It is not used as the controlled R.F2 row because it used
the legacy seed-42 sample set and trainer, whereas the reduced-data runs use
the current seed-0 cache and trainer.

## R.F3 condition

`vicuna_reference_n1800_e15` reuses the exact full-setting prompts and
teacher-forces the dataset reference response. It
collects answer-token-to-image-token attention in one forward pass. It does
not perform autoregressive answer generation.

All conditions are evaluated on the complete ChartQA, DocVQA, TextVQA, GQA,
COCO Caption, NoCaps, and TextCaps evaluation sets at 20% total prompt-KV
retention with batch size 1 and seed 0. The first three tasks fill Tables R.F2
and R.F3; the other four measure whether the adaptation trend also holds
outside the three VQA benchmarks.

The 7- and 10-epoch conditions evaluate the final epoch checkpoint rather than
the lowest-validation-loss checkpoint, so their results measure the requested
training budgets exactly.
