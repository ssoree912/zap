# OneVision scorer transfer and data efficiency

This experiment tests whether the LLaVA-1.5 answer-attention scorer provides
a useful warm start for LLaVA-OneVision-Qwen2 and measures the OneVision
student's teacher-data efficiency.

## Controlled matrix

| GPU | Initialization | Total teacher samples | Samples per dataset | Epochs |
|---:|---|---:|---:|---:|
| 0 | scratch | 450 | 150 | 15 |
| 1 | LLaVA-1.5 warm start | 450 | 150 | 15 |
| 2 | scratch | 900 | 300 | 15 |
| 3 | LLaVA-1.5 warm start | 900 | 300 | 15 |

All cases use the same OneVision answer-only teacher cache, seed, optimizer,
learning rate, validation ratio, and architecture. Checkpoints are saved after
epochs 1, 3, 5, 10, and 15. ChartQA evaluation uses image-token keep ratio
0.1 at epochs 1, 3, 5, and 15.

The existing scratch/1800/15-epoch checkpoint and ChartQA result are used as
the full-data reference.

## Scratch-only follow-up

The warm-start evaluation workers were stopped after the epoch-1 ChartQA
results at the user's request. Scratch ChartQA evaluation continues for epochs
3, 5, and 15. A shared four-GPU queue evaluates TextVQA, DocVQA, and GQA for
the scratch 450- and 900-sample scorers at epochs 1, 3, 5, and the actual
best-validation checkpoint. The scratch-450 best checkpoint is epoch 14; the
scratch-900 best checkpoint is epoch 15. ChartQA best is also evaluated for
scratch-450 because its epoch-15 snapshot is not its best checkpoint.

## Warm-start rule

- map 28 OneVision layers to 32 LLaVA-1.5 layers by relative decoder depth;
- copy common MLP, normalization, biases, and pointwise-convolution weights;
- average each 2D depthwise kernel over its vertical axis to initialize the
  corresponding 1D kernel;
- retain overlapping columns for 4096-to-3584 backbone-facing projections.

Each warm run writes `warm_start_report.json` with the exact layer mapping and
transferred parameter fraction.
