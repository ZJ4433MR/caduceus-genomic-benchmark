# Hyperparameter Reporting

For every model in the paper, report both the final values and the searched
ranges.

Minimum fields:

- Model name.
- Pretraining context length.
- Model dimension.
- Number of layers.
- Number of heads where applicable.
- Kernel size and state dimension where applicable.
- Optimizer.
- Learning rate.
- Weight decay.
- Scheduler.
- Warmup steps.
- Max steps or epochs.
- Batch size and global batch size.
- Dropout and drop path.
- Random seeds.
- Checkpoint selection rule.

The same-budget comparison should explicitly show that Mamba2-1k,
Mamba2-RevPh-1k, and MLBN-1k share the intended pretraining context and
training-budget constraints.

