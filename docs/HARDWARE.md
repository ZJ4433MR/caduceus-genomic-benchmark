# Hardware And Software Reporting

The AAAI reproducibility checklist asks for the computing infrastructure used
for experiments.

For each reported run, record:

- GPU model and count.
- GPU memory.
- CPU model if available.
- System RAM if available.
- Operating system.
- CUDA version.
- PyTorch version.
- mamba-ssm, causal-conv1d, flash-attn, transformers, and PyTorch Lightning
  versions.
- Whether mixed precision was used.
- Wall-clock time for pretraining and finetuning where available.

Suggested raw-result columns:

```text
hardware_id,gpu_model,gpu_count,gpu_memory_gb,cuda,torch,mamba_ssm,
causal_conv1d,flash_attn,transformers,precision,wall_time_hours
```

