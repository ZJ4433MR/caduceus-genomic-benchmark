# Checkpoints

Large checkpoint files are not included in this repository.

Expected local layout:

```text
checkpoints/
  mlbn_1k/
    last.ckpt
  mamba2_1k/
    last.ckpt
  mamba2_revph_1k/
    last.ckpt
  caduceus/
    <optional cached public Hugging Face models>
```

For public Caduceus baselines, scripts can also read directly from Hugging Face
model identifiers when network access is available.

