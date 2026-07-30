# Hardware and Software

## Software

- Ubuntu 22.04.3
- Singularity 4.2.1
- Python 3.10.13
- CUDA 12.1 and cuDNN 8
- PyTorch 2.2.0
- PyTorch Lightning 1.8.6
- Hydra 1.3.2
- Mamba-SSM 1.2.0.post1
- causal-conv1d 1.2.0.post2
- FlashAttention 2.5.6
- Triton 2.2.0
- NumPy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2

The full pinned environment is `caduceus_env.yml`.

## Compute Profiles

| Stage | Accelerator profile | Parallelism |
|---|---|---|
| GB fine-tuning/evaluation | NVIDIA L40 | one GPU per task/seed |
| VEP 1-kb pretraining and 131-kb embedding | NVIDIA L40 | eight GPUs |
| VEP component pretraining | NVIDIA A800 | four GPUs |
| VEP SVC grid | CPU | 28 CPU cores, up to 400 GB RAM |
| ETGP 1-kb direct fine-tuning | NVIDIA L40 | eight GPUs |
| ETGP 2-/5-kb continuation | NVIDIA A800 | four GPUs |
| ETGP 2-/5-kb direct fine-tuning | NVIDIA L40 or A800 | global batch 8 |

Representative direct 450-kb ETGP runs require approximately 13--27 hours per
epoch/profile depending on GPU type and context checkpoint. VEP embedding
shards require approximately 5--6 hours each on L40 GPUs, with the complete
run split across eight shards. Exact timing varies with filesystem and kernel
build; scientific configurations do not depend on scheduler-specific paths.
