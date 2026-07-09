# Installation

The recommended review setup uses the Conda environment file:

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
make smoke
```

The core dependency versions are:

- Python 3.8
- PyTorch 2.2.0
- CUDA 12.1
- Transformers 4.38.1
- PyTorch Lightning 1.8.6
- mamba-ssm 1.2.0.post1
- causal-conv1d 1.2.0.post2
- flash-attn 2.5.6

If a package manager cannot build the CUDA extension packages on the target
machine, install compatible wheels for the local CUDA/PyTorch combination before
running full-scale experiments.

The full AAAI experiments were run on GPU nodes. Reviewers can still validate
the package structure and imports with `make smoke` before running larger jobs.

