# Installation

## Reference Platform

The reported experiments used Ubuntu 22.04.3, Python 3.10.13, CUDA 12.1,
cuDNN 8, PyTorch 2.2.0, and the package versions pinned in
`caduceus_env.yml`. GPU kernels were built with:

- `mamba-ssm==1.2.0.post1`
- `causal-conv1d==1.2.0.post2`
- `flash-attn==2.5.6`
- `triton==2.2.0`

## Create the Environment

```bash
conda env create -f caduceus_env.yml
conda activate flipped-gari-aaai27
```

CUDA extension installation is compiler- and GPU-specific. If the three
kernel packages fail during the first environment solve, create the
environment without them, activate it, and install them after confirming that
`nvcc --version` reports CUDA 12.1:

```bash
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
pip install flash-attn==2.5.6 --no-build-isolation
```

## Verify the Installation

The package-level checks do not require a GPU:

```bash
python scripts/validate_package.py
python scripts/test_window_readout.py
```

The full import check requires a CUDA-capable environment:

```bash
python -c "import torch, mamba_ssm, flash_attn; print(torch.__version__)"
python -c "from src.models.sequence.dna_embedding import DNAEmbeddingModelMLBN"
```

## Storage

Approximate external storage requirements are:

- hg38 masked-nucleotide pretraining assets: about 3 GB compressed/unpacked.
- GenomicBenchmarks: task-dependent, downloaded by the official package.
- Causal-eQTL VEP: about 11 GB before embeddings.
- DNALongBench ETGP: about 0.94 GB from Harvard Dataverse.
- VEP embeddings and long-sequence checkpoints: tens of GB depending on profiles.

Set data and output roots on a filesystem with adequate capacity. No
machine-specific path is required by the code or launchers in this package.
