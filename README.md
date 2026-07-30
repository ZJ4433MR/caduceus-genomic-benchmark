# Flipped-GARI Code and Data Package

This anonymous artifact accompanies the AAAI-27 submission
**Flipped-GARI: Sequence-Reversal-Aware Transfer from Short-Context
Pretraining to Long-Range Genomic Prediction**.

The review deliverable is a self-contained code-and-metadata ZIP. Full public
datasets are retrieved from their official sources. The artifact does not rely
on a project website or an anonymous source-control link; the repository
branch used to assemble the ZIP is only an internal staging area.

## What Is Included

- The Flipped-GARI implementation and the training/evaluation framework.
- Exact public-name model configurations for GenomicBenchmarks, VEP, and ETGP.
- Four configurations for the VEP component diagnostic.
- Masked-nucleotide pretraining, downstream training, and evaluation entry points.
- Public-data download scripts, dataset/version/license metadata, a small
  CC0-licensed representative ETGP metadata subset, and synthetic smoke data.
- Checkpoint SHA-256 identifiers. Checkpoint binaries and third-party raw data
  are excluded because of archive size and redistribution constraints.
- An implementation-to-paper map and a line-by-line reproducibility-checklist map.
- An anonymous-archive builder and package validator.

## Quick Audit

```bash
conda env create -f caduceus_env.yml
conda activate flipped-gari-aaai27
python scripts/validate_package.py
python scripts/test_window_readout.py
```

To build the reviewer ZIP:

```bash
python scripts/build_anonymous_archive.py
```

The output is `dist/flipped-gari-aaai27-code-data.zip`. The builder excludes
version-control metadata, raw downloaded datasets, checkpoints, caches, and
machine-specific paths, then writes a SHA-256 manifest inside the archive.

## Reproduction Paths

Start with [INSTALLATION.md](INSTALLATION.md), then follow
[REPRODUCE.md](REPRODUCE.md). Exact experiment settings are collected in
[docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md), and the data boundary is
explained in [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).

The code retains a few `mlbn` class names and checkpoint keys solely for
compatibility with checkpoints produced before the public model name was fixed.
All user-facing configurations use **Flipped-GARI**. See
[docs/LEGACY_IDENTIFIERS.md](docs/LEGACY_IDENTIFIERS.md).

## Artifact Boundary

This package covers the experiments described in the submitted paper and
Technical Supplement:

1. Five-split GenomicBenchmarks classification under original-order and
   sequence-reversed evaluation.
2. Causal-eQTL VEP at 131 kb, including pretraining, input-context, feature-source,
   and component diagnostics.
3. Direct full-sequence 450-kb DNALongBench ETGP.

Every included long-range ETGP path processes the complete input sequence
directly.

## License

The source code is distributed under the Apache License 2.0 in [LICENSE](LICENSE).
Third-party code and data retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
