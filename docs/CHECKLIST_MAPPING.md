# Reproducibility Checklist Mapping

This file maps the affirmative checklist answers to concrete artifact paths.

| Checklist item | Evidence in this package |
|---|---|
| Hyperparameter ranges and selection | `docs/HYPERPARAMETERS.md`; public-name configs under `configs/model/` |
| Data preprocessing code | `scripts/download_*`; `src/dataloaders/`; `vep_embeddings.py` |
| Complete source code | `src/`, `caduceus/`, `train.py`, VEP and evaluation scripts |
| Public source under research license | `LICENSE`, `THIRD_PARTY_NOTICES.md` |
| Comments mapping implementation to paper | `docs/IMPLEMENTATION_MAP.md` plus symbol-level comments |
| Random seeds | GB 1--5; VEP 1--10; ETGP 2222/3333 in configs, launchers, and result CSVs |
| Hardware/software requirements | `INSTALLATION.md`, `caduceus_env.yml`, `docs/HARDWARE.md` |
| Evaluation code | GB evaluation scripts, `vep_svm_eval.py`, ETGP global-metric evaluator |
| Number of runs | `docs/HYPERPARAMETERS.md` and machine-readable `results/` |
| Variation estimates | GB and VEP sample SD files; ETGP raw-run and recomputed summary CSVs |
| Statistical tests | None claimed where sample size/protocol does not justify a test |
| Final hyperparameters | `docs/HYPERPARAMETERS.md` and exact Hydra configuration files |
| Data access and licensing | `docs/DATA_AND_LICENSES.md`, `data/manifests/datasets.csv` |
| Checkpoint identity | `checkpoints/manifest.csv` |

Run `python scripts/validate_package.py` to check that every referenced file
exists and that the anonymous archive boundary is respected.
