# Implementation-to-Paper Map

| Paper concept | Implementation |
|---|---|
| Canonical and flipped internal streams | `FlippedGARIEncoder` in `src/models/sequence/mlbn.py` |
| Shared Mamba-2 processing | shared pipeline modules constructed by the Flipped-GARI encoder |
| Coordinate-frame conversion | flip/align operations in the intermediate block forward path |
| Layer-wise cross-stream interaction | `CrossAttention_pro` and intermediate block update logic |
| Discrepancy-aware terminal fusion | terminal query construction controlled by `discrepancy_mode` |
| Public downstream backbone | `DNAEmbeddingModelFlippedGARI` registered as `dna_embedding_flipped_gari` |
| Masked-nucleotide prediction | `FlippedGARILMHeadModel`, `src/dataloaders/utils/mlm.py`, and `dataset.mlm=true` |
| Original-order/sequence-reversed GB evaluation | `reverse_sequence` in the GB dataset and `scripts/evaluate_genomic_benchmark_split.py` |
| VEP coordinate restoration and paired embedding extraction | `vep_embeddings.py` |
| VEP scaler/SVC protocol | `vep_svm_eval.py` |
| Direct 450-kb ETGP loading | `DNALongBenchDataset` and `configs/pipeline/dnalongbench_450k.yaml` |
| Full-split ETGP AUROC/AUPRC | `scripts/evaluate_dnalongbench450k_global_metrics.py` |
| Four component-diagnostic cells | `configs/model/diagnostics/flipped_gari_vep_a*b*.yaml` |

The implementation file retains a few internal class names containing `MLBN`
for compatibility with the experimental checkpoint loader. Public Python names,
Hydra registry entries, commands, and reviewer-facing documentation use
Flipped-GARI.
