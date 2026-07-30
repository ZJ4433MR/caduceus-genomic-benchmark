# Implementation-to-Paper Map

| Paper concept | Implementation |
|---|---|
| Canonical and flipped internal streams | `MLBN_encoder` in `src/models/sequence/mlbn.py` |
| Shared Mamba-2 processing | shared pipeline modules constructed by the Flipped-GARI encoder |
| Coordinate-frame conversion | flip/align operations in the intermediate block forward path |
| Layer-wise cross-stream interaction | `CrossAttention_pro` and intermediate block update logic |
| Discrepancy-aware terminal fusion | terminal query construction controlled by `discrepancy_mode` |
| Public downstream backbone | `DNAEmbeddingModelMLBN` with registry alias `dna_embedding_flipped_gari` |
| Masked-nucleotide prediction | `MLBNLMHeadModel`, `src/dataloaders/utils/mlm.py`, and `dataset.mlm=true` |
| Original-order/sequence-reversed GB evaluation | `reverse_sequence` in the GB dataset and `scripts/evaluate_genomic_benchmark_split.py` |
| VEP coordinate restoration and paired embedding extraction | `vep_embeddings.py` |
| VEP scaler/SVC protocol | `vep_svm_eval.py` |
| Direct 450-kb ETGP loading | `DNALongBenchDataset` and `configs/pipeline/dnalongbench_450k.yaml` |
| Full-split ETGP AUROC/AUPRC | `scripts/evaluate_dnalongbench450k_global_metrics.py` |
| Four component-diagnostic cells | `configs/model/diagnostics/flipped_gari_vep_a*b*.yaml` |

The Python class names containing `MLBN` are legacy serialization identifiers,
not a second model. Public Hydra aliases and all reviewer-facing documentation
use Flipped-GARI.
