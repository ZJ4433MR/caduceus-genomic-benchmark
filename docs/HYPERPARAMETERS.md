# Hyperparameters

## Masked-Nucleotide Pretraining

All profiles use AdamW with betas `(0.9, 0.95)`, weight decay 0.1,
mask probability 0.15, no sequence-reversal augmentation, and no
reverse-complement augmentation. The released pretraining profiles use seed
2222.

| Profile | Context | d | Blocks | Steps | Peak LR | Warmup | Global / per-GPU batch | GPUs | Precision | Clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| GB | 1,024 bp | 80 | 3 | 16,000 | 1.5e-3 | 1,600 | 1,024 / 32 | 4 | FP16 | 1.0 |
| VEP | 1,024 bp | 192 | 12 | 16,000 | 1.0e-4 | 3,200 | 1,024 / 8 | 8 | FP32 | 0.5 |
| ETGP base | 1,024 bp | 192 | 13 | 16,000 | 1.0e-4 | 3,200 | 1,024 / 8 | 8 | FP32 | 0.5 |
| ETGP 2-kb continuation | 2,048 bp | 192 | 13 | 2,000 | 1.0e-4 | 100 | 512 / 4 | 4 | FP32 | 0.5 |
| ETGP 5-kb continuation | 5,120 bp | 192 | 13 | 5,000 | 1.0e-4 | 100 | 200 / 1 | 4 | FP32 | 0.5 |

The continuation profiles use cosine horizon 8,000 and minimum LR 1.0e-5.

## GenomicBenchmarks

All tasks use 10 epochs, AdamW, weight decay 0.01, split seeds 1--5, no
orientation augmentation, and checkpoint selection by original-order
validation accuracy.

| Task | Public abbreviation | LR | Batch | Readout |
|---|---|---:|---:|---|
| `dummy_mouse_enhancers_ensembl` | ME | 1.0e-3 | 64 | pooled |
| `demo_coding_vs_intergenomic_seqs` | CvI | 1.0e-3 | 256 | pooled |
| `demo_human_or_worm` | HvW | 1.0e-3 | 256 | pooled |
| `human_enhancers_cohn` | HE-C | 1.0e-3 | 256 | pooled |
| `human_enhancers_ensembl` | HE-E | 5.0e-4 | 256 | pooled |
| `human_ensembl_regulatory` | Reg. | 1.0e-3 | 256 | pooled |
| `human_nontata_promoters` | Non-TATA | 1.5e-3 | 256 | pooled |
| `human_ocr_ensembl` | OCR | 5.0e-4 | 256 | pooled, dropout 0.05 |

The submitted aggregate uses completed formal task-wise configurations. Each
individual run uses one original-order-validation-selected checkpoint for both
evaluation views, but the two aggregate view rows are not a strictly paired
orientation effect.

## Causal-eQTL VEP

- Frozen Flipped-GARI encoder: `d=192`, 12 blocks.
- Primary downstream input: 131,072 bp.
- Primary/pretraining-study readout: 1,536 bp centered on the variant.
- Context-study readout: 1,024 bp for 1-, 16-, and 131-kb inputs.
- Training subset: 5,000 sampled variants per distance stratum.
- Probe seeds: 1--10 for primary studies.
- Probe: `StandardScaler` followed by RBF-SVC.
- Grid: `C in {0.1, 0.3, 1, 3, 5, 10, 30, 100}`.
- Reported metric: hard-label AUROC.
- Feature vector: reference and alternative pooled representations with tissue
  covariates for the primary protocol.

The retained VEP benchmark implementation selects the displayed setting from
the evaluation grid for each stated condition. The component diagnostic
transfers the original-order-selected setting unchanged to the
sequence-reversed evaluation. Its four cells are each pretrained once with
seed 2222; SVC protocol seeds 1 and 2 reuse the fixed 5,000-example pool and
therefore are protocol repeats rather than independent encoder replicates.

## Direct 450-kb ETGP

- Complete downstream input: 450,000 bp.
- Full-backbone fine-tuning with activation checkpointing.
- BF16, global batch 8, per-GPU microbatch 1.
- Five fine-tuning epochs for every context and seed.
- AdamW betas `(0.9, 0.95)`, weight decay 0.1, clip 1.0.
- Task-head peak LR: 6.0e-4.
- Backbone LR: 6.0e-4 at 1 kb, 5.0e-5 at 2 kb, and 2.0e-5 at 5 kb.
- Scheduler: proportional cosine decay over all five epochs with 15% warmup.
- Seeds: 2222 and 3333 for every context.
- Metrics: AUROC and AUPRC from continuous scores over the complete split.

For each context and seed, the checkpoint with the highest full-split
validation AUROC is selected and evaluated once on the test split. The two
seed-level test metrics are summarized by their mean and sample standard
deviation.
