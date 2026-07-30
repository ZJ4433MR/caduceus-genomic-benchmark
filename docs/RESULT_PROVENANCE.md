# Result Provenance and Reconciliation

## GenomicBenchmarks

`results/raw/genomicbenchmarks/fixed_profile_seed_metrics.csv` is a complete
five-seed, two-view record for one fixed Flipped-GARI profile.

The submitted Flipped-GARI table row is a task-wise summary over completed
formal configurations. Each individual training run selects one checkpoint
using original-order validation accuracy and evaluates that checkpoint on
both views. The final original-order and sequence-reversed aggregate rows use
different task-wise configuration mixtures, so they are not a strictly paired
orientation effect. Selection labels are retained in
`flipped_gari_task_variation.csv`.

## VEP

The package includes ten-subset hard-label SVC grids for the primary,
pretraining, context-length, and feature-source analyses. The historical
benchmark protocol scans regularization settings on the stated evaluation
condition. The component diagnostic additionally reports
sequence-reversed results at the original-order-selected setting.

The current `vep_embeddings.py` exposes `--window_size_bp` and refuses a
readout wider than the available input. The common-readout context study must
be generated with `--window_size_bp=1024`. Existing context grid files in this
draft were produced before this guard was added and must be replaced by the
planned 1-/16-/131-kb rerun before the final reviewer ZIP is uploaded.

## ETGP

`main_table_as_submitted.csv` preserves the values displayed in the submitted
paper. `flipped_gari_available_runs.csv` records the underlying direct runs
currently available for audit, and `recomputed_available_run_summary.csv`
recomputes their means and sample SDs.

The two-run AUROC means reproduce the displayed 2-kb and 5-kb values after
rounding. Their two-run AUPRC means do not reproduce the displayed AUPRC
values: the available-run means are approximately 0.420 and 0.434, whereas
the submitted table displays 0.445 and 0.446. The latter originated from
earlier single-run summaries. This difference must be reconciled in the
Technical Supplement and final code/data archive; no result file in this
package silently relabels one provenance as the other.
