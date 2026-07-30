# Reported Results

## GenomicBenchmarks

`results/raw/genomicbenchmarks/fixed_profile_seed_metrics.csv` contains the
five-seed, two-view records used for the packaged GenomicBenchmarks summary.

The Flipped-GARI table row is a task-wise summary over the formal
configurations. Each individual training run selects one checkpoint
using original-order validation accuracy and evaluates that checkpoint on
both views. Task-wise configuration labels are retained in
`flipped_gari_task_variation.csv`.

## VEP

The package includes the paper-reported VEP table and context-length summary,
along with the completed ten-subset hard-label SVC grids for the primary,
pretraining, feature-source, and component analyses. The context launcher uses
the common 1,024-bp readout for 1-, 16-, and 131-kb inputs. The component
diagnostic transfers the original-order-selected setting unchanged to
sequence-reversed evaluation.

## ETGP

`results/processed/etgp/main_table.csv` contains the paper-reported
DNALongBench comparison for the 1-, 2-, and 5-kb Flipped-GARI profiles and the
published external baselines.
