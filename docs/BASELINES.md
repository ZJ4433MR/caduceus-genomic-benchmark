# Baseline Roles

The paper separates published strong baselines from controlled same-budget
baselines.

## Published Strong Baselines

These baselines are used to test whether a shorter-context MLBN model can be
more stable under order reversal than public long-context RC-aware models.

- Caduceus-Ph-131k:
  public Caduceus post-hoc/conjoined baseline trained with 131,072-token
  context for 50k pretraining steps.
- Caduceus-PS-131k:
  public Caduceus parameter-sharing baseline with structural
  reverse-complement equivariance, also trained with 131,072-token context for
  50k pretraining steps.

These are not treated as same-pretraining-budget controls.

## Controlled Same-Budget Baselines

These baselines isolate whether the MLBN architecture helps under the same 1k
pretraining context.

- Mamba2-1k.
- Mamba2-RevPh-1k.
- MLBN-1k.

This is the primary mechanism-control comparison.

## Expert Models

DNALongBench expert models are included only as context where reported by the
benchmark. They are not the primary mechanism comparison because they can use
task-specific signals beyond sequence-only DNA foundation models.

