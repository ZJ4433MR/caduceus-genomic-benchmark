# Result File Schema

Raw result CSV files should use long format:

```text
benchmark,task,model,comparison_role,pretrain_context,seed,orientation,metric,value
```

Recommended values:

- `orientation`: `normal` or `flipped`.
- `comparison_role`: `published_strong_baseline`, `same_budget_control`,
  `ablation`, or `context_expert`.
- `metric`: for example `accuracy`, `AUROC`, `AUPRC`, `MCC`, or `F1`.

`scripts/aggregate_results.py` computes:

- normal mean and standard deviation.
- flipped mean and standard deviation.
- drop = normal - flipped.
- worst_case = min(normal, flipped).

Keep seed-level raw files in `results/raw/` and generated paper tables in
`results/tables/`.

