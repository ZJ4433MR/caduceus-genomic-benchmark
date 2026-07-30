# Results

- `raw/` contains per-seed metrics or complete hyperparameter grids retained
  from experiment outputs.
- `processed/` contains the machine-readable paper-result tables.
- `figures/` is reserved for plots regenerated from the processed CSV files.

No genomic sequence, embedding tensor, model checkpoint, or participant-level
record is stored here.

Run:

```bash
python scripts/verify_reported_results.py
```

to recompute aggregate checks from these files.
