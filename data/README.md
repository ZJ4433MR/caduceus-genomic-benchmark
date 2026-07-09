# Data Layout

Large datasets are not stored in this repository.

Expected local layout:

```text
data/
  hg38/
    hg38.ml.fa
    human-sequences.bed
  genomic_benchmarks/
    <GenomicBenchmarks task directories>
  dnalongbench/
    data_long_range_dna/
      eQTL/
      ETGP/
```

Use the helper scripts where available:

```bash
bash scripts/download_hg38_pretrain_data.sh
python scripts/download_dnalongbench_450k_data.py --help
```

The dataset manifest in `data/manifests/datasets.csv` records the dataset role,
task type, sequence length, metric, and availability route expected by the
paper.

