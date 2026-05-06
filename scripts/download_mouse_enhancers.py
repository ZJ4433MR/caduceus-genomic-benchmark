#!/usr/bin/env python
from genomic_benchmarks.loc2seq import download_dataset

print(download_dataset("dummy_mouse_enhancers_ensembl", version=0, dest_path="data/genomic_benchmarks"))
