# Third-Party Notices

This package contains or interoperates with third-party software and data.
The Apache-2.0 license in `LICENSE` applies only to source code distributed
under that license. It does not replace the terms of third-party assets.

## Source Code

- Caduceus framework components: Apache License 2.0.
- Mamba and Mamba-2 kernels: licenses distributed by their upstream projects.
- PyTorch, Lightning, Hydra, Hugging Face, scikit-learn, and other dependencies:
  their respective upstream licenses.

## Data

- GenomicBenchmarks: downloaded through the official
  `genomic-benchmarks==0.0.9` package; each underlying dataset remains subject
  to its source terms.
- Causal-eQTL VEP: `InstaDeepAI/genomics-long-range-benchmark`, distributed
  under CC BY-NC-SA 4.0 by its publisher.
- DNALongBench ETGP: Harvard Dataverse DOI `10.7910/DVN/CTEQXX`, version 1,
  released under CC0 1.0. Six unmodified metadata rows are included under
  `data/representative/` with source identifiers.
- hg38 pretraining assets: obtained from the public Basenji/Caduceus storage
  endpoint; the reference genome and annotations remain subject to their
  source terms.

No complete third-party raw dataset is redistributed in this archive. Apart
from the small CC0 ETGP metadata subset above, download scripts retrieve data
directly from the official sources.
