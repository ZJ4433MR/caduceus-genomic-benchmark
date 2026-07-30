#!/bin/bash

# Download the hg38 files used for masked-nucleotide pretraining.

set -Eeo pipefail

DATA_DIR="${HG38_DATA_DIR:-data/raw/hg38}"
HG38_FASTA_URL="${HG38_FASTA_URL:-https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz}"
HG38_BED_URL="${HG38_BED_URL:-https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed}"

mkdir -p "${DATA_DIR}"

cd "${DATA_DIR}" || exit

if [ ! -f hg38.ml.fa ]; then
  echo "Downloading hg38.ml.fa.gz into ${DATA_DIR}"
  curl -L --fail --retry 5 --retry-delay 10 -C - \
    "${HG38_FASTA_URL}" \
    -o hg38.ml.fa.gz
  echo "Decompressing hg38.ml.fa.gz"
  gunzip -f hg38.ml.fa.gz
else
  echo "hg38.ml.fa already exists"
fi

if [ ! -f human-sequences.bed ]; then
  echo "Downloading human-sequences.bed into ${DATA_DIR}"
  curl -L --fail --retry 5 --retry-delay 10 \
    "${HG38_BED_URL}" \
    -o human-sequences.bed
else
  echo "human-sequences.bed already exists"
fi

ls -lh hg38.ml.fa human-sequences.bed
