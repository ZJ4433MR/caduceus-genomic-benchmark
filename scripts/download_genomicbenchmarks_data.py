#!/usr/bin/env python3
"""Download the eight GenomicBenchmarks tasks through the official package."""

from __future__ import annotations

import argparse
from pathlib import Path

from genomic_benchmarks.data_check import is_downloaded
from genomic_benchmarks.loc2seq import download_dataset


TASKS = (
    "dummy_mouse_enhancers_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/genomic_benchmarks"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if is_downloaded(task, cache_path=args.root):
            print(f"already present: {task}")
            continue
        print(f"download: {task}")
        download_dataset(task, version=0, dest_path=args.root)


if __name__ == "__main__":
    main()
