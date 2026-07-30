#!/usr/bin/env python3
"""Check internal consistency of the packaged paper-result tables."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def verify_gb() -> None:
    rows = read_csv("results/processed/genomicbenchmarks/main_table.csv")
    task_columns = ["ME", "CvI", "HvW", "HE-C", "HE-E", "Reg", "Non-TATA", "OCR"]
    for row in rows:
        recomputed = statistics.fmean(float(row[column]) for column in task_columns)
        assert_close(
            recomputed,
            float(row["average"]),
            8e-5,
            f"GB {row['model']} {row['evaluation_view']}",
        )
    print(f"GB_OK rows={len(rows)}")


def verify_vep() -> None:
    rows = read_csv("results/processed/vep/main_table.csv")
    for row in rows:
        recomputed = statistics.fmean(
            float(row[column]) for column in ("near_mean", "mid_mean", "distal_mean")
        )
        assert_close(
            recomputed,
            float(row["macro_auroc"]),
            7e-5,
            f"VEP {row['model']} {row['evaluation_view']}",
        )
    context_rows = read_csv("results/processed/vep/context_length_table.csv")
    for row in context_rows:
        recomputed = statistics.fmean(
            float(row[column]) for column in ("near_auroc", "mid_auroc", "distal_auroc")
        )
        assert_close(
            recomputed,
            float(row["macro_auroc"]),
            7e-5,
            f"VEP context {row['input_length_kb']}kb",
        )
    print(f"VEP_OK rows={len(rows)} context_rows={len(context_rows)}")


def verify_etgp() -> None:
    rows = read_csv("results/processed/etgp/main_table.csv")
    flipped_contexts = {
        row["pretraining_context_kb"]
        for row in rows
        if row["model"] == "Flipped-GARI"
    }
    if flipped_contexts != {"1", "2", "5"}:
        raise AssertionError(
            f"ETGP Flipped-GARI contexts: expected 1, 2, and 5 kb, got {sorted(flipped_contexts)}"
        )
    print(f"ETGP_OK rows={len(rows)}")


def main() -> None:
    verify_gb()
    verify_vep()
    verify_etgp()
    print("RESULT_CHECKS_COMPLETE")


if __name__ == "__main__":
    main()
