#!/usr/bin/env python3
"""Recompute packaged table aggregates and report provenance differences."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
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
    print(f"VEP_OK rows={len(rows)}")


def verify_etgp() -> None:
    raw = read_csv("results/raw/etgp/flipped_gari_available_runs.csv")
    expected = {
        row["pretraining_context_kb"]: row
        for row in read_csv("results/processed/etgp/recomputed_available_run_summary.csv")
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["pretraining_context_kb"]].append(row)

    for context, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        auroc = [float(row["test_auroc"]) for row in rows]
        auprc = [float(row["test_auprc"]) for row in rows]
        summary = expected[context]
        assert_close(statistics.fmean(auroc), float(summary["test_auroc_mean"]), 5e-11, f"ETGP {context}kb AUROC")
        assert_close(statistics.fmean(auprc), float(summary["test_auprc_mean"]), 5e-11, f"ETGP {context}kb AUPRC")
        if len(rows) > 1:
            assert_close(
                statistics.stdev(auroc),
                float(summary["test_auroc_sample_sd"]),
                5e-11,
                f"ETGP {context}kb AUROC SD",
            )
            assert_close(
                statistics.stdev(auprc),
                float(summary["test_auprc_sample_sd"]),
                5e-11,
                f"ETGP {context}kb AUPRC SD",
            )

    submitted = {
        row["pretraining_context_kb"]: row
        for row in read_csv("results/processed/etgp/main_table_as_submitted.csv")
        if row["model"] == "Flipped-GARI"
    }
    for context in ("2", "5"):
        aggregate = expected[context]
        submitted_row = submitted[context]
        print(
            "ETGP_RECONCILIATION "
            f"context={context}kb "
            f"submitted_AUROC={submitted_row['test_auroc']} "
            f"available_run_mean_AUROC={float(aggregate['test_auroc_mean']):.6f} "
            f"submitted_AUPRC={submitted_row['test_auprc']} "
            f"available_run_mean_AUPRC={float(aggregate['test_auprc_mean']):.6f}"
        )
    print(f"ETGP_OK raw_runs={len(raw)}")


def main() -> None:
    verify_gb()
    verify_vep()
    verify_etgp()
    print("RESULT_CHECKS_COMPLETE")


if __name__ == "__main__":
    main()
