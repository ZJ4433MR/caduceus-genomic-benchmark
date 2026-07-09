#!/usr/bin/env python
"""Aggregate normal/flipped seed-level runs into paper-table summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "benchmark",
    "task",
    "model",
    "comparison_role",
    "pretrain_context",
    "seed",
    "orientation",
    "metric",
    "value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Seed-level long-format CSV.")
    parser.add_argument("--output", required=True, help="Aggregated output CSV.")
    return parser.parse_args()


def check_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_list}")

    valid_orientations = {"normal", "flipped"}
    bad = sorted(set(df["orientation"]) - valid_orientations)
    if bad:
        raise ValueError(f"Unexpected orientation values: {bad}")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "benchmark",
        "task",
        "model",
        "comparison_role",
        "pretrain_context",
        "metric",
        "orientation",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    index_cols = [
        "benchmark",
        "task",
        "model",
        "comparison_role",
        "pretrain_context",
        "metric",
    ]
    mean_wide = summary.pivot(index=index_cols, columns="orientation", values="mean")
    std_wide = summary.pivot(index=index_cols, columns="orientation", values="std")
    count_wide = summary.pivot(index=index_cols, columns="orientation", values="count")

    out = mean_wide.reset_index()
    out = out.rename(columns={"normal": "normal_mean", "flipped": "flipped_mean"})
    out["normal_std"] = std_wide.get("normal").to_numpy()
    out["flipped_std"] = std_wide.get("flipped").to_numpy()
    out["normal_n"] = count_wide.get("normal").to_numpy()
    out["flipped_n"] = count_wide.get("flipped").to_numpy()
    out["drop_mean"] = out["normal_mean"] - out["flipped_mean"]
    out["worst_case_mean"] = out[["normal_mean", "flipped_mean"]].min(axis=1)

    ordered = [
        "benchmark",
        "task",
        "model",
        "comparison_role",
        "pretrain_context",
        "metric",
        "normal_mean",
        "normal_std",
        "normal_n",
        "flipped_mean",
        "flipped_std",
        "flipped_n",
        "drop_mean",
        "worst_case_mean",
    ]
    return out[ordered].sort_values(["benchmark", "task", "metric", "model"])


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    check_schema(df)
    table = aggregate(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(table)} rows")


if __name__ == "__main__":
    main()

