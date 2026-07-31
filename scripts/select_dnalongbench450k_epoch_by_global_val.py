#!/usr/bin/env python3
"""Select one ETGP checkpoint using complete-validation-split metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--metrics-dir-name", default="global_metrics_epoch_selection"
    )
    parser.add_argument("--link-name", default="selected_val_auroc.ckpt")
    return parser.parse_args()


def load_candidates(
    run_dir: Path, metrics_dir_name: str, epochs: int
) -> list[dict[str, object]]:
    candidates = []
    for epoch_number in range(1, epochs + 1):
        checkpoint = run_dir / "checkpoints" / f"epoch-{epoch_number - 1}.ckpt"
        metrics_path = (
            run_dir
            / metrics_dir_name
            / f"epoch-{epoch_number}"
            / "val_metrics.json"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing validation metrics: {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        auroc = float(metrics["roc_auc_macro"])
        auprc = float(metrics["auprc_macro"])
        if not math.isfinite(auroc) or not math.isfinite(auprc):
            raise ValueError(f"Non-finite metrics in {metrics_path}")
        candidates.append(
            {
                "epoch": epoch_number,
                "checkpoint": str(checkpoint.resolve()),
                "val_metrics_path": str(metrics_path.resolve()),
                "val_auroc": auroc,
                "val_auprc": auprc,
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    candidates = load_candidates(run_dir, args.metrics_dir_name, args.epochs)
    selected = max(
        candidates,
        key=lambda row: (
            float(row["val_auroc"]),
            float(row["val_auprc"]),
            -int(row["epoch"]),
        ),
    )

    link_path = run_dir / "checkpoints" / args.link_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(Path(str(selected["checkpoint"])).name)

    summary = {
        "selection_rule": (
            "maximum complete-validation AUROC; validation AUPRC tie-break; "
            "earlier epoch final tie-break"
        ),
        "selected": selected,
        "selected_checkpoint_link": str(link_path),
        "candidates": candidates,
    }
    summary_path = run_dir / args.metrics_dir_name / "selection.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
