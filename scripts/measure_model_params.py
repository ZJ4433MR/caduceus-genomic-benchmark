#!/usr/bin/env python
"""Measure Caduceus sequence-classification parameter counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caduceus.configuration_caduceus import CaduceusConfig  # noqa: E402
from caduceus.modeling_caduceus import CaduceusForSequenceClassification  # noqa: E402


DEFAULT_MODELS = {
    "PH": "kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
    "PS": "kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
}


def measure_model(name: str, model_id: str, num_labels: int) -> dict[str, object]:
    config = CaduceusConfig.from_pretrained(model_id)
    config.num_labels = num_labels
    config.problem_type = "single_label_classification"
    model = CaduceusForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    backbone = sum(parameter.numel() for parameter in model.caduceus.parameters())
    return {
        "name": name,
        "model_id": model_id,
        "total_params": total,
        "trainable_params": trainable,
        "backbone_params": backbone,
        "classifier_params": total - backbone,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--output", default="outputs/downstream/first_genomic_benchmark/model_params.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [measure_model(name, model_id, args.num_labels) for name, model_id in DEFAULT_MODELS.items()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
