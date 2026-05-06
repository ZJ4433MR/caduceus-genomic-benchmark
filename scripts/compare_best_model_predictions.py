#!/usr/bin/env python
"""Compare saved PH/PS predictions on the first GenomicBenchmarks task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caduceus.modeling_caduceus import CaduceusForSequenceClassification  # noqa: E402
from caduceus.tokenization_caduceus import CaduceusTokenizer  # noqa: E402
from scripts.finetune_first_genomic_benchmark import SequenceDataset, load_split, move_batch  # noqa: E402


@torch.no_grad()
def predict(model_dir: Path, dataset: SequenceDataset, batch_size: int) -> list[int]:
    model = CaduceusForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(input_ids=batch["input_ids"]).logits
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/genomic_benchmarks/dummy_mouse_enhancers_ensembl")
    parser.add_argument("--ps-dir", default="outputs/downstream/first_genomic_benchmark/best_model")
    parser.add_argument("--ph-dir", default="outputs/downstream/first_genomic_benchmark_ph/best_model")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    examples, _ = load_split(Path(args.data_dir), "test")
    labels = [example.label for example in examples]
    tokenizer = CaduceusTokenizer(model_max_length=args.max_length, padding_side="left")
    dataset = SequenceDataset(examples, tokenizer, args.max_length)

    ps_predictions = predict(Path(args.ps_dir), dataset, args.batch_size)
    ph_predictions = predict(Path(args.ph_dir), dataset, args.batch_size)
    ps_correct = sum(pred == label for pred, label in zip(ps_predictions, labels))
    ph_correct = sum(pred == label for pred, label in zip(ph_predictions, labels))
    same_predictions = sum(ps == ph for ps, ph in zip(ps_predictions, ph_predictions))

    result = {
        "test_size": len(labels),
        "ps_correct": ps_correct,
        "ph_correct": ph_correct,
        "ps_accuracy": ps_correct / len(labels),
        "ph_accuracy": ph_correct / len(labels),
        "same_predictions": same_predictions,
        "different_predictions": len(labels) - same_predictions,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
