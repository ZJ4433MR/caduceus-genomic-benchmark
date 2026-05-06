#!/usr/bin/env python
"""Fine-tune Caduceus on the first GenomicBenchmarks downstream task.

This intentionally avoids the original Lightning checkpoint path and uses the
public Hugging Face Caduceus weights, so pre-training data is not needed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from genomic_benchmarks.loc2seq import download_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caduceus.configuration_caduceus import CaduceusConfig  # noqa: E402
from caduceus.modeling_caduceus import CaduceusForSequenceClassification  # noqa: E402
from caduceus.tokenization_caduceus import CaduceusTokenizer  # noqa: E402


@dataclass(frozen=True)
class Example:
    sequence: str
    label: int


class SequenceDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: CaduceusTokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        encoded = self.tokenizer(
            example.sequence,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(example.label, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_sequence(path: Path) -> str:
    return "".join(line.strip().upper() for line in path.read_text().splitlines() if line.strip())


def load_split(base_dir: Path, split: str) -> tuple[list[Example], list[str]]:
    split_dir = base_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    label_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    examples: list[Example] = []
    for label, label_name in enumerate(label_names):
        for seq_path in sorted((split_dir / label_name).iterdir()):
            if seq_path.is_file():
                examples.append(Example(sequence=read_sequence(seq_path), label=label))
    return examples, label_names


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: CaduceusForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    use_amp = fp16 and device.type == "cuda"
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
        preds.extend(outputs.logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(labels, preds)) if labels else float("nan"),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = args.proxy

    cache_dir = Path(args.data_cache).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_dir = cache_dir / args.dataset
    if not base_dir.exists():
        downloaded = download_dataset(args.dataset, version=0, dest_path=str(cache_dir))
        base_dir = Path(downloaded) if downloaded else base_dir
    if not base_dir.exists():
        base_dir = cache_dir / args.dataset

    train_examples, label_names = load_split(base_dir, "train")
    test_examples, _ = load_split(base_dir, "test")
    indices = list(range(len(train_examples)))
    labels = [ex.label for ex in train_examples]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = CaduceusTokenizer(model_max_length=args.max_length, padding_side="left")
    train_dataset = SequenceDataset(train_examples, tokenizer, args.max_length)
    test_dataset = SequenceDataset(test_examples, tokenizer, args.max_length)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    config = CaduceusConfig.from_pretrained(args.model_id, cache_dir=args.hf_cache)
    config.num_labels = len(label_names)
    config.problem_type = "single_label_classification"
    config.use_cache = False

    model = CaduceusForSequenceClassification.from_pretrained(
        args.model_id,
        config=config,
        cache_dir=args.hf_cache,
        ignore_mismatched_sizes=True,
    )
    if args.freeze_backbone:
        for parameter in model.caduceus.parameters():
            parameter.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    use_amp = args.fp16 and device.type == "cuda"

    metadata = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "labels": label_names,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_examples),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "device": str(device),
        "freeze_backbone": args.freeze_backbone,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)
            progress.set_postfix(loss=f"{np.mean(running_losses[-20:]):.4f}")

        val_metrics = evaluate(model, val_loader, device, args.fp16)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(running_losses)),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(epoch_record, indent=2))

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            model.save_pretrained(output_dir / "best_model")

    best_model = CaduceusForSequenceClassification.from_pretrained(output_dir / "best_model", config=config)
    best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device, args.fp16)

    result = {
        **metadata,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dummy_mouse_enhancers_ensembl")
    parser.add_argument("--model-id", default="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3")
    parser.add_argument("--output-dir", default="outputs/downstream/first_genomic_benchmark")
    parser.add_argument("--data-cache", default="data/genomic_benchmarks")
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--max-length", type=positive_int, default=1024)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--eval-batch-size", type=positive_int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
