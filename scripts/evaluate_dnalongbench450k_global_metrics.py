"""Evaluate DNALongBench 450k checkpoints with global binary metrics.

The training loop logs functional AUROC/AUPRC per batch and Lightning averages
those values. For long-range tasks with batch size 1 this can become NaN when a
batch contains only one class. This script replays val/test once, collects all
logits and labels, then computes sklearn metrics over the full split.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

from train import SequenceLightningModule
import src.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Hydra run directory.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path. Defaults to run-dir/checkpoints/last.ckpt.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--output-dir", default=None, help="Directory for metrics and prediction npz files.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--precision", choices=["32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only: limit batches per split per rank.")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--save-predictions", action="store_true", help="Save logits/labels as npz.")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank))))
    if world_size <= 1:
        return False, 0, 1, 0

    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29589")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank


def cleanup_distributed(is_dist):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def load_config(run_dir, args):
    config_path = Path(run_dir) / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    config = utils.train.process_config(config)
    OmegaConf.set_struct(config, False)

    # This is a full fine-tuned checkpoint, not a pretraining checkpoint. Disable
    # backbone-only checkpoint hooks before loading the Lightning state dict.
    config.train.pretrained_model_path = None
    if config.train.get("pretrained_model_state_hook", None) is not None:
        config.train.pretrained_model_state_hook.update({"_name_": None})

    if args.batch_size is not None:
        config.dataset.batch_size_eval = args.batch_size
    if args.num_workers is not None:
        config.loader.num_workers = args.num_workers
    config.loader.drop_last = False
    config.loader.shuffle = False
    config.loader.pin_memory = bool(config.loader.get("pin_memory", True))
    return config


def load_model(config, ckpt_path, device, strict):
    model = SequenceLightningModule(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(
            "CHECKPOINT_LOAD_WARN "
            + json.dumps({"missing": list(missing), "unexpected": list(unexpected)}, sort_keys=True)
        )
    model.to(device)
    model.eval()
    return model


def split_dataset(model, split):
    if split == "val":
        return model.dataset.dataset_val
    if split == "test":
        return model.dataset.dataset_test
    raise ValueError(f"Unsupported split {split!r}")


def make_loader(dataset, config, indices):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=int(config.dataset.batch_size_eval),
        shuffle=False,
        num_workers=int(config.loader.get("num_workers", 0)),
        pin_memory=bool(config.loader.get("pin_memory", True)),
        drop_last=False,
    )


def evaluate_split(model, split, config, device, precision, rank, world_size, max_batches):
    dataset = split_dataset(model, split)
    indices = list(range(rank, len(dataset), world_size))
    loader = make_loader(dataset, config, indices)

    local_indices = []
    logits_all = []
    y_all = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_to_device(batch, device)
            model._process_state(batch, batch_idx, training=False)
            with autocast_context(device, precision):
                logits, y, _ = model.forward(batch)
            logits = logits.view(-1, logits.shape[-1]).float().detach().cpu()
            y = y.view(-1).detach().cpu()

            start = batch_idx * int(config.dataset.batch_size_eval)
            batch_indices = indices[start : start + y.numel()]
            local_indices.append(np.asarray(batch_indices, dtype=np.int64))
            logits_all.append(logits.numpy())
            y_all.append(y.numpy())

    if logits_all:
        local = {
            "indices": np.concatenate(local_indices),
            "logits": np.concatenate(logits_all, axis=0),
            "y": np.concatenate(y_all),
        }
    else:
        local = {
            "indices": np.empty((0,), dtype=np.int64),
            "logits": np.empty((0, 2), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
        }

    if dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]

    if rank != 0:
        return None

    indices_all = np.concatenate([item["indices"] for item in gathered])
    logits_all = np.concatenate([item["logits"] for item in gathered], axis=0)
    y_all = np.concatenate([item["y"] for item in gathered])
    order = np.argsort(indices_all)
    return indices_all[order], logits_all[order], y_all[order]


def compute_metrics(split, indices, logits, y, run_dir, ckpt_path):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    prob_pos = probs[:, 1]
    pred = probs.argmax(axis=1)
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    result = {
        "split": split,
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "n": int(y.shape[0]),
        "target_counts": {str(label): int((y == label).sum()) for label in labels},
        "pred_counts": {str(label): int((pred == label).sum()) for label in labels},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "f1_micro": float(f1_score(y, pred, average="micro")),
        "mcc": float(matthews_corrcoef(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc_macro"] = float(roc_auc_score(y, prob_pos))
        result["roc_auc_micro"] = result["roc_auc_macro"]
        result["auprc_macro"] = float(average_precision_score(y, prob_pos))
        result["auprc_micro"] = result["auprc_macro"]
        result["f1_binary"] = float(f1_score(y, pred, average="binary"))
        result["log_loss"] = float(log_loss(y, probs, labels=[0, 1]))
    else:
        result["roc_auc_macro"] = float("nan")
        result["roc_auc_micro"] = float("nan")
        result["auprc_macro"] = float("nan")
        result["auprc_micro"] = float("nan")
        result["f1_binary"] = float("nan")
        result["log_loss"] = float("nan")
    return result, probs, pred


def main():
    args = parse_args()
    is_dist, rank, world_size, local_rank = setup_distributed()
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        run_dir = Path(args.run_dir)
        ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "last.ckpt"
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "global_metrics"
        config = load_config(run_dir, args)
        model = load_model(config, ckpt_path, device, args.strict)

        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)

        for split in args.splits:
            payload = evaluate_split(
                model=model,
                split=split,
                config=config,
                device=device,
                precision=args.precision,
                rank=rank,
                world_size=world_size,
                max_batches=args.max_batches,
            )
            if rank != 0:
                continue

            indices, logits, y = payload
            metrics, probs, pred = compute_metrics(split, indices, logits, y, run_dir, ckpt_path)
            metrics_path = output_dir / f"{split}_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            if args.save_predictions:
                np.savez_compressed(
                    output_dir / f"{split}_predictions.npz",
                    indices=indices,
                    logits=logits,
                    probs=probs,
                    y=y,
                    pred=pred,
                )
            print("DNALONGBENCH_GLOBAL_METRICS " + json.dumps(metrics, sort_keys=True))
    finally:
        cleanup_distributed(is_dist)


if __name__ == "__main__":
    main()
