"""Evaluate a genomic benchmark checkpoint with global split metrics.

The training loop logs functional metrics per batch and Lightning averages
those batch values. MCC and F1 are not additive, so this script collects all
predictions for a split and computes sklearn metrics once over the full split.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from train import SequenceLightningModule
import src.utils as utils


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _batch_logits(model, batch, batch_idx):
    model._process_state(batch, batch_idx, training=False)
    logits, y, _ = model.forward(batch)
    return logits.view(-1, logits.shape[-1]), y.view(-1)


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = utils.train.process_config(config)

    split = config.eval.get("split", "test")
    ckpt_path = config.eval.ckpt_path
    out_path = config.eval.get("out_path", None)

    if config.train.get("pretrained_model_state_hook", None) is not None:
        config.train.pretrained_model_state_hook.update({"_name_": None})

    model = SequenceLightningModule(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if split == "test":
        dataloader = model.dataset.test_dataloader(**config.loader)
    elif split == "val":
        dataloader = model.dataset.val_dataloader(**config.loader)
    else:
        raise ValueError(f"Unsupported eval.split={split!r}; expected 'val' or 'test'")

    y_all = []
    pred_all = []
    batch_mcc = []
    batch_f1_binary = []
    batch_accuracy = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_to_device(batch, device)
            logits, y = _batch_logits(model, batch, batch_idx)
            pred = torch.argmax(logits, dim=-1)

            y_np = y.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            y_all.append(y_np)
            pred_all.append(pred_np)

            batch_mcc.append(float(matthews_corrcoef(y_np, pred_np)))
            batch_accuracy.append(float(accuracy_score(y_np, pred_np)))
            try:
                batch_f1_binary.append(float(f1_score(y_np, pred_np, average="binary")))
            except ValueError:
                batch_f1_binary.append(float("nan"))

    y_all = np.concatenate(y_all)
    pred_all = np.concatenate(pred_all)
    labels = sorted(set(y_all.tolist()) | set(pred_all.tolist()))

    result = {
        "split": split,
        "ckpt": ckpt_path,
        "dataset": config.dataset.get("dataset_name", None),
        "metric_config": config.dataset.get("metric", None),
        "n": int(y_all.shape[0]),
        "labels": labels,
        "target_counts": {str(k): int((y_all == k).sum()) for k in labels},
        "pred_counts": {str(k): int((pred_all == k).sum()) for k in labels},
        "confusion_matrix": confusion_matrix(y_all, pred_all, labels=labels).tolist(),
        "global_mcc": float(matthews_corrcoef(y_all, pred_all)),
        "mean_batch_mcc": float(np.mean(batch_mcc)),
        "std_batch_mcc": float(np.std(batch_mcc, ddof=1)) if len(batch_mcc) > 1 else 0.0,
        "global_accuracy": float(accuracy_score(y_all, pred_all)),
        "mean_batch_accuracy": float(np.mean(batch_accuracy)),
        "global_balanced_accuracy": float(balanced_accuracy_score(y_all, pred_all)),
        "global_f1_macro": float(f1_score(y_all, pred_all, average="macro")),
        "global_f1_micro": float(f1_score(y_all, pred_all, average="micro")),
    }
    if len(labels) == 2:
        result["global_f1_binary"] = float(f1_score(y_all, pred_all, average="binary"))
        result["mean_batch_f1_binary"] = float(np.nanmean(batch_f1_binary))

    line = "GLOBAL_GENOMIC_BENCHMARK_EVAL " + json.dumps(result, sort_keys=True)
    print(line)

    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
