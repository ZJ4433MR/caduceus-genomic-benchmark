"""Run the VEP SVM evaluation from dumped embeddings.

This mirrors the original ``vep_svm.ipynb`` evaluation loop, but keeps it
scriptable for cluster runs after ``vep_embeddings.py`` finishes.
"""

import argparse
import json
import random
import time
from os import path as osp
from typing import Tuple

import fsspec
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DIST_TO_TSS = (
    (0, 30_000, "0 - 30k"),
    (30_000, 100_000, "30 - 100k"),
    (100_000, np.inf, "100k+"),
)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_pt(path: str) -> dict:
    with fsspec.open(path, "rb") as f:
        return torch.load(f, map_location="cpu")


def dataset_nan_filter(data: dict, data_keys) -> dict:
    mask_out = None
    for data_key in data_keys:
        if data_key is None or data_key not in data:
            continue
        key_mask = torch.any(data[data_key].isnan(), dim=1)
        mask_out = key_mask if mask_out is None else torch.logical_or(mask_out, key_mask)
    if mask_out is None:
        return data
    return {key: value[~mask_out] for key, value in data.items()}


def dataset_tss_filter(data: dict, min_distance: int, max_distance: float) -> dict:
    distance_mask = (
        (data["distance_to_nearest_tss"] >= min_distance)
        & (data["distance_to_nearest_tss"] <= max_distance)
    )
    return {key: value[distance_mask] for key, value in data.items()}


def build_features(data: dict, key: str, conjoin: bool, use_tissue: bool) -> np.ndarray:
    x = tensor_to_numpy(data[key]).astype(np.float32, copy=False)
    if conjoin:
        x = (x + tensor_to_numpy(data[f"rc_{key}"]).astype(np.float32, copy=False)) / 2.0
    if use_tissue:
        tissue = tensor_to_numpy(data["tissue_embed"])[..., None].astype(np.float32, copy=False)
        x = np.concatenate([x, tissue], axis=-1)
    return x


def run_eval(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_key = args.train_key or args.key
    test_key = args.test_key or args.key
    train_embed_path = args.train_embed_path or args.embed_path
    test_embed_path = args.test_embed_path or args.embed_path
    train_path = osp.join(args.path_to_outputs, train_embed_path, "train_embeds_combined.pt")
    test_path = osp.join(args.path_to_outputs, test_embed_path, "test_embeds_combined.pt")

    train_required_keys = [train_key]
    test_required_keys = [test_key]
    if args.conjoin_train:
        train_required_keys.append(f"rc_{train_key}")
    if args.conjoin_test:
        test_required_keys.append(f"rc_{test_key}")

    print(f"Loading train embeddings: {train_path}")
    train_raw = dataset_nan_filter(load_pt(train_path), data_keys=train_required_keys)
    print(f"Loading test embeddings: {test_path}")
    test_raw = dataset_nan_filter(load_pt(test_path), data_keys=test_required_keys)
    print(
        f"Loaded {args.model_name}: train={len(train_raw[train_key])}, "
        f"test={len(test_raw[test_key])}, train_key={train_key}, "
        f"test_key={test_key}, feature_shape={tuple(test_raw[test_key].shape[1:])}"
    )

    rows = []
    for bucket_id, (min_dist, max_dist, bucket_name) in enumerate(DIST_TO_TSS):
        train_bucket = dataset_tss_filter(train_raw, min_dist, max_dist)
        test_bucket = dataset_tss_filter(test_raw, min_dist, max_dist)
        print(
            f"Bucket {bucket_name}: train={len(train_bucket[train_key])}, "
            f"test={len(test_bucket[test_key])}"
        )

        y_full = tensor_to_numpy(train_bucket["labels"])
        y_test = tensor_to_numpy(test_bucket["labels"])
        x_test = build_features(test_bucket, test_key, args.conjoin_test, args.use_tissue)

        x_full = build_features(train_bucket, train_key, args.conjoin_train, args.use_tissue)
        c_values = (
            [args.fixed_cs_by_bucket[bucket_id]]
            if args.fixed_cs_by_bucket is not None
            else args.cs
        )
        for c_value in c_values:
            for seed in args.seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                sample_size = min(args.sample_size, len(x_full))
                replace = args.sample_size > len(x_full)
                mask = np.random.choice(len(x_full), size=args.sample_size, replace=replace)
                if not replace and sample_size != args.sample_size:
                    mask = mask[:sample_size]

                clf = make_pipeline(StandardScaler(), SVC(C=c_value, random_state=seed))
                start = time.time()
                clf.fit(x_full[mask], y_full[mask])
                if args.score_mode == "hard_label":
                    y_score = clf.predict(x_test)
                else:
                    y_score = clf.decision_function(x_test)
                auroc = roc_auc_score(y_test, y_score)
                elapsed = time.time() - start
                print(
                    f"{args.model_name} bucket={bucket_name} C={c_value} seed={seed} "
                    f"AUROC={auroc:.9f} elapsed={elapsed:.3f}s"
                )
                rows.append(
                    {
                        "model_name": args.model_name,
                        "embed_path": args.embed_path,
                        "train_embed_path": train_embed_path,
                        "test_embed_path": test_embed_path,
                        "train_key": train_key,
                        "test_key": test_key,
                        "score_mode": args.score_mode,
                        "bucket_id": bucket_id,
                        "Distance to TSS": bucket_name,
                        "use_tissue": args.use_tissue,
                        "C": c_value,
                        "seed": seed,
                        "AUROC": auroc,
                    }
                )

    metrics = pd.DataFrame(rows)
    by_hparam = (
        metrics.groupby(["model_name", "Distance to TSS", "use_tissue", "C"], as_index=False)
        .agg(AUROC=("AUROC", "mean"), AUROC_std=("AUROC", "std"))
    )
    best_ids = by_hparam.groupby(["model_name", "Distance to TSS"])["AUROC"].idxmax()
    best = by_hparam.loc[best_ids].sort_values("Distance to TSS")
    macro = float(best["AUROC"].mean())
    best["macro_AUROC_mean_over_buckets"] = macro
    return metrics, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path_to_outputs", default="./outputs/downstream/vep_embeddings")
    parser.add_argument("--embed_path", required=True)
    parser.add_argument("--train_embed_path", default=None)
    parser.add_argument("--test_embed_path", default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--key", default="concat_avg_ws")
    parser.add_argument("--train_key", default=None)
    parser.add_argument("--test_key", default=None)
    parser.add_argument("--cs", nargs="+", type=float, default=[1, 5, 10])
    parser.add_argument(
        "--fixed_cs_by_bucket",
        nargs=3,
        type=float,
        default=None,
        metavar=("C_0_30K", "C_30_100K", "C_100K_PLUS"),
        help="Use one fixed C for each TSS bucket instead of selecting C on this evaluation set.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument(
        "--score_mode",
        choices=["continuous", "hard_label"],
        default="continuous",
        help="Use SVC decision scores or predicted 0/1 labels when computing AUROC.",
    )
    parser.add_argument("--use_tissue", dest="use_tissue", action="store_true", default=True)
    parser.add_argument("--no-use_tissue", dest="use_tissue", action="store_false")
    parser.add_argument("--conjoin_train", dest="conjoin_train", action="store_true", default=False)
    parser.add_argument("--no-conjoin_train", dest="conjoin_train", action="store_false")
    parser.add_argument("--conjoin_test", dest="conjoin_test", action="store_true", default=False)
    parser.add_argument("--no-conjoin_test", dest="conjoin_test", action="store_false")
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.output_prefix or osp.join(
        args.path_to_outputs, f"SVM_results_{args.model_name.replace(' ', '_')}"
    )
    metrics, best = run_eval(args)

    metrics_path = f"{prefix}.csv"
    best_path = f"{prefix}_best.csv"
    summary_path = f"{prefix}_summary.json"
    metrics.to_csv(metrics_path, index=False)
    best.to_csv(best_path, index=False)
    summary = {
        "model_name": args.model_name,
        "embed_path": args.embed_path,
        "train_embed_path": args.train_embed_path or args.embed_path,
        "test_embed_path": args.test_embed_path or args.embed_path,
        "key": args.key,
        "train_key": args.train_key or args.key,
        "test_key": args.test_key or args.key,
        "score_mode": args.score_mode,
        "fixed_cs_by_bucket": args.fixed_cs_by_bucket,
        "macro_AUROC_mean_over_buckets": float(best["macro_AUROC_mean_over_buckets"].iloc[0]),
        "best_by_bucket": best.to_dict(orient="records"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Best hyperparameters by TSS bucket:")
    print(best.to_string(index=False))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved best summary: {best_path}")
    print(f"Saved JSON summary: {summary_path}")


if __name__ == "__main__":
    main()
