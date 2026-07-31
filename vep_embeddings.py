"""Dump model embeddings for VEP classification task.

"""

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import urllib.request
from urllib.error import URLError
from functools import partial
from os import path as osp
from typing import Dict, Iterable, Iterator, Optional

import fsspec
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from datasets import load_dataset, load_from_disk
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator

from src.utils.train import get_logger

try:
    from caduceus.tokenization_caduceus import CaduceusTokenizer
except ImportError:
    # Some runtime-only JanusDNA environments do not have the optional Mamba
    # kernels needed by caduceus.__init__.  Load the tokenizer module directly.
    import importlib.util

    tokenizer_path = osp.join(osp.dirname(__file__), "caduceus", "tokenization_caduceus.py")
    tokenizer_spec = importlib.util.spec_from_file_location("caduceus_tokenization_direct", tokenizer_path)
    tokenizer_module = importlib.util.module_from_spec(tokenizer_spec)
    assert tokenizer_spec.loader is not None
    tokenizer_spec.loader.exec_module(tokenizer_module)
    CaduceusTokenizer = tokenizer_module.CaduceusTokenizer

VEP_TSS_BUCKETS = (
    (0, 30_000, "0 - 30k"),
    (30_001, 100_000, "30 - 100k"),
    (100_001, None, "100k+"),
)
PREPARED_VEP_METADATA_FILENAME = "prepared_vep_metadata.json"
PREPARED_VEP_MANIFEST_FILENAME = "train_subset_manifest.json"
PREPARED_VEP_SCHEMA_VERSION = 1
log = get_logger(__name__)

_DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def string_reverse_complement(sequence: str) -> str:
    return sequence.translate(_DNA_COMPLEMENT)[::-1]


class DistributedSequentialSampler(Sampler[int]):
    """Shard evaluation data across ranks without dropping or padding samples."""

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None):
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else (
            dist.get_world_size() if dist.is_initialized() else 1
        )
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        if self.num_replicas < 1:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


class IndexedDataset(Dataset):
    """Attach a stable split-local index so orientation dumps can be aligned safely."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        example = dict(self.dataset[index])
        example["sample_index"] = index
        return example


def pooled_window_bounds(window_size_tokens: int) -> tuple[int, int]:
    """Return an exclusive range containing exactly ``window_size_tokens`` offsets."""
    if window_size_tokens < 1:
        raise ValueError(f"window_size_tokens must be positive, got {window_size_tokens}")
    left = window_size_tokens // 2
    right = window_size_tokens - left - 1
    return -left, right + 1

try:
    import enformer_pytorch
except ImportError:
    enformer_pytorch = None


def is_flipped_gari_local(model_name_or_path: Optional[str]) -> bool:
    """Return true when using the local Flipped-GARI checkpoint loader."""
    return (model_name_or_path or "").lower() in {
        "flipped-gari-local",
        "flipped_gari_local",
    }


def is_janusdna_local(model_name_or_path: Optional[str]) -> bool:
    """Return true when using a local JanusDNA checkpoint loader."""
    return (model_name_or_path or "").lower() in {"janusdna-local", "janusdna_local", "local-janusdna"}


def install_retrying_urlretrieve(max_attempts: int, timeout: int):
    """Install retrying urllib downloads for large reference-genome assets."""
    socket.setdefaulttimeout(timeout)
    original_urlretrieve = urllib.request.urlretrieve
    hg38_urls = [
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    ]

    def cached_reference_candidates(target_name: str) -> Iterable[str]:
        search_roots = [
            os.getenv("VEP_REFERENCE_CACHE"),
            os.getenv("HF_DATASETS_CACHE"),
        ]
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            search_roots.append(osp.join(hf_home, "datasets"))

        seen = set()
        for root in search_roots:
            if not root:
                continue
            root = osp.abspath(root)
            if root in seen:
                continue
            seen.add(root)
            if osp.isfile(root):
                if osp.basename(root) == target_name:
                    yield root
                continue
            if not osp.isdir(root):
                continue

            direct = osp.join(root, target_name)
            if osp.isfile(direct):
                yield direct
            for dirpath, _, filenames in os.walk(root):
                if target_name in filenames:
                    yield osp.join(dirpath, target_name)

    def try_cached_reference(url: str, filename: Optional[str], reporthook=None):
        target_name = url.rstrip("/").rsplit("/", 1)[-1]
        if target_name != "hg38.fa.gz" or filename is None:
            return None

        target = osp.abspath(filename)
        for cached_path in cached_reference_candidates(target_name):
            cached_path = osp.abspath(cached_path)
            try:
                if cached_path == target:
                    if osp.getsize(cached_path) > 100_000_000:
                        log.warning(f"Using existing cached reference: {target}")
                        return filename, None
                    continue
                if osp.getsize(cached_path) <= 100_000_000:
                    continue
                os.makedirs(osp.dirname(target), exist_ok=True)
                log.warning(f"Copying cached reference {cached_path} -> {target}")
                shutil.copyfile(cached_path, target)
                if reporthook is not None:
                    size = osp.getsize(target)
                    reporthook(1, size, size)
                return filename, None
            except OSError as exc:
                log.warning(f"Skipping cached reference candidate {cached_path}: {exc}")
        return None

    def retrying_urlretrieve(url, filename=None, reporthook=None, data=None):
        urls = [url]
        if "hgdownload" in url and url.endswith("/hg38.fa.gz"):
            urls = list(dict.fromkeys([url] + hg38_urls))
            cached = try_cached_reference(url, filename, reporthook)
            if cached is not None:
                return cached
        errors = []
        for candidate in urls:
            for attempt in range(1, max_attempts + 1):
                try:
                    log.warning(
                        f"Downloading {candidate} via urlretrieve attempt {attempt}/{max_attempts}"
                    )
                    return original_urlretrieve(candidate, filename, reporthook, data)
                except Exception as exc:
                    errors.append(f"{candidate}: {repr(exc)}")
                    if attempt < max_attempts:
                        time.sleep(min(60, 5 * attempt))
        raise URLError("All urlretrieve attempts failed: " + " | ".join(errors[-5:]))

    urllib.request.urlretrieve = retrying_urlretrieve


class DNAEmbeddingModel(nn.Module):
    """Wrapper around HF model.

    Args:
        model_name_or_path: str, path to HF model.
    """
    def __init__(
            self,
            model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        # Enformer uses different library for loading
        if "enformer" in model_name_or_path.lower():
            if enformer_pytorch is None:
                raise ImportError("enformer_pytorch is required when loading Enformer models.")
            self.backbone = enformer_pytorch.from_pretrained(
                model_name_or_path,
                use_tf_gamma=False,
                use_checkpointing=True
            )
        # NT model is not compatible with AutoModel class
        elif "nucleotide-transformer" in model_name_or_path.lower():
            # NT LM `backbone` is under the `.esm` attribute
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids):
        """Backbone forward pass to retrieve last_hidden_state."""
        if "enformer" in self.model_name_or_path.lower():
            # Enformer forward pass has different signature
            return self.backbone(input_ids, return_embeddings=True)[1]
        return self.backbone(input_ids).last_hidden_state


class FlippedGARILocalEmbeddingModel(nn.Module):
    """Local Flipped-GARI loader retaining legacy checkpoint key names."""

    def __init__(self, config_path: str, checkpoint_path: str):
        super().__init__()
        if not config_path:
            raise ValueError("--flipped_gari_config is required for the local Flipped-GARI loader")
        if not checkpoint_path:
            raise ValueError("--flipped_gari_checkpoint is required for the local Flipped-GARI loader")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        config.pop("_name_", None)
        from src.models.sequence.dna_embedding import DNAEmbeddingModelFlippedGARI

        self.backbone = DNAEmbeddingModelFlippedGARI(**config)
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = self.backbone.state_dict()
        remapped = {}
        dropped = []
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("model.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            if clean_key.startswith("lm_head.") or "torchmetrics" in clean_key:
                continue
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                remapped[clean_key] = value
            else:
                dropped.append(key)
        missing, unexpected = self.backbone.load_state_dict(remapped, strict=False)
        print(
            "FLIPPED_GARI_LOCAL_LOAD "
            f"checkpoint={checkpoint_path} loaded={len(remapped)} "
            f"missing={len(missing)} unexpected={len(unexpected)} dropped={len(dropped)}"
        )
        if missing:
            print(f"FLIPPED_GARI_LOCAL_MISSING_KEYS={missing[:20]}")
        if unexpected:
            print(f"FLIPPED_GARI_LOCAL_UNEXPECTED_KEYS={unexpected[:20]}")
        if missing or unexpected or dropped:
            raise RuntimeError(
                "Flipped-GARI checkpoint audit failed: every embedding/encoder/norm key must load "
                "with the expected shape. Only lm_head/torchmetrics keys may be discarded. "
                f"missing={missing[:20]} unexpected={unexpected[:20]} dropped={dropped[:20]}"
            )

    def forward(self, input_ids):
        output = self.backbone(input_ids)
        if isinstance(output, tuple):
            return output[0]
        return output


class JanusDNALocalEmbeddingModel(nn.Module):
    """Local JanusDNA backbone loader for non-HuggingFace checkpoints."""

    def __init__(self, janusdna_root: str, config_path: str, checkpoint_path: str):
        super().__init__()
        if not janusdna_root:
            raise ValueError("--janusdna_root is required for model_name_or_path=janusdna-local")
        if not config_path:
            raise ValueError("--janusdna_config is required for model_name_or_path=janusdna-local")
        if not checkpoint_path:
            raise ValueError("--janusdna_checkpoint is required for model_name_or_path=janusdna-local")
        janusdna_root = osp.abspath(janusdna_root)
        if janusdna_root not in sys.path:
            sys.path.insert(0, janusdna_root)

        from janusdna.configuration_janusdna import JanusDNAConfig
        from janusdna.modeling_janusdna import JanusDNAModel

        with open(config_path, "r", encoding="utf-8") as f:
            config_blob = json.load(f)
        config = dict(config_blob.get("config", config_blob))
        config.pop("_target_", None)
        # Router-loss logits and training checkpointing are unnecessary for frozen embedding extraction.
        config["output_router_logits"] = False
        config["gradient_checkpointing"] = False
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            # The public JanusDNA checkpoint records FlashAttention2 as its HF
            # attention implementation.  Keep the architecture unchanged when
            # flash_attn is present; otherwise avoid an initialization-time
            # import error in inference-only smoke tests.
            config["_attn_implementation"] = "eager"
        self.backbone = JanusDNAModel(JanusDNAConfig(**config))
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = self.backbone.state_dict()
        remapped = {}
        dropped = []
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("model.model.", "model.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            if clean_key.startswith("lm_head.") or "torchmetrics" in clean_key:
                continue
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                remapped[clean_key] = value
            else:
                dropped.append(key)
        missing, unexpected = self.backbone.load_state_dict(remapped, strict=False)
        print(
            "JANUSDNA_LOCAL_LOAD "
            f"checkpoint={checkpoint_path} loaded={len(remapped)} "
            f"missing={len(missing)} unexpected={len(unexpected)} dropped={len(dropped)}"
        )
        if missing:
            print(f"JANUSDNA_LOCAL_MISSING_KEYS={missing[:20]}")
        if unexpected:
            print(f"JANUSDNA_LOCAL_UNEXPECTED_KEYS={unexpected[:20]}")
        if dropped:
            print(f"JANUSDNA_LOCAL_DROPPED_KEYS={dropped[:20]}")
        if missing or unexpected or dropped:
            raise RuntimeError(
                "JanusDNA checkpoint audit failed: every backbone key must load "
                "with the expected shape. "
                f"missing={missing[:20]} unexpected={unexpected[:20]} dropped={dropped[:20]}"
            )

    def forward(self, input_ids):
        output = self.backbone(input_ids, return_dict=False)
        hidden_states = output[0] if isinstance(output, tuple) else output
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = torch.stack(hidden_states, dim=0).sum(dim=0)
        return hidden_states


class EnformerTokenizer:
    """Enformer tokenizer."""
    # Order is important here! (See: https://github.com/lucidrains/enformer-pytorch?tab=readme-ov-file#usage)
    pad_token = "P"  # Padding token should be a character to avoid issues with tokenization
    encode_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, pad_token: -1}

    @classmethod
    def encode(
            cls, seq: str, max_length: Optional[int] = None, truncation: Optional[bool] = False
    ) -> Iterable[int]:
        """Convert bp to token ids."""
        if max_length is not None:
            assert max_length >= 0, "max_length should be a positive integer."
            if len(seq) < max_length:
                seq = seq + cls.pad_token * (max_length - len(seq))
            elif truncation:
                seq = seq[:max_length]
        return [cls.encode_map[bp] for bp in seq.upper()]

    @classmethod
    def batch_encode_plus(
            cls, seqs: Iterable[str], max_length: Optional[int] = None, truncation: Optional[bool] = False,
            **kwargs,  # ensures compatibility with HF tokenizer-like API
    ) -> Dict[str, Iterable[Iterable[int]]]:
        """Batch encode sequences using HF tokenizer-like API."""
        input_ids = [cls.encode(seq, max_length=max_length, truncation=truncation) for seq in seqs]
        return {"input_ids": input_ids}


def setup_distributed():
    """Set environment variables for distributed runs."""
    dist.init_process_group("nccl")


def cleanup_distributed():
    """Clean up processes from distributed runs."""
    dist.destroy_process_group()


def fsspec_exists(filename):
    """Check if file exists in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


# Processing functions
def recast_chromosome_tissue_dist2TSS(examples):
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }


def tokenize_variants(examples, tokenizer, max_length: int):
    """Tokenize sequence.

    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
    """

    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    ref_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    ref_flip_tokenized = tokenizer.batch_encode_plus(
        [seq[::-1] for seq in examples["ref_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_flip_tokenized = tokenizer.batch_encode_plus(
        [seq[::-1] for seq in examples["alt_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )

    return {
        "ref_input_ids": ref_tokenized["input_ids"],
        "alt_input_ids": alt_tokenized["input_ids"],
        "ref_rc_input_ids": ref_rc_tokenized["input_ids"],
        "alt_rc_input_ids": alt_rc_tokenized["input_ids"],
        "ref_flip_input_ids": ref_flip_tokenized["input_ids"],
        "alt_flip_input_ids": alt_flip_tokenized["input_ids"],
    }


def find_first_difference(ref_values, alt_values, center: Optional[int] = None) -> int:
    """Find the first differing position, checking a center guess first."""
    n = min(len(ref_values), len(alt_values))
    if n == 0:
        return -1
    if center is None:
        center = n // 2
    center = max(0, min(center, n - 1))
    if ref_values[center] != alt_values[center]:
        return center

    # NumPy performs the long equality scan in C, which matters for 131k VEP inputs.
    if isinstance(ref_values, str) and isinstance(alt_values, str):
        ref_array = np.frombuffer(ref_values.encode("ascii"), dtype=np.uint8, count=n)
        alt_array = np.frombuffer(alt_values.encode("ascii"), dtype=np.uint8, count=n)
    else:
        ref_array = np.asarray(ref_values[:n])
        alt_array = np.asarray(alt_values[:n])
    diff = np.flatnonzero(ref_array != alt_array)
    if len(diff) == 0:
        return -1
    return int(diff[0])


def find_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = len(examples["ref_input_ids"]) // 2
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        idx = find_first_difference(examples["ref_input_ids"], examples["alt_input_ids"], center=idx)
    # Same as above, but for reverse complement
    rc_idx = len(examples["ref_rc_input_ids"]) // 2 - 1
    if examples["ref_rc_input_ids"][rc_idx] == examples["alt_rc_input_ids"][rc_idx]:
        rc_idx = find_first_difference(examples["ref_rc_input_ids"], examples["alt_rc_input_ids"], center=rc_idx)
    return {"variant_idx": idx, "rc_variant_idx": rc_idx}


def select_fixed_vep_train_subset(dataset, args):
    """Apply one paired, label-stratified train subset before tokenization/embedding."""
    if args.train_samples_per_tss_bucket is not None:
        train = dataset["train"]
        source_indices = np.arange(len(train), dtype=np.int64)
        distances = np.asarray(train["distance_to_nearest_tss"])
        label_column = "labels" if "labels" in train.column_names else "label"
        labels = np.asarray(train[label_column])
        selected = []
        bucket_manifest = []
        for bucket_id, (min_distance, max_distance, bucket_name) in enumerate(VEP_TSS_BUCKETS):
            mask = distances >= min_distance
            if max_distance is not None:
                mask &= distances <= max_distance
            candidates = np.flatnonzero(mask)
            sample_size = min(args.train_samples_per_tss_bucket, len(candidates))
            if sample_size < len(candidates):
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=sample_size,
                    random_state=args.train_subset_seed + bucket_id,
                )
                try:
                    local_indices, _ = next(splitter.split(np.zeros(len(candidates)), labels[candidates]))
                    chosen = candidates[local_indices]
                except ValueError:
                    rng = np.random.default_rng(args.train_subset_seed + bucket_id)
                    chosen = rng.choice(candidates, size=sample_size, replace=False)
            else:
                chosen = candidates
            selected.extend(chosen.tolist())
            unique_labels, label_counts = np.unique(labels[chosen], return_counts=True)
            bucket_manifest.append({
                "bucket_id": bucket_id,
                "bucket": bucket_name,
                "available": int(len(candidates)),
                "selected": int(len(chosen)),
                "label_counts": {
                    str(label): int(count) for label, count in zip(unique_labels.tolist(), label_counts.tolist())
                },
            })
        selected = np.asarray(sorted(selected), dtype=np.int64)
        dataset["train"] = train.select(selected.tolist())
        if not dist.is_initialized() or dist.get_rank() == 0:
            manifest_dir = osp.join(args.downstream_save_dir, args.name)
            os.makedirs(manifest_dir, exist_ok=True)
            manifest_path = osp.join(manifest_dir, "train_subset_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "seed": args.train_subset_seed,
                    "samples_per_tss_bucket": args.train_samples_per_tss_bucket,
                    "label_column": label_column,
                    "source_index_definition": "post-N-filter, pre-subset train split index",
                    "selected_source_indices": source_indices[selected].tolist(),
                    "buckets": bucket_manifest,
                }, handle, indent=2)
            log.warning(f"Wrote fixed VEP train subset manifest: {manifest_path}")
        log.warning(
            f"Selected fixed VEP train subset: {len(selected)} examples "
            f"({args.train_samples_per_tss_bucket} per TSS bucket where available)."
        )
    return dataset


def load_vep_dataset(args, datasets_cache_root):
    """Load and lightly normalize the VEP dataset without tokenizing it."""
    install_retrying_urlretrieve(args.download_retries, args.download_timeout)
    dataset = load_dataset(
        "InstaDeepAI/genomics-long-range-benchmark",
        task_name=args.vep_task_name,
        sequence_length=args.seq_len,
        cache_dir=datasets_cache_root,
        load_from_cache=False,
    )
    log.warning("Dataset loaded. Cached to disk:")
    log.warning(osp.dirname(list(dataset.cache_files.values())[0][0]["filename"]))
    try:
        del dataset["validation"]  # `validation` split is empty
    except KeyError:
        pass

    dataset = dataset.filter(
        lambda example: example["ref_forward_sequence"].count('N') < 0.005 * args.seq_len,
        desc="Filter N's"
    )
    dataset = dataset.map(
        recast_chromosome_tissue_dist2TSS,
        remove_columns=["chromosome", "tissue", "distance_to_nearest_tss"],
        desc="Recast chromosome"
    )
    return select_fixed_vep_train_subset(dataset, args)


def expected_prepared_vep_metadata(args):
    """Return the immutable settings that identify a reusable raw VEP dataset."""
    return {
        "schema_version": PREPARED_VEP_SCHEMA_VERSION,
        "vep_task_name": args.vep_task_name,
        "seq_len": args.seq_len,
        "train_samples_per_tss_bucket": args.train_samples_per_tss_bucket,
        "train_subset_seed": args.train_subset_seed,
    }


def _validate_prepared_vep_metadata(args, prepared_path):
    metadata_path = osp.join(prepared_path, PREPARED_VEP_METADATA_FILENAME)
    if not osp.isfile(metadata_path):
        raise FileNotFoundError(
            f"Prepared VEP metadata is missing: {metadata_path}. "
            "Rebuild the cache with --prepare_dataset_only."
        )
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = expected_prepared_vep_metadata(args)
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Prepared VEP dataset settings do not match {prepared_path}: {mismatches}"
        )
    return metadata


def load_prepared_vep_dataset(args):
    """Load a shared raw-sequence VEP dataset and audit its immutable settings."""
    prepared_path = osp.abspath(args.prepared_dataset_path)
    if not osp.isdir(prepared_path):
        raise FileNotFoundError(
            f"Prepared VEP dataset does not exist: {prepared_path}. "
            "Run once with --prepare_dataset_only before distributed embedding extraction."
        )
    metadata = _validate_prepared_vep_metadata(args, prepared_path)
    dataset = load_from_disk(prepared_path)
    split_sizes = {split_name: len(split) for split_name, split in dataset.items()}
    if split_sizes != metadata.get("split_sizes"):
        raise RuntimeError(
            f"Prepared VEP split sizes do not match metadata: "
            f"loaded={split_sizes}, metadata={metadata.get('split_sizes')}"
        )

    if not dist.is_initialized() or dist.get_rank() == 0:
        source_manifest = osp.join(prepared_path, PREPARED_VEP_MANIFEST_FILENAME)
        if args.train_samples_per_tss_bucket is not None and not osp.isfile(source_manifest):
            raise FileNotFoundError(f"Prepared VEP subset manifest is missing: {source_manifest}")
        if osp.isfile(source_manifest):
            output_dir = osp.join(args.downstream_save_dir, args.name)
            os.makedirs(output_dir, exist_ok=True)
            shutil.copyfile(
                source_manifest,
                osp.join(output_dir, PREPARED_VEP_MANIFEST_FILENAME),
            )
    log.warning(f"Loaded audited shared VEP dataset from {prepared_path}: {split_sizes}")
    return dataset


def prepare_shared_vep_dataset(args):
    """Build one atomically published raw-sequence dataset for all model cells."""
    if not args.prepared_dataset_path:
        raise ValueError("--prepared_dataset_path is required with --prepare_dataset_only.")
    prepared_path = osp.abspath(args.prepared_dataset_path)
    if osp.isdir(prepared_path):
        metadata = _validate_prepared_vep_metadata(args, prepared_path)
        dataset = load_from_disk(prepared_path)
        split_sizes = {split_name: len(split) for split_name, split in dataset.items()}
        if split_sizes != metadata.get("split_sizes"):
            raise RuntimeError(
                f"Existing prepared VEP cache is incomplete: loaded={split_sizes}, "
                f"metadata={metadata.get('split_sizes')}"
            )
        log.warning(f"Prepared VEP dataset already exists and passed audit: {prepared_path}")
        return
    if osp.exists(prepared_path):
        raise RuntimeError(f"Prepared VEP target exists but is not a directory: {prepared_path}")

    datasets_cache_root = os.getenv("HF_DATASETS_CACHE")
    if datasets_cache_root is None:
        hf_home = os.getenv("HF_HOME", osp.expanduser("~/.cache/huggingface"))
        datasets_cache_root = osp.join(hf_home, "datasets")
    parent = osp.dirname(prepared_path)
    os.makedirs(parent, exist_ok=True)
    prefix = f".{osp.basename(prepared_path)}.preparing-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as staging_root:
        prep_args = argparse.Namespace(**vars(args))
        prep_args.downstream_save_dir = staging_root
        prep_args.name = "manifest"
        dataset = load_vep_dataset(prep_args, datasets_cache_root)
        staged_dataset = osp.join(staging_root, "dataset")
        dataset.save_to_disk(staged_dataset)

        source_manifest = osp.join(
            staging_root, "manifest", PREPARED_VEP_MANIFEST_FILENAME
        )
        if args.train_samples_per_tss_bucket is not None:
            if not osp.isfile(source_manifest):
                raise RuntimeError(f"VEP preparation did not write its subset manifest: {source_manifest}")
            shutil.copyfile(
                source_manifest,
                osp.join(staged_dataset, PREPARED_VEP_MANIFEST_FILENAME),
            )

        metadata = expected_prepared_vep_metadata(args)
        metadata["split_sizes"] = {
            split_name: len(split) for split_name, split in dataset.items()
        }
        with open(
            osp.join(staged_dataset, PREPARED_VEP_METADATA_FILENAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        # The target is published only after every split and audit file is complete.
        os.rename(staged_dataset, prepared_path)
    _validate_prepared_vep_metadata(args, prepared_path)
    log.warning(f"Published shared VEP dataset atomically: {prepared_path}")


def prepare_dataset(args, tokenizer):
    """Prepare or load the tokenized dataset."""
    # Data Preprocessing
    num_tokens = args.seq_len // args.bp_per_token

    # Load data
    datasets_cache_root = os.getenv("HF_DATASETS_CACHE")
    if datasets_cache_root is None:
        hf_home = os.getenv("HF_HOME", osp.expanduser("~/.cache/huggingface"))
        datasets_cache_root = osp.join(hf_home, "datasets")
    cache_dir = osp.join(
        datasets_cache_root, "InstaDeepAI___genomics-long-range-benchmark",
        args.vep_task_name, f"seqlen={args.seq_len}"
    )
    if "nucleotide-transformer" in args.model_name_or_path.lower():  # NT uses 6-mers, so tokenization is different
        preprocessed_cache_file = osp.join(cache_dir, "6mer_token_preprocessed")

    elif "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        preprocessed_cache_file = osp.join(cache_dir, "enformer_char_token_preprocessed")
    else:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed")
    log.warning(f"Cache dir: {cache_dir}")
    log.warning(f"Cache dir preprocessed: {preprocessed_cache_file}")

    if args.stream_tokenize:
        log.warning("Using streaming tokenization; full tokenized VEP dataset will not be saved to disk.")
        if args.prepared_dataset_path:
            return load_prepared_vep_dataset(args)
        return load_vep_dataset(args, datasets_cache_root)

    if not fsspec_exists(preprocessed_cache_file):
        if dist.get_rank() == 0:
            dataset = load_vep_dataset(args, datasets_cache_root)
            dataset = dataset.map(
                partial(tokenize_variants, tokenizer=tokenizer, max_length=num_tokens),
                batch_size=args.tokenize_batch_size,
                batched=True,
                remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize"
            )
            dataset = dataset.map(find_variant_idx, desc="Find variant idx")
            dataset.save_to_disk(preprocessed_cache_file)
    dist.barrier()  # Processes need to wait for dataset to be saved to disk (if not already done)
    dataset = load_from_disk(preprocessed_cache_file)
    log.warning(f"Loaded preprocessed dataset from {preprocessed_cache_file}")
    log.warning(dataset)
    return dataset


def get_backbone_model(args, device):
    """Get the backbone model."""

    if is_flipped_gari_local(args.model_name_or_path):
        model = FlippedGARILocalEmbeddingModel(
            config_path=args.flipped_gari_config,
            checkpoint_path=args.flipped_gari_checkpoint,
        )
    elif is_janusdna_local(args.model_name_or_path):
        model = JanusDNALocalEmbeddingModel(
            janusdna_root=args.janusdna_root,
            config_path=args.janusdna_config,
            checkpoint_path=args.janusdna_checkpoint,
        )
    else:
        model = DNAEmbeddingModel(
            model_name_or_path=args.model_name_or_path,
        )
    model.eval()
    return DDP(model.to(device))


def concat_storage_dict_values(storage_dict):
    """Helper method that combines lists of tensors in storage_dict into a single torch.Tensor."""
    return {key: torch.cat(storage_dict[key], dim=0) for key in storage_dict.keys()}


def streaming_tokenize_collate(batch, tokenizer, max_length: int, use_sequence_variant_idx: bool):
    """Collate raw VEP examples and tokenize the sequence fields on the fly."""
    sequence_batch = {
        "ref_forward_sequence": [example["ref_forward_sequence"] for example in batch],
        "alt_forward_sequence": [example["alt_forward_sequence"] for example in batch],
    }
    tokenized = tokenize_variants(sequence_batch, tokenizer=tokenizer, max_length=max_length)

    if use_sequence_variant_idx:
        variant_idx = [
            find_first_difference(ref_seq, alt_seq, center=len(ref_seq) // 2)
            for ref_seq, alt_seq in zip(sequence_batch["ref_forward_sequence"], sequence_batch["alt_forward_sequence"])
        ]
    else:
        variant_idx = [
            find_first_difference(ref_ids, alt_ids, center=len(ref_ids) // 2)
            for ref_ids, alt_ids in zip(tokenized["ref_input_ids"], tokenized["alt_input_ids"])
        ]

    output = {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in tokenized.items()
    }
    output["variant_idx"] = torch.tensor(variant_idx, dtype=torch.long)
    output["rc_variant_idx"] = torch.tensor(
        [max_length - idx - 1 if idx >= 0 else -1 for idx in variant_idx],
        dtype=torch.long,
    )
    output["labels"] = torch.tensor([
        example["labels"] if "labels" in example else example["label"]
        for example in batch
    ])
    output["sample_index"] = torch.tensor([example["sample_index"] for example in batch], dtype=torch.long)
    for key in ["chromosome", "distance_to_nearest_tss", "tissue_embed"]:
        output[key] = torch.tensor([example[key] for example in batch])
    return output


def dump_embeddings(args, dataset, model, tokenizer, device):
    """Dump embeddings to disk."""
    def extract_embeddings(item_ref, item_alt, variant_idx):
        """Extract embedding representation from last layer outputs

        Args:
            item_ref: torch.Tensor, shape (batch_size, seq_len, hidden_size) Ref embedding
            item_alt: torch.Tensor, shape (batch_size, seq_len, hidden_size) Alt embedding
            variant_idx: torch.Tensor, shape (batch_size,) Index of variant
        Returns:
            layer_metrics: dict, with values to save to disk
        """
        layer_metrics = {}

        # Compute windowed statistics
        if "enformer" in args.model_name_or_path.lower():
            window_size = args.window_size_bp // 128  # Enformer's receptive field is 128
            # We also need to override variant_idx since Enformer model reduces to target_length of 896
            variant_idx = torch.ones_like(variant_idx) * item_ref.size(1) // 2
        else:
            if args.window_size_bp % args.bp_per_token:
                raise ValueError(
                    f"window_size_bp={args.window_size_bp} must be divisible by "
                    f"bp_per_token={args.bp_per_token}"
                )
            window_size = args.window_size_bp // args.bp_per_token

        start, end = pooled_window_bounds(window_size)
        expanded_indices = torch.arange(start, end, device=item_ref.device).unsqueeze(0) + \
                           variant_idx.unsqueeze(1).to(item_ref.device)
        if expanded_indices.min().item() < 0 or expanded_indices.max().item() >= item_ref.size(1):
            raise ValueError(
                "The requested variant-centered readout exceeds the available sequence. "
                "Set --window_size_bp no larger than the downstream input length."
            )
        tokens_window_ref = torch.gather(
            item_ref, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        tokens_window_alt = torch.gather(
            item_alt, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        layer_metrics["concat_avg_ws"] = torch.cat([tokens_window_ref, tokens_window_alt], dim=-1)
        return layer_metrics

    embeds_path = osp.join(args.downstream_save_dir, args.name)
    os.makedirs(embeds_path, exist_ok=True)
    paired_key = f"{args.eval_orientation}_concat_avg_ws"
    include_paired = not args.skip_paired

    num_tokens = args.seq_len // args.bp_per_token
    collate_fn = DefaultDataCollator(return_tensors="pt")
    if args.stream_tokenize:
        collate_fn = partial(
            streaming_tokenize_collate,
            tokenizer=tokenizer,
            max_length=num_tokens,
            use_sequence_variant_idx=args.bp_per_token == 1,
        )

    dataloader_params = {
        "batch_size": args.embed_dump_batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False,
    }

    # Process label_encoder = preprocessing.LabelEncoder()
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(dataset["test"]["tissue"])
    train_tissue_embed = label_encoder.transform(dataset["train"]["tissue"])
    dataset["train"] = dataset["train"].add_column("tissue_embed", train_tissue_embed)
    test_tissue_embed = label_encoder.transform(dataset["test"]["tissue"])
    dataset["test"] = dataset["test"].add_column("tissue_embed", test_tissue_embed)

    selected_splits = set(args.splits)
    for split_name, split in dataset.items():
        if split_name not in selected_splits:
            continue
        combined_path = osp.join(embeds_path, f"{split_name}_embeds_combined.pt")
        existing_combined = None
        if include_paired and fsspec_exists(combined_path):
            with fsspec.open(combined_path, "rb") as f:
                existing_combined = torch.load(f, map_location="cpu")
            if paired_key in existing_combined:
                log.warning(f"{split_name} already contains {paired_key}, skipping.")
                continue
            if "sample_index" not in existing_combined:
                raise RuntimeError(
                    f"Cannot append {paired_key} to legacy cache {combined_path}: "
                    "sample_index is missing. Re-dump this split with the audited cache format."
                )
            log.warning(f"{split_name} is missing {paired_key}; dumping it for an incremental merge.")

        indexed_split = IndexedDataset(split)
        manual_num_shards = args.manual_num_shards
        manual_shard_index = args.manual_shard_index
        sampler = DistributedSequentialSampler(
            indexed_split,
            num_replicas=manual_num_shards,
            rank=manual_shard_index,
        )
        shard_rank = sampler.rank
        include_canonical = existing_combined is None
        if not include_canonical:
            log.warning(f"Reusing canonical embeddings from {combined_path}.")

        dl = DataLoader(indexed_split, **dataloader_params, sampler=sampler)

        storage_dict = {
            "sample_index": [],
            "chromosome": [],
            "labels": [],
            "distance_to_nearest_tss": [],
            "tissue_embed": [],
        }
        if include_paired:
            storage_dict[paired_key] = []
        if include_canonical:
            storage_dict["concat_avg_ws"] = []

        autocast_enabled = args.autocast_dtype != "fp32"
        autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.autocast_dtype]
        torch.cuda.reset_peak_memory_stats(device)
        split_start = time.perf_counter()

        with torch.no_grad():

            for batch_idx, batch in tqdm(
                    enumerate(dl), total=len(dl), desc=f"[RANK {dist.get_rank()}] Embedding {split_name}",
                    disable=dist.get_rank() != 0  # Only rank 0 updates pbar
            ):
                for key in ["sample_index", "chromosome", "labels", "distance_to_nearest_tss", "tissue_embed"]:
                    storage_dict[key].append(batch[key].to("cpu", non_blocking=True))
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    output_alt = model(batch["alt_input_ids"].to(device))
                    output_ref = model(batch["ref_input_ids"].to(device))
                    if include_paired and args.rcps:
                        num_channels = output_alt.size(-1)
                        if args.eval_orientation == "rc":
                            # Flip along length and channel dims to preserve RC equivariance.
                            output_alt_paired = output_alt[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
                            output_ref_paired = output_ref[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
                        else:
                            output_alt_paired = model(batch["alt_flip_input_ids"].to(device))
                            output_ref_paired = model(batch["ref_flip_input_ids"].to(device))
                            output_alt_paired = output_alt_paired[..., :num_channels // 2].contiguous().flip(dims=[1])
                            output_ref_paired = output_ref_paired[..., :num_channels // 2].contiguous().flip(dims=[1])
                        output_alt = output_alt[..., :num_channels // 2]
                        output_ref = output_ref[..., :num_channels // 2]

                    elif include_paired:
                        paired_alt_key = (
                            "alt_rc_input_ids" if args.eval_orientation == "rc"
                            else "alt_flip_input_ids"
                        )
                        paired_ref_key = (
                            "ref_rc_input_ids" if args.eval_orientation == "rc"
                            else "ref_flip_input_ids"
                        )
                        # Flip along length dim so variant_idx aligns after RC or plain reversal.
                        output_alt_paired = model(batch[paired_alt_key].to(device)).contiguous().flip(dims=[1])
                        output_ref_paired = model(batch[paired_ref_key].to(device)).contiguous().flip(dims=[1])

                metrics = extract_embeddings(
                    item_ref=output_ref,
                    item_alt=output_alt,
                    variant_idx=batch["variant_idx"],
                )
                if include_canonical:
                    for key, value in metrics.items():
                        storage_dict[key].append(value.to("cpu", non_blocking=True))

                if include_paired:
                    metrics_paired = extract_embeddings(
                        item_ref=output_ref_paired,
                        item_alt=output_alt_paired,
                        variant_idx=batch["variant_idx"],
                    )
                    for key, value in metrics_paired.items():
                        storage_dict[f"{args.eval_orientation}_{key}"].append(
                            metrics_paired[key].to("cpu", non_blocking=True)
                        )

                if batch_idx % 100 == 0:
                    # Every machine should print progress updates
                    print(
                        f"[RANK {dist.get_rank()} SHARD {shard_rank}/{sampler.num_replicas}] "
                        f"Completed index: {batch_idx}/{len(dl)}"
                    )

            storage_dict_temp = concat_storage_dict_values(storage_dict)
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - split_start
            samples_dumped = int(storage_dict_temp["sample_index"].numel())
            if args.skip_paired:
                encoded_sequences_per_example = 2
            else:
                encoded_sequences_per_example = 2 if args.rcps and args.eval_orientation == "rc" else 4
            model_tokens = samples_dumped * encoded_sequences_per_example * num_tokens
            performance = {
                "split": split_name,
                "orientation": args.eval_orientation,
                "skip_paired": args.skip_paired,
                "rank": dist.get_rank(),
                "shard_rank": shard_rank,
                "num_shards": sampler.num_replicas,
                "samples_dumped": samples_dumped,
                "encoded_sequences_per_example": encoded_sequences_per_example,
                "sequence_length_bp": args.seq_len,
                "effective_tokens_per_sequence": num_tokens,
                "model_tokens_processed": model_tokens,
                "elapsed_seconds": elapsed_seconds,
                "model_tokens_per_second": model_tokens / elapsed_seconds if elapsed_seconds else 0.0,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            rank_path = osp.join(embeds_path, f"{split_name}_embeds_{args.eval_orientation}_{shard_rank}.pt")
            with fsspec.open(rank_path, "wb") as f:
                torch.save(storage_dict_temp, f)
            performance_path = osp.join(
                embeds_path,
                f"{split_name}_performance_{args.eval_orientation}_{shard_rank}.json",
            )
            with fsspec.open(performance_path, "w") as f:
                json.dump(performance, f, indent=2, sort_keys=True)
            print(f"[RANK {dist.get_rank()}] Saved {split_name} to {rank_path}")
            log.warning("VEP_DUMP_PERFORMANCE %s", json.dumps(performance, sort_keys=True))

def _sort_storage_by_sample_index(storage_dict: dict) -> dict:
    sample_index = storage_dict.get("sample_index")
    if sample_index is None:
        raise RuntimeError("Embedding cache is missing sample_index; cannot verify orientation alignment.")
    if sample_index.ndim != 1 or torch.unique(sample_index).numel() != sample_index.numel():
        raise RuntimeError("Embedding cache has duplicated sample_index values.")
    order = torch.argsort(sample_index)
    return {key: value[order] for key, value in storage_dict.items()}


def combine_embeddings(embeds_path, orientation, splits=("train", "test"), expected_sizes=None):
    """Merge one orientation's rank shards into the audited combined cache."""
    for split in splits:
        rank_prefix = f"{split}_embeds_{orientation}_"
        rank_files = [
            filename for filename in fsspec_listdir(embeds_path)
            if osp.basename(filename).startswith(rank_prefix) and filename.endswith(".pt")
        ]
        if not rank_files:
            log.warning(f"No {orientation} rank files found for {split} in {embeds_path}; skipping combine.")
            continue

        shards = []
        for filename in rank_files:
            log.warning(f"Loading data from: {filename}")
            with fsspec.open(filename, "rb") as f:
                shards.append(torch.load(f, map_location="cpu"))
        new_data = _sort_storage_by_sample_index(concat_storage_dict_values({
            key: [shard[key] for shard in shards]
            for key in shards[0].keys()
        }))

        expected_size = expected_sizes.get(split) if expected_sizes else None
        if expected_size is not None and new_data["sample_index"].numel() != expected_size:
            raise RuntimeError(
                f"{split} emitted {new_data['sample_index'].numel()} samples, expected {expected_size}."
            )

        combined_path = osp.join(embeds_path, f"{split}_embeds_combined.pt")
        paired_key = f"{orientation}_concat_avg_ws"
        if fsspec_exists(combined_path):
            with fsspec.open(combined_path, "rb") as f:
                combined = _sort_storage_by_sample_index(torch.load(f, map_location="cpu"))
            if not torch.equal(combined["sample_index"], new_data["sample_index"]):
                raise RuntimeError(f"{split} {orientation} cache sample indices do not match the canonical cache.")
            combined[paired_key] = new_data[paired_key]
            storage_dict = combined
        else:
            storage_dict = new_data

        log.warning(f"Saving combined data to: {combined_path}")
        with fsspec.open(combined_path, "wb") as f:
            torch.save(storage_dict, f)


def main(args):
    """Main entry point."""
    if (args.manual_num_shards is None) != (args.manual_shard_index is None):
        raise ValueError("--manual_num_shards and --manual_shard_index must be set together.")
    if args.manual_num_shards is not None:
        if args.manual_num_shards < 1:
            raise ValueError("--manual_num_shards must be positive.")
        if not 0 <= args.manual_shard_index < args.manual_num_shards:
            raise ValueError("--manual_shard_index must be in [0, manual_num_shards).")

    # Reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Init distributed
    log.warning("Initializing distributed...")
    dist.init_process_group("nccl")
    print(f"[RANK {dist.get_rank()}] Distributed initialized: rank {dist.get_rank()}")  # All processes print this
    # Setup device
    device = torch.device(f"cuda:{dist.get_rank()}")
    print(f"[RANK {dist.get_rank()}] Using device: {device}.")  # All processes print this

    # Init tokenizer
    if is_flipped_gari_local(args.model_name_or_path) or is_janusdna_local(args.model_name_or_path):
        tokenizer = CaduceusTokenizer(model_max_length=args.seq_len)
    elif "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        tokenizer = EnformerTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Get dataset
    dist.barrier()
    dataset = prepare_dataset(args, tokenizer)

    # Get model
    dist.barrier()
    model = get_backbone_model(args, device)
    log.warning("Model loaded.")

    # Dump embeddings
    dist.barrier()
    dump_embeddings(args, dataset, model, tokenizer, device)

    # Combine embeddings into single file
    dist.barrier()
    if dist.get_rank() == 0 and not args.skip_combine:
        expected_sizes = {split_name: len(split) for split_name, split in dataset.items() if split_name in args.splits}
        combine_embeddings(
            osp.join(args.downstream_save_dir, args.name),
            orientation=args.eval_orientation,
            splits=args.splits,
            expected_sizes=expected_sizes,
        )
    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_len", type=int, default=131072,
                        help="Sequence length (in bp)..")
    parser.add_argument("--bp_per_token", type=int, default=1,
                        help="Number of base pairs per token.")
    parser.add_argument("--window_size_bp", type=int, default=1536,
                        help="Variant-centered readout width in base pairs. Use 1024 for the common-readout context study.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--flipped_gari_config", dest="flipped_gari_config", type=str, default=None,
                        help="Path to the local Flipped-GARI downstream config.")
    parser.add_argument("--flipped_gari_checkpoint", dest="flipped_gari_checkpoint", type=str, default=None,
                        help="Path to the local Flipped-GARI checkpoint.")
    parser.add_argument("--janusdna_root", type=str, default=None,
                        help="Path to a local JanusDNA repository for model_name_or_path=janusdna-local.")
    parser.add_argument("--janusdna_config", type=str, default=None,
                        help="Path to local JanusDNA model_config.json for model_name_or_path=janusdna-local.")
    parser.add_argument("--janusdna_checkpoint", type=str, default=None,
                        help="Path to local JanusDNA pretraining checkpoint for model_name_or_path=janusdna-local.")
    parser.add_argument("--vep_task_name", type=str, default="variant_effect_causal_eqtl",
                        help="Task key for InstaDeepAI/genomics-long-range-benchmark VEP data.")
    parser.add_argument("--download_retries", type=int, default=4,
                        help="Number of urlretrieve attempts per reference-genome URL.")
    parser.add_argument("--download_timeout", type=int, default=300,
                        help="Socket timeout in seconds for reference-genome downloads.")
    parser.add_argument("--tokenize_batch_size", type=int, default=16,
                        help="Batch size for VEP sequence tokenization.")
    parser.add_argument("--train_samples_per_tss_bucket", type=int, default=None,
                        help="Before tokenization/embedding, keep a fixed stratified train subset per TSS bucket.")
    parser.add_argument("--train_subset_seed", type=int, default=2222,
                        help="Seed for the fixed pre-embedding VEP train subset.")
    parser.add_argument("--stream_tokenize", default=False, action="store_true",
                        help="Tokenize VEP examples inside the embedding DataLoader instead of saving token ids.")
    parser.add_argument("--prepared_dataset_path", type=str, default=None,
                        help="Shared raw-sequence VEP DatasetDict created by --prepare_dataset_only.")
    parser.add_argument("--prepare_dataset_only", default=False, action="store_true",
                        help="Build and audit --prepared_dataset_path without initializing distributed GPUs.")
    parser.add_argument("--eval_orientation", choices=["rc", "flip"], default="rc",
                        help="Paired orientation to dump: reverse-complement (rc) or plain reversal (flip).")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
                        help="Dataset splits to dump. Useful for adding sequence-reversed test embeddings to an existing train dump.")
    parser.add_argument("--autocast_dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="Autocast dtype for embedding forward passes. Use bf16/fp32 for long Flipped-GARI runs.")
    parser.add_argument("--downstream_save_dir", type=str, default="./outputs/downstream/vep_embeddings",
                        help="Directory to save downstream task.")
    parser.add_argument("--name", type=str, default=None, help="Embeddings model name.")
    parser.add_argument("--rcps", default=False, action="store_true", help="Use RCPS.")
    parser.add_argument("--no-rcps", dest="rcps", action="store_false", help="Do not use RCPS.")
    parser.add_argument("--embed_dump_batch_size", type=int, default=1,
                        help="Batch size for embedding dump.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--manual_num_shards", type=int, default=None,
                        help="Override distributed world size for manually submitted shard jobs.")
    parser.add_argument("--manual_shard_index", type=int, default=None,
                        help="Override distributed rank for manually submitted shard jobs.")
    parser.add_argument("--skip_combine", default=False, action="store_true",
                        help="Only write this shard; do not combine split shards at the end.")
    parser.add_argument("--skip_paired", default=False, action="store_true",
                        help="Dump only canonical ref/alt embeddings and skip RC/flip paired forward passes.")
    opts, _ = parser.parse_known_args()
    log.warning("*** Args ************************")
    for k, v in vars(opts).items():
        log.warning(f"  - {k}: {v}")
    log.warning("******************************\n")

    if opts.prepare_dataset_only:
        prepare_shared_vep_dataset(opts)
    else:
        main(opts)
