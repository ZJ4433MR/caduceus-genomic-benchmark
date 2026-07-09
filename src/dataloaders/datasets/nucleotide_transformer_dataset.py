"""Nucleotide Transformer Benchmarks Dataset.

From: https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
"""

import glob
import os

import torch
from datasets import load_dataset

from src.dataloaders.utils.rc import coin_flip, string_reverse_complement


def plain_sequence_reverse(sequence):
    return sequence[::-1]


NT_DATASET_REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks"


def _filter_task(dataset, dataset_name):
    if dataset_name is not None and "task" in dataset.column_names:
        dataset = dataset.filter(lambda row: row["task"] == dataset_name)
    return dataset


def _load_from_local_parquet(dest_path, dataset_name, split):
    if dest_path is None:
        return None

    roots = [
        os.path.join(dest_path, "nucleotide_transformer_downstream_tasks"),
        dest_path,
    ]
    patterns = []
    for root in roots:
        patterns.extend(
            [
                os.path.join(root, "default", split, "*.parquet"),
                os.path.join(root, split, "*.parquet"),
            ]
        )

    parquet_files = []
    for pattern in patterns:
        parquet_files.extend(glob.glob(pattern))
    parquet_files = sorted(set(parquet_files))
    if not parquet_files:
        return None

    dataset = load_dataset("parquet", data_files={split: parquet_files}, split=split)
    return _filter_task(dataset, dataset_name)


def _load_nucleotide_transformer_dataset(dataset_name, split, dest_path=None):
    local_dataset = _load_from_local_parquet(dest_path, dataset_name, split)
    if local_dataset is not None:
        return local_dataset

    try:
        return load_dataset(NT_DATASET_REPO, name=dataset_name, split=split)
    except Exception as original_error:
        try:
            dataset = load_dataset(NT_DATASET_REPO, name="default", split=split)
        except Exception:
            raise original_error
        return _filter_task(dataset, dataset_name)


class NucleotideTransformerDataset(torch.utils.data.Dataset):

    """
    Loop through fasta file for sequence.
    Returns a generator that retrieves the sequence.
    """

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2,  # default binary classification
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        reverse_sequence=False,
        reverse_aug=False,
        return_lengths=False,
        return_augs=False,
        dest_path=None,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        assert not (conjoin_train and conjoin_test), "conjoin_train and conjoin_test cannot both be True"
        if (conjoin_train or conjoin_test) and rc_aug:
            print("When using conjoin, we turn off rc_aug.")
            rc_aug = False
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.reverse_sequence = reverse_sequence
        self.reverse_aug = reverse_aug
        self.return_lengths = return_lengths

        self.split = split

        # For NT tasks, use local parquet shards when available so cluster jobs do
        # not depend on live HuggingFace connectivity.
        self.seqs = _load_nucleotide_transformer_dataset(dataset_name, split, dest_path=dest_path)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seqs[idx]["sequence"]  # only one sequence
        y = self.seqs[idx]["label"]
        seq_len = min(len(x), self.max_length)

        if self.reverse_sequence or (self.reverse_aug and self.split == "train" and coin_flip()):
            x = plain_sequence_reverse(x)

        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(
            x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq_ids.append(self.tokenizer.sep_token_id)

        if self.conjoin_train or (self.conjoin_test and self.split != "train"):
            x_rc = string_reverse_complement(x)
            seq_rc = self.tokenizer(
                x_rc,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )
            seq_rc_ids = seq_rc["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq_rc_ids.append(self.tokenizer.sep_token_id)
            seq_ids = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)

        else:
            # convert to tensor
            seq_ids = torch.LongTensor(seq_ids)

        # need to wrap in list
        target = torch.LongTensor([y])

        # `seq` has shape:
        #     - (seq_len,) if not conjoining
        #     - (seq_len, 2) for conjoining
        if self.return_lengths:
            return seq_ids, target, {"lengths": torch.LongTensor([seq_len])}
        return seq_ids, target
