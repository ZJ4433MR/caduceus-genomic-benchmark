"""Genomic Benchmarks Dataset.

From: https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
"""

import csv
import gzip
from pathlib import Path

import torch
from genomic_benchmarks.data_check import is_downloaded
from genomic_benchmarks.loc2seq import download_dataset

from src.dataloaders.utils.rc import coin_flip, string_reverse_complement

SEQUENCE_COLUMNS = ("sequence", "seq", "text")
COORDINATE_COLUMNS = {"region", "start", "end"}


class GenomicBenchmarkDataset(torch.utils.data.Dataset):
    """
    Loop through bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    """

    def __init__(
            self,
            split,
            max_length,
            dataset_name="human_nontata_promoters",
            d_output=2,  # default binary classification
            dest_path=None,
            tokenizer=None,
            tokenizer_name=None,
            use_padding=None,
            add_eos=False,
            rc_aug=False,
            conjoin_train=False,
            conjoin_test=False,
            reverse_sequence=False,
            return_augs=False,
            return_mask=False,
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
        self.return_mask = return_mask

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        self.split = split

        # use Path object
        dataset_root = Path(dest_path) / dataset_name
        base_path = dataset_root / split
        if self._split_contains_coordinate_csv(base_path):
            dataset_root = self._ensure_sequence_dataset(dataset_name, Path(dest_path))
            base_path = dataset_root / split

        self.all_seqs = []
        self.all_labels = []
        entries = sorted(base_path.iterdir())
        label_mapper = {x.stem: i for i, x in enumerate(entries)}

        for entry in entries:
            label_type = entry.stem
            if entry.is_dir():
                for path in sorted(entry.iterdir()):
                    with open(path, "r") as f:
                        content = f.read()
                    self.all_seqs.append(content)
                    self.all_labels.append(label_mapper[label_type])
            elif self._is_csv_file(entry):
                for content in self._iter_sequences_from_csv(entry):
                    self.all_seqs.append(content)
                    self.all_labels.append(label_mapper[label_type])
            else:
                raise ValueError(
                    f"Unsupported GenomicBenchmarks class entry: {entry}. "
                    "Expected a class directory or a class CSV file."
                )

    @staticmethod
    def _is_csv_file(path):
        suffixes = [suffix.lower() for suffix in path.suffixes]
        return suffixes[-1:] == [".csv"] or suffixes[-2:] == [".csv", ".gz"]

    @classmethod
    def _split_contains_coordinate_csv(cls, split_path):
        if not split_path.exists():
            return False
        for path in split_path.iterdir():
            if path.is_file() and cls._is_csv_file(path):
                header = cls._read_csv_header(path)
                if header is not None and cls._is_coordinate_header(header):
                    return True
        return False

    @classmethod
    def _ensure_sequence_dataset(cls, dataset_name, dest_path):
        sequence_dest = dest_path / "_sequence_cache"
        sequence_root = sequence_dest / dataset_name
        if not cls._has_sequence_dataset(sequence_root):
            print(
                f"Converting coordinate-only GenomicBenchmarks dataset {dataset_name} "
                f"to sequence cache at {sequence_root}"
            )
            download_dataset(dataset_name, version=0, dest_path=sequence_dest)
        return sequence_root

    @classmethod
    def _has_sequence_dataset(cls, dataset_root):
        return (
            cls._split_has_sequence_entries(dataset_root / "train")
            and cls._split_has_sequence_entries(dataset_root / "test")
        )

    @classmethod
    def _split_has_sequence_entries(cls, split_path):
        if not split_path.exists():
            return False
        for entry in split_path.iterdir():
            if entry.is_dir():
                return True
            if entry.is_file() and cls._is_csv_file(entry):
                header = cls._read_csv_header(entry)
                if header is None or any(name in header for name in SEQUENCE_COLUMNS):
                    return True
        return False

    @staticmethod
    def _read_csv_header(path):
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                row = [cell.strip().lower() for cell in row]
                if row and any(row):
                    return row
        return None

    @staticmethod
    def _is_coordinate_header(header):
        return COORDINATE_COLUMNS.issubset(set(header)) and not any(
            name in header for name in SEQUENCE_COLUMNS
        )

    @staticmethod
    def _iter_sequences_from_csv(path):
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", newline="") as f:
            reader = csv.reader(f)
            header = None
            seq_idx = None
            for row in reader:
                row = [cell.strip() for cell in row]
                if not row or not any(row):
                    continue
                if header is None:
                    lower = [cell.lower() for cell in row]
                    if GenomicBenchmarkDataset._is_coordinate_header(lower):
                        raise ValueError(
                            f"{path} contains genomic coordinates, not sequences. "
                            "The dataset should have been converted with loc2seq first."
                        )
                    for name in SEQUENCE_COLUMNS:
                        if name in lower:
                            seq_idx = lower.index(name)
                            header = row
                            break
                    if header is not None:
                        continue
                    header = []
                if seq_idx is None:
                    seq_idx = GenomicBenchmarkDataset._guess_sequence_column(row)
                sequence = row[seq_idx].strip()
                if sequence:
                    yield sequence

    @staticmethod
    def _guess_sequence_column(row):
        dna_alphabet = set("acgtnACGTN")
        for i, cell in enumerate(row):
            if cell and set(cell) <= dna_alphabet:
                return i
        return max(range(len(row)), key=lambda i: len(row[i]))

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        y = self.all_labels[idx]

        if self.reverse_sequence:
            x = x[::-1]

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
        if self.return_mask:
            return seq_ids, target, {"mask": torch.BoolTensor(seq["attention_mask"])}
        else:
            return seq_ids, target
