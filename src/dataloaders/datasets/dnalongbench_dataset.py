"""DNALONGBENCH long-range classification datasets.

This module implements only the 450k sequence-classification tasks used in the
current experiments: enhancer-target gene prediction (ETGP) and eQTL
prediction. It mirrors the parsing logic from the DNALONGBENCH loaders while
returning token ids for this repository's sequence backbones.
"""

from pathlib import Path

import pandas as pd
import torch
from kipoiseq import Interval
from pyfaidx import Fasta

from src.dataloaders.utils.rc import coin_flip, string_reverse, string_reverse_complement

try:
    import tabix
except ImportError as exc:  # pragma: no cover - import checked in env smoke tests
    tabix = None
    _TABIX_IMPORT_ERROR = exc
else:
    _TABIX_IMPORT_ERROR = None


SUPPORTED_EQTL_CELL_TYPES = [
    "Adipose_Subcutaneous",
    "Artery_Tibial",
    "Cells_Cultured_fibroblasts",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Thyroid",
    "Whole_Blood",
]


def _parse_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_config(config_file):
    config = {}
    with open(config_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            key, value, data_type = parts[0], parts[1], parts[2]
            if data_type == "int":
                value = int(value)
            elif data_type == "float":
                value = float(value)
            elif data_type == "bool":
                value = _parse_bool(value)
            elif data_type == "list":
                value = [item for item in value.split(",") if item]
            else:
                value = str(value)
            config[key] = value
    return config


class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta_file = str(fasta_file)
        self.fasta = None
        self._chromosome_sizes = None

    def _ensure_open(self):
        if self.fasta is None:
            self.fasta = Fasta(self.fasta_file)
            self._chromosome_sizes = {key: len(value) for key, value in self.fasta.items()}

    def extract(self, interval: Interval) -> str:
        self._ensure_open()
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(
            interval.chrom,
            max(interval.start, 0),
            min(interval.end, chromosome_length),
        )
        sequence = str(
            self.fasta.get_seq(
                trimmed_interval.chrom,
                trimmed_interval.start + 1,
                trimmed_interval.stop,
            ).seq
        ).upper()
        pad_upstream = "N" * max(-interval.start, 0)
        pad_downstream = "N" * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream


class _DNALongBenchBase(torch.utils.data.Dataset):
    target2int = {"positive": 1, "negative": 0, 1: 1, 0: 0, "1": 1, "0": 0}

    def __init__(
        self,
        *,
        root_path,
        config_file,
        split,
        max_length=450000,
        tokenizer=None,
        tokenizer_name="char",
        use_padding=True,
        rc_aug=False,
        reverse_aug=False,
        reverse_sequence=False,
        conjoin_train=False,
        conjoin_test=False,
    ):
        if tabix is None:
            raise ImportError("pytabix is required for DNALONGBENCH ETGP/eQTL datasets.") from _TABIX_IMPORT_ERROR

        self.root_path = Path(root_path)
        self.config_file = Path(config_file)
        self.split = split
        self.max_length = int(max_length)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.use_padding = use_padding
        self.rc_aug = rc_aug
        self.reverse_aug = reverse_aug
        self.reverse_sequence = reverse_sequence
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        if self.rc_aug and self.reverse_aug:
            raise ValueError("`rc_aug` and `reverse_aug` are mutually exclusive for DNALONGBENCH datasets.")
        if (self.conjoin_train or self.conjoin_test) and self.reverse_aug:
            raise ValueError("RC conjoining and plain reverse augmentation should not be enabled together.")

        self.config = parse_config(self.config_file)
        if "seq_len_cutoff" not in self.config:
            self.config["seq_len_cutoff"] = self.max_length
        self.config["seq_len_cutoff"] = min(int(self.config["seq_len_cutoff"]), self.max_length)

        self.fasta_reader = FastaStringExtractor(self.root_path / self.config["genome_fa"])
        self._blacklist = None
        self._blacklist_path = self.root_path / self.blacklist_config_key

        self.rows = self._load_rows()

    @property
    def blacklist_config_key(self):
        raise NotImplementedError

    @property
    def records_config_key(self):
        raise NotImplementedError

    def _ensure_blacklist(self):
        if self._blacklist is None:
            self._blacklist = tabix.open(str(self._blacklist_path))
        return self._blacklist

    def _load_rows(self):
        df = pd.read_csv(self.root_path / self.config[self.records_config_key], sep="\t", header=0)
        rows = []
        for _, row in df.iterrows():
            if row["subset"] != self.split:
                continue
            if row["gene_chrom"] != row["region_chrom"]:
                continue
            if row["gene_strand"] not in {"+", "-"}:
                continue
            record = self._record_coordinates(row)
            if record["distance"] > self.config["seq_len_cutoff"]:
                continue
            rows.append(row.to_dict())
        return rows

    def __len__(self):
        return len(self.rows)

    def _record_coordinates(self, row):
        gene_start = int(row["gene_start"])
        gene_end = int(row["gene_end"])
        region_start = int(row["region_start"])
        region_end = int(row["region_end"])
        if row["gene_strand"] == "+":
            tss_start = gene_start
        else:
            tss_start = gene_end - 1
        tss_end = tss_start + 1

        tss_start -= int(self.config["tss_flank_upstream"])
        tss_end += int(self.config["tss_flank_downstream"])
        region_start_flank = region_start - int(self.config["region_flank_upstream"])
        region_end_flank = region_end + int(self.config["region_flank_downstream"])
        distance = max(0, max(tss_start, region_start_flank) - min(tss_end, region_end_flank))
        sequence_start = min(tss_start, region_start_flank)
        sequence_end = max(tss_end, region_end_flank)
        return {
            "gene_start": gene_start,
            "tss_start": tss_start,
            "tss_end": tss_end,
            "region_start": region_start,
            "region_end": region_end,
            "region_start_flank": region_start_flank,
            "region_end_flank": region_end_flank,
            "distance": distance,
            "sequence_start": sequence_start,
            "sequence_end": sequence_end,
        }

    def _mask_blacklist(self, chrom, sequence, coords):
        if coords["distance"] <= 0:
            return sequence
        if coords["tss_start"] > coords["region_end_flank"]:
            query_start = coords["region_end_flank"]
            query_end = coords["tss_start"]
        else:
            query_start = coords["tss_end"]
            query_end = coords["region_start_flank"]
        query_start = max(0, int(query_start))
        query_end = max(query_start, int(query_end))
        if query_end <= query_start:
            return sequence

        try:
            query_result = self._ensure_blacklist().query(chrom, query_start, query_end)
        except Exception:
            return sequence

        sequence_start = coords["sequence_start"]
        for overlap in query_result:
            overlap_start = int(overlap[1])
            overlap_end = int(overlap[2])
            rel_start = overlap_start - sequence_start
            rel_end = overlap_end - sequence_start
            if 0 < rel_start < len(sequence) and 0 < rel_end < len(sequence):
                sequence = sequence[:rel_start] + "N" * (rel_end - rel_start) + sequence[rel_end:]
        return sequence

    def _pad_or_truncate(self, sequence):
        if len(sequence) <= self.max_length:
            return sequence + "N" * (self.max_length - len(sequence))
        return sequence[: self.max_length]

    def _maybe_augment(self, sequence):
        if self.reverse_sequence or (self.reverse_aug and self.split == "train" and coin_flip()):
            return string_reverse(sequence)
        if (self.rc_aug or self.conjoin_test) and self.split == "train" and coin_flip():
            return string_reverse_complement(sequence)
        return sequence

    def _tokenize(self, sequence):
        tokenized = self.tokenizer(
            sequence,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )
        return torch.LongTensor(tokenized["input_ids"])

    def _tokenize_with_optional_conjoin(self, sequence):
        sequence_ids = self._tokenize(sequence)
        if self.conjoin_train or (self.conjoin_test and self.split != "train"):
            sequence_rc_ids = self._tokenize(string_reverse_complement(sequence))
            return torch.stack((sequence_ids, sequence_rc_ids), dim=1)
        return sequence_ids

    def _target(self, row):
        return torch.tensor(self.target2int[row["target"]], dtype=torch.long)


class DNALongBenchETGPDataset(_DNALongBenchBase):
    @property
    def blacklist_config_key(self):
        return self.config["enhancer_tabix_file"]

    @property
    def records_config_key(self):
        return "EPI_file"

    def __getitem__(self, idx):
        row = self.rows[idx]
        coords = self._record_coordinates(row)
        sequence = self.fasta_reader.extract(
            Interval(row["region_chrom"], coords["sequence_start"], coords["sequence_end"])
        )
        sequence = self._mask_blacklist(row["region_chrom"], sequence, coords)
        if coords["gene_start"] > coords["region_end_flank"]:
            sequence = string_reverse_complement(sequence)
        sequence = self._pad_or_truncate(sequence)
        sequence = self._maybe_augment(sequence)
        return self._tokenize_with_optional_conjoin(sequence), self._target(row)


class DNALongBenchEQTLDataset(_DNALongBenchBase):
    @property
    def blacklist_config_key(self):
        return self.config["eQTL_tabix_file"]

    @property
    def records_config_key(self):
        return "eQTL_file"

    def __getitem__(self, idx):
        row = self.rows[idx]
        coords = self._record_coordinates(row)
        sequence = self.fasta_reader.extract(
            Interval(row["region_chrom"], coords["sequence_start"], coords["sequence_end"])
        )
        variant_start = int(row["region_start"])
        variant_end = int(row["region_end"])
        rel_start = variant_start - coords["sequence_start"]
        rel_end = variant_end - coords["sequence_start"]
        ref_allele = str(row["allele1"]).upper()
        alt_allele = str(row["allele2"]).upper()
        observed_ref = sequence[rel_start:rel_end].upper()
        if observed_ref != ref_allele:
            raise ValueError(
                f"Reference allele mismatch for {row['region_id']}: expected {ref_allele}, got {observed_ref}"
            )

        sequence = self._mask_blacklist(row["region_chrom"], sequence, coords)
        variant_sequence = sequence[:rel_start] + alt_allele + sequence[rel_end:]
        if coords["gene_start"] > coords["region_end_flank"]:
            sequence = string_reverse_complement(sequence)
            variant_sequence = string_reverse_complement(variant_sequence)
        sequence = self._pad_or_truncate(sequence)
        variant_sequence = self._pad_or_truncate(variant_sequence)
        if self.reverse_sequence or (self.reverse_aug and self.split == "train" and coin_flip()):
            sequence = string_reverse(sequence)
            variant_sequence = string_reverse(variant_sequence)
        elif (self.rc_aug or self.conjoin_test) and self.split == "train" and coin_flip():
            sequence = string_reverse_complement(sequence)
            variant_sequence = string_reverse_complement(variant_sequence)
        return torch.stack(
            [
                self._tokenize_with_optional_conjoin(sequence),
                self._tokenize_with_optional_conjoin(variant_sequence),
            ],
            dim=0,
        ), self._target(row)
