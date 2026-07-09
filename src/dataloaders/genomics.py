"""Dataloaders for genomics datasets, including pretraining and downstream tasks.

    - Adapted from:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
    - Adapted from:
        https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

import copy
from typing import Any, List, Union

import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from caduceus.tokenization_caduceus import CaduceusTokenizer
import src.utils.train
from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.datasets.genomic_bench_dataset import GenomicBenchmarkDataset
from src.dataloaders.datasets.dnalongbench_dataset import (
    DNALongBenchETGPDataset,
    DNALongBenchEQTLDataset,
    SUPPORTED_EQTL_CELL_TYPES,
)
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.datasets.nucleotide_transformer_dataset import NucleotideTransformerDataset
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler

logger = src.utils.train.get_logger(__name__)


class HG38(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "hg38"  # this name is how the dataset config finds the right dataloader

    def __init__(self, bed_file, fasta_file, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2,
                 rc_aug=False, plain_flip_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, shuffle=False,
                 num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 mlm=False, mlm_probability=0.15,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.plain_flip_aug = plain_flip_aug
        if self.rc_aug and self.plain_flip_aug:
            raise ValueError("`rc_aug` and `plain_flip_aug` are mutually exclusive.")
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.bed_file = bed_file
        self.fasta_file = fasta_file

        # handle if file paths are None (default paths)
        if self.bed_file is None:
            self.bed_file = default_data_path / self._name_ / "human-sequences.bed"
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / "hg38.ml.fa"

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.mlm = mlm
        self.mlm_probability = mlm_probability

        # To be instantiated in `setup`
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            logger.info("**Using Char-level tokenizer**")
            # self.tokenizer = CharacterTokenizer(
            #     characters=["A", "C", "G", "T", "N"],
            #     model_max_length=self.max_length,
            #     add_special_tokens=False,
            # )
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.max_length,
                add_special_tokens=False
            )
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, "dataset_train"):
            self.dataset_train.fasta.seqs.close()
            del self.dataset_train.fasta.seqs

        # delete old datasets to free memory
        if hasattr(self, "dataset_test"):
            self.dataset_test.fasta.seqs.close()
            del self.dataset_test.fasta.seqs

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HG38Dataset(split=split,
                        bed_file=self.bed_file,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        rc_aug=self.rc_aug,
                        plain_flip_aug=self.plain_flip_aug,
                        return_augs=False,
                        mlm=self.mlm,
                        mlm_probability=self.mlm_probability, )
            for split, max_len in
            zip(["train", "valid", "test"], [self.max_length, self.max_length_val, self.max_length_test])
        ]

        return

    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(
                self.dataset_train,
                **distributed_sampler_kwargs
            ) if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    "epoch": self.fast_forward_epochs,
                    "counter": self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        loader = self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                   shuffle=shuffle, sampler=sampler, **kwargs)
        return loader

    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval, **kwargs)

    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
            # TD [2022-08-07] ["epoch_loop.batch_progress"]["total"]["completed"] is 1 iteration
            # behind, so we're using the optimizer"s progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
                "completed"]
        # At this point the train loader hasn't been constructed yet


class GenomicBenchmark(HG38):
    _name_ = "genomic_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(
            self, dataset_name, train_val_split_seed,
            dest_path=None, tokenizer_name="char", d_output=None, rc_aug=False,
            conjoin_train=False, conjoin_test=False,
            reverse_val=False, reverse_test=False,
            max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
            padding_side="left", val_ratio=0.0005, val_split_seed=2357, add_eos=False,
            detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
            shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
            fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs
    ):

        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.reverse_val = reverse_val
        self.reverse_test = reverse_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )

        # Create all splits: torch datasets (only train/test in this benchmark, val created below)
        self.dataset_train, self.dataset_test = [
            GenomicBenchmarkDataset(
                split=split,
                max_length=max_len,
                dataset_name=self.dataset_name,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                dest_path=self.dest_path,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                reverse_sequence=False,
                return_augs=False
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        val_data, train_data = torch.utils.data.random_split(
            list(zip(self.dataset_train.all_seqs, self.dataset_train.all_labels)),
            lengths=[0.1, 0.9],
            generator=torch.Generator().manual_seed(self.train_val_split_seed)
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.all_seqs = [train_data[i][0] for i in range(len(train_data))]
        self.dataset_train.all_labels = [train_data[i][1] for i in range(len(train_data))]

        self.dataset_val.all_seqs = [val_data[i][0] for i in range(len(val_data))]
        self.dataset_val.all_labels = [val_data[i][1] for i in range(len(val_data))]
        self.dataset_val.split = "val"
        self.dataset_val.reverse_sequence = self.reverse_val
        self.dataset_test.reverse_sequence = self.reverse_test


class NucleotideTransformer(HG38):
    _name_ = "nucleotide_transformer"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, train_val_split_seed,
                 tokenizer_name="char", d_output=None, rc_aug=False,
                 conjoin_train=False, conjoin_test=False, return_lengths=False,
                 reverse_val=False, reverse_test=False, reverse_aug=False,
                 max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                 padding_side="left", val_ratio=0.0005, val_split_seed=2357, add_eos=False,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 dest_path=None,
                 shuffle=True, shuffle_eval=None, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.train_val_split_seed = train_val_split_seed
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.return_lengths = return_lengths
        self.reverse_val = reverse_val
        self.reverse_test = reverse_test
        self.reverse_aug = reverse_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle_eval = shuffle_eval if shuffle_eval is not None else shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )

        # Create all splits: torch datasets (only train/test in this benchmark)
        # self.dataset_train, self.dataset_val = [
        self.dataset_train, self.dataset_test = [
            NucleotideTransformerDataset(
                split=split,
                max_length=max_len,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                dataset_name=self.dataset_name,
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                dest_path=self.dest_path,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                reverse_sequence=False,
                reverse_aug=self.reverse_aug,
                return_lengths=self.return_lengths,
                return_augs=False
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        ds_train_val_split = self.dataset_train.seqs.train_test_split(
            test_size=0.1,
            seed=self.train_val_split_seed
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.seqs = ds_train_val_split["train"]

        self.dataset_val.split = "val"
        self.dataset_val.seqs = ds_train_val_split["test"]
        self.dataset_val.reverse_sequence = self.reverse_val
        self.dataset_test.reverse_sequence = self.reverse_test


class DNALongBench450K(HG38):
    _name_ = "dnalongbench_450k"
    l_output = 0

    def __init__(
            self, task_name, data_root, subset=None, train_val_split_seed=1,
            tokenizer_name="char", d_output=2, rc_aug=False, reverse_aug=False,
            conjoin_train=False, conjoin_test=False,
            reverse_val=False, reverse_test=False, max_length=450000, use_padding=True,
            padding_side="right", add_eos=False, batch_size=1, batch_size_eval=None,
            num_workers=0, shuffle=True, shuffle_eval=False, pin_memory=False,
            drop_last=False, *args, **kwargs):
        self.task_name = task_name
        self.data_root = data_root
        self.subset = subset
        self.train_val_split_seed = train_val_split_seed
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.reverse_aug = reverse_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.reverse_val = reverse_val
        self.reverse_test = reverse_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.padding_side = padding_side
        self.add_eos = add_eos
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle_eval = shuffle_eval
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage=None):
        if self.tokenizer_name == "char":
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length,
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == "caduceus":
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.max_length,
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        dataset_cls, root_path, config_file = self._resolve_dataset()
        self.dataset_train = dataset_cls(
            root_path=root_path,
            config_file=config_file,
            split="train",
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            use_padding=self.use_padding,
            rc_aug=self.rc_aug,
            reverse_aug=self.reverse_aug,
            conjoin_train=self.conjoin_train,
            conjoin_test=self.conjoin_test,
        )
        self.dataset_val = dataset_cls(
            root_path=root_path,
            config_file=config_file,
            split="valid",
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            use_padding=self.use_padding,
            reverse_sequence=self.reverse_val,
            conjoin_train=self.conjoin_train,
            conjoin_test=self.conjoin_test,
        )
        self.dataset_test = dataset_cls(
            root_path=root_path,
            config_file=config_file,
            split="test",
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            use_padding=self.use_padding,
            reverse_sequence=self.reverse_test,
            conjoin_train=self.conjoin_train,
            conjoin_test=self.conjoin_test,
        )

    def _resolve_dataset(self):
        from pathlib import Path

        data_root = Path(self.data_root)
        if self.task_name in {"etgp", "enhancer_target_gene_prediction"}:
            candidates = [
                data_root / "enhancer_target_gene",
                data_root / "enhancer_promoter_interaction" / "CRISPRi_EPI",
                data_root,
            ]
            config_name = "CRISPRi_EPI_K562_hg19.config"
            dataset_cls = DNALongBenchETGPDataset
        elif self.task_name in {"eqtl", "eqtl_prediction"}:
            subset = self.subset or SUPPORTED_EQTL_CELL_TYPES[0]
            if subset not in SUPPORTED_EQTL_CELL_TYPES:
                raise ValueError(f"Unsupported eQTL subset {subset}; expected one of {SUPPORTED_EQTL_CELL_TYPES}.")
            candidates = [data_root / "eQTL", data_root / "eqtl", data_root]
            config_name = f"gtex_hg38.{subset}.config"
            dataset_cls = DNALongBenchEQTLDataset
        else:
            raise ValueError("DNALongBench450K task_name must be etgp or eqtl.")

        for root_path in candidates:
            config_file = root_path / "config" / config_name
            if config_file.exists():
                return dataset_cls, root_path, config_file
        raise FileNotFoundError(
            f"Could not find DNALONGBENCH config {config_name} under candidates: "
            + ", ".join(str(path) for path in candidates)
        )

    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        kwargs.setdefault("pin_memory", self.pin_memory)
        kwargs.setdefault("drop_last", self.drop_last)
        kwargs.setdefault("num_workers", self.num_workers)
        sampler = self._distributed_sampler(self.dataset_train, shuffle=self.shuffle)
        return self._data_loader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            **kwargs,
        )

    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        kwargs.setdefault("pin_memory", self.pin_memory)
        kwargs["drop_last"] = False
        kwargs.setdefault("num_workers", self.num_workers)
        sampler = self._distributed_sampler(self.dataset_val, shuffle=False)
        return self._data_loader(
            self.dataset_val,
            batch_size=self.batch_size_eval,
            shuffle=self.shuffle_eval if sampler is None else False,
            sampler=sampler,
            **kwargs,
        )

    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        kwargs.setdefault("pin_memory", self.pin_memory)
        kwargs["drop_last"] = False
        kwargs.setdefault("num_workers", self.num_workers)
        sampler = self._distributed_sampler(self.dataset_test, shuffle=False)
        return self._data_loader(
            self.dataset_test,
            batch_size=self.batch_size_eval,
            shuffle=self.shuffle_eval if sampler is None else False,
            sampler=sampler,
            **kwargs,
        )

    @staticmethod
    def _distributed_sampler(dataset: Dataset, shuffle: bool = False):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                return DistributedSampler(dataset, shuffle=shuffle)
        return None
