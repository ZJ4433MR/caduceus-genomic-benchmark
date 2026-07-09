"""Evaluate a GenomicBenchmarks checkpoint on a selected split."""

import hydra
from omegaconf import OmegaConf

from train import SequenceLightningModule, create_trainer
import src.utils as utils


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = utils.train.process_config(config)

    split = config.eval.get("split", "test")
    ckpt_path = config.eval.ckpt_path

    if split == "val":
        config.train.update({"remove_val_loader_in_eval": False})
        config.train.update({"remove_test_loader_in_eval": True})
    elif split == "test":
        config.train.update({"remove_val_loader_in_eval": True})
        config.train.update({"remove_test_loader_in_eval": False})
    else:
        raise ValueError(f"Unsupported eval.split={split!r}; expected 'val' or 'test'")

    if config.train.get("pretrained_model_state_hook", None) is not None:
        config.train.pretrained_model_state_hook.update({"_name_": None})

    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # This codebase evaluates held-out test loaders through Lightning's
    # validation loop so the internal metric prefix stays "test".
    metrics = trainer.validate(model, ckpt_path=ckpt_path)

    print(
        "GENOMIC_BENCHMARK_EVAL "
        f"split={split} "
        f"reverse_val={config.dataset.get('reverse_val', False)} "
        f"reverse_test={config.dataset.get('reverse_test', False)} "
        f"ckpt={ckpt_path} "
        f"metrics={metrics}"
    )


if __name__ == "__main__":
    main()
