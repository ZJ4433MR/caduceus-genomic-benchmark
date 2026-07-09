"""Evaluate a GenomicBenchmarks checkpoint on a flipped validation split."""

import hydra
from omegaconf import OmegaConf

from train import SequenceLightningModule, create_trainer
import src.utils as utils


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = utils.train.process_config(config)
    config.train.update({"remove_val_loader_in_eval": False})
    config.train.update({"remove_test_loader_in_eval": True})

    ckpt_path = config.eval.ckpt_path
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)
    metrics = trainer.validate(model, ckpt_path=ckpt_path)
    print(f"FLIP_VAL_METRICS ckpt={ckpt_path} metrics={metrics}")


if __name__ == "__main__":
    main()
