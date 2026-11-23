"""spk test: test a finished spk train run's best checkpoint on its test split
(``configs/test.yaml``)."""

import os
from glob import glob

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from schnetpack.cli import _run_config
from schnetpack.cli.train import _datamodule, _task, test_checkpoint


@hydra.main(config_path="../configs", config_name="test", version_base="1.2")
def main(config: DictConfig):
    """spk test: test the best checkpoint of a spk train run on the run's test split
    and write ``<run>/test_metrics.json``, as spk train does after training."""
    run = _run_config(config.run_dir, config.data_dir)
    run.trainer.accelerator = config.device
    os.chdir(config.run_dir)

    # save_top_k=1: the best checkpoint is the one besides last.ckpt
    best = [c for c in glob("checkpoints/*.ckpt") if c != "checkpoints/last.ckpt"]
    if len(best) != 1:
        raise RuntimeError(f"expected one best checkpoint in checkpoints/: {best}")

    if run.get("matmul_precision") is not None:
        torch.set_float32_matmul_precision(run.matmul_precision)
    datamodule = _datamodule(run)
    trainer: Trainer = hydra.utils.instantiate(
        run.trainer, logger=False, callbacks=[], _convert_="partial"
    )
    test_checkpoint(trainer, datamodule, _task(run), best[0])
