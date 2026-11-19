"""spktest: test a finished spktrain run's best checkpoint on its test split."""

import argparse
import logging
import os
from glob import glob

import hydra
import torch
from pytorch_lightning import Trainer

from schnetpack.cli import _run_config
from schnetpack.cli.train import _datamodule, test_checkpoint


def main():
    """spktest: test the best checkpoint of a spktrain run on the run's test split
    and write ``<run_dir>/test_metrics.json``, as spktrain does after training."""
    parser = argparse.ArgumentParser(
        description="Test a spktrain run's best checkpoint on its test split."
    )
    parser.add_argument("run_dir", help="the run folder spktrain wrote")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="key=value changes of the run's config.yaml, for paths that only "
        "existed where it trained, e.g. run.data_dir=/data data.cache_dir=null",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = _run_config(args.run_dir, args.overrides)
    os.chdir(args.run_dir)

    # save_top_k=1: the best checkpoint is the one besides last.ckpt
    best = [c for c in glob("checkpoints/*.ckpt") if c != "checkpoints/last.ckpt"]
    if len(best) != 1:
        raise RuntimeError(f"expected one best checkpoint in checkpoints/: {best}")

    if config.get("matmul_precision") is not None:
        torch.set_float32_matmul_precision(config.matmul_precision)
    datamodule = _datamodule(config)
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=False, callbacks=[], _convert_="partial"
    )
    task_cls = hydra.utils.get_class(config.task._target_)
    test_checkpoint(trainer, datamodule, task_cls, best[0])
