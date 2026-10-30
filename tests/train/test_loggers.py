import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from schnetpack.utils.script import log_hyperparameters


def test_log_hyperparameters_covers_every_logger(tmp_path):
    """Every logger must get the config and then ignore the hparams of the task,
    which Lightning logs afterwards and which hold objects YAML cannot store."""
    loggers = [
        CSVLogger(tmp_path, name="csv"),
        TensorBoardLogger(tmp_path, name="tensorboard"),
    ]
    trainer = pl.Trainer(logger=loggers, accelerator="cpu")

    log_hyperparameters(OmegaConf.create({"lr": 0.1}), model=None, trainer=trainer)
    for logger in loggers:
        logger.log_hyperparams({"model": object()})

    for hparams in (loggers[0].experiment.hparams, loggers[1].hparams):
        assert hparams == {"lr": 0.1}
