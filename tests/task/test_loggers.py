import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from schnetpack.lightning.utils import log_hyperparameters


def test_log_hyperparameters_covers_every_logger(tmp_path):
    """Every logger must get the config."""
    loggers = [
        CSVLogger(tmp_path, name="csv"),
        TensorBoardLogger(tmp_path, name="tensorboard"),
    ]
    trainer = pl.Trainer(logger=loggers, accelerator="cpu")

    log_hyperparameters(OmegaConf.create({"lr": 0.1}), trainer=trainer)

    for hparams in (loggers[0].experiment.hparams, loggers[1].hparams):
        assert hparams == {"lr": 0.1}
