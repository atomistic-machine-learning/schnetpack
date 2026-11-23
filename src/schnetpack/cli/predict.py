"""spk predict: predict with a trained model on an ASE db (``configs/predict.yaml``)."""

import logging

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer

from schnetpack import properties
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.lightning import PredictionWriter
from schnetpack.model.compile import CompiledPotential
from schnetpack.model_format import load_model
from schnetpack.transform import CastTo32, MatScipyNeighborList, SubtractCenterOfMass

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="predict", version_base="1.2")
def main(config: DictConfig):
    """spk predict: run ``model_dir`` (a model folder or a spk train run folder) on every
    structure of ``db`` and write the predictions to ``out``. The neighbor list
    uses the cutoff of the model's representation."""
    model = load_model(config.model_dir)
    cutoff = getattr(getattr(model, "representation", None), "cutoff", None)
    if cutoff is None:
        raise ValueError(f"{config.model_dir}: the model has no representation.cutoff")

    log.info(f"Load data from `{config.db}`")
    dataset = ASEAtomsData(
        config.db,
        transforms=[
            SubtractCenterOfMass(),
            MatScipyNeighborList(cutoff=float(cutoff)),
            CastTo32(),
        ],
    )
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)
    model_fn = CompiledPotential(model) if config.compile else model

    class WrapperLM(LightningModule):
        def __init__(self, model, enable_grad=config.enable_grad):
            super().__init__()
            self.model = model
            self.enable_grad = enable_grad

        def forward(self, x):
            return model_fn(x)

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            torch.set_grad_enabled(self.enable_grad)
            results = self(batch)
            results[properties.idx_m] = batch[properties.idx][batch[properties.idx_m]]
            results = {k: v.detach().cpu() for k, v in results.items()}
            return results

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.out,
                write_interval=config.write_interval,
                write_idx=config.write_idx_m,
            )
        ],
        logger=False,
        _convert_="partial",
    )
    trainer.predict(
        WrapperLM(model, config.enable_grad),
        dataloaders=loader,
        ckpt_path=config.ckpt_path,
    )
