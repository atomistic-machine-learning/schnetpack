"""spkpredict: predict with a trained model on an ASE db (``configs/predict.yaml``)."""

import logging

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer

from schnetpack import properties
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.lightning import PredictionWriter
from schnetpack.model.compile import CompiledPotential
from schnetpack.utils import load_model

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="predict", version_base="1.2")
def main(config: DictConfig):
    log.info(f"Load data from `{config.data.datapath}`")
    dataset: ASEAtomsData = hydra.utils.instantiate(config.data)
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)

    model = load_model("best_model")

    model_fn = CompiledPotential(model) if config.get("compile", False) else model

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
                output_dir=config.outputdir,
                write_interval=config.write_interval,
                write_idx=config.write_idx_m,
            )
        ],
        default_root_dir=".",
        _convert_="partial",
    )
    trainer.predict(
        WrapperLM(model, config.enable_grad),
        dataloaders=loader,
        ckpt_path=config.ckpt_path,
    )
