import torch
import torch.nn as nn

import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from omegaconf import DictConfig


class SinglePropertyModel(LightningModule):
    def __init__(
        self,
        datamodule: LightningDataModule,
        representation: DictConfig,
        output: DictConfig,
        optimizer: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters("representation", "output", "optimizer")

        self.optimizer = optimizer
        self.representation = hydra.utils.instantiate(representation)

        self.output = hydra.utils.instantiate(output.module)
        self.loss_fn = hydra.utils.instantiate(output.loss)
        self.pred_property = output.property

    def forward(self, inputs):
        inputs.update(self.representation(inputs))
        pred = self.output(inputs)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val/loss": loss}

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optimizer, params=self.parameters())
