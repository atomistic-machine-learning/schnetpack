import pytorch_lightning.metrics

import schnetpack as spk

import hydra
import copy
import torch
from pytorch_lightning import LightningModule, LightningDataModule
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


class SinglePropertyModel(LightningModule):
    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: DictConfig,
        output: DictConfig,
        schedule: DictConfig,
        optimizer: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters("representation", "output", "optimizer")

        self.optimizer = optimizer
        self.schedule = schedule
        self.representation = hydra.utils.instantiate(representation)

        self.loss_fn = hydra.utils.instantiate(output.loss)
        self.pred_property = output.property

        if output.requires_atomref:
            atomrefs = datamodule.train_dataset.atomrefs
            atomref = atomrefs[output.property][:, None]
        else:
            atomrefs = None
            atomref = None

        if output.requires_stats:
            log.info("Calculate stats...")
            stats = spk.data.calculate_stats(
                datamodule.train_dataloader(),
                divide_by_atoms={output.property: output.divide_stats_by_atoms},
                atomref=atomrefs,
            )[output.property]
            log.info(f"{output.property} (mean / stddev): {stats[0]}, {stats[1]}")

            self.output = hydra.utils.instantiate(
                output.module,
                atomref=atomref,
                mean=torch.tensor(stats[0], dtype=torch.float32),
                stddev=torch.tensor(stats[1], dtype=torch.float32),
            )
        else:
            self.output = hydra.utils.instantiate(
                output.module, atomref=atomref[:, None]
            )

        self.metric = pytorch_lightning.metrics.MeanAbsoluteError()

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
        self.log(
            "val/mae",
            self.metric(pred, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"val/loss": loss}

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/mae",
            self.metric(pred, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"test/loss": loss}

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        schedule = hydra.utils.instantiate(self.schedule.scheduler, optimizer=optimizer)

        optimconf = {
            "optimizer": optimizer,
            "lr_scheduler": schedule,
        }

        if self.schedule.monitor:
            optimconf["monitor"] = self.schedule.monitor
        return optimconf
