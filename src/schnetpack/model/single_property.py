import logging

import hydra
import pytorch_lightning.metrics
import torch

import schnetpack as spk
from schnetpack.model.base import AtomisticModel

log = logging.getLogger(__name__)

__all__ = ["SinglePropertyModel"]


class SinglePropertyModel(AtomisticModel):
    """
    AtomisticModel for models that predict single chemical properties, e.g. for QM9 benchmarks.
    """

    def build_model(
        self,
    ):
        self.representation = hydra.utils.instantiate(self._representation_cfg)
        self.output = hydra.utils.instantiate(self._output_cfg.module)

        self.loss_fn = hydra.utils.instantiate(self._output_cfg.loss)
        self.pred_property = self._output_cfg.property

        self.metrics = torch.nn.ModuleDict(
            {
                name: hydra.utils.instantiate(metric)
                for name, metric in self._output_cfg.metrics.items()
            }
        )

    def forward(self, inputs):
        inputs.update(self.representation(inputs))
        pred = self.output(inputs)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for name, metric in self.metrics.items():
            self.log(
                f"train_{name}",
                metric(pred, target),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log(
                f"val_{name}",
                metric(pred, target),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log(
                f"test_{name}",
                metric(pred, target),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self._optimizer_cfg, params=self.parameters()
        )
        schedule = hydra.utils.instantiate(
            self._schedule_cfg.scheduler, optimizer=optimizer
        )

        optimconf = {
            "optimizer": optimizer,
            "lr_scheduler": schedule,
        }

        if self._schedule_cfg.monitor:
            optimconf["monitor"] = self._schedule_cfg.monitor
        return optimconf

    # TODO: add eval mode post-processing
