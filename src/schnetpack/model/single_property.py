import logging

import hydra
import pytorch_lightning.metrics
import torch

import schnetpack as spk

from schnetpack.model.base import AtomisticModel

log = logging.getLogger(__name__)


class SinglePropertyModel(AtomisticModel):
    """
    LightningModule for models that predict single chemical properties, e.g. for QM9 benchmarks.
    """

    def build_model(
        self,
    ):
        self.representation = hydra.utils.instantiate(self._representation_cfg)
        self.loss_fn = hydra.utils.instantiate(self._output_cfg.loss)
        self.pred_property = self._output_cfg.property

        if self._output_cfg.requires_atomref:
            atomrefs = self.datamodule.train_dataset.atomrefs
            atomref = atomrefs[self._output_cfg.property][:, None]
        else:
            atomrefs = None
            atomref = None

        if self._output_cfg.requires_stats:
            log.info("Calculate stats...")
            stats = spk.data.calculate_stats(
                self.datamodule.train_dataloader(),
                divide_by_atoms={
                    self._output_cfg.property: self._output_cfg.divide_stats_by_atoms
                },
                atomref=atomrefs,
            )[self._output_cfg.property]
            log.info(
                f"{self._output_cfg.property} (mean / stddev): {stats[0]}, {stats[1]}"
            )

            self.output = hydra.utils.instantiate(
                self._output_cfg.module,
                atomref=atomref,
                mean=torch.tensor(stats[0], dtype=torch.float32),
                stddev=torch.tensor(stats[1], dtype=torch.float32),
            )
        else:
            atomref = atomref[:, None] if atomref else None
            self.output = hydra.utils.instantiate(
                self._output_cfg.module, atomref=atomref
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_mae",
            self.metric(pred, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.pred_property]
        loss = self.loss_fn(pred, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_mae",
            self.metric(pred, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
