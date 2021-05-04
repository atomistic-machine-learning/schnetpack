import pytorch_lightning as pl
import schnetpack as spk
import torch
import torch.nn as nn
import torch.optim as opt
from schnetpack import (
    Properties,
)
from schnetpack.nn.neighbors import (
    atom_distances,
)
from typing import (
    List,
    Dict,
)


class AtomisticModel(pl.LightningModule):
    def __init__(
        self,
        representation: nn.Module,
        outputs: List[nn.Module],
        losses: List[Dict],
        validation_metrics: List[Dict] = [],
        lr: float = 1e-3,
        lr_decay: float = 0.5,
        lr_patience: int = 100,
        lr_monitor="training/ema_val_loss",
        ema_decay=0.9,
    ):
        super().__init__()

        self.representation = representation
        self.output_modules = nn.ModuleList(outputs)

        self.losses = losses
        self.validation_metrics = validation_metrics

        self.loss_metrics = nn.ModuleList([l["metric"] for l in losses])
        self.val_metrics = nn.ModuleList([l["metric"] for l in validation_metrics])

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor

        self.save_hyperparameters()

        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])

        self.ema_loss = None
        self.ema_decay = ema_decay

    def calculate_loss(
        self,
        batch,
        result,
    ):
        loss = torch.tensor(
            0.0,
            device=self.device,
        )
        for loss_dict in self.losses:
            loss_fn = loss_dict["metric"]

            if "target" in loss_dict.keys():
                pred = result[loss_dict["prediction"]]
                tnames = loss_dict["target"].split(",")
                targets = [batch[t] for t in tnames]
                if len(targets) == 1:
                    targets = targets[0]

                loss_i = loss_fn(
                    pred,
                    targets,
                )
            else:
                loss_i = loss_fn(result[loss_dict["prediction"]])
            loss += loss_dict["loss_weight"] * loss_i

        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        self._enable_grads(batch)

        batch["representation"] = self.representation(batch)

        result = {}
        for output_model in self.output_modules:
            result.update(output_model(batch))

        loss = self.calculate_loss(
            batch,
            result,
        )

        self.log(
            "training/loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        val_batch = batch.copy()
        val_batch["representation"] = self.representation(val_batch)

        result = {}
        for output_model in self.output_modules:
            result.update(output_model(val_batch))
        torch.set_grad_enabled(False)

        val_loss = (
            self.calculate_loss(
                val_batch,
                result,
            )
            .detach()
            .item()
        )
        self.log_metrics(
            val_batch,
            result,
            "validation",
        )
        torch.set_grad_enabled(False)
        return val_loss

    def validation_epoch_end(
        self,
        validation_step_outputs,
    ):
        val_epoch_loss = 0.0
        for val_loss in validation_step_outputs:
            val_epoch_loss += val_loss
        val_epoch_loss /= len(validation_step_outputs)
        self.log(
            "training/val_loss",
            val_epoch_loss,
            on_step=False,
            on_epoch=True,
        )

        if self.ema_loss is None:
            self.ema_loss = val_epoch_loss
        else:
            self.ema_loss = (
                self.ema_decay * self.ema_loss + (1 - self.ema_decay) * val_epoch_loss
            )
        self.log(
            "training/ema_val_loss",
            self.ema_loss,
            on_step=False,
            on_epoch=True,
        )

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch = batch.copy()
        batch["representation"] = self.representation(batch)

        result = {}
        for output_model in self.output_modules:
            result.update(output_model(batch))
        torch.set_grad_enabled(False)

        val_loss = (
            self.calculate_loss(
                batch,
                result,
            )
            .detach()
            .item()
        )
        self.log_metrics(
            batch,
            result,
            "test",
        )
        torch.set_grad_enabled(False)
        return val_loss

    def forward(
        self,
        batch,
    ):
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch = batch.copy()
        batch["representation"] = self.representation(batch)

        result = {}
        for output_model in self.output_modules:
            result.update(output_model(batch))
        torch.set_grad_enabled(False)
        return result

    def log_metrics(
        self,
        batch,
        result,
        mode,
    ):
        for metric_dict in self.validation_metrics:
            loss_fn = metric_dict["metric"]

            if "target" in metric_dict.keys():
                pred = result[metric_dict["prediction"]]

                tnames = metric_dict["target"].split(",")
                targets = [batch[t] for t in tnames]
                if len(targets) == 1:
                    targets = targets[0]

                loss_i = (
                    loss_fn(
                        pred,
                        targets,
                    )
                    .detach()
                    .item()
                )
            else:
                loss_i = loss_fn(result[metric_dict["prediction"]]).detach().item()

            if hasattr(
                loss_fn,
                "name",
            ):
                lossname = loss_fn.name
            else:
                lossname = type(loss_fn).__name__.split(".")[-1]

            self.log(
                mode + "/" + lossname + "_" + metric_dict["prediction"],
                loss_i,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(
        self,
    ):
        optimizer = opt.Adam(
            self.parameters(),
            lr=self.lr,
        )
        scheduler = {
            "scheduler": opt.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def _enable_grads(
        self,
        batch,
    ):
        if self.requires_dr:
            batch[Properties.R].requires_grad_()
