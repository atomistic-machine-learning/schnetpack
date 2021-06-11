import logging
from typing import Dict, Optional, List, Type, Any

import torch
import torch.nn as nn
from torch.autograd import grad

import schnetpack as spk
from schnetpack import properties
from schnetpack.model.base import AtomisticModel

log = logging.getLogger(__name__)

__all__ = ["PESModel"]


class PESModel(AtomisticModel):
    """
    AtomisticModel for potential energy surfaces.

    """

    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: nn.Module,
        output: nn.Module,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        energy_property: str = None,
        energy_loss_fn: Optional[nn.Module] = None,
        energy_weight: Optional[float] = None,
        energy_metrics: Optional[Dict] = None,
        forces_property: Optional[str] = None,
        forces_loss_fn: Optional[nn.Module] = None,
        forces_weight: Optional[float] = None,
        forces_metrics: Optional[Dict] = None,
        stress_property: Optional[str] = None,
        stress_loss_fn: Optional[nn.Module] = None,
        stress_weight: Optional[float] = None,
        stress_metrics: Optional[Dict] = None,
        scheduler_cls: Type = None,
        scheduler_args: Dict[str, Any] = None,
        scheduler_monitor: Optional[str] = None,
        postprocess: Optional[List[spk.transform.Transform]] = None,
    ):
        """
        Args:
            datamodule: pytorch_lightning module for dataset
            representation: nn.Module for atomistic representation
            output: nn.Module for computation of physical properties
                from atomistic representation
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            energy_property: name of energy property in dataset
            energy_loss_fn: loss function for computation of energy loss
            energy_weight: weighting of energy loss
            energy_metrics: dict of metrics for energy predictions with metric
                name as keys and callable as values
            forces_property: name of forces property in datamodule
            forces_loss_fn: loss function for computation of forces loss
            forces_weight: weighting of forces loss
            forces_metrics: dict of metrics for forces predictions with metric name as keys and callable as values
            stress_property: name of stress property in datamodule
            stress_loss_fn: loss function for computation of stress loss
            stress_weight: weighting of stress loss
            stress_metrics: dict of metrics for stress predictions with metric name as keys and callable as values
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            postprocess: list of postprocessors to be applied to model for predictions
        """
        super(PESModel, self).__init__(
            datamodule=datamodule,
            representation=representation,
            output=output,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            postprocess=postprocess,
        )
        # todo: datamodule optional?
        self.targets = {
            properties.energy: energy_property,
            properties.forces: forces_property,
            properties.stress: stress_property,
        }

        self.losses = nn.ModuleDict()
        self.loss_weights = {}
        if energy_loss_fn is not None:
            self.losses[properties.energy] = energy_loss_fn
            self.loss_weights[properties.energy] = energy_weight
        if forces_loss_fn is not None:
            self.losses[properties.forces] = forces_loss_fn
            self.loss_weights[properties.forces] = forces_weight
        if stress_loss_fn is not None:
            self.losses[properties.stress] = stress_loss_fn
            self.loss_weights[properties.stress] = stress_weight

        self.metrics = nn.ModuleDict()
        if energy_metrics is not None:
            self.metrics.update(
                {
                    properties.energy + ":" + name: metric
                    for name, metric in energy_metrics.items()
                }
            )
        if forces_metrics is not None:
            self.metrics.update(
                {
                    properties.forces + ":" + name: metric
                    for name, metric in forces_metrics.items()
                }
            )
        if stress_metrics is not None:
            self.metrics.update(
                {
                    properties.stress + ":" + name: metric
                    for name, metric in stress_metrics.items()
                }
            )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        R = inputs[properties.R]
        inputs[properties.Rij].requires_grad_()
        inputs.update(self.representation(inputs))
        Epred = self.output(inputs)
        results = {properties.energy: Epred}

        if (
            self.targets[properties.forces] is not None
            or self.targets[properties.stress] is not None
        ):
            go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
            dEdRij = grad(
                [Epred],
                [inputs[properties.Rij]],
                grad_outputs=go,
                create_graph=self.training,
            )[0]

            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdRij is None:
                dEdRij = torch.zeros_like(inputs[properties.Rij])

            if self.targets[properties.forces] is not None and dEdRij is not None:
                Fpred = torch.zeros_like(R)
                Fpred = Fpred.index_add(0, inputs[properties.idx_i], dEdRij)
                Fpred = Fpred.index_add(0, inputs[properties.idx_j], -dEdRij)
                results[properties.forces] = Fpred

            if self.targets[properties.stress] is not None:
                stress_i = torch.zeros(
                    (R.shape[0], 3, 3), dtype=R.dtype, device=R.device
                )

                # sum over j
                stress_i = stress_i.index_add(
                    0,
                    inputs[properties.idx_i],
                    dEdRij[:, None, :] * inputs[properties.Rij][:, :, None],
                )

                # sum over i
                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1
                stress = torch.zeros(
                    (maxm, 3, 3), dtype=stress_i.dtype, device=stress_i.device
                )
                stress = stress.index_add(0, idx_m, stress_i)

                cell_33 = inputs[properties.cell].view(maxm, 3, 3)
                volume = (
                    torch.sum(
                        cell_33[:, 0, :]
                        * torch.cross(cell_33[:, 1, :], cell_33[:, 2, :], dim=1),
                        dim=1,
                        keepdim=True,
                    )
                    .expand(maxm, 3)
                    .reshape(maxm * 3, 1)
                )
                results[properties.stress] = stress.reshape(maxm * 3, 3) / volume

        results = self.postprocess(inputs, results)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for pname, ploss_fn in self.losses.items():
            loss_p = self.loss_weights[pname] * ploss_fn(
                pred[pname], batch[self.targets[pname]]
            )
            loss += loss_p
        return loss

    def log_metrics(self, batch, pred, subset):
        for metric_name, pmetric in self.metrics.items():
            prop, name = metric_name.split(":")
            self.log(
                f"{subset}_{metric_name}".replace(":", "_"),

                pmetric(pred[prop], batch[self.targets[prop]]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(batch, pred, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics(batch, pred, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics(batch, pred, "test")
        return {"test_loss": loss}
