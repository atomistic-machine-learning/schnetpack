import logging

import hydra
import torch
from torch.autograd import grad
from typing import Dict, Optional, List, Callable
import torch.nn as nn
from functools import partial
from schnetpack import structure
from schnetpack.model.base import AtomisticModel, optimizer_factory, scheduler_factory, Properties
from omegaconf import DictConfig

import schnetpack as spk

log = logging.getLogger(__name__)

__all__ = ["PESModel"]


class PESModel(AtomisticModel):
    """
    AtomisticModel for potential energy surfaces
    """

    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: nn.Module,
        output: nn.Module,
        energy_property: str = "energy",
        energy_loss_fn: Optional[Callable] = None,
        energy_weight: Optional[float] = None,
        energy_metrics: Optional[List[Callable]] = None,
        forces_property: str = None,
        forces_loss_fn: Optional[Callable] = None,
        forces_weight: Optional[float] = None,
        forces_metrics: Optional[List[Callable]] = None,
        stress_property: str = None,
        stress_loss_fn: Optional[Callable] = None,
        stress_weight: Optional[float] = None,
        stress_metrics: Optional[List[Callable]] = None,
        optimizer: Optional = None,
        scheduler: Optional = None,
        scheduler_monitor: Optional[str] = None,
        postprocess: Optional[list[spk.transform.Transform]] = None,
    ):
        super(PESModel, self).__init__(
            datamodule=datamodule,
            representation=representation,
            output=output,
            postprocess=postprocess,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_monitor = scheduler_monitor

        self.targets = {
            Properties.energy: energy_property,
            Properties.forces: forces_property,
            Properties.stress: stress_property,
        }

        self.losses = {Properties.energy: (energy_weight, energy_loss_fn)}
        if forces_loss_fn is not None:
            self.losses[Properties.forces] = (forces_weight, forces_loss_fn)
        if stress_loss_fn is not None:
            self.losses[Properties.stress] = (stress_weight, stress_loss_fn)

        self._collect_metrics(energy_metrics, forces_metrics, stress_metrics)

    def _collect_metrics(self, energy_metrics, forces_metrics, stress_metrics):
        self.metrics = {}
        if energy_metrics is not None:
            self.metrics[Properties.energy] = nn.ModuleDict({
                Properties.energy + "_" + name: metric for name, metric in energy_metrics.items()
            })

        if forces_metrics is not None:
            self.metrics[Properties.forces] = nn.ModuleDict({
                Properties.forces + "_" + name: metric for name, metric in forces_metrics.items()
            })
        if stress_metrics is not None:
            self.metrics[Properties.stress] = nn.ModuleDict({
                Properties.stress + "_" + name: metric for name, metric in stress_metrics.items()
            })

    def forward(self, inputs: Dict[str, torch.Tensor]):
        R = inputs[structure.R]
        inputs[structure.Rij].requires_grad_()
        inputs.update(self.representation(inputs))
        Epred = self.output(inputs)
        results = {Properties.energy: Epred}

        if self.targets[Properties.forces] is not None or self.targets[Properties.stress] is not None:
            go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
            dEdRij = grad(
                [Epred],
                [inputs[structure.Rij]],
                grad_outputs=go,
                create_graph=self.training,
            )[0]

            if self.targets[Properties.forces] is not None and dEdRij is not None:
                Fpred_i = torch.zeros_like(R)
                Fpred_i = Fpred_i.index_add(0, inputs[structure.idx_i], dEdRij)

                Fpred_j = torch.zeros_like(R)
                Fpred_j = Fpred_j.index_add(0, inputs[structure.idx_j], dEdRij)
                Fpred = Fpred_i - Fpred_j
                results[Properties.forces] = Fpred

        if self.targets[Properties.stress] is not None:
            stress_i = torch.zeros((R.shape[0], 3, 3), dtype=R.dtype, device=R.device)

            # sum over j
            stress_i = stress_i.index_add(
                0,
                inputs[structure.idx_i],
                dEdRij[:, None, :] * inputs[structure.Rij][:, :, None],
            )

            # sum over i
            idx_m = inputs[structure.idx_m]
            maxm = int(idx_m[-1]) + 1
            stress = torch.zeros(
                (maxm, 3, 3), dtype=stress_i.dtype, device=stress_i.device
            )
            stress = stress.index_add(0, idx_m, stress_i)

            cell_33 = inputs[structure.cell].view(maxm, 3, 3)
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

            results[Properties.stress] = stress.reshape(maxm * 3, 3) / volume

        results = self.postprocess(inputs, results)

        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for pname, v in self.losses.items():
            weight, fn = v
            loss_p = weight * fn(pred[pname], batch[self.targets[pname]])
            loss += loss_p
        return loss

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        for prop, pmetrics in self.metrics.items():
            for name, pmetric in pmetrics.items():
                self.log(
                    f"train_{name}",
                    pmetric(pred[prop], batch[self.targets[prop]]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for prop, pmetrics in self.metrics.items():
            for name, pmetric in pmetrics.items():
                self.log(
                    f"val_{name}",
                    pmetric(pred[prop], batch[self.targets[prop]]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for prop, pmetrics in self.metrics.items():
            for name, pmetric in pmetrics.items():
                self.log(
                    f"test_{name}",
                    pmetric(pred[prop], batch[self.targets[prop]]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        schedule = self.scheduler(optimizer=optimizer)

        optimconf = {"scheduler": schedule, "name": "lr_schedule"}
        if self.schedule_monitor:
            optimconf["monitor"] = self.schedule_monitor
        return [optimizer], [optimconf]

    # TODO: add eval mode post-processing
