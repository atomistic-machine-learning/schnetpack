from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.transform import Transform

import torch
import pytorch_lightning as pl
import torch.nn as nn

import schnetpack as spk

__all__ = ["AtomisticModel"]


class AtomisticModel(pl.LightningModule):
    """
    Base class for all SchNetPack models.

    To define a new model, override and implement build_model, which has to parse the defined hydra configs
    and instantiate the pytorch modules etc. Define  forward, training_step, optimizer etc in the appropriate
    methods (see https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
    """

    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: nn.Module,
        output: nn.Module,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        postprocess: Optional[List[Transform]] = None,
    ):
        """
        Args:
            datamodule: pytorch_lightning module for dataset
            representation: nn.Module for atomistic representation
            output: nn.Module for computation of physical properties from atomistic representation
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            postprocess: list of postprocessors to be applied to model for predictions
        """
        super().__init__()
        self.datamodule = datamodule
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor

        self.save_hyperparameters("representation", "output", "postprocess")
        self.representation = representation
        self.cutoff = representation.cutoff
        self.output = output
        self.pp = postprocess or []

        self.inference_mode = False

    def setup(self, stage=None):
        self.postprocessors = torch.nn.ModuleList()
        for pp in self.pp:
            pp.postprocessor()
            pp.datamodule(self.datamodule)
            self.postprocessors.append(pp)

    def postprocess(
        self, inputs: Dict[str, torch.Tensor], results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.inference_mode:
            for pp in self.postprocessors:
                results = pp(inputs, results)
        return results

    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[torch.ScriptModule, Dict[str, torch.ScriptModule]]:
        imode = self.inference_mode
        self.inference_mode = True
        script = super().to_torchscript(file_path, method, example_inputs, **kwargs)
        self.inference_mode = imode
        return script

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)

            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            return [optimizer], [optimconf]
        else:
            return optimizer
