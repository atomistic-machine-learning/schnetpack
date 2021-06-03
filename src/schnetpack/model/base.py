from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable

import torch
from pytorch_lightning import LightningModule
import torch.nn as nn

import schnetpack as spk

__all__ = ["AtomisticModel", "Properties"]


class Properties:
    energy = "energy"
    forces = "forces"
    stress = "stress"
    dipole_moments = "dipole_moments"


class AtomisticModel(LightningModule):
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
        postprocess: Optional[list[spk.transform.Transform]] = None,
        optimizer: Optional[Callable] = None,
        scheduler: Optional[Callable] = None,
        scheduler_monitor: Optional[str] = None,
    ):
        """
        Args:
            datamodule: pytorch_lightning module for dataset
            representation: nn.Module for atomistic representation
            output: nn.Module for computation of physical properties from atomistic representation
            postprocess: list of postprocessors to be applied to model for predictions
            optimizer: functools.partial-function with torch optimizer and args for creation of optimizer with
                optimizer(params=self.parameters())
            scheduler: functools.partial-function with scheduler and args for creation of scheduler with
                scheduler(optimizer=optimizer)
            scheduler_monitor: name of metric to be observed and used for lr drops of scheduler
            postprocess: list of postprocessors to be applied to model for predictions
        """
        super().__init__()
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_monitor = scheduler_monitor

        self.save_hyperparameters("representation", "output", "postprocess")
        self.representation = representation
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
        optimizer = self.optimizer(params=self.parameters())
        schedule = self.scheduler(optimizer=optimizer)

        optimconf = {"scheduler": schedule, "name": "lr_schedule"}
        if self.schedule_monitor:
            optimconf["monitor"] = self.schedule_monitor
        return [optimizer], [optimconf]

    # TODO: add eval mode post-processing
