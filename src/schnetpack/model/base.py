from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable

import hydra.utils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch.nn as nn

import schnetpack as spk

__all__ = ["AtomisticModel", "Properties"]


class Properties:
    energy = "energy"
    forces = "forces"
    stress = "stress"
    dipole_moments = "dipole_moments"


class ModelTarget:
    def __init__(self, name, loss_fn, loss_weight, metrics=None):
        self.name = name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = metrics or []


def optimizer_factory(optimizer, **kwargs):
    def factory(params):
        return optimizer(params=params, **kwargs)
    return factory


def scheduler_factory(scheduler, **kwargs):
    def factory(optimizer):
        return scheduler(optimizer, **kwargs)
    return factory


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
    ):
        super().__init__()
        self.save_hyperparameters(
            "representation", "output", "postprocess"
        )
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
