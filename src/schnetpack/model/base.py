from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import hydra.utils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

import schnetpack as spk


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
        representation: DictConfig,
        output: DictConfig,
        schedule: DictConfig,
        optimizer: DictConfig,
        postprocess: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            "representation", "output", "optimizer", "schedule", "postprocess"
        )
        self._representation_cfg = representation
        self._output_cfg = output
        self._schedule_cfg = schedule
        self._optimizer_cfg = optimizer
        self._postproc_cfg = postprocess or []
        self.inference_mode = False

        self.build_model(datamodule)
        self.build_postprocess(datamodule)

    @abstractmethod
    def build_model(self, datamodule):
        """Parser dict configs and instantiate the model"""
        pass

    def build_postprocess(self, datamodule: spk.data.AtomsDataModule):
        self.postprocessors = torch.nn.ModuleList()
        for pp in self._postproc_cfg:
            pp = hydra.utils.instantiate(pp)
            pp.postprocessor()
            pp.datamodule(datamodule)
            self.postprocessors.append(pp)

    def postprocess(
        self, inputs: Dict[str, torch.Tensor], results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.inference_mode:
            for pp in self.postprocessors:
                results = pp(results, inputs)
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
