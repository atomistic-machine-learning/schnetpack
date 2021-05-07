from abc import abstractmethod
from typing import Optional

import hydra.utils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

import schnetpack as spk
import torch


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
        self.datamodule = datamodule
        self._representation_cfg = representation
        self._output_cfg = output
        self._schedule_cfg = schedule
        self._optimizer_cfg = optimizer
        self._postproc_cfg = postprocess or []

        self.build_model()
        self.build_postprocess()

    @abstractmethod
    def build_model(
        self,
    ):
        """Parser dict configs and instantiate the model"""
        pass

    def build_postprocess(
        self,
    ):
        self.postprocessors = torch.nn.ModuleList()
        for pp in self._postproc_cfg:
            pp = hydra.utils.instantiate(pp)
            pp.postprocessor()
            self.postprocessors.append(pp)

    def predict(self, inputs):
        result = self(inputs)
        for pp in self.postprocessors:
            result = pp(result, inputs)
        return result
