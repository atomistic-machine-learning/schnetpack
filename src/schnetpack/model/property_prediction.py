import torch
import torch.nn as nn

import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from omegaconf import DictConfig


class PropertyPredictionModel(LightningModule):
    def __init__(
        self,
        datamodule: LightningDataModule,
        representation: DictConfig,
        outputs: DictConfig,
        optimizer: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters("representation", "outputs", "optimizer")

        self.optimizer = optimizer
        self.representation = hydra.utils.instantiate(representation)

        self.outputs = nn.ModuleList()
        self.repr_keys = []
        self.loss_fns = []
        self.loss_weight = []
        for output in outputs:
            if "representation_keys" in output:
                self.repr_keys.append(output.representation_keys)
                n_in = [
                    self.representation.size[key][-1]
                    for key in output.representation_keys
                ]
            else:
                n_in = [self.representation.size[-1]]
                self.repr_keys.append(None)

            self.outputs.append(hydra.utils.instantiate(output.module, n_in=n_in))
            self.loss_fns.append(hydra.utils.instantiate(output.loss))
            self.loss_weight.append(output.loss_weight)

    def forward(self, inputs):
        representation = self.representation(inputs)
        return representation

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # return loss
        pass

    def configure_optimizers(self):
        pass
        # return hydra.instantiate(self.optimizer, params=self.parameters)
