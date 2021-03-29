import torch
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

        self.outputs = hydra.utils.instantiate(outputs, n_in=self.representation.size)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # return loss
        pass

    def configure_optimizers(self):
        pass
        # return hydra.instantiate(self.optimizer, params=self.parameters)
