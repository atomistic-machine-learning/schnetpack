from typing import Optional, Any
import pytorch_lightning as pl
from .atoms import *
import torch


class AtomsDataModule(pl.LightningDataModule):
    def __init__(self, datapath: str, format: Optional[AtomsDataFormat]):
        super().__init__()
        self.datapath, self.format = resolve_format(datapath, format)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_transforms(self):
        pass
