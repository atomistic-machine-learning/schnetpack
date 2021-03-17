from typing import Optional
import pytorch_lightning as pl
from .atoms import *


class AtomsDataModule(pl.LightningDataModule):
    def __init__(self, datapath: str, format: Optional[AtomsDataFormat]):
        self.datapath = datapath
        self.format = format
