import os
from typing import Optional, Any, List
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from schnetpack.data import *
import torch

from schnetpack.data import (
    AtomsDataFormat,
    resolve_format,
    create_dataset,
    load_dataset,
)


class AtomsDataModuleError(Exception):
    pass


class AtomsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datapath: str,
        format: Optional[AtomsDataFormat],
        environment_provider: Optional[torch.nn.Module],
        load_properties: List[str] = None,
        convert_float32: bool = True,
        transform_fn: Optional[torch.nn.Module] = None,
        train_transform_fn: Optional[torch.nn.Module] = None,
        val_transform_fn: Optional[torch.nn.Module] = None,
        test_transform_fn: Optional[torch.nn.Module] = None,
    ):
        if not transform_fn or not (
            train_transform_fn and val_transform_fn and test_transform_fn
        ):
            if environment_provider:
                raise AtomsDataModuleError(
                    "Either `custom_transform_fn` or"
                    " `environment_provider` needs to be given."
                )
            transforms = [environment_provider]
            if convert_float32:
                transforms.append(CastTo32())
            transform_fn = torch.nn.Sequential(*transforms)

        super().__init__(
            train_transforms=train_transform_fn or transform_fn,
            val_transforms=val_transform_fn or transform_fn,
            test_transforms=test_transform_fn or transform_fn,
        )
        self.datapath, self.format = resolve_format(datapath, format)
        self.load_properties = load_properties


class QM9(AtomsDataModule):
    """QM9 benchmark database for organic molecules.

    The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
    from including C, O, N, F. This class adds convenient functions to download QM9 from
    figshare and load the data into pytorch.

    References:
        .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"

    def __init__(
        self,
        datapath: str,
        format: Optional[AtomsDataFormat],
        environment_provider: Optional[torch.nn.Module],
        load_properties: List[str] = None,
        remove_uncharacterized: bool = False,
        convert_float32: bool = True,
        transform_fn: Optional[torch.nn.Module] = None,
        train_transform_fn: Optional[torch.nn.Module] = None,
        val_transform_fn: Optional[torch.nn.Module] = None,
        test_transform_fn: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            datapath: path to database (or target directory for download).
            format:
            environment_provider: define how neighborhood is calculated.
            load_properties: reduced set of properties to be loaded
            remove_uncharacterized: do not include uncharacterized molecules.
            convert_float32:
            transform_fn:
            train_transform_fn:
            val_transform_fn:
            test_transform_fn:
        """
        super().__init__(
            datapath=datapath,
            format=format,
            environment_provider=environment_provider,
            load_properties=load_properties,
            convert_float32=convert_float32,
            transform_fn=transform_fn,
            train_transform_fn=train_transform_fn,
            val_transform_fn=val_transform_fn,
            test_transform_fn=test_transform_fn,
        )
        self.remove_uncharacterized = remove_uncharacterized

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            self.dataset = create_dataset(self.datapath, self.format)
        else:
            self.dataset = load_dataset(self.datapath, self.format)
