import torch

from schnetpack.data import *
from schnetpack.datasets.md17 import GDMLDataModule

from typing import Optional, Dict, List

all = ["MD22"]


class MD22(GDMLDataModule):
    """
    MD22 benchmark data set for extended molecules containing molecular forces.

    References:
        .. [#md22_1] http://quantum-machine.org/gdml/#datasets

    """

    def __init__(
        self,
        datapath: str,
        molecule: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
        """
        atomrefs = {
            self.energy: [
                0.0,
                -313.5150902000774,
                0.0,
                0.0,
                0.0,
                0.0,
                -23622.587180094913,
                -34219.46811826416,
                -47069.30768969713,
            ]
        }
        datasets_dict = {
            "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
            "DHA": "md22_DHA.npz",
            "stachyose": "md22_stachyose.npz",
            "AT-AT": "md22_AT-AT.npz",
            "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
            "buckyball-catcher": "md22_buckyball-catcher.npz",
            "double-walled_nanotube": "md22_double-walled_nanotube.npz",
        }

        super(MD22, self).__init__(
            datasets_dict=datasets_dict,
            download_url="http://www.quantum-machine.org/gdml/repo/datasets/",
            tmpdir="md22",
            molecule=molecule,
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            atomrefs=atomrefs,
            **kwargs,
        )
