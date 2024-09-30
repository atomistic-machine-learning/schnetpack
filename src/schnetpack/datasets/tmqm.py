import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
from typing import List, Optional, Dict
from urllib import request as request
import gzip

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz
from ase.io import read
from tqdm import tqdm

import torch
from schnetpack.data import *
import schnetpack.properties as structure
from schnetpack.data import AtomsDataModuleError, AtomsDataModule

__all__ = ["TMQM"]


class TMQM(AtomsDataModule):
    """tmQM database of Ballcells 2020 of inorganic CSD structures.



    References:

        .. [#tmqm_1] https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041

    """

    # properties
    #  electronic and dispersion energies,
    # highest occupied molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO) energies,
    # HOMO/LUMO gap,
    # dipole moment, and natural charge of the metal center; GFN2-xTB polarizabilities are also provided.

    # these strings match the names in the header of the csv file
    csd_code = "CSD_code"  # should go into key-value pair
    energy = "Electronic_E"
    dispersion = "Dispersion_E"
    homo = "HOMO_Energy"
    lumo = "LUMO_Energy"
    gap = "HL_Gap"
    mu = "Dipole_M"
    alpha = "Polarizability"
    qm = "Metal_q"

    def __init__(
        self,
        datapath: str,
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
            remove_uncharacterized: do not include uncharacterized molecules.
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
        """
        super().__init__(
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
            **kwargs,
        )

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                TMQM.energy: "Ha",
                TMQM.dispersion: "Ha",
                TMQM.homo: "Ha",
                TMQM.lumo: "Ha",
                TMQM.gap: "Ha",
                TMQM.mu: "Debye",
                TMQM.alpha: "a0 a0 a0",
                TMQM.qm: "e",
            }

            tmpdir = tempfile.mkdtemp("tmQM")

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data(self, tmpdir, dataset: BaseAtomsData):
        tar_path = os.path.join(tmpdir, "tmQM_X1.xyz.gz")
        url = [
            "https://github.com/bbskjelstad/tmqm/raw/master/data/tmQM_X1.xyz.gz",
            "https://github.com/bbskjelstad/tmqm/raw/master/data/tmQM_X2.xyz.gz",
        ]

        url_y = "https://github.com/bbskjelstad/tmqm/raw/master/data/tmQM_y.csv"

        tmp_xyz_file = os.path.join(tmpdir, "tmQM_X.xyz")
        tmp_properties_file = os.path.join(tmpdir, "tmQM_y.csv")

        atomslist = []

        for u in url:
            request.urlretrieve(u, tar_path)
            with gzip.open(tar_path, "rb") as f_in:
                with open(tmp_xyz_file, "wb") as f_out:
                    lines = f_in.readlines()
                    # remove empty lines
                    lines = [line for line in lines if line.strip()]
                    f_out.writelines(lines)

            atomslist.extend(read(tmp_xyz_file, index=":"))

        # download proeprties in tmQM_y.csv
        request.urlretrieve(url_y, tmp_properties_file)

        # CSV format
        # CSD_code;Electronic_E;Dispersion_E;Dipole_M;Metal_q;HL_Gap;HOMO_Energy;LUMO_Energy;Polarizability
        # WIXKOE;-2045.524942;-0.239239;4.233300;2.109340;0.131080;-0.162040;-0.030960;598.457913
        # DUCVIG;-2430.690317;-0.082134;11.754400;0.759940;0.124930;-0.243580;-0.118650;277.750698
        # KINJOG;-3467.923206;-0.137954;8.301700;1.766500;0.140140;-0.236460;-0.096320;393.442545

        # read csv
        prop_list = []
        key_value_pairs_list = []

        with open(tmp_properties_file, "r") as file:
            lines = file.readlines()
            keys = lines[0].strip("\n").split(";")

            for l in lines[1:]:
                properties = l.split(";")
                prop_dict = {
                    k: np.array([float(v)]) for k, v in zip(keys[1:], properties[1:])
                }
                key_value_pairs = {k: v for k, v in zip(keys[0], properties[0])}
                prop_list.append(prop_dict)
                key_value_pairs_list.append(key_value_pairs)

        dataset.add_systems(property_list=prop_list, atoms_list=atomslist)
