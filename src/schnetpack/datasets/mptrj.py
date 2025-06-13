import logging
import os
import shutil
import tempfile
import zipfile
from typing import List, Optional, Dict
from urllib import request as request

from ase.io import read
from ase import Atoms
import numpy as np

import torch
import schnetpack.properties as structure
from matbench_discovery.data import ase_atoms_from_zip
from schnetpack.data import *
import ase

__all__ = ["MPTraj"]


class MPTraj(AtomsDataModule):
    """
    MPTRJ Dataset loader (custom .extxyz inside .zip) using SchNetPack.
    """

    energy = "energy"
    forces = "forces"
    stress = "stress"
    MaterialId = "material_id"

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

        # Dataset specific configuration
        self.datasets_dict = {
            "mptrj": "mp/2024-09-03-mp-trj.extxyz.zip",
        }
        self.download_url = "https://figshare.com/files/49034296"
        self.molecule = "mptrj"
        self.tmpdir = "mptrj_tmp"
        self.atomrefs = {
            self.energy: [0.0]
            * 119  # Replace with real atom reference values if available
        }

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                self.energy: "kcal/mol",
                self.forces: "kcal/mol/Ang",
                self.stress: "kcal/mol/Ang^3",
                self.MaterialId: None,
            }

            tmpdir = tempfile.mkdtemp(self.tmpdir)

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=self.atomrefs,
            )
            dataset.update_metadata(molecule=self.molecule)

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data(self, tmpdir, dataset: BaseAtomsData):
        filename = self.datasets_dict[self.molecule]
        url = self.download_url
        local_path = os.path.join(tmpdir, os.path.basename(filename))
        print(local_path)

        logging.info(f"Downloading {filename} from {url}...")
        request.urlretrieve(url, local_path)

        logging.info("Loading structures from zip file...")
        atoms_list = ase_atoms_from_zip(
            zip_filename=local_path,
            filename_to_info=True,
            # limit=200,  # remove this limit to read all the structures
        )

        property_list = []
        key_value_pairs_list = []
        for atoms in atoms_list:
            properties = {
                self.energy: np.array([atoms.get_total_energy()]),
                self.forces: atoms.get_forces(),
                self.stress: atoms.get_stress(),
                structure.Z: atoms.get_atomic_numbers(),
                structure.R: atoms.get_positions(),
                structure.cell: atoms.get_cell(),
                structure.pbc: atoms.get_pbc(),
            }

            property_list.append(properties)
            key_value_pairs_list.append({"material_id": atoms.info.get("material_id")})

        logging.info("Write atoms to db...")
        dataset.add_systems(
            property_list=property_list,
            key_value_list=key_value_pairs_list,
        )
        logging.info("Done.")
