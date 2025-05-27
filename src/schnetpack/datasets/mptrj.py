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


class MPTrajDataModule(AtomsDataModule):
    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        datasets_dict: Dict[str, str],
        download_url: str,
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
        tmpdir: str = "gdml_tmp",
        atomrefs: Optional[Dict[str, List[float]]] = None,
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

        self.datasets_dict = datasets_dict
        self.download_url = download_url
        self.atomrefs = atomrefs
        self.tmpdir = tmpdir

        if molecule not in self.datasets_dict:
            raise AtomsDataModuleError(
                f"Molecule '{molecule}' not found in datasets_dict."
            )
        self.molecule = molecule

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                self.energy: "kcal/mol",
                self.forces: "kcal/mol/Ang",
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

        logging.info(f"Downloading {filename} from {url}...")
        request.urlretrieve(url, local_path)

        logging.info("Loading structures from zip file...")
        atoms_list = ase_atoms_from_zip(
            zip_filename=local_path,
            filename_to_info=True,
            limit=20000,  # remove this limit to read all the structures
        )

        property_list = []
        key_value_pairs_list = []
        for atoms in atoms_list:
            # Extract metadata from info dictionary
            # info = atoms.info
            # metadata = {
            #     "material_id": info.get("material_id"),
            #     "formula": info.get("formula"),
            #     "task_id": info.get("task_id"),
            #     "calc_id": info.get("calc_id"),
            #     "ionic_step": info.get("ionic_step"),
            #     "frame_id": info.get("frame_id")
            # }

            properties = {
                self.energy: np.array([atoms.get_total_energy()]),
                self.forces: atoms.get_forces(),
                structure.Z: atoms.get_atomic_numbers(),
                structure.R: atoms.get_positions(),
                structure.cell: atoms.get_cell(),
                structure.pbc: atoms.get_pbc(),
            }

            # Add optional properties if available
            # if atoms.get_stress() is not None:
            #     properties["stress"] = atoms.get_stress()
            # if atoms.get_initial_magnetic_moments() is not None:
            #     properties["magmoms"] = atoms.get_initial_magnetic_moments()

            property_list.append(properties)
            key_value_pairs_list.append({"material_id": atoms.info.get("material_id")})

        logging.info("Write atoms to db...")
        dataset.add_systems(
            property_list=property_list
        )  # key_value_list=key_value_pairs_list)
        logging.info("Done.")


class MPTraj(MPTrajDataModule):
    """
    MPTRJ Dataset loader (custom .extxyz inside .zip) using SchNetPack.
    """

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
        datasets_dict = {
            "mptrj": "mp/2024-09-03-mp-trj.extxyz.zip",
        }

        atomrefs = {
            self.energy: [0.0]
            * 10  # Replace with real atom reference values if available
        }

        super(MPTraj, self).__init__(
            datasets_dict=datasets_dict,
            download_url="https://figshare.com/files/49034296",
            tmpdir="mptrj_tmp",
            molecule="mptrj",
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
