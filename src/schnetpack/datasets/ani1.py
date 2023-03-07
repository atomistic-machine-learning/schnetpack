import logging
import os
import shutil
import tempfile
from typing import List, Optional, Dict
from urllib import request as request

import numpy as np
from ase import Atoms

import torch
import tarfile
import h5py

from schnetpack.data import *

log = logging.getLogger(__name__)


class ANI1(AtomsDataModule):
    """
    ANI1 benchmark database.
    This class adds convenience functions to download ANI1 from figshare and
    load the data into pytorch.

    References:

        .. [#ani1] https://arxiv.org/abs/1708.04987

    """

    energy = "energy"

    self_energies = {
        "H": -0.500607632585,
        "C": -37.8302333826,
        "N": -54.5680045287,
        "O": -75.0362229210,
    }

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_heavy_atoms: int = 8,
        high_energies: bool = False,
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
        **kwargs
    ):
        """

        Args:
            datapath: path to dataset
            num_heavy_atoms: number of heavy atoms. (See 'Table 1' in Ref. [#ani1]_)
            high_energies: add high energy conformations. (See 'Technical Validation' of Ref. [#ani1]_)
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
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).

        """
        self.num_heavy_atoms = num_heavy_atoms
        self.high_energies = high_energies

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
            **kwargs
        )

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                ANI1.energy: "Hartree",
            }
            atomrefs = self._create_atomrefs()

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=atomrefs,
            )
            tmpdir = tempfile.mkdtemp("ani1")

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data(self, tmpdir, dataset: BaseAtomsData):
        logging.info("downloading ANI-1 data...")
        tar_path = os.path.join(tmpdir, "ANI1_release.tar.gz")
        raw_path = os.path.join(tmpdir, "data")
        url = "https://ndownloader.figshare.com/files/9057631"

        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()

        logging.info("parse files...")
        for i in range(1, self.num_heavy_atoms + 1):
            file_name = os.path.join(raw_path, "ANI-1_release", "ani_gdb_s0%d.h5" % i)
            logging.info("start to parse %s" % file_name)
            self._load_h5_file(file_name, dataset)

        logging.info("done...")

    def _load_h5_file(self, file_name, dataset):
        atoms_list = []
        properties_list = []

        store = h5py.File(file_name)
        for file_key in store:
            for molecule_key in store[file_key]:
                molecule_group = store[file_key][molecule_key]
                species = "".join([str(s)[-2] for s in molecule_group["species"]])
                positions = molecule_group["coordinates"]
                energies = molecule_group["energies"]

                # loop over conformations
                for i in range(energies.shape[0]):
                    atm = Atoms(species, positions[i])
                    energy = energies[i]
                    properties = {self.energy: np.array([energy])}
                    atoms_list.append(atm)
                    properties_list.append(properties)

                # high energy conformations as described in 'Technical Validation'
                # section of https://arxiv.org/abs/1708.04987
                if self.high_energies:
                    high_energy_positions = molecule_group["coordinatesHE"]
                    high_energies = molecule_group["energiesHE"]

                    # loop over high energy conformations
                    for i in range(high_energies.shape[0]):
                        atm = Atoms(species, high_energy_positions[i])
                        high_energy = high_energies[i]
                        properties = {self.energy: np.array([high_energy])}
                        atoms_list.append(atm)
                        properties_list.append(properties)

        # write data to ase db
        dataset.add_systems(atoms_list=atoms_list, property_list=properties_list)

    def _create_atomrefs(self):
        atref = np.zeros((100,))

        # converts units to eV (which are set to one in ase)
        atref[1] = self.self_energies["H"]
        atref[6] = self.self_energies["C"]
        atref[7] = self.self_energies["N"]
        atref[8] = self.self_energies["O"]

        return {ANI1.energy: atref.tolist()}
