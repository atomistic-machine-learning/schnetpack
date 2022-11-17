import logging
import os
import shutil
import tempfile
from typing import List, Optional, Dict
from urllib import request as request

import numpy as np
from ase import Atoms

import torch
import schnetpack.properties as structure

from schnetpack.data import *

__all__ = ["MD17"]


class GDMLDataModule(AtomsDataModule):
    """
    Base class for GDML type data (e.g. MD17 or MD22). Requires a dictionary translating between molecule and filenames
    and an URL under which the molecular datasets can be found.
    """

    energy = "energy"
    forces = "forces"

    # properties
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
        """
        Args:
            datasets_dict: dictionary mapping molecule names to dataset names.
            download_url: URL where individual molecule datasets can me found.
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
            tmpdir: name of temporary directory used for parsing.
            atomrefs: properties of free atoms
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

        self.datasets_dict = datasets_dict
        self.download_url = download_url
        self.atomrefs = atomrefs
        self.tmpdir = tmpdir

        self.existing_datasets = self.datasets_dict.keys()

        if molecule not in self.datasets_dict.keys():
            raise AtomsDataModuleError("Molecule {} is not supported!".format(molecule))

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
            md = dataset.metadata
            if "molecule" not in md:
                raise AtomsDataModuleError(
                    "Not a valid GDML dataset! The molecule needs to be specified in the metadata."
                )
            if md["molecule"] != self.molecule:
                raise AtomsDataModuleError(
                    f"The dataset at the given location does not contain the specified molecule: "
                    + f"`{md['molecule']}` instead of `{self.molecule}`"
                )

    def _download_data(
        self,
        tmpdir,
        dataset: BaseAtomsData,
    ):
        logging.info("Downloading {} data".format(self.molecule))
        rawpath = os.path.join(tmpdir, self.datasets_dict[self.molecule])
        url = self.download_url + self.datasets_dict[self.molecule]

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(rawpath)

        numbers = data["z"]
        property_list = []
        for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
            ats = Atoms(positions=positions, numbers=numbers)
            properties = {
                self.energy: energies,
                self.forces: forces,
                structure.Z: ats.numbers,
                structure.R: ats.positions,
                structure.cell: ats.cell,
                structure.pbc: ats.pbc,
            }
            property_list.append(properties)

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")


class MD17(GDMLDataModule):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    References:
        .. [#md17_1] http://quantum-machine.org/gdml/#datasets

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

        datasets_dict = dict(
            aspirin="aspirin_dft.npz",
            # aspirin_ccsd='aspirin_ccsd.zip',
            azobenzene="azobenzene_dft.npz",
            benzene="benzene_dft.npz",
            ethanol="ethanol_dft.npz",
            # ethanol_ccsdt='ethanol_ccsd_t.zip',
            malonaldehyde="malonaldehyde_dft.npz",
            # malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
            naphthalene="naphthalene_dft.npz",
            paracetamol="paracetamol_dft.npz",
            salicylic_acid="salicylic_dft.npz",
            toluene="toluene_dft.npz",
            # toluene_ccsdt='toluene_ccsd_t.zip',
            uracil="uracil_dft.npz",
        )

        super(MD17, self).__init__(
            datasets_dict=datasets_dict,
            download_url="http://www.quantum-machine.org/gdml/data/npz/",
            tmpdir="md17",
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
