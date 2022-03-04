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


class MD17(AtomsDataModule):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    References:
        .. [#md17_1] http://quantum-machine.org/gdml/#datasets

    """

    energy = "energy"
    forces = "forces"

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
    existing_datasets = datasets_dict.keys()

    # properties
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

        if molecule not in MD17.datasets_dict.keys():
            raise AtomsDataModuleError("Molecule {} is not supported!".format(molecule))

        self.molecule = molecule

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                MD17.energy: "kcal/mol",
                MD17.forces: "kcal/mol/Ang",
            }

            tmpdir = tempfile.mkdtemp("md17")

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )
            dataset.update_metadata(molecule=self.molecule)

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)
            md = dataset.metadata
            if "molecule" not in md:
                raise AtomsDataModuleError(
                    "Not a valid MD17 dataset! The molecule needs to be specified in the metadata."
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
        url = (
            "http://www.quantum-machine.org/gdml/data/npz/"
            + self.datasets_dict[self.molecule]
        )

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(rawpath)

        numbers = data["z"]
        property_list = []
        for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
            ats = Atoms(positions=positions, numbers=numbers)
            properties = {MD17.energy: energies, MD17.forces: forces}
            properties[structure.Z] = ats.numbers
            properties[structure.R] = ats.positions
            properties[structure.cell] = ats.cell
            properties[structure.pbc] = ats.pbc
            property_list.append(properties)

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")
