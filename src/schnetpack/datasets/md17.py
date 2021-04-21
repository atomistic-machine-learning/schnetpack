import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
from typing import List
from urllib import request as request

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz

from schnetpack.data import *
import schnetpack.structure as structure
from schnetpack.data import AtomsDataModuleError, AtomsDataModule


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
        num_train: int,
        num_val: int,
        num_test: int = -1,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[torch.nn.Module] = None,
        train_transforms: Optional[torch.nn.Module] = None,
        val_transforms: Optional[torch.nn.Module] = None,
        test_transforms: Optional[torch.nn.Module] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        """
        Args:
            datapath: path to database (or target directory for download).
            format:
            load_properties: reduced set of properties to be loaded
            remove_uncharacterized: do not include uncharacterized molecules.
            transforms:
            train_transforms:
            val_transforms:
            test_transforms:
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
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
