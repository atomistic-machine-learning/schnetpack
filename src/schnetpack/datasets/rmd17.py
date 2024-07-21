import logging
import os
import shutil
import tempfile
import tarfile
from typing import List, Optional, Dict
from urllib import request as request

import numpy as np
from ase import Atoms

import torch
import schnetpack.properties as structure

from schnetpack.data import *

__all__ = ["rMD17"]


class rMD17(AtomsDataModule):
    """
    Revised MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    References:
        .. [#md17_1] https://figshare.com/articles/dataset/
            Revised_MD17_dataset_rMD17_/12672038?file=24013628
        .. [#md17_2] http://quantum-machine.org/gdml/#datasets

    """

    energy = "energy"
    forces = "forces"

    atomrefs = {
        energy: [
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
        aspirin="rmd17_aspirin.npz",
        azobenzene="rmd17_azobenzene.npz",
        benzene="rmd17_benzene.npz",
        ethanol="rmd17_ethanol.npz",
        malonaldehyde="rmd17_malonaldehyde.npz",
        naphthalene="rmd17_naphthalene.npz",
        paracetamol="rmd17_paracetamol.npz",
        salicylic_acid="rmd17_salicylic.npz",
        toluene="rmd17_toluene.npz",
        uracil="rmd17_uracil.npz",
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
        split_id: Optional[int] = None,
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
            val_batch_size: validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then
                batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers
                (overrides num_workers).
            num_test_workers: Number of test data loader workers
                (overrides num_workers).
            distance_unit: Unit of the atom positions and cell as a string
                (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for
                faster performance.
            split_id: The id of the predefined rMD17 train/test splits (0-4).
        """

        if split_id is not None:
            splitting = SubsamplePartitions(
                split_partition_sources=["known", "known", "test"], split_id=split_id
            )
        else:
            splitting = RandomSplit()

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
            splitting=splitting,
            **kwargs,
        )

        if molecule not in rMD17.datasets_dict.keys():
            raise AtomsDataModuleError("Molecule {} is not supported!".format(molecule))

        self.molecule = molecule

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                rMD17.energy: "kcal/mol",
                rMD17.forces: "kcal/mol/Ang",
            }

            tmpdir = tempfile.mkdtemp("md17")

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=rMD17.atomrefs,
            )
            dataset.update_metadata(molecule=self.molecule)

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)
            md = dataset.metadata
            if "molecule" not in md:
                raise AtomsDataModuleError(
                    "Not a valid rMD17 dataset! The molecule needs to be specified in "
                    + "the metadata."
                )
            if md["molecule"] != self.molecule:
                raise AtomsDataModuleError(
                    f"The dataset at the given location does not contain the specified "
                    + f"molecule: `{md['molecule']}` instead of `{self.molecule}`"
                )

    def _download_data(
        self,
        tmpdir,
        dataset: BaseAtomsData,
    ):
        logging.info("Downloading {} data".format(self.molecule))
        raw_path = os.path.join(tmpdir, "rmd17")
        tar_path = os.path.join(tmpdir, "rmd17.tar.gz")
        url = "https://figshare.com/ndownloader/files/23950376"
        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        logging.info("Extracting data...")
        tar = tarfile.open(tar_path)
        tar.extract(
            path=raw_path, member=f"rmd17/npz_data/{self.datasets_dict[self.molecule]}"
        )

        logging.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(
            os.path.join(
                raw_path, "rmd17", "npz_data", self.datasets_dict[self.molecule]
            )
        )

        numbers = data["nuclear_charges"]
        property_list = []
        for positions, energies, forces in zip(
            data["coords"], data["energies"], data["forces"]
        ):
            ats = Atoms(positions=positions, numbers=numbers)
            properties = {
                rMD17.energy: np.array([energies]),
                rMD17.forces: forces,
                structure.Z: ats.numbers,
                structure.R: ats.positions,
                structure.cell: ats.cell,
                structure.pbc: ats.pbc,
            }
            property_list.append(properties)

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")

        train_splits = []
        test_splits = []
        for i in range(1, 6):
            tar.extract(path=raw_path, member=f"rmd17/splits/index_train_0{i}.csv")
            tar.extract(path=raw_path, member=f"rmd17/splits/index_test_0{i}.csv")

            train_split = (
                np.loadtxt(
                    os.path.join(raw_path, "rmd17", "splits", f"index_train_0{i}.csv")
                )
                .flatten()
                .astype(int)
                .tolist()
            )
            train_splits.append(train_split)
            test_split = (
                np.loadtxt(
                    os.path.join(raw_path, "rmd17", "splits", f"index_test_0{i}.csv")
                )
                .flatten()
                .astype(int)
                .tolist()
            )
            test_splits.append(test_split)

        dataset.update_metadata(splits={"known": train_splits, "test": test_splits})

        tar.close()
        logging.info("Done.")
