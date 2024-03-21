import hashlib
import logging
import lzma
import os
import re
import shutil
import tempfile
from typing import Dict, List, Optional
from urllib import request as request

import h5py
import numpy as np
import progressbar
import torch
from ase import Atoms
from tqdm import tqdm

from schnetpack.data import *
from schnetpack.data import AtomsDataModule

__all__ = ["QM7X"]

# Helper functions
pbar = None


def show_progress(block_num: int, block_size: int, total_size: int):
    """
    progress callback for files downloads
    """
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download_and_check(url: str, tar_path: str, checksum: str):
    """
    Download file from url to tar_path and check md5 checksum.
    """

    file = url.split("/")[-1]

    # check if file already exists and has correct checksum
    if os.path.exists(tar_path):
        md5_sum = hashlib.md5(open(tar_path, "rb").read()).hexdigest()
        if md5_sum == checksum:
            logging.info(
                f"File {file} already exists and has correct checksum. Skipping download."
            )
            return
        else:
            logging.info(
                f"File {file} already exists but has wrong checksum. Redownloading."
            )
            os.remove(tar_path)

    logging.info(f"Downloading {url} ...")
    request.urlretrieve(url, tar_path, show_progress)

    if hashlib.md5(open(tar_path, "rb").read()).hexdigest() == checksum:
        logging.info("Done.")
    else:
        raise RuntimeError(
            f"Checksum of downloaded file {file} does not match. Please try again."
        )


def extract_xz(source: str, target: str):
    """
    helper to extract xz files.
    """
    s_file = source.split("/")[-1]
    t_file = target.split("/")[-1]

    if os.path.exists(target):
        logging.info(f"File {t_file} already exists. Skipping extraction.")
        return

    logging.info(f"Extracting {s_file} ...")

    try:
        with lzma.open(source) as fin, open(target, mode="wb") as fout:
            shutil.copyfileobj(fin, fout)
    except:
        if os.path.exists(target):
            os.remove(target)
        raise RuntimeError(f"Could not extract file {s_file}. Please try again.")

    logging.info("Done.")


class QM7X(AtomsDataModule):
    """
    QM7-X a comprehensive dataset of > 40 physicochemical properties for ~4.2 M equilibrium and non-equilibrium
    structure of small organic molecules with up to seven non-hydrogen (C, N, O, S, Cl) atoms.
    This class adds convenient functions to download QM7-X and load the data into pytorch.

    References:

        .. [#qm7x_1] https://zenodo.org/record/4288677

    """

    # more molecular and atomic properties can be found in the original paper and added here
    # Notice that adding more properties can drastically increase the size of the dataset
    # adding more properties here requires to add them to the property_unit_dict
    # and there key mapping in the raw dataset in property_dataset_keys.

    forces = "forces"  # total ePBE0+MBD forces
    energy = "energy"  # ePBE0+MBD: total energy after convergence of the PBE0 exchange-correlation functional and the MBD dispersion correction
    Eat = "Eat"  # atomization energy using PBE0 energy per atom and ePBE0+MBD total energy
    EPBE0 = "EPBE0"  # ePBE0: total energy at the level of PBE0
    EMBD = "EMBD"  # eMBD: total energy at the level of MBD
    FPBE0 = "FMBD"  # FPBE0: total ePBE0 forces
    FMBD = "FMBD"  # FMBD: total eMBD forces
    RMSD = "rmsd"  # root mean square deviation of the atomic positions from the equilibrium structure

    property_unit_dict = {
        forces: "eV/Ang",
        energy: "eV",
        Eat: "eV",
        EPBE0: "eV",
        EMBD: "eV",
        FPBE0: "eV/Ang",
        FMBD: "eV/Ang",
        RMSD: "Ang",
    }

    # the original keys in the raw dataset to query the properties
    property_dataset_keys = {
        forces: "totFOR",
        energy: "ePBE0+MBD",
        Eat: "eAT",
        EPBE0: "ePBE0",
        EMBD: "eMBD",
        FPBE0: "pbe0FOR",
        FMBD: "vdwFOR",
        RMSD: "sRMSD",
    }

    # atom energies (atomrefs) from PBE0
    EPBE0_atom = {
        1: -13.641404161,
        6: -1027.592489146,
        7: -1484.274819088,
        8: -2039.734879322,
        16: -10828.707468187,
        17: -12516.444619523,
    }

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        raw_data_path: str = None,
        remove_duplicates: bool = True,
        only_equilibrium: bool = False,
        only_non_equilibrium: bool = False,
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
        splitting: Optional[SplittingStrategy] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            raw_data_path: path to raw data. If None use tmp dir otherwise persist data and not remove it.
            remove_duplicates: remove duplicated equilibrium structures with different non-equilibrium structures
            only_equilibrium: only use equilibrium structures
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
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
            splitting: Method to generate train/validation/test partitions
                (default: GroupSplit(splitting_key="smiles_id"))
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
            splitting=splitting or GroupSplit(splitting_key="smiles_id"),
            **kwargs,
        )

        self.raw_data_path = raw_data_path
        self.remove_duplicates = remove_duplicates
        self.duplicates_ids = None
        self.only_equilibrium = only_equilibrium
        self.only_non_equilibrium = only_non_equilibrium

    def _download_duplicates_ids(self, tar_dir: str):
        """
        download duplicates ids for QM7-X
        """
        url = f"https://zenodo.org/record/4288677/files/DupMols.dat"
        tar_path = os.path.join(tar_dir, "DupMols.dat")
        checksum = "5d886ccac38877c8cb26c07704dd1034"

        download_and_check(url, tar_path, checksum)

        # fetch duplicates ids
        dup_mols = []
        for line in open(tar_path, "r"):
            dup_mols.append(line.rstrip("\n")[:-4])
        self.duplicates_ids = dup_mols

    def _download_data(self, tar_dir: str, ignore_extracted: bool = True) -> List[str]:
        """
        download data and extract them
        """
        file_ids = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000"]

        # file fingerprints to check integrity
        checksums = [
            "b50c6a5d0a4493c274368cf22285503e",
            "4418a813daf5e0d44aa5a26544249ee6",
            "f7b5aac39a745f11436047c12d1eb24e",
            "26819601705ef8c14080fa7fc69decd4",
            "85ac444596b87812aaa9e48d203d0b70",
            "787fc4a9036af0e67c034a30ad854c07",
            "5ecce00a188410d06b747cb683d8d347",
            "c893ae88b8f5c32541c3f024fc1daa45",
        ]

        logging.info("Downloading QM7-X data files ...")

        # download files
        for i, file_id in enumerate(file_ids):
            if ignore_extracted and os.path.exists(
                os.path.join(tar_dir, f"{file_id}.hdf5")
            ):
                logging.info(
                    f"File {file_id}.xz exists in extracted version {file_id}.hdf5 already, skipping download."
                )
                continue

            url = f"https://zenodo.org/record/4288677/files/{file_id}.xz"

            tar_path = os.path.join(tar_dir, f"{file_id}.xz")
            download_and_check(url, tar_path, checksums[i])

        # extract the compressed files
        extracted = []
        for i, file_id in enumerate(file_ids):
            xz_path = os.path.join(tar_dir, f"{file_id}.xz")
            hd_path = os.path.join(tar_dir, f"{file_id}.hdf5")

            extract_xz(xz_path, hd_path)

            extracted.append(hd_path)

        return extracted

    def _parse_data(self, files: List[str], dataset: BaseAtomsData):
        """
        Parse the downloaded data files and add them to the dataset.
        """

        # parse the data files

        for file in files:
            logging.info(f"Parsing {file.split('/')[-1]} ...")

            atoms_list = []
            property_list = []
            groups_ids = {
                "smiles_id": [],
                "stereo_iso_id": [],
                "conform_id": [],
                "step_id": [],
            }

            with h5py.File(file, "r") as mol_dict:
                for mol_id, mol in tqdm(mol_dict.items()):
                    for conf_id, conf in mol.items():
                        # exclude equilibrium duplicates
                        trunc_id = conf_id[::-1].split("-", 1)[-1][::-1]
                        if self.remove_duplicates and trunc_id in self.duplicates_ids:
                            continue

                        ats = Atoms(positions=conf["atXYZ"], numbers=conf["atNUM"])
                        properties = {
                            key: np.array(
                                conf[QM7X.property_dataset_keys[key]], dtype=np.float64
                            )
                            for key in QM7X.property_unit_dict.keys()
                        }

                        # get the hierarchical ids for each system
                        if "opt" in conf_id:
                            conf_id = (
                                conf_id[:-3] + "d0"
                            )  # repalce the 'opt' key with id 'd0'
                        ids = map(lambda x: int(x), re.findall(r"\d+", conf_id))

                        atoms_list.append(ats)
                        property_list.append(properties)

                        # save the hierarchical ids for each system in same order as the systems
                        for i, j in zip(groups_ids.keys(), ids):
                            groups_ids[i].append(j)

            # add the data to the dataset
            logging.info(f"Write parsed data from {file.split('/')[-1]} to db ...")

            dataset.add_systems(property_list=property_list, atoms_list=atoms_list)

            # add the hierarchical ids to the metadata
            md = dataset.metadata
            if "groups_ids" in md.keys():
                for key, ids in groups_ids.items():
                    groups_ids[key] = md["groups_ids"][key] + ids

                # add the ids as in the database of the new added systems
                last_id = md["groups_ids"]["id"][-1]
                sys_ids = list(range(last_id + 1, last_id + len(atoms_list) + 1))
                groups_ids["id"] = md["groups_ids"]["id"] + sys_ids
            else:
                groups_ids["id"] = list(range(1, len(atoms_list) + 1))

            dataset.update_metadata(groups_ids=groups_ids)

            logging.info("Done.")

    def prepare_data(self):
        """
        prepare data for pytorch lightning data module
        """
        if not os.path.exists(self.datapath):
            tar_dir = self.raw_data_path or tempfile.mkdtemp("qm7x")

            atomrefs = {
                QM7X.energy: [
                    QM7X.EPBE0_atom[i] if i in QM7X.EPBE0_atom else 0.0
                    for i in range(0, 18)
                ]
            }

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=QM7X.property_unit_dict,
                atomrefs=atomrefs,
            )

            hd_files = self._download_data(tar_dir)
            if self.remove_duplicates:
                self._download_duplicates_ids(tar_dir)
            self._parse_data(hd_files, dataset)

            if self.raw_data_path is None:
                shutil.rmtree(tar_dir)

    def setup(self, stage=None):
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
            )

            # use subset of equilibrium structures

            if self.only_equilibrium or self.only_non_equilibrium:
                step_ids = self.dataset.metadata["groups_ids"]["step_id"]

                if len(step_ids) != len(self.dataset):
                    raise ValueError(
                        "The dataset size does not match the size of step ids arrays in meta data."
                    )

                if self.only_equilibrium:
                    eq_indices = [i for i, s in enumerate(step_ids) if s == 0]
                else:
                    eq_indices = [i for i, s in enumerate(step_ids) if s != 0]

                self.dataset = self.dataset.subset(eq_indices)

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            self._test_dataset = self.dataset.subset(self.test_idx)

        self._setup_transforms()
