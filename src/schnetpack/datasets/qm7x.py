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
from ase import Atoms

from schnetpack.transform.base import Transform
from schnetpack.data.atoms import DownloadableASEAtomsData, AtomsDataError

__all__ = ["QM7X"]

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


def download_and_check(url: str, target_path: str, checksum: str):
    """
    Download file from url to tar_path and check md5 checksum.
    """
    file_name = url.split("/")[-1]

    if os.path.exists(target_path):
        md5_sum = hashlib.md5(open(target_path, "rb").read()).hexdigest()
        if md5_sum == checksum:
            logging.info(
                f"File {file_name} already exists and has correct checksum. Skipping download."
            )
            return
        logging.info(
            f"File {file_name} already exists but has wrong checksum. Redownloading."
        )
        os.remove(target_path)

    logging.info(f"Downloading {url} ...")
    request.urlretrieve(url, target_path, show_progress)

    if hashlib.md5(open(target_path, "rb").read()).hexdigest() != checksum:
        raise RuntimeError(
            f"Checksum of downloaded file {file_name} does not match. Please try again."
        )
    logging.info("Done.")


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
    except Exception as e:
        if os.path.exists(target):
            os.remove(target)
        raise RuntimeError(f"Could not extract file {s_file}. Please try again.") from e

    logging.info("Done.")


class QM7X(DownloadableASEAtomsData):
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
        raw_data_path: Optional[str] = None,
        remove_duplicates: bool = True,
        only_equilibrium: bool = False,
        only_non_equilibrium: bool = False,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            raw_data_path: path to raw data
            remove_duplicates: do not include duplicate molecules
            only_equilibrium: only include equilibrium molecules
            only_non_equilibrium: only include non-equilibrium molecules
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """

        if only_equilibrium and only_non_equilibrium:
            raise AtomsDataError(
                "only_equilibrium and only_non_equilibrium cannot both be True."
            )

        self.raw_data_path = raw_data_path
        self.remove_duplicates = remove_duplicates
        self.duplicates_ids = None
        self.only_equilibrium = only_equilibrium
        self.only_non_equilibrium = only_non_equilibrium

        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

        # initialize without subset first, then apply dataset-specific filtering
        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

        self._apply_structure_filter(original_subset_idx=subset_idx)

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        return {
            QM7X.forces: "eV/Ang",
            QM7X.energy: "eV",
            QM7X.Eat: "eV",
            QM7X.EPBE0: "eV",
            QM7X.EMBD: "eV",
            QM7X.FPBE0: "eV/Ang",
            QM7X.FMBD: "eV/Ang",
            QM7X.RMSD: "Ang",
        }

    def _apply_structure_filter(self, original_subset_idx: Optional[List[int]]) -> None:
        effective_subset = original_subset_idx

        if self.only_equilibrium or self.only_non_equilibrium:
            step_ids = self.metadata["groups_ids"]["step_id"]

            if len(step_ids) != self.conn.count():
                raise AtomsDataError(
                    "Dataset size does not match size of step_id metadata."
                )

            if self.only_equilibrium:
                filtered = [i for i, s in enumerate(step_ids) if s == 0]
            else:
                filtered = [i for i, s in enumerate(step_ids) if s != 0]

            if effective_subset is None:
                effective_subset = filtered
            else:
                filtered_set = set(filtered)
                effective_subset = [i for i in effective_subset if i in filtered_set]

        self.subset_idx = effective_subset

    def download(self) -> None:
        """
        Download the QM7-X dataset and create the ASEAtomsData object.
        """
        tar_dir = self.raw_data_path or tempfile.mkdtemp("qm7x")
        atomrefs = {
            QM7X.energy: [
                QM7X.EPBE0_atom[i] if i in QM7X.EPBE0_atom else 0.0
                for i in range(0, 18)
            ]
        }

        # Write the PBE0 single-atom reference energies into the DB metadata,
        # like the other datasets do.
        md = self.metadata
        md["atomrefs"] = atomrefs
        self._set_metadata(md)

        hd_files = self._download_data(tar_dir)
        if self.remove_duplicates:
            self._download_duplicates_ids(tar_dir)
        self._parse_data(hd_files)

        if self.raw_data_path is None:
            shutil.rmtree(tar_dir, ignore_errors=True)

    def _download_duplicates_ids(self, tar_dir: str):
        """
        download duplicates ids for QM7-X
        """
        url = "https://zenodo.org/record/4288677/files/DupMols.dat"
        target_path = os.path.join(tar_dir, "DupMols.dat")
        checksum = "5d886ccac38877c8cb26c07704dd1034"

        download_and_check(url, target_path, checksum)

        # fetch duplicates ids
        dup_mols = []
        with open(target_path, "r") as f:
            for line in f:
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

        for i, file_id in enumerate(file_ids):
            if ignore_extracted and os.path.exists(
                os.path.join(tar_dir, f"{file_id}.hdf5")
            ):
                logging.info(
                    f"File {file_id}.hdf5 already exists. Skipping download of {file_id}.xz."
                )
                continue

            url = f"https://zenodo.org/record/4288677/files/{file_id}.xz"
            xz_path = os.path.join(tar_dir, f"{file_id}.xz")
            download_and_check(url, xz_path, checksums[i])

        # extract the compressed files
        extracted = []
        for file_id in file_ids:
            xz_path = os.path.join(tar_dir, f"{file_id}.xz")
            hd_path = os.path.join(tar_dir, f"{file_id}.hdf5")
            extract_xz(xz_path, hd_path)
            extracted.append(hd_path)

        return extracted

    def _parse_data(self, files: List[str]):
        """
        Parse the downloaded data files and add them to the dataset.
        """
        for file in files:
            logging.info(f"Parsing {os.path.basename(file)} ...")

            atoms_list = []
            property_list = []
            groups_ids = {
                "smiles_id": [],
                "stereo_iso_id": [],
                "conform_id": [],
                "step_id": [],
            }

            with h5py.File(file, "r") as mol_dict:
                for _mol_id, mol in mol_dict.items():
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
                            for key in QM7X._native_property_units().keys()
                        }

                        # get the hierarchical ids for each system
                        if "opt" in conf_id:
                            conf_id = conf_id[:-3] + "d0"

                        ids = map(lambda x: int(x), re.findall(r"\d+", conf_id))

                        atoms_list.append(ats)
                        property_list.append(properties)

                        # save the hierarchical ids for each system in same order as the systems
                        for key, idx in zip(groups_ids.keys(), ids):
                            groups_ids[key].append(idx)

            logging.info(f"Write parsed data from {os.path.basename(file)} to db ...")
            self.add_systems(property_list=property_list, atoms_list=atoms_list)

            # add the hierarchical ids to the metadata
            md = self.metadata
            if "groups_ids" in md:
                for key, ids in groups_ids.items():
                    groups_ids[key] = md["groups_ids"][key] + ids

                # add the ids as in the database of the new added systems
                last_id = md["groups_ids"]["id"][-1]
                sys_ids = list(range(last_id + 1, last_id + len(atoms_list) + 1))
                groups_ids["id"] = md["groups_ids"]["id"] + sys_ids
            else:
                groups_ids["id"] = list(range(1, len(atoms_list) + 1))

            self.update_metadata(groups_ids=groups_ids)
            logging.info("Done.")
