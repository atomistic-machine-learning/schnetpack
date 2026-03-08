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
import torch
import numpy as np
import progressbar
from ase import Atoms

import schnetpack.properties as structure
from schnetpack.data import AtomsDataFormat
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError

__all__ = ["QM7X"]

pbar = None


def show_progress(block_num: int, block_size: int, total_size: int):
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


class QM7X(ASEAtomsData):
    """
    QM7-X dataset of equilibrium and non-equilibrium structures of small organic molecules.
    """

    forces = "forces"
    energy = "energy"
    Eat = "Eat"
    EPBE0 = "EPBE0"
    EMBD = "EMBD"
    FPBE0 = "FPBE0"
    FMBD = "FMBD"
    RMSD = "rmsd"

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
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
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
            format: dataset format
            load_properties: subset of properties to load
            transforms: Transform applied to each system separately before batching
            subset_idx: indices of the subset to load
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            **kwargs: additional keyword arguments.
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
        self.format = format

        self.download(
            datapath=datapath,
            distance_unit=distance_unit or "Ang",
        )

        # initialize without subset first, then apply dataset-specific filtering
        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            subset_idx=None,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

        self._apply_structure_filter(original_subset_idx=subset_idx)

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

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        """
        Download the QM7-X dataset and create the ASEAtomsData object.
        """
        if os.path.exists(datapath):
            _ = ASEAtomsData(datapath, self.format, load_structure=False)
            return

        tar_dir = self.raw_data_path or tempfile.mkdtemp("qm7x")
        try:
            atomrefs = {
                QM7X.energy: [
                    QM7X.EPBE0_atom[i] if i in QM7X.EPBE0_atom else 0.0
                    for i in range(0, 18)
                ]
            }

            dataset = ASEAtomsData.create(
                datapath=datapath,
                distance_unit=distance_unit,
                property_unit_dict=QM7X.property_unit_dict,
                atomrefs=atomrefs,
            )

            hd_files = self._download_data(tar_dir)
            if self.remove_duplicates:
                self._download_duplicates_ids(tar_dir)
            self._parse_data(hd_files, dataset)

        finally:
            if self.raw_data_path is None:
                shutil.rmtree(tar_dir, ignore_errors=True)

    def _download_duplicates_ids(self, tar_dir: str):
        url = "https://zenodo.org/record/4288677/files/DupMols.dat"
        target_path = os.path.join(tar_dir, "DupMols.dat")
        checksum = "5d886ccac38877c8cb26c07704dd1034"

        download_and_check(url, target_path, checksum)

        dup_mols = []
        with open(target_path, "r") as f:
            for line in f:
                dup_mols.append(line.rstrip("\n")[:-4])

        self.duplicates_ids = dup_mols

    def _download_data(self, tar_dir: str, ignore_extracted: bool = True) -> List[str]:
        file_ids = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000"]
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

        extracted = []
        for file_id in file_ids:
            xz_path = os.path.join(tar_dir, f"{file_id}.xz")
            hd_path = os.path.join(tar_dir, f"{file_id}.hdf5")
            extract_xz(xz_path, hd_path)
            extracted.append(hd_path)

        return extracted

    def _parse_data(self, files: List[str], dataset: ASEAtomsData):
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

                        if "opt" in conf_id:
                            conf_id = conf_id[:-3] + "d0"

                        ids = map(lambda x: int(x), re.findall(r"\d+", conf_id))

                        atoms_list.append(ats)
                        property_list.append(properties)

                        for key, idx in zip(groups_ids.keys(), ids):
                            groups_ids[key].append(idx)

            logging.info(f"Write parsed data from {os.path.basename(file)} to db ...")
            dataset.add_systems(property_list=property_list, atoms_list=atoms_list)

            md = dataset.metadata
            if "groups_ids" in md:
                for key, ids in groups_ids.items():
                    groups_ids[key] = md["groups_ids"][key] + ids

                last_id = md["groups_ids"]["id"][-1]
                sys_ids = list(range(last_id + 1, last_id + len(atoms_list) + 1))
                groups_ids["id"] = md["groups_ids"]["id"] + sys_ids
            else:
                groups_ids["id"] = list(range(1, len(atoms_list) + 1))

            dataset.update_metadata(groups_ids=groups_ids)
            logging.info("Done.")
