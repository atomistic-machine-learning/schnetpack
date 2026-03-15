import logging
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
from ase import Atoms
from ase.db import connect

import schnetpack.properties as structure
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["rMD17"]


class rMD17(ASEAtomsData):
    """
    Revised MD17 benchmark dataset for molecular dynamics of small molecules
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

    datasets_dict = {
        "aspirin": "rmd17_aspirin.npz",
        "azobenzene": "rmd17_azobenzene.npz",
        "benzene": "rmd17_benzene.npz",
        "ethanol": "rmd17_ethanol.npz",
        "malonaldehyde": "rmd17_malonaldehyde.npz",
        "naphthalene": "rmd17_naphthalene.npz",
        "paracetamol": "rmd17_paracetamol.npz",
        "salicylic_acid": "rmd17_salicylic.npz",
        "toluene": "rmd17_toluene.npz",
        "uracil": "rmd17_uracil.npz",
    }

    download_urls = [
        "https://figshare.com/ndownloader/files/23950376",
        "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1",
    ]

    def __init__(
        self,
        datapath: str,
        molecule: str,
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
            molecule: name of the molecule
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """

        if molecule not in self.datasets_dict.keys():
            raise AtomsDataError(f"Molecule {molecule} is not supported!")

        self.molecule = molecule

        self.download(
            datapath=datapath,
            distance_unit=distance_unit or "Ang",
        )

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

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        return {
            rMD17.energy: "kcal/mol",
            rMD17.forces: "kcal/mol/Ang",
        }

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        """
        Ensure the ASE DB exists and matches the requested molecule.
        """
        if os.path.exists(datapath):
            with connect(datapath, use_lock_file=False) as conn:
                md = conn.metadata

            if "molecule" not in md:
                raise AtomsDataError(
                    "Not a valid rMD17 dataset. Metadata must contain `molecule`."
                )

            if md["molecule"] != self.molecule:
                raise AtomsDataError(
                    f"The dataset at the given location contains `{md['molecule']}` "
                    f"instead of `{self.molecule}`."
                )
            return

        tmpdir = tempfile.mkdtemp("rmd17")
        dataset = self.create(
            datapath=datapath,
            distance_unit=distance_unit,
            property_unit_dict=self._native_property_units(),
            atomrefs=self.atomrefs,
        )
        dataset.update_metadata(molecule=self.molecule)
        self._download_data(tmpdir, dataset)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_data(self, tmpdir: str, dataset: ASEAtomsData) -> None:
        logging.info("Downloading %s data...", self.molecule)

        raw_path = os.path.join(tmpdir, "rmd17")
        tar_path = os.path.join(tmpdir, "rmd17.tar")

        self._download_archive(tar_path)
        logging.info("Done.")

        logging.info("Extracting data...")
        os.makedirs(raw_path, exist_ok=True)

        with tarfile.open(tar_path, mode="r:*") as tar:
            tar.extract(
                path=raw_path,
                member=f"rmd17/npz_data/{self.datasets_dict[self.molecule]}",
            )

            logging.info("Parsing molecule %s", self.molecule)

            data = np.load(
                os.path.join(
                    raw_path,
                    "rmd17",
                    "npz_data",
                    self.datasets_dict[self.molecule],
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
                        os.path.join(
                            raw_path, "rmd17", "splits", f"index_train_0{i}.csv"
                        )
                    )
                    .flatten()
                    .astype(int)
                    .tolist()
                )
                train_splits.append(train_split)

                test_split = (
                    np.loadtxt(
                        os.path.join(
                            raw_path, "rmd17", "splits", f"index_test_0{i}.csv"
                        )
                    )
                    .flatten()
                    .astype(int)
                    .tolist()
                )
                test_splits.append(test_split)

        dataset.update_metadata(splits={"known": train_splits, "test": test_splits})
        logging.info("Done.")

    def _download_archive(self, destination: str) -> None:
        last_error = None

        for url in self.download_urls:
            try:
                logging.info("Downloading from: %s", url)
                req = Request(url)
                with urlopen(req, timeout=600) as resp, open(destination, "wb") as f:
                    shutil.copyfileobj(resp, f)

                if not os.path.exists(destination):
                    raise RuntimeError("Download did not create a file.")

                size = os.path.getsize(destination)
                ctype = (resp.headers.get("Content-Type") or "").lower()

                if size == 0:
                    raise RuntimeError("Downloaded file is empty.")

                if "text/html" in ctype:
                    raise RuntimeError(
                        f"Got HTML instead of archive (Content-Type={ctype})."
                    )

                return

            except (HTTPError, URLError, RuntimeError) as e:
                last_error = e
                logging.warning("Download failed from %s: %s", url, e)

        raise AtomsDataError(
            f"rMD17 download failed from all sources. Last error: {last_error}"
        )
