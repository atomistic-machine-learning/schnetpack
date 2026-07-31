import logging
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
from urllib import request as request
from ase.db import connect

import h5py
import numpy as np
from ase import Atoms

from schnetpack.data.atoms import DownloadableASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["ANI1"]

log = logging.getLogger(__name__)


class ANI1(DownloadableASEAtomsData):
    """
    ANI1 benchmark dataset.
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
        num_heavy_atoms: int = 8,
        high_energies: bool = False,
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
            num_heavy_atoms: number of heavy atoms
            high_energies: whether to include high-energy conformations
            load_properties: subset of properties to load
            transforms: Transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...).
        """
        self.num_heavy_atoms = num_heavy_atoms
        self.high_energies = high_energies
        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

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
            ANI1.energy: "Hartree",
        }

    def _check_db(self) -> None:
        """
        Ensure the ANI1 ASE DB exists.
        """
        super()._check_db()
        with connect(self.datapath, use_lock_file=False) as conn:
            md = conn.metadata

        # DBs created with older schnetpack versions do not carry the
        # `num_heavy_atoms` / `high_energies` metadata keys. Only enforce the
        # consistency check when the keys are present, so legacy ani1.db
        # files keep loading; warn instead of rejecting them.
        md_num_heavy_atoms = md.get("num_heavy_atoms")
        if md_num_heavy_atoms is None:
            log.warning(
                "The ANI1 database does not store `num_heavy_atoms` (created "
                "with an older schnetpack version) — cannot verify that it "
                f"matches the requested num_heavy_atoms={self.num_heavy_atoms}."
            )
        elif md_num_heavy_atoms != self.num_heavy_atoms:
            raise AtomsDataError(
                f"Existing ANI1 dataset was created with num_heavy_atoms={md_num_heavy_atoms}, "
                f"but requested num_heavy_atoms={self.num_heavy_atoms}."
            )

        md_high_energies = md.get("high_energies")
        if md_high_energies is None:
            log.warning(
                "The ANI1 database does not store `high_energies` (created "
                "with an older schnetpack version) — cannot verify that it "
                f"matches the requested high_energies={self.high_energies}."
            )
        elif md_high_energies != self.high_energies:
            raise AtomsDataError(
                f"Existing ANI1 dataset was created with high_energies={md_high_energies}, "
                f"but requested high_energies={self.high_energies}."
            )

    def download(self) -> None:
        """
        Download ANI1 data and populate the ASE DB.
        """
        tmpdir = tempfile.mkdtemp("ani1")

        md = self.metadata
        md["atomrefs"] = self._create_atomrefs()
        md["num_heavy_atoms"] = self.num_heavy_atoms
        md["high_energies"] = self.high_energies
        self._set_metadata(md)

        self._download_data(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_data(self, tmpdir: str) -> None:
        logging.info("Downloading ANI-1 data...")
        tar_path = os.path.join(tmpdir, "ANI1_release.tar.gz")
        raw_path = os.path.join(tmpdir, "data")
        url = "https://ndownloader.figshare.com/files/9057631"

        request.urlretrieve(url, tar_path)
        if not os.path.exists(tar_path):
            raise AtomsDataError(f"Download failed, file not found: {tar_path}")

        if os.path.getsize(tar_path) == 0:
            raise AtomsDataError(f"Downloaded file is empty: {tar_path}")

        logging.info("Done.")

        with tarfile.open(tar_path) as tar:
            tar.extractall(raw_path)

        logging.info("Parsing files...")
        for i in range(1, self.num_heavy_atoms + 1):
            file_name = os.path.join(raw_path, "ANI-1_release", f"ani_gdb_s0{i}.h5")
            logging.info("Start to parse %s", file_name)
            self._load_h5_file(file_name)

        logging.info("Done.")

    def _load_h5_file(self, file_name: str) -> None:
        atoms_list = []
        properties_list = []

        with h5py.File(file_name, "r") as store:
            for file_key in store:
                for molecule_key in store[file_key]:
                    molecule_group = store[file_key][molecule_key]
                    species = "".join([str(s)[-2] for s in molecule_group["species"]])
                    positions = molecule_group["coordinates"]
                    energies = molecule_group["energies"]

                    # regular conformations
                    for i in range(energies.shape[0]):
                        atm = Atoms(species, positions[i])
                        energy = energies[i]
                        properties = {self.energy: np.array([energy])}
                        atoms_list.append(atm)
                        properties_list.append(properties)

                    # high-energy conformations
                    # section of https://arxiv.org/abs/1708.04987
                    if self.high_energies:
                        high_energy_positions = molecule_group["coordinatesHE"]
                        high_energies = molecule_group["energiesHE"]

                        for i in range(high_energies.shape[0]):
                            atm = Atoms(species, high_energy_positions[i])
                            high_energy = high_energies[i]
                            properties = {self.energy: np.array([high_energy])}
                            atoms_list.append(atm)
                            properties_list.append(properties)

        self.add_systems(atoms_list=atoms_list, property_list=properties_list)

    def _create_atomrefs(self) -> Dict[str, List[float]]:
        atref = np.zeros((100,))
        atref[1] = self.self_energies["H"]
        atref[6] = self.self_energies["C"]
        atref[7] = self.self_energies["N"]
        atref[8] = self.self_energies["O"]
        return {ANI1.energy: atref.tolist()}
