import logging
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
from urllib import request as request

import torch
import h5py
import numpy as np
from ase import Atoms

from schnetpack.data import AtomsDataFormat
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError, load_dataset
from schnetpack.transform.base import Transform

__all__ = ["ANI1"]

log = logging.getLogger(__name__)


class ANI1(ASEAtomsData):
    """
    ANI1 benchmark dataset.

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
            num_heavy_atoms: number of heavy atoms
            high_energies: whether to include high-energy conformations
            format: dataset format
            load_properties: subset of properties to load
            transforms: Transform applied to each system separately before batching
            subset_idx: indices of the subset to load
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            **kwargs: additional keyword arguments.
        """
        self.num_heavy_atoms = num_heavy_atoms
        self.high_energies = high_energies
        self.format = format

        self.download(
            datapath=datapath,
            distance_unit=distance_unit or "Ang",
        )

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
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

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        """
        Ensure the ANI1 ASE DB exists.
        """
        if os.path.exists(datapath):
            _ = ASEAtomsData(datapath, load_structure=False)
            return

        tmpdir = tempfile.mkdtemp("ani1")
        try:
            dataset = ASEAtomsData.create(
                datapath=datapath,
                distance_unit=distance_unit,
                property_unit_dict=self._native_property_units(),
                atomrefs=self._create_atomrefs(),
            )
            self._download_data(tmpdir, dataset)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_data(self, tmpdir: str, dataset: ASEAtomsData) -> None:
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
            self._load_h5_file(file_name, dataset)

        logging.info("Done.")

    def _load_h5_file(self, file_name: str, dataset: ASEAtomsData) -> None:
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

        dataset.add_systems(atoms_list=atoms_list, property_list=properties_list)

    def _create_atomrefs(self) -> Dict[str, List[float]]:
        atref = np.zeros((100,))
        atref[1] = self.self_energies["H"]
        atref[6] = self.self_energies["C"]
        atref[7] = self.self_energies["N"]
        atref[8] = self.self_energies["O"]
        return {ANI1.energy: atref.tolist()}
