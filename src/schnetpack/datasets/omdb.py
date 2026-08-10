import logging
import os
import tarfile
from typing import List, Optional, Dict

import numpy as np
from ase.io import read

from schnetpack.data.atoms import DownloadableASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["OrganicMaterialsDatabase"]


class OrganicMaterialsDatabase(DownloadableASEAtomsData):
    """
    Organic Materials Database (OMDB) of bulk organic crystals.
    Registration to the OMDB is free for academic users. This database contains DFT
    (PBE) band gap (OMDB-GAP1 database) for 12500 non-magnetic materials.

    The dataset is described in [#omdb]_.

    References:

    .. [#omdb] Bart Olsthoorn, R. Matthias Geilhufe, Stanislav S. Borysov, Alexander V. Balatsky.
       Band gap prediction for large organic crystal structures with machine learning.
       https://arxiv.org/abs/1810.12814
    """

    BandGap = "band_gap"

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        raw_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching.
            train_transforms: overrides transform_fn for training.
            val_transforms: overrides transform_fn for validation.
            test_transforms: overrides transform_fn for testing.
            subset_idx: indices of the subset to load.
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...).
            raw_path: path to raw tar.gz file with the data
        """
        self.raw_path = raw_path
        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            load_structure=True,
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
        return {OrganicMaterialsDatabase.BandGap: "eV"}

    def download(self) -> None:
        """
        Convert the OMDB raw archive into an ASE DB.
        """
        if self.raw_path is None or not os.path.exists(self.raw_path):
            raise AtomsDataError(
                "The path to the raw dataset is not provided or invalid and the db-file does "
                "not exist!"
            )
        self._convert()

    def _convert(self) -> None:
        """
        Converts .tar.gz to a .db file
        """
        logging.info("Converting %s to a .db file..", self.raw_path)

        extract_dir = os.path.dirname(self.raw_path) or "."
        with tarfile.open(self.raw_path, "r:gz") as tar:
            names = tar.getnames()
            tar.extractall(path=extract_dir)

        structures_path = os.path.join(extract_dir, "structures.xyz")
        bandgaps_path = os.path.join(extract_dir, "bandgaps.csv")

        structures = read(structures_path, index=":")
        y = np.loadtxt(bandgaps_path)

        atoms_list = []
        property_list = []
        for i, at in enumerate(structures):
            atoms_list.append(at)
            property_list.append(
                {OrganicMaterialsDatabase.BandGap: np.array([y[i]], dtype=np.float64)}
            )

        self.add_systems(atoms_list=atoms_list, property_list=property_list)

        for name in names:
            path = os.path.join(extract_dir, name)
            if os.path.exists(path):
                os.remove(path)
