import logging
import os
from typing import List, Optional, Dict

import numpy as np
from ase import Atoms

from schnetpack.data.atoms import DownloadableASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["MaterialsProject"]


class MaterialsProject(DownloadableASEAtomsData):
    """
    Materials Project (MP) database of bulk crystals.
    This class adds convenient functions to download Materials Project data into
    pytorch.

    References:

        .. [#matproj] https://materialsproject.org/
    """

    # properties
    EformationPerAtom = "formation_energy_per_atom"
    EPerAtom = "energy_per_atom"
    BandGap = "band_gap"
    TotalMagnetization = "total_magnetization"
    MaterialId = "material_id"
    CreatedAt = "created_at"

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
        apikey: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
            apikey: api key use to get data
        """
        # Only validate the API key when a download is actually needed —
        # opening an already-downloaded materials_project.db without a key is
        # a valid workflow.
        if not os.path.exists(datapath):
            self._validate_apikey(apikey)

        self.apikey = apikey
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
    def _validate_apikey(apikey: Optional[str]) -> None:
        if apikey is None:
            raise AtomsDataError(
                "No API key provided. Visit https://next-gen.materialsproject.org/ "
                "to get an API key."
            )

        if len(apikey) == 16:
            raise AtomsDataError(
                "You are using a legacy API key. This API is deprecated and no longer "
                "supported by Materials Project. Please use the next-gen API instead. "
                "Visit https://next-gen.materialsproject.org/ to get a valid API key."
            )

        if len(apikey) != 32:
            raise AtomsDataError(
                "Invalid API key. MaterialsProject requires an API key of 32 characters. "
                f"Your API key contains {len(apikey)} characters. "
                "Visit https://next-gen.materialsproject.org/ to get a valid API key."
            )

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        return {
            MaterialsProject.EformationPerAtom: "eV",
            MaterialsProject.EPerAtom: "eV",
            MaterialsProject.BandGap: "eV",
            MaterialsProject.TotalMagnetization: "None",
        }

    def download(self) -> None:
        """
        Download Materials Project entries and store them in the ASE DB.
        """
        atoms_list = []
        properties_list = []
        atoms_metadata_list = []

        try:
            from pymatgen.core import Structure
            from mp_api.client import MPRester
        except Exception as e:
            raise ImportError(
                "To download Materials Project data, install `mp-api` and `pymatgen`."
            ) from e

        with MPRester(self.apikey) as m:
            query = m.materials.summary.search(
                num_sites=(0, 300),
                num_elements=(1, 9),
                fields=[
                    "structure",
                    "energy_per_atom",
                    "formation_energy_per_atom",
                    "total_magnetization",
                    "band_gap",
                    "material_id",
                    "warnings",
                ],
            )

            for q in query:
                s = q.structure
                if isinstance(s, Structure):
                    atoms_list.append(
                        Atoms(
                            numbers=s.atomic_numbers,
                            positions=s.cart_coords,
                            cell=s.lattice.matrix,
                            pbc=True,
                        )
                    )
                    properties_list.append(
                        {
                            MaterialsProject.EPerAtom: np.array([q.energy_per_atom]),
                            MaterialsProject.EformationPerAtom: np.array(
                                [q.formation_energy_per_atom]
                            ),
                            MaterialsProject.TotalMagnetization: np.array(
                                [q.total_magnetization]
                            ),
                            MaterialsProject.BandGap: np.array([q.band_gap]),
                        }
                    )
                    atoms_metadata_list.append(
                        {
                            MaterialsProject.MaterialId: str(q.material_id),
                        }
                    )

        logging.info("Write atoms to db...")
        self.add_systems(
            atoms_list=atoms_list,
            property_list=properties_list,
            atoms_metadata_list=atoms_metadata_list,
        )
        logging.info("Done.")
