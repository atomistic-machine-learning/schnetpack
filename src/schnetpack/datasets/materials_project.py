import logging
import os
from typing import List, Optional, Dict

import torch
import numpy as np
from ase import Atoms

from schnetpack.data import AtomsDataFormat
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["MaterialsProject"]


class MaterialsProject(ASEAtomsData):
    """
    Materials Project (MP) database of bulk crystals.

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
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        apikey: Optional[str] = None,
        **kwargs,
    ):
        if apikey is not None and len(apikey) == 16:
            raise DeprecationWarning(
                "You are using a legacy API key. This API is deprecated and no longer "
                "supported by Materials Project. Please use the next-gen API instead. "
                "Visit https://next-gen.materialsproject.org/ to get a valid API key."
            )

        if apikey is not None and len(apikey) != 32:
            raise AtomsDataError(
                "Invalid API key. MaterialsProject requires an API key of 32 characters. "
                f"Your API key contains {len(apikey)} characters. "
                "Visit https://next-gen.materialsproject.org/ to get a valid API key."
            )

        self.apikey = apikey
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
            MaterialsProject.EformationPerAtom: "eV",
            MaterialsProject.EPerAtom: "eV",
            MaterialsProject.BandGap: "eV",
            MaterialsProject.TotalMagnetization: "None",
        }

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        """
        Ensure the Materials Project ASE DB exists.
        """
        if os.path.exists(datapath):
            _ = ASEAtomsData(datapath, self.format, load_structure=False)
            return

        if self.apikey is None:
            raise AtomsDataError(
                "No API key provided. Visit https://next-gen.materialsproject.org/ "
                "to get an API key."
            )

        dataset = ASEAtomsData.create(
            datapath=datapath,
            distance_unit=distance_unit,
            property_unit_dict=self._native_property_units(),
        )

        self._download_data_nextgen(dataset)

    def _download_data_nextgen(self, dataset: ASEAtomsData) -> None:
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
        dataset.add_systems(
            atoms_list=atoms_list,
            property_list=properties_list,
            atoms_metadata_list=atoms_metadata_list,
        )
        logging.info("Done.")
