import logging
import os
from typing import List, Optional, Dict
import warnings

from ase import Atoms

import torch
import numpy as np
from schnetpack.data import *
from schnetpack.data import AtomsDataModuleError, AtomsDataModule


__all__ = ["MaterialsProject"]


class MaterialsProject(AtomsDataModule):
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
    MaterialId = ("material_id",)
    CreatedAt = "created_at"

    def __init__(
        self,
        datapath: str,
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
        apikey: Optional[str] = None,
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
            apikey: Materials project key needed to download the data.
        """
        if apikey is not None and len(apikey) == 16:
            raise DeprecationWarning(
                "You are using a legacy API key. This API is deprecated and no longer supported by MaterialsProject. "
                "Please use the nextgen API instead. "
                "Visit https://next-gen.materialsproject.org/ to get a valid API-key. "
            )
        if apikey is not None and len(apikey) != 32:
            raise AtomsDataModuleError(
                "Invalid API-key. MaterialsProject requires an API-key of 32 characters. "
                f"Your API-key contains {len(apikey)} characters. "
                "Visit https://next-gen.materialsproject.org/ to get a valid API-key. "
            )

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
            **kwargs,
        )
        self.apikey = apikey

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            # check if apikey is provided
            if self.apikey is None:
                raise AtomsDataModuleError(
                    "No API-key provided, visit https://next-gen.materialsproject.org/ to get an API-key."
                )

            # initialize dataset
            property_unit_dict = {
                MaterialsProject.EformationPerAtom: "eV",
                MaterialsProject.EPerAtom: "eV",
                MaterialsProject.BandGap: "eV",
                MaterialsProject.TotalMagnetization: "None",
            }
            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )

            self._download_data_nextgen(dataset)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data_nextgen(self, dataset: BaseAtomsData):
        """
        Downloads dataset provided it does not exist in self.path
        Returns:
            works (bool): true if download succeeded or file already exists
        """
        # collect data
        atms_list = []
        properties_list = []
        atoms_metadata_list = []
        try:
            from pymatgen.core import Structure
            import pymatgen as pmg
            from mp_api.client import MPRester

        except:
            raise ImportError(
                "In order to download Materials Project data, you have to install "
                "mp-api and pymatgen packages"
            )

        with MPRester(self.apikey) as m:
            query = m.materials.summary.search(
                num_sites=(0, 300, 30),
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
                if type(s) is Structure:
                    atms_list.append(
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
                            "material_id": q.material_id,
                        }
                    )

        # write systems to database
        logging.info("Write atoms to db...")
        dataset.add_systems(
            atoms_list=atms_list,
            property_list=properties_list,
            atoms_metadata_list=atoms_metadata_list,
        )
        logging.info("Done.")
