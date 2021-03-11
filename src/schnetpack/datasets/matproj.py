import os

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import eV

import schnetpack as spk
from schnetpack.data import AtomsDataError, AtomsDataSubset
from schnetpack.datasets import DownloadableAtomsData

__all__ = ["MaterialsProject"]


class MaterialsProject(DownloadableAtomsData):
    """Materials Project (MP) database of bulk crystals.

    This class adds convenient functions to download Materials Project data into
    pytorch.

    Args:
        dbpath (str): path to directory containing database.
        cutoff (float): cutoff for bulk interactions.
        apikey (str, optional): materials project key needed to download the data.
        download (bool, optional): enable downloading if database does not exists.
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        load_only (list, optional): reduced set of properties to be loaded
        collect_triples (bool, optional): Set to True if angular features are needed.
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).

    """

    # properties
    EformationPerAtom = "formation_energy_per_atom"
    EPerAtom = "energy_per_atom"
    BandGap = "band_gap"
    TotalMagnetization = "total_magnetization"

    def __init__(
        self,
        dbpath,
        apikey=None,
        download=True,
        subset=None,
        load_only=None,
        collect_triples=False,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
    ):

        available_properties = [
            MaterialsProject.EformationPerAtom,
            MaterialsProject.EPerAtom,
            MaterialsProject.BandGap,
            MaterialsProject.TotalMagnetization,
        ]

        units = [eV, eV, eV, 1.0]

        self.apikey = apikey

        super(MaterialsProject, self).__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            environment_provider=environment_provider,
            collect_triples=collect_triples,
            available_properties=available_properties,
            units=units,
            download=download,
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return MaterialsProject(
            dbpath=self.dbpath,
            download=False,
            subset=subidx,
            load_only=self.load_only,
            collect_triples=self.collect_triples,
            environment_provider=self.environment_provider,
        )

    def at_timestamp(self, timestamp):
        """
        Returns a new dataset that only consists of items created before
        the given timestamp.

        Args:
            timestamp (str): timestamp

        Returns:
            schnetpack.datasets.matproj.MaterialsProject: dataset with subset of
                original data
        """
        with connect(self.dbpath) as conn:
            rows = conn.select(columns=["id", "key_value_pairs"])
            idxs = []
            timestamps = []
            for row in rows:
                idxs.append(row.id - 1)
                timestamps.append(row.key_value_pairs["created_at"])
        idxs = np.array(idxs)
        timestamps = np.array(timestamps)
        return AtomsDataSubset(self, idxs[timestamps <= timestamp])

    def _download(self):
        """
        Downloads dataset provided it does not exist in self.path

        Returns:
            works (bool): true if download succeeded or file already exists
        """
        if self.apikey is None:
            raise AtomsDataError(
                "Provide a valid API key in order to download the "
                "Materials Project data!"
            )
        try:
            from pymatgen.ext.matproj import MPRester
            from pymatgen.core import Structure
            import pymatgen as pmg
        except:
            raise ImportError(
                "In order to download Materials Project data, you have to install "
                "pymatgen"
            )

        # collect data
        atms_list = []
        properties_list = []
        key_value_pairs_list = []
        with MPRester(self.apikey) as m:
            for N in range(1, 9):
                for nsites in range(0, 300, 30):
                    ns = {"$lt": nsites + 31, "$gt": nsites}
                    query = m.query(
                        criteria={
                            "nelements": N,
                            "is_compatible": True,
                            "nsites": ns,
                        },
                        properties=[
                            "structure",
                            "energy_per_atom",
                            "formation_energy_per_atom",
                            "total_magnetization",
                            "band_gap",
                            "material_id",
                            "warnings",
                            "created_at",
                        ],
                    )
                    for k, q in enumerate(query):
                        s = q["structure"]
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
                                    MaterialsProject.EPerAtom: q["energy_per_atom"],
                                    MaterialsProject.EformationPerAtom: q[
                                        "formation_energy_per_atom"
                                    ],
                                    MaterialsProject.TotalMagnetization: q[
                                        "total_magnetization"
                                    ],
                                    MaterialsProject.BandGap: q["band_gap"],
                                }
                            )
                            key_value_pairs_list.append(
                                {
                                    "material_id": q["material_id"],
                                    "created_at": q["created_at"],
                                }
                            )

        # write systems to database
        self.add_systems(
            atms_list,
            property_list=properties_list,
            key_value_pairs_list=key_value_pairs_list,
        )

        self.set_metadata({})
