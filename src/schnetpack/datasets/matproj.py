import os

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import eV

import schnetpack as spk
from schnetpack.data import AtomsDataError
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
        subset (list, optional): indices to subset. Set to None for entire database.
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

        with connect(self.dbpath) as con:
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
                            ],
                        )

                        for k, q in enumerate(query):
                            s = q["structure"]
                            if type(s) is Structure:
                                at = Atoms(
                                    numbers=s.atomic_numbers,
                                    positions=s.cart_coords,
                                    cell=s.lattice.matrix,
                                    pbc=True,
                                )
                                con.write(
                                    at,
                                    data={
                                        MaterialsProject.EPerAtom: q["energy_per_atom"],
                                        MaterialsProject.EformationPerAtom: q[
                                            "formation_energy_per_atom"
                                        ],
                                        MaterialsProject.TotalMagnetization: q[
                                            "total_magnetization"
                                        ],
                                        MaterialsProject.BandGap: q["band_gap"],
                                    },
                                )
        self.set_metadata({})
