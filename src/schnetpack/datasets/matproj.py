import os

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import eV

from schnetpack.data import DownloadableAtomsData
from schnetpack.environment import AseEnvironmentProvider

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
        properties (list, optional): properties in mp, e.g. formation_energy_per_atom.
        collect_triples (bool, optional): Set to True if angular features are needed.

    """

    # properties
    EformationPerAtom = "formation_energy_per_atom"
    EPerAtom = "energy_per_atom"
    BandGap = "band_gap"
    TotalMagnetization = "total_magnetization"

    def __init__(
        self,
        dbpath,
        cutoff,
        apikey=None,
        download=True,
        subset=None,
        properties=None,
        collect_triples=False,
    ):

        available_properties = [
            MaterialsProject.EformationPerAtom,
            MaterialsProject.EPerAtom,
            MaterialsProject.BandGap,
            MaterialsProject.TotalMagnetization,
        ]

        units = [eV, eV, eV, 1.0]

        self.cutoff = cutoff
        self.apikey = apikey

        environment_provider = AseEnvironmentProvider(cutoff)

        super(MaterialsProject, self).__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=properties,
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
            cutoff=self.cutoff,
            download=False,
            subset=subidx,
            properties=self.load_only,
            collect_triples=self.collect_triples,
        )

    def _download(self):
        """
        Downloads dataset provided it does not exist in self.path

        Returns:
            works (bool): true if download succeeded or file already exists
        """
        try:
            from pymatgen.ext.matproj import MPRester
            from pymatgen.core import Structure
            import pymatgen as pmg
        except:
            raise ImportError(
                "In order to download Materials Project data, you have to install pymatgen"
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
