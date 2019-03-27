import os

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import eV

from schnetpack.data import DownloadableAtomsData
from schnetpack.environment import AseEnvironmentProvider

__all__ = ["MaterialsProject"]


class MaterialsProject(DownloadableAtomsData):
    """ Materials Project data repository of bulk crystals.

        This class adds convenience functions to download Materials Project data into pytorch.

        Args:
            path (str): path to directory containing mp database.
            cutoff (float): cutoff for bulk interactions
            apikey (str): materials project key needed to download the data (default: None)
            download (bool): enable downloading if database does not exists (default: True)
            subset (list): indices of subset. Set to None for entire dataset (default: None)
            properties (list): properties, e.g. formation_energy_per_atom

    """

    # properties
    EformationPerAtom = "formation_energy_per_atom"
    EPerAtom = "energy_per_atom"
    BandGap = "band_gap"
    TotalMagnetization = "total_magnetization"

    available_properties = [EformationPerAtom, EPerAtom, BandGap, TotalMagnetization]

    units = dict(zip(available_properties, [eV, eV, eV, 1.0]))

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
        self.cutoff = cutoff
        self.apikey = apikey

        self.dbpath = dbpath

        environment_provider = AseEnvironmentProvider(cutoff)

        if properties is None:
            properties = MaterialsProject.available_properties

        if download and not os.path.exists(self.dbpath):
            self._download()

        super(MaterialsProject, self).__init__(
            self.dbpath, subset, properties, environment_provider, collect_triples
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return MaterialsProject(
            self.dbpath,
            self.cutoff,
            download=False,
            subset=subidx,
            properties=self.required_properties,
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
