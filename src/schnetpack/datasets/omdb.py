import os
import tarfile

import numpy as np
from ase.io import read
from ase.db import connect
from ase.units import eV

from schnetpack.data import DownloadableAtomsData
from schnetpack.environment import AseEnvironmentProvider

__all__ = ["OrganicMaterialsDatabase"]


class OrganicMaterialsDatabase(DownloadableAtomsData):
    """Organic Materials Database (OMDB) of bulk organic crystals.

    Registration to the OMDB is free for academic users. This database contains DFT
    (PBE) band gap (OMDB-GAP1 database) for 12500 non-magnetic materials.

    Args:
        path (str): path to directory containing database.
        cutoff (float): cutoff for bulk interactions.
        download (bool, optional): enable downloading if database does not exists.
        subset (list): indices to subset. Set to None for entire database.
        properties (list, optional): properties in omdb, e.g. band_gap.
        collect_triples (bool, optional): Set to True if angular features are needed.

    References:
        arXiv: https://arxiv.org/abs/1810.12814 "Band gap prediction for large organic
        crystal structures with machine learning" Bart Olsthoorn, R. Matthias Geilhufe,
        Stanislav S. Borysov, Alexander V. Balatsky (Submitted on 30 Oct 2018)

    """

    BandGap = "band_gap"

    properties = [BandGap]

    units = dict(zip(properties, [eV]))

    def __init__(
        self,
        path,
        cutoff,
        download=True,
        subset=None,
        properties=[],
        collect_triples=False,
    ):
        self.path = path
        self.cutoff = cutoff

        self.dbpath = self.path.replace(".tar.gz", ".db")

        if not os.path.exists(self.path) and not os.path.exists(self.dbpath):
            raise FileNotFoundError(
                "Download OMDB dataset (e.g. OMDB-GAP1.tar.gz) from https://omdb.diracmaterials.org/dataset/ and set datapath to this file"
            )

        environment_provider = AseEnvironmentProvider(cutoff)

        if download and not os.path.exists(self.dbpath):
            # Convert OMDB .tar.gz into a .db file
            self._convert()

        super(OrganicMaterialsDatabase, self).__init__(
            self.dbpath, subset, properties, environment_provider, collect_triples
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return OrganicMaterialsDatabase(
            self.path,
            self.cutoff,
            download=False,
            subset=subidx,
            properties=self.properties,
            collect_triples=self.collect_triples,
        )

    def _convert(self):
        """
        Converts .tar.gz to a .db file
        """
        print("Converting %s to a .db file.." % self.path)
        tar = tarfile.open(self.path, "r:gz")
        names = tar.getnames()
        tar.extractall()
        tar.close()

        structures = read("structures.xyz", index=":")
        Y = np.loadtxt("bandgaps.csv")
        [os.remove(name) for name in names]

        with connect(self.dbpath) as con:
            for i, at in enumerate(structures):
                con.write(at, data={OrganicMaterialsDatabase.BandGap: Y[i]})
