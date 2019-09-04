import logging
import os
import re
import shutil
import tarfile
import tempfile
from urllib import request as request

import numpy as np
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV

import schnetpack as spk
from schnetpack.datasets import DownloadableAtomsData

__all__ = ["QM9"]


class QM9(DownloadableAtomsData):
    """QM9 benchmark database for organic molecules.

    The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
    from including C, O, N, F. This class adds convenient functions to download QM9 from
    figshare and load the data into pytorch.

    Args:
        dbpath (str): path to directory containing database.
        download (bool, optional): enable downloading if database does not exists.
        subset (list, optional): indices to subset. Set to None for entire database.
        load_only (list, optional): reduced set of properties to be loaded
        collect_triples (bool, optional): Set to True if angular features are needed.
        remove_uncharacterized (bool, optional): remove uncharacterized molecules.
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).

    References:
        .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"

    reference = {zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5}

    def __init__(
        self,
        dbpath,
        download=True,
        subset=None,
        load_only=None,
        collect_triples=False,
        remove_uncharacterized=False,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
        **kwargs
    ):

        self.remove_uncharacterized = remove_uncharacterized

        available_properties = [
            QM9.A,
            QM9.B,
            QM9.C,
            QM9.mu,
            QM9.alpha,
            QM9.homo,
            QM9.lumo,
            QM9.gap,
            QM9.r2,
            QM9.zpve,
            QM9.U0,
            QM9.U,
            QM9.H,
            QM9.G,
            QM9.Cv,
        ]

        units = [
            1.0,
            1.0,
            1.0,
            Debye,
            Bohr ** 3,
            Hartree,
            Hartree,
            Hartree,
            Bohr ** 2,
            Hartree,
            Hartree,
            Hartree,
            Hartree,
            Hartree,
            1.0,
        ]

        super().__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            collect_triples=collect_triples,
            download=download,
            available_properties=available_properties,
            units=units,
            environment_provider=environment_provider,
            **kwargs
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return QM9(
            dbpath=self.dbpath,
            download=False,
            subset=subidx,
            load_only=self.load_only,
            collect_triples=self.collect_triples,
            remove_uncharacterized=False,
            environment_provider=self.environment_provider,
        )

    def _download(self):
        if self.remove_uncharacterized:
            evilmols = self._load_evilmols()
        else:
            evilmols = None

        self._load_data(evilmols)

        atref, labels = self._load_atomrefs()
        self.set_metadata({"atomrefs": atref.tolist(), "atref_labels": labels})

    def _load_atomrefs(self):
        logging.info("Downloading GDB-9 atom references...")
        at_url = "https://ndownloader.figshare.com/files/3195395"
        tmpdir = tempfile.mkdtemp("gdb9")
        tmp_path = os.path.join(tmpdir, "atomrefs.txt")

        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        atref = np.zeros((100, 6))
        labels = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                atref[z, 0] = float(l.split()[1])
                atref[z, 1] = float(l.split()[2]) * Hartree / eV
                atref[z, 2] = float(l.split()[3]) * Hartree / eV
                atref[z, 3] = float(l.split()[4]) * Hartree / eV
                atref[z, 4] = float(l.split()[5]) * Hartree / eV
                atref[z, 5] = float(l.split()[6])
        return atref, labels

    def _load_evilmols(self):
        logging.info("Downloading list of uncharacterized molecules...")
        at_url = "https://ndownloader.figshare.com/files/3195404"
        tmpdir = tempfile.mkdtemp("gdb9")
        tmp_path = os.path.join(tmpdir, "uncharacterized.txt")

        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        evilmols = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                evilmols.append(int(line.split()[0]))
        return np.array(evilmols)

    def _load_data(self, evilmols=None):
        logging.info("Downloading GDB-9 data...")
        tmpdir = tempfile.mkdtemp("gdb9")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        url = "https://ndownloader.figshare.com/files/3195389"

        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
        )

        all_atoms = []
        all_properties = []

        irange = np.arange(len(ordered_files), dtype=np.int)
        if evilmols is not None:
            irange = np.setdiff1d(irange, evilmols - 1)

        for i in irange:
            xyzfile = os.path.join(raw_path, ordered_files[i])

            if (i + 1) % 10000 == 0:
                logging.info("Parsed: {:6d} / 133885".format(i + 1))
            properties = {}
            tmp = os.path.join(tmpdir, "tmp.xyz")

            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(self.available_properties, l):
                    properties[pn] = np.array([float(p) * self.units[pn]])
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace("*^", "e"))

            with open(tmp, "r") as f:
                ats = list(read_xyz(f, 0))[0]
            all_atoms.append(ats)
            all_properties.append(properties)

        logging.info("Write atoms to db...")
        self.add_systems(all_atoms, all_properties)
        logging.info("Done.")

        shutil.rmtree(tmpdir)

        return True
