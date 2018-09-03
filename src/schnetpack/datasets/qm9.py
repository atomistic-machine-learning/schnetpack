import logging
import os
import re
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError

import numpy as np
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider


class QM9(AtomsData):
    """ QM9 benchmark dataset for organic molecules with up to nine heavy atoms from {C, O, N, F}.

        This class adds convenience functions to download QM9 from figshare and load the data into pytorch.

        Args:
            path (str): path to directory containing qm9 database.
            download (bool): enable downloading if database does not exists (default: True)
            subset (list): indices of subset. Set to None for entire dataset (default: None)
            properties (list): properties in qm9, e.g. U0
            pair_provider (BaseEnvironmentProvider):
            remove_uncharacterized (bool): remove uncharacterized molecules from dataset (according to [#qm9_1]_)

        References:
            .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = 'rotational_constant_A'
    B = 'rotational_constant_B'
    C = 'rotational_constant_C'
    mu = 'dipole_moment'
    alpha = 'isotropic_polarizability'
    homo = 'homo'
    lumo = 'lumo'
    gap = 'gap'
    r2 = 'electronic_spatial_extent'
    zpve = 'zpve'
    U0 = 'energy_U0'
    U = 'energy_U'
    H = 'enthalpy_H'
    G = 'free_energy'
    Cv = 'heat_capacity'

    properties = [
        A, B, C, mu, alpha,
        homo, lumo, gap, r2, zpve,
        U0, U, H, G, Cv
    ]

    reference = {
        zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5
    }

    units = dict(
        zip(properties,
            [
                1., 1., 1., Debye, Bohr ** 3,
                Hartree, Hartree, Hartree,
                                   Bohr ** 2, Hartree,
                Hartree, Hartree, Hartree,
                Hartree, 1.
            ]
            )
    )

    def __init__(self, path, download=True, subset=None, properties=[], collect_triples=False,
                 remove_uncharacterized=False):
        self.path = path
        self.dbpath = os.path.join(self.path, 'qm9.db')
        self.atomref_path = os.path.join(self.path, 'atomref.npz')
        self.evilmols_path = os.path.join(self.path, 'evilmols.npy')

        environment_provider = SimpleEnvironmentProvider()

        if download:
            self._download()

        if remove_uncharacterized:
            if subset is None:
                with connect(self.dbpath) as con:
                    subset = np.arange(con.count())
            else:
                subset = np.array(subset)
            evilmols = np.load(self.evilmols_path)

            # attention:  1-indexing vs 0-indexing
            subset = np.setdiff1d(subset, evilmols - 1)

        super().__init__(self.dbpath, subset, properties, environment_provider, collect_triples)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return QM9(self.path, False, subidx, self.properties, self.collect_triples, False)

    def _download(self):
        works = True
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.atomref_path):
            works = works and self._load_atomrefs()
        if not os.path.exists(self.dbpath):
            works = works and self._load_data()
        if not os.path.exists(self.evilmols_path):
            works = works and self._load_evilmols()
        return works

    def get_reference(self, property):
        """
        Returns atomref for property.

        Args:
            property: property in the qm9 dataset

        Returns:
            list: list with atomrefs
        """
        if property not in QM9.reference:
            atomref = None
        else:
            col = QM9.reference[property]
            atomref = np.load(self.atomref_path)['atom_ref'][:, col:col + 1]
        return atomref

    def _load_atomrefs(self):
        logging.info('Downloading GDB-9 atom references...')
        at_url = 'https://ndownloader.figshare.com/files/3195395'
        tmpdir = tempfile.mkdtemp('gdb9')
        tmp_path = os.path.join(tmpdir, 'atomrefs.txt')

        try:
            request.urlretrieve(at_url, tmp_path)
            logging.info("Done.")
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, at_url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, at_url)
            return False

        atref = np.zeros((100, 6))
        labels = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                atref[z, 0] = float(l.split()[1])
                atref[z, 1] = float(l.split()[2]) * Hartree / eV
                atref[z, 2] = float(l.split()[3]) * Hartree / eV
                atref[z, 3] = float(l.split()[4]) * Hartree / eV
                atref[z, 4] = float(l.split()[5]) * Hartree / eV
                atref[z, 5] = float(l.split()[6])
        np.savez(self.atomref_path, atom_ref=atref, labels=labels)
        return True

    def _load_evilmols(self):
        logging.info('Downloading list of evil molecules...')
        at_url = 'https://ndownloader.figshare.com/files/3195404'
        tmpdir = tempfile.mkdtemp('gdb9')
        tmp_path = os.path.join(tmpdir, 'uncharacterized.txt')

        try:
            request.urlretrieve(at_url, tmp_path)
            logging.info("Done.")
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, at_url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, at_url)
            return False

        evilmols = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                evilmols.append(int(line.split()[0]))
        np.save(self.evilmols_path, np.array(evilmols))

    def _load_data(self):
        logging.info('Downloading GDB-9 data...')
        tmpdir = tempfile.mkdtemp('gdb9')
        tar_path = os.path.join(tmpdir, 'gdb9.tar.gz')
        raw_path = os.path.join(tmpdir, 'gdb9_xyz')
        url = 'https://ndownloader.figshare.com/files/3195389'

        try:
            request.urlretrieve(url, tar_path)
            logging.info("Done.")
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, url)
            return False

        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()

        logging.info('Parse xyz files...')
        with connect(self.dbpath) as con:
            ordered_files = sorted(os.listdir(raw_path), key=lambda x: (int(re.sub('\D', '', x)), x))
            for i, xyzfile in enumerate(ordered_files):
                xyzfile = os.path.join(raw_path, xyzfile)

                if (i + 1) % 10000 == 0:
                    logging.info('Parsed: {:6d} / 133885'.format(i + 1))
                properties = {}
                tmp = os.path.join(tmpdir, 'tmp.xyz')

                with open(xyzfile, 'r') as f:
                    lines = f.readlines()
                    l = lines[1].split()[2:]
                    for pn, p in zip(self.properties, l):
                        properties[pn] = float(p) * self.units[pn]
                    with open(tmp, "wt") as fout:
                        for line in lines:
                            fout.write(line.replace('*^', 'e'))

                with open(tmp, 'r') as f:
                    ats = list(read_xyz(f, 0))[0]

                con.write(ats, data=properties)
        logging.info('Done.')

        shutil.rmtree(tmpdir)

        return True
