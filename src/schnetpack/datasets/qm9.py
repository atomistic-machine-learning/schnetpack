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

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider

__all__ = ['QM9']


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

    def __init__(self, dbpath, download=True, subset=None, properties=[],
                 collect_triples=False, remove_uncharacterized=False):
        self.dbpath = dbpath
        self.required_properties = properties
        environment_provider = SimpleEnvironmentProvider()

        if not os.path.exists(dbpath) and download:
            self._download(remove_uncharacterized)

        super().__init__(self.dbpath, subset, self.required_properties,
                         environment_provider,
                         collect_triples)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return QM9(self.dbpath, False, subidx, self.required_properties,
                   self.collect_triples, False)

    def _download(self, remove_uncharacterized):
        if remove_uncharacterized:
            evilmols = self._load_evilmols()
        else:
            evilmols = None

        self._load_data(evilmols)

        atref, labels = self._load_atomrefs()
        self.set_metadata({
            'atomrefs': atref.tolist(), 'atref_labels': labels
        })

    def _load_atomrefs(self):
        logging.info('Downloading GDB-9 atom references...')
        at_url = 'https://ndownloader.figshare.com/files/3195395'
        tmpdir = tempfile.mkdtemp('gdb9')
        tmp_path = os.path.join(tmpdir, 'atomrefs.txt')

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
        logging.info('Downloading list of uncharacterized molecules...')
        at_url = 'https://ndownloader.figshare.com/files/3195404'
        tmpdir = tempfile.mkdtemp('gdb9')
        tmp_path = os.path.join(tmpdir, 'uncharacterized.txt')

        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        evilmols = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                evilmols.append(int(line.split()[0]))
        return np.array(evilmols)

    def _load_data(self, evilmols=None):
        logging.info('Downloading GDB-9 data...')
        tmpdir = tempfile.mkdtemp('gdb9')
        tar_path = os.path.join(tmpdir, 'gdb9.tar.gz')
        raw_path = os.path.join(tmpdir, 'gdb9_xyz')
        url = 'https://ndownloader.figshare.com/files/3195389'

        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info('Parse xyz files...')
        ordered_files = sorted(os.listdir(raw_path),
                               key=lambda x: (int(re.sub('\D', '', x)), x))

        all_atoms = []
        all_properties = []

        irange = np.arange(len(ordered_files), dtype=np.int)
        if evilmols is not None:
            irange = np.setdiff1d(irange, evilmols - 1)

        for i in irange:
            xyzfile = os.path.join(raw_path, ordered_files[i])

            if (i + 1) % 10000 == 0:
                logging.info('Parsed: {:6d} / 133885'.format(i + 1))
            properties = {}
            tmp = os.path.join(tmpdir, 'tmp.xyz')

            with open(xyzfile, 'r') as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(self.properties, l):
                    properties[pn] = np.array([float(p) * self.units[pn]])
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace('*^', 'e'))

            with open(tmp, 'r') as f:
                ats = list(read_xyz(f, 0))[0]
            all_atoms.append(ats)
            all_properties.append(properties)

        logging.info('Write atoms to db...')
        self.add_systems(all_atoms, all_properties)
        logging.info('Done.')

        shutil.rmtree(tmpdir)

        return True
