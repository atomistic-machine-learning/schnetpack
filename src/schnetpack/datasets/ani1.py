import logging
import os
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError

import h5py
import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import Hartree

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider


class ANI1(AtomsData):
    """ ANI1 dataset.

        This class adds convenience functions to download ANI1 from figshare and load the data into pytorch.

        Args:
            path (str): path to directory containing ANI1 database.
            download (bool): enable downloading if database does not exists (default: True)
            subset (list): indices of subset. Set to None for entire dataset (default: None)
            properties (list): properties in ani, e.g. energy (default: ['energy'])
            calculate_triples (bool): enables triples for Behler type networks (default: False)
            num_heavy_atoms (int): number of heavy atoms (see Table 1 in Ref. [#ani1]_) (default: 1)
            high_energies (bool): add high energy conformations (see 'Technical Validation' of Ref. [#ani1]_) (default: False)

        References:
            .. [#ani1] https://arxiv.org/abs/1708.04987

    """

    # properties:
    energy = 'energy'

    properties = [energy]

    reference = {
        energy: 0
    }

    units = dict(zip(properties, [Hartree]))

    self_energies = {'H': -0.500607632585, 'C': -37.8302333826, 'N': -54.5680045287, 'O': -75.0362229210}

    def __init__(self, path, download=True, subset=None, properties=['energy'], collect_triples=False,
                 num_heavy_atoms=8, high_energies=False):
        self.path = path
        self.atomref_path = os.path.join(self.path, "atomrefs.npz")
        self.dbpath = os.path.join(self.path, 'ani1.db')

        self.num_heavy_atoms = num_heavy_atoms
        self.high_energies = high_energies

        environment_provider = SimpleEnvironmentProvider()

        if download:
            self._download()

        super().__init__(self.dbpath, subset, properties, environment_provider, collect_triples)

    def _download(self):
        works = True
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.dbpath):
            works = works and self._load_data()

        if not os.path.exists(self.atomref_path):
            works = works and self._create_atoms_ref()

        return works

    def _load_data(self):
        logging.info('downloading ANI-1 data...')
        tmpdir = tempfile.mkdtemp('ani1')
        tar_path = os.path.join(tmpdir, 'ANI1_release.tar.gz')
        raw_path = os.path.join(tmpdir, 'data')
        url = 'https://ndownloader.figshare.com/files/9057631'

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

        logging.info('parse files...')
        for i in range(1, self.num_heavy_atoms + 1):
            file_name = os.path.join(raw_path, "ANI-1_release", "ani_gdb_s0%d.h5" % i)
            logging.info("start to parse %s" % file_name)
            self._load_h5_file(file_name)

        logging.info('done...')

        shutil.rmtree(tmpdir)

        return True

    def _load_h5_file(self, file_name):
        with connect(self.dbpath) as con:
            store = h5py.File(file_name)
            for file_key in store:
                for molecule_key in store[file_key]:
                    molecule_group = store[file_key][molecule_key]
                    species = "".join([str(s)[-2] for s in molecule_group['species']])
                    positions = molecule_group['coordinates']
                    energies = molecule_group['energies']

                    # loop over conformations
                    for i in range(energies.shape[0]):
                        atm = Atoms(species, positions[i])
                        energy = energies[i] * self.units[self.energy]

                        properties = {self.energy: energy}

                        con.write(atm, data=properties)

                    # high energy conformations as described in 'Technical Validation'
                    # section of https://arxiv.org/abs/1708.04987
                    if self.high_energies:
                        high_energy_positions = molecule_group['coordinatesHE']
                        high_energies = molecule_group['energiesHE']

                        # loop over high energy conformations
                        for i in range(high_energies.shape[0]):
                            atm = Atoms(species, high_energy_positions[i])
                            high_energy = high_energies[i] * self.units[self.high_energy]

                            properties = {self.energy: high_energy}

                            con.write(atm, data=properties)

    def _create_atoms_ref(self):
        file_name = os.path.join(self.path, "atomrefs.npz")

        atref = np.zeros((100, 6))
        labels = self.properties

        # converts units to eV (which are set to one in ase)
        atref[1, :] = self.self_energies['H'] * self.units['energy']
        atref[6, :] = self.self_energies['C'] * self.units['energy']
        atref[7, :] = self.self_energies['N'] * self.units['energy']
        atref[8, :] = self.self_energies['O'] * self.units['energy']

        np.savez(file_name, atom_ref=atref, labels=labels)

        return True

    def get_reference(self, property):
        """
        Returns atomref for property

        Args:
            property: property in the Ani1 dataset

        Returns:
            list: list with atomrefs
        """
        col = ANI1.reference[property]
        atomref = np.load(self.atomref_path)['atom_ref'][:, col:col + 1]
        return atomref
