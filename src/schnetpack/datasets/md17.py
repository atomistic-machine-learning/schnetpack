import logging
import os
import shutil
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError

import numpy as np
from ase import Atoms

from schnetpack.data import AtomsData, AtomsDataError
from schnetpack.environment import SimpleEnvironmentProvider

__all__ = ['MD17']


class MD17(AtomsData):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    Args:
        path (str): path to database
        dataset (str): Name of molecule to load into database. Allowed are:
                            aspirin
                            benzene
                            ethanol
                            malonaldehyde
                            naphthalene
                            salicylic_acid
                            toluene
                            uracil
        subset (list): indices of subset. Set to None for entire dataset
            (default: None)
        download (bool): set true if dataset should be downloaded
            (default: True)
        calculate_triples (bool): set true if triples for angular functions
            should be computed (default: False)
        parse_all (bool): set true to generate the ase dbs of all molecules in
            the beginning (default: False)

    See: http://quantum-machine.org/datasets/
    """

    energy = 'energy'
    forces = 'forces'
    properties = [energy, forces]

    datasets_dict = dict(aspirin='aspirin_dft.npz',
                         aspirin_ccsd='aspirin_ccsd.zip',
                         azobenzene='azobenzene_dft.npz',
                         benzene='benzene_dft.npz',
                         ethanol='ethanol_dft.npz',
                         ethanol_ccsdt='ethanol_ccsd_t.zip',
                         malonaldehyde='malonaldehyde_dft.npz',
                         malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
                         naphthalene='naphthalene_dft.npz',
                         paracetamol='paracetamol_dft.npz',
                         salicylic_acid='salicylic_dft.npz',
                         toluene='toluene_dft.npz',
                         toluene_ccsdt='toluene_ccsd_t.zip',
                         uracil='uracil_dft.npz'
                         )

    existing_datasets = datasets_dict.keys()

    def __init__(self, dbpath, dataset, subset=None, download=True,
                 collect_triples=False, properties=None):
        self.collect_triples = collect_triples

        environment_provider = SimpleEnvironmentProvider()

        if properties is None:
            properties = MD17.properties

        super(MD17, self).__init__(dbpath, subset, properties,
                                   environment_provider,
                                   collect_triples)

        if download:
            self.download(dataset)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]
        return MD17(self.dbdir, self.dataset, subset=subidx, download=False,
                    collect_triples=self.collect_triples)

    def download(self, dataset):
        """
        download data if not already on disk.
        """
        if not os.path.exists(self.dbpath):
            self._load_data(dataset)
        else:
            raise AtomsDataError(
                "Dataset exists already at path " + self.dbpath)

    def _load_data(self, dataset):

        logging.info("Downloading {} data".format(dataset))
        tmpdir = tempfile.mkdtemp("MD")
        rawpath = os.path.join(tmpdir, self.datasets_dict[dataset])
        url = "http://www.quantum-machine.org/gdml/data/npz/" + \
              self.datasets_dict[dataset]

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(dataset))

        data = np.load(rawpath)

        numbers = data['z']
        atoms_list = []
        properties_list = []
        for positions, energies, forces in zip(data['R'], data['E'],
                                               data['F']):
            properties_list.append(dict(energy=energies, forces=forces))
            atoms_list.append(Atoms(positions=positions, numbers=numbers))

        self.add_systems(atoms_list, properties_list)

        logging.info("Cleanining up the mess...")
        logging.info('{} molecule done'.format(dataset))
        shutil.rmtree(tmpdir)

        return True
