import logging
import os
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError
from ase import Atoms
import numpy as np
from ase.units import kcal, mol

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider
from schnetpack.datasets.extxyz import parse_extxyz


class MD17(AtomsData):
    """
    MD17 benchmark data set for molecular dynamics of small molecules containing molecular forces.

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
        subset (list): indices of subset. Set to None for entire dataset (default: None)
        download (bool): set true if dataset should be downloaded (default: True)
        calculate_triples (bool): set true if triples for angular functions should be computed (default: False)
        parse_all (bool): set true to generate the ase dbs of all molecules in the beginning (default: False)

    See: http://quantum-machine.org/datasets/
    """

    datasets_dict = dict(aspirin_dft='aspirin_dft.npz',
                         aspirin_ccsd='aspirin_ccsd.zip',
                         azobenzene_dft='azobenzene_dft.npz',
                         benzene='benzene_dft.npz',
                         ethanol_dft='ethanol_dft.npz',
                         ethanol_ccsd_t='ethanol_ccsd_t.zip',
                         malonaldehyde_dft='malonaldehyde_dft.npz',
                         malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
                         naphthalene_dft='naphthalene_dft.npz',
                         paracetamol_dft='paracetamol_dft.npz',
                         salicylic_acid_dft='salicylic_dft.npz',
                         toluene_dft='toluene_dft.npz',
                         toluene_ccsd_t='toluene_ccsd_t.zip',
                         uracil_dft='uracil_dft.npz'
                         )

    def __init__(self, datapath, dataset, subset=None, download=True, collect_triples=False, parse_all=False,
                 properties=None):
        self.load_all = parse_all
        self.datapath = datapath

        if dataset not in self.datasets_dict.keys():
            raise ValueError("Unknown dataset specification {:s}".format(dataset))

        self.dataset = dataset
        self.database = dataset + ".db"
        self.dbpath = os.path.join(self.datapath, self.database)
        self.collect_triples = collect_triples

        environment_provider = SimpleEnvironmentProvider()

        if properties is None:
            properties = ["energy", "forces"]

        super(MD17, self).__init__(self.datapath, self.dbpath, subset, properties, environment_provider,
                                   collect_triples)

        if download:
            self.download()

    #energies = "energy"
    #forces = "forces"

    #properties = [
    #    energies, forces
    #]

    #units = dict(
    #    zip(properties, [kcal / mol, kcal / mol / 1.0])
    #)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]
        return MD17(self.datapath, self.dataset, subset=subidx, download=False, collect_triples=self.collect_triples)

    def download(self):
        """
        download data if not already on disk.
        """
        success = True
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        if not os.path.exists(self.dbpath):
            success = success and self._load_data()

        return success

    def _load_data(self):

        for molecule in self.datasets_dict.keys():
            # if requested, convert only the required molecule
            if not self.load_all:
                if molecule != self.dataset:
                    continue

            logging.info("Downloading {} data".format(molecule))
            tmpdir = tempfile.mkdtemp("MD")
            rawpath = os.path.join(tmpdir, self.datasets_dict[molecule])
            url = "http://www.quantum-machine.org/gdml/data/npz/" + self.datasets_dict[molecule]

            try:
                request.urlretrieve(url, rawpath)
            except HTTPError as e:
                logging.error("HTTP Error:", e.code, url)
                return False
            except URLError as e:
                logging.error("URL Error:", e.reason, url)
                return False

            logging.info("Parsing molecule {:s}".format(molecule))

            data = np.load(rawpath)

            name = data['name']
            numbers = data['z']
            for positions, energies, forces in zip(data['R'], data['E'], data['F']):
                properties = dict(energy=energies,
                                  forces=forces)
                atoms = Atoms(positions=positions,
                              numbers=numbers,
                              symbols=name)
                self.add_atoms(atoms, **properties)

        logging.info("Cleanining up the mess...")
        logging.info('{} molecule done'.format(molecule))
        shutil.rmtree(tmpdir)

        return True


if __name__ == '__main__':
    md17 = MD17('./test', 'aspirin_ccsd')
