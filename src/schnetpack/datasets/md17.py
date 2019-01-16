import logging
import os
import shutil
import tempfile
from urllib import request as request

import numpy as np
from ase import Atoms

from schnetpack.data import AtomsData

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
    available_properties = [energy, forces]

    datasets_dict = dict(aspirin='aspirin_dft.npz',
                         #aspirin_ccsd='aspirin_ccsd.zip',
                         azobenzene='azobenzene_dft.npz',
                         benzene='benzene_dft.npz',
                         ethanol='ethanol_dft.npz',
                         #ethanol_ccsdt='ethanol_ccsd_t.zip',
                         malonaldehyde='malonaldehyde_dft.npz',
                         #malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
                         naphthalene='naphthalene_dft.npz',
                         paracetamol='paracetamol_dft.npz',
                         salicylic_acid='salicylic_dft.npz',
                         toluene='toluene_dft.npz',
                         #toluene_ccsdt='toluene_ccsd_t.zip',
                         uracil='uracil_dft.npz'
                         )

    existing_datasets = datasets_dict.keys()

    def __init__(self, datapath, molecule=None, subset=None, download=True,
                 collect_triples=False, properties=None):
        self.datapath = datapath
        self.molecule = molecule
        dbpath = os.path.join(datapath, 'md17', molecule + '.db')

        super(MD17, self).__init__(dbpath=dbpath, subset=subset,
                                   required_properties=properties,
                                   collect_triples=collect_triples,
                                   download=download)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]
        return MD17(datapath=self.datapath, molecule=self.molecule,
                    subset=subidx, download=False,
                    collect_triples=self.collect_triples,
                    properties=self.required_properties)

    def _download(self):

        logging.info("Downloading {} data".format(self.molecule))
        tmpdir = tempfile.mkdtemp("MD")
        rawpath = os.path.join(tmpdir, self.datasets_dict[self.molecule])
        url = "http://www.quantum-machine.org/gdml/data/npz/" + \
              self.datasets_dict[self.molecule]

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(self.molecule))

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
        logging.info('{} molecule done'.format(self.molecule))
        shutil.rmtree(tmpdir)
