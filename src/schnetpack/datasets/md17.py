import logging
import os
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError

import numpy as np
from ase.units import kcal, mol

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider
from .extxyz import parse_extxyz


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
    existing_datasets = [
        "aspirin",
        "benzene",
        "ethanol",
        "malonaldehyde",
        "naphthalene",
        "salicylic_acid",
        "toluene",
        "uracil"
    ]

    def __init__(self, datapath, dataset, subset=None, download=True, collect_triples=False, parse_all=False):
        self.load_all = parse_all
        self.datapath = datapath

        if dataset not in self.existing_datasets:
            raise ValueError("Unknown dataset specification {:s}".format(dataset))

        self.dataset = dataset
        self.database = dataset + ".db"
        self.dbpath = os.path.join(self.datapath, self.database)
        self.collect_triples = collect_triples

        environment_provider = SimpleEnvironmentProvider()

        if download:
            self.download()

        properties = ["energy", "forces"]

        super(MD17, self).__init__(self.dbpath, subset, properties, environment_provider, collect_triples)

    E = "energy"
    F = "forces"

    properties = [
        E, F
    ]

    units = dict(
        zip(properties, [kcal / mol, kcal / mol / 1.0])
    )

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
        logging.info("Downloading MD database...")
        tmpdir = tempfile.mkdtemp("MD")
        tarpath = os.path.join(self.datapath, "md17.tar.xz")
        rawpath = os.path.join(tmpdir, "md17")
        url = "http://www.quantum-machine.org/data/md17.tar.xz"

        try:
            request.urlretrieve(url, tarpath)
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, url)
            return False

        logging.info("Extracting files...")

        tar = tarfile.open(tarpath)
        tar.extractall(tmpdir)
        tar.close()

        logging.info('Done!')

        for molecule in MD17.existing_datasets:
            # if requested, convert only the required molecule
            if not self.load_all:
                if molecule != self.dataset:
                    continue
            logging.info("Parsing molecule {:s}".format(molecule))
            moldb = os.path.join(self.datapath, molecule + ".db")
            molraw = os.path.join(rawpath, molecule + '.xyz')

            parse_extxyz(moldb, molraw)

            logging.info("Cleaning temporary directory {:s}".format(molraw))

        logging.info("Cleanining up the mess...")
        logging.info('Data loaded.')

        shutil.rmtree(tmpdir)

        return True
