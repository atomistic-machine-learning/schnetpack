import logging
import os
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider


class ISO17(AtomsData):
    """
    ISO17 benchmark data set for molecular dynamics of C7O2H10 isomers containing molecular forces.

    Args:
        path (str): Path to database
        fold (str): Fold of data to load. Allowed are:
                        reference - 80% of steps of 80% of MD trajectories
                        reference_eq - equilibrium conformations of those molecules
                        test_within - remaining 20% unseen steps of reference trajectories
                        test_other - remaining 20% unseen MD trajectories
                        test_eq - equilibrium conformations of test trajectories
        subset (list): indices of subset. Set to None for entire dataset (default: None)
        download (bool): set to true if dataset should be downloaded. (default: True)
        calculate_triples (false): set to true to compute triples for angular functions (default: true)

    See: http://quantum-machine.org/datasets/
    """
    existing_folds = [
        "reference",
        "reference_eq",
        "test_within",
        "test_other",
        "test_eq"
    ]

    def __init__(self, path, fold, subset=None, download=True, collect_triples=False):
        if fold not in self.existing_folds:
            raise ValueError("Fold {:s} does not exist".format(fold))

        self.path = path
        self.datapath = os.path.join(self.path, "iso17")

        self.database = fold + ".db"

        self.dbpath = os.path.join(self.datapath, self.database)

        environment_provider = SimpleEnvironmentProvider()

        if download:
            self.download()

        properties = ["total_energy", "atomic_forces"]

        super().__init__(self.dbpath, subset, properties, environment_provider, collect_triples)

    E = "total_energy"
    F = "atomic_forces"

    properties = [
        E, F
    ]

    units = dict(
        zip(properties, [1.0, 1.0])
    )

    def download(self):
        r"""
        download dataset if not already on disk
        """
        success = True
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.datapath):
            success = success and self._download()
        if not os.path.exists(self.dbpath):
            success = success and self._download()

        return success

    def _download(self):
        logging.info("Downloading ISO17 database...")
        tmpdir = tempfile.mkdtemp("iso17")
        tarpath = os.path.join(tmpdir, "iso17.tar.gz")
        url = "http://www.quantum-machine.org/datasets/iso17.tar.gz"

        try:
            request.urlretrieve(url, tarpath)
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, url)
            return False

        tar = tarfile.open(tarpath)
        tar.extractall(self.path)
        tar.close()

        shutil.rmtree(tmpdir)

        return True