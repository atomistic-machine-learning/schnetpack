import logging
import os
import shutil
import tarfile
import tempfile
import numpy as np
from urllib import request as request
from urllib.error import HTTPError, URLError

import schnetpack as spk
from schnetpack.datasets import DownloadableAtomsData


class ISO17(DownloadableAtomsData):
    """
    ISO17 benchmark data set for molecular dynamics of C7O2H10 isomers
    containing molecular forces.

    Args:
        datapath (str): Path to database directory
        fold (str): Fold of data to load. Allowed are:
                        reference - 80% of steps of 80% of MD trajectories
                        reference_eq - equilibrium conformations of those
                                       molecules
                        test_within - remaining 20% unseen steps of reference trajectories
                        test_other - remaining 20% unseen MD trajectories
                        test_eq - equilibrium conformations of test trajectories
        subset (list): indices of subset. Set to None for entire dataset (default: None)
        load_only (list, optional): reduced set of properties to be loaded
        download (bool): set to true if dataset should be downloaded. (default: True)
        collect_triples (false): set to true to compute triples for angular functions
            (default: true)
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).

    See: http://quantum-machine.org/datasets/
    """

    existing_folds = [
        "reference",
        "reference_eq",
        "test_within",
        "test_other",
        "test_eq",
    ]

    E = "total_energy"
    F = "atomic_forces"

    def __init__(
        self,
        datapath,
        fold,
        download=True,
        load_only=None,
        subset=None,
        collect_triples=False,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
    ):

        if fold not in self.existing_folds:
            raise ValueError("Fold {:s} does not exist".format(fold))

        available_properties = [ISO17.E, ISO17.F]
        units = [1.0, 1.0]

        self.path = datapath
        self.fold = fold
        dbpath = os.path.join(datapath, "iso17", fold + ".db")

        super().__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            collect_triples=collect_triples,
            download=download,
            available_properties=available_properties,
            units=units,
            environment_provider=environment_provider,
        )

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices
        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return type(self)(
            self.path,
            self.fold,
            download=False,
            properties=self.load_only,
            subset=subidx,
            collect_triples=self.collect_triples,
            environment_provider=self.environment_provider,
        )

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
