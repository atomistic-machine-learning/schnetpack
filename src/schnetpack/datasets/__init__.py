r"""
Classes wrapping various standard benchmark datasets.
"""

from schnetpack import AtomsData
from schnetpack.data.atoms import logger
from schnetpack.environment import SimpleEnvironmentProvider


class DownloadableAtomsData(AtomsData):
    """
        Base class for online datasets that can be automatically downloaded.

        Args:
            dbpath (str): path to directory containing database.
            subset (list, optional): indices to subset. Set to None for entire database.
            available_properties (list, optional): complete set of physical properties
                that are contained in the database.
            load_only (list, optional): reduced set of properties to be loaded
            units (list, optional): definition of units for all available properties
            environment_provider (spk.environment.BaseEnvironmentProvider): define how
                neighborhood is calculated
                (default=spk.environment.SimpleEnvironmentProvider).
            collect_triples (bool, optional): Set to True if angular features are needed.
            center_positions (bool): subtract center of mass from all positions
                (default=True)
            download (bool): If true, automatically download dataset
                if it does not exist.

    """

    def __init__(
        self,
        dbpath,
        subset=None,
        load_only=None,
        available_properties=None,
        units=None,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        center_positions=True,
        download=False,
    ):

        super(DownloadableAtomsData, self).__init__(
            dbpath=dbpath,
            subset=subset,
            available_properties=available_properties,
            load_only=load_only,
            units=units,
            environment_provider=environment_provider,
            collect_triples=collect_triples,
            center_positions=center_positions,
        )
        if download:
            self.download()

    def download(self):
        """
        Wrapper function for the download method.
        """
        if os.path.exists(self.dbpath):
            logger.info(
                "The dataset has already been downloaded and stored "
                "at {}".format(self.dbpath)
            )
        else:
            logger.info("Starting download")
            folder = os.path.dirname(os.path.abspath(self.dbpath))
            if not os.path.exists(folder):
                os.makedirs(folder)
            self._download()

    def _download(self):
        """
        To be implemented in deriving classes.
        """
        raise NotImplementedError


from .ani1 import *
from .iso17 import *
from .matproj import *
from .omdb import *
from .md17 import *
from .qm9 import *
from .extxyz import ExtXYZ
