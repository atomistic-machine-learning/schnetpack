import os

from schnetpack.data import DownloadableAtomsData
from schnetpack.data.parsing import extxyz_to_db
from schnetpack.environment import SimpleEnvironmentProvider

__all__ = ["ExtXYZ"]


class ExtXYZ(DownloadableAtomsData):
    """
    Loader for MD data in extended XYZ format

    :param path: Path to database
    """

    E = "energy"
    F = "forces"
    available_properties = [E, F]

    def __init__(
        self,
        dbpath,
        xyzpath,
        subset=None,
        properties=[],
        environment_provider=SimpleEnvironmentProvider(),
        pair_provider=None,
        center_positions=True,
    ):
        if not os.path.exists(dbpath):
            os.makedirs(os.path.dirname(dbpath), exist_ok=True)
            extxyz_to_db(dbpath, xyzpath, db_properties=self.available_properties)
        super(ExtXYZ, self).__init__(
            dbpath,
            subset,
            properties,
            environment_provider,
            pair_provider,
            center_positions,
        )
