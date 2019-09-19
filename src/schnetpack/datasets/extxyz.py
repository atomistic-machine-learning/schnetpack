import os

from schnetpack.datasets import DownloadableAtomsData
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

    def __init__(
        self,
        dbpath,
        xyzpath,
        subset=None,
        properties=None,
        environment_provider=SimpleEnvironmentProvider(),
        pair_provider=None,
        centering_function=True,
    ):
        available_properties = [ExtXYZ.E, ExtXYZ.F]
        units = [1.0, 1.0]

        if not os.path.exists(dbpath):
            os.makedirs(os.path.dirname(dbpath), exist_ok=True)
            extxyz_to_db(dbpath, xyzpath)

        super(ExtXYZ, self).__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=properties,
            environment_provider=environment_provider,
            collect_triples=pair_provider,
            centering_function=centering_function,
            available_properties=available_properties,
            units=units,
        )
