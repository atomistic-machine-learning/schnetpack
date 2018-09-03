import os

from ase.db import connect
from ase.io.extxyz import read_xyz

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider

__all__ = ['ExtXYZ', 'parse_extxyz']


def parse_extxyz(dbpath, xyzpath):
    r"""Parses file in XYZ format and writes content to sqllite database

    Args:
        dbpath(str): path to sqllite database
        xyzpath (str): path to file with xyz file format
    """
    with connect(dbpath, use_lock_file=False) as conn:
        with open(xyzpath) as f:
            for at in read_xyz(f, index=slice(None)):
                e = at.get_total_energy()
                f = at.get_forces()
                conn.write(at, data={ExtXYZ.E: e, ExtXYZ.F: f})


class ExtXYZ(AtomsData):
    '''
    Loader for MD data in extended XYZ format

    :param path: Path to database
    '''

    E = "energy"
    F = "forces"

    def __init__(self, dbpath, xyzpath, subset=None, properties=[], environment_provider=SimpleEnvironmentProvider(),
                 pair_provider=None, center_positions=True):
        if not os.path.exists(dbpath):
            os.makedirs('/'.join(dbpath.split('/')[:-1]))
            parse_extxyz(dbpath, xyzpath)
        super(ExtXYZ, self).__init__(dbpath, subset, properties, environment_provider, pair_provider, center_positions)
