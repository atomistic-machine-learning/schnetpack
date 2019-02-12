import os

from ase.db import connect
from ase.io.extxyz import read_xyz

from schnetpack.data import AtomsData
from schnetpack.environment import SimpleEnvironmentProvider

__all__ = ['ExtXYZ', 'parse_extxyz']


def extend_xyz(xyz_path, properties):
    """
    Convert an xyz-file to extended xyz.

    Args:
        xyz_path (str): path to the xyz file
        properties (str): property-string

    """
    ext_xyz_path = xyz_path[:-3] + 'extxyz'
    new_file = open(ext_xyz_path, 'w')
    with open(xyz_path, 'r') as xyz_file:
        while True:
            # read first line of molecule
            first_line = xyz_file.readline()
            # check if it is not blank
            if first_line == '':
                break
            # get number of atoms
            n_atoms = int(first_line.strip('\n'))
            # get energy from comment line
            energy = float(xyz_file.readline().strip('/n'))
            # write to new filew
            new_file.writelines(str(n_atoms) + '\n')
            new_file.writelines(' '.join([properties,
                                          'energy={}'.format(energy)]) + '\n')
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
    # close new file
    new_file.close()


def parse_extxyz(dbpath, xyzpath):
    r"""Parses file in XYZ format and writes content to sqllite database

    Args:
        dbpath(str): path to sqlite database
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
    available_properties = [E, F]

    def __init__(self, dbpath, xyzpath, subset=None, properties=[],
                 environment_provider=SimpleEnvironmentProvider(),
                 pair_provider=None, center_positions=True):
        if not os.path.exists(dbpath):
            os.makedirs(os.path.dirname(dbpath), exist_ok=True)
            parse_extxyz(dbpath, xyzpath)
        super(ExtXYZ, self).__init__(dbpath, subset, properties,
                                     environment_provider, pair_provider,
                                     center_positions)
