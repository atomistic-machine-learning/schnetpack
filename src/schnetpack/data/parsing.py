import os
from ase.db import connect
from ase.io.extxyz import read_xyz
from tqdm import tqdm
import tempfile


def xyz_to_extxyz(xyz_path, extxyz_path, atomic_properties,
                  molecular_properties=[]):
    """
    Convert a xyz-file to extxyz.

    Args:
        xyz_path (str): path to the xyz file
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    new_file = open(extxyz_path, 'w')
    with open(xyz_path, 'r') as xyz_file:
        while True:
            first_line = xyz_file.readline()
            if first_line == '':
                break
            n_atoms = int(first_line.strip('\n'))
            molecular_data = xyz_file.readline().strip('/n').split()
            assert len(molecular_data) == \
                len(molecular_properties), ('The number of datapoints and '
                                       'properties do not match!')
            comment = ' '.join(['{}={}'.format(prop, val) for prop, val in
                                zip(molecular_properties, molecular_data)])
            new_file.writelines(str(n_atoms) + '\n')
            new_file.writelines(' '.join([atomic_properties, comment]) + '\n')
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
    new_file.close()


def extxyz_to_db(extxyz_path, db_path):
    r"""
    Convertes en extxyz-file to an ase database

    Args:
        db_path(str): path to sqlite database
        xyzpath (str): path to file with xyz file format
        db_properties (list): physical properties that will be included to
            the database
    """
    with connect(db_path, use_lock_file=False) as conn:
        with open(extxyz_path) as f:
            for at in tqdm(read_xyz(f, index=slice(None)), 'creating ase db'):
                data = {}
                if at.has('forces'):
                    data['forces'] = at.get_forces()
                data.update(at.info)
                conn.write(at, data=data)


def xyz_to_db(xyz_path, db_path, atomic_properties, molecular_properties=[]):
    """
    Convertes a xyz-file to an ase database.

    Args:
        xyz_path (str): path to the xyz file
        db_path(str): path to sqlite database
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
        db_properties:
    """
    extxyz_path = os.path.join(tempfile.mkdtemp(), 'temp.extxyz')
    xyz_to_extxyz(xyz_path, extxyz_path, atomic_properties,
                  molecular_properties)
    extxyz_to_db(extxyz_path, db_path)


def generate_db(file_path, db_path, atomic_properties=None,
                molecular_properties=None):
    """
    Convert a file with molecular information to an ase database. Currently
    supports .xyz and .extxyz.

    Args:
        file_path (str): path to the input file
        db_path(str): path to sqlite database
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
        db_properties (list): physical properties that will be included to
            the database
    """
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.xyz':
        xyz_to_db(file_path, db_path, atomic_properties,
                  molecular_properties)
    elif file_extension == '.extxyz':
        extxyz_to_db(file_path, db_path)
    else:
        raise NotImplementedError
