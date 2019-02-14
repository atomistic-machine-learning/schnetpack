from ase.db import connect
from ase.io.extxyz import read_xyz


def extend_xyz(xyz_path, properties, molecular_props=[]):
    """
    Convert an xyz-file to extended xyz.

    Args:
        xyz_path (str): path to the xyz file
        properties (str): property-string
        molecular_props (list): molecular properties contained in the comment
            line

    """
    ext_xyz_path = xyz_path[:-4] + '_ext.xyz'
    new_file = open(ext_xyz_path, 'w')
    with open(xyz_path, 'r') as xyz_file:
        while True:
            first_line = xyz_file.readline()
            if first_line == '':
                break
            n_atoms = int(first_line.strip('\n'))
            molecular_data = xyz_file.readline().strip('/n').split()
            assert len(molecular_data) == \
                len(molecular_props), ('The number of datapoints and '
                                       'properties do not match!')
            comment = ' '.join(['{}={}'.format(prop, val) for prop, val in
                                zip(molecular_props, molecular_data)])
            new_file.writelines(str(n_atoms) + '\n')
            new_file.writelines(' '.join([properties, comment]) + '\n')
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
    new_file.close()


def ext_xyz_to_db(dbpath, xyzpath, properties=[]):
    r"""Parses file in XYZ format and writes content to sqllite database

    Args:
        dbpath(str): path to sqlite database
        xyzpath (str): path to file with xyz file format
    """
    with connect(dbpath, use_lock_file=False) as conn:
        with open(xyzpath) as f:
            for at in read_xyz(f, index=slice(None)):
                data = {}
                if 'energy' in properties:
                    data['energy'] = at.get_total_energy()
                if 'forces' in properties:
                    data['forces'] = at.get_forces()
                conn.write(at, data=data)
