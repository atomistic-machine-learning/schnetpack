import urllib.request
import os
import zipfile
from schnetpack.datasets.extxyz import parse_extxyz, extend_xyz


# Parameters
url = 'http://quantum-machine.org/gdml/data/xyz/ethanol_dft.zip'
data_dir = './data'
zip_file = os.path.join(data_dir, 'ethanol.zip')
db_path = os.path.join(data_dir, 'ethanol.db')
xyz_path = os.path.join(data_dir, 'ethanol.xyz')

# create the data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# download the ZIP-file
if not os.path.exists(zip_file):
    urllib.request.urlretrieve(url, zip_file)

# unzip the data
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(data_dir)

# create extended xyz file
properties = 'Properties=species:S:1:pos:R:3:forces:R:3'
extend_xyz(xyz_path=xyz_path, properties=properties)

# create ASE database
ext_xyz_path = xyz_path[:-3] + 'extxyz'
parse_extxyz(dbpath=db_path, xyzpath=ext_xyz_path)

