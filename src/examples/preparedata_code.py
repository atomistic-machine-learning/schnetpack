import urllib.request
import os
import zipfile
from schnetpack.data.parsing import extend_xyz, ext_xyz_to_db


# Parameters
url = 'http://quantum-machine.org/gdml/data/xyz/ethanol_dft.zip'
data_dir = './data'
zip_file = os.path.join(data_dir, 'ethanol.zip')
db_path = os.path.join(data_dir, 'ethanol.db')
xyz_path = os.path.join(data_dir, 'ethanol_test.xyz')

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
molecular_props = ['energy']
extend_xyz(xyz_path=xyz_path, properties=properties,
           molecular_props=molecular_props)

# create ASE database
ext_xyz_path = xyz_path[:-4] + '_ext.xyz'
ext_xyz_to_db(dbpath=db_path, xyzpath=ext_xyz_path, properties=[])
