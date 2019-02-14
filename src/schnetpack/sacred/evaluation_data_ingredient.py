from sacred import Ingredient
from schnetpack.data.parsing import ext_xyz_to_db
from schnetpack.data.atoms import AtomsData

eval_data_ing = Ingredient('evaluation_data')


@eval_data_ing.config
def config():
    data_path = 'data/ethanol_test_ext.xyz'
    data_type = 'xyz'


@eval_data_ing.capture
def get_eval_data(data_path, data_type):
    if data_type == 'xyz':
        path_to_db = data_path[:-4] + '.db'
        ext_xyz_to_db(path_to_db, data_path)
    elif data_type == 'db':
        path_to_db = data_path
    else:
        raise NotImplementedError
    return AtomsData(path_to_db)
