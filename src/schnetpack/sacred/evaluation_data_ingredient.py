import os
from sacred import Ingredient
from schnetpack.data.parsing import generate_db
from schnetpack.data.atoms import AtomsData


eval_data_ing = Ingredient('dataset')


@eval_data_ing.config
def config():
    path = 'data/ethanol_test_ext.xyz'


@eval_data_ing.capture
def get_eval_data(path):
    data_type = os.path.splitext(path)[1]
    if data_type in ['.xyz', '.extxyz']:
        path_to_db = path[:-4] + '.db'
        generate_db(path, path_to_db)
    elif data_type == '.db':
        path_to_db = path
    else:
        raise NotImplementedError
    return AtomsData(path_to_db)
