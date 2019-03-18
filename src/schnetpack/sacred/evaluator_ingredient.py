import os
import torch
from sacred import Ingredient
from schnetpack.evaluation import NPZEvaluator, DBEvaluator
from schnetpack.sacred.dataloader_ingredient import evaluation_loader_ing,\
    build_eval_loader
from schnetpack.sacred.evaluation_data_ingredient import eval_data_ing,\
    get_eval_data


evaluator_ing = Ingredient('evaluator', ingredients=[evaluation_loader_ing,
                                                     eval_data_ing])


@evaluator_ing.config
def config():
    """configuration of the evaluator ingredient"""


@evaluator_ing.capture
def build_evaluator(_log, model_path, in_path, out_path):
    """
    Create the evaluator object.

    Args:
        model_path (str): path to the trained model
        in_path (str): path to input file
        out_path (str): path to the output file

    Returns:
        Evaluator object
    """
    file_type = os.path.splitext(out_path)[1]
    _log.info('loading data...')

    data = get_eval_data(path=in_path)
    dataloader = build_eval_loader(data)
    _log.info('loading model...')
    model = torch.load(model_path)
    if file_type == '.npz':
        return get_npz_evaluator(model=model, dataloader=dataloader,
                                 out_path=out_path)
    elif file_type == '.db':
        return get_db_evaluator(model=model, dataloader=dataloader,
                                out_path=out_path)
    else:
        raise NotImplementedError


@evaluator_ing.capture
def get_npz_evaluator(model, dataloader, out_path):
    """
    Build evaluator for npz output format.
    """
    return NPZEvaluator(model, dataloader, out_path)


@evaluator_ing.capture
def get_db_evaluator(model, dataloader, out_path):
    """
    Build evaluator for ase.db output format.
    """
    return DBEvaluator(model, dataloader, out_path)
