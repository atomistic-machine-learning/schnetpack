import os
import torch
from sacred import Ingredient

from schnetpack.evaluation import NPZEvaluator, DBEvaluator
from schnetpack.sacred.dataloader_ingredient import evaluation_loader_ing,\
    build_eval_loader
from schnetpack.data.loader import AtomsLoader
from schnetpack.sacred.evaluation_data_ingredient import eval_data_ing,\
    get_eval_data


evaluator_ing = Ingredient('evaluator', ingredients=[evaluation_loader_ing,
                                                     eval_data_ing])


@evaluator_ing.config
def config():
    """configuration of the evaluator ingredient"""
    out_file = 'evaluation.db'


@evaluator_ing.named_config
def npz():
    out_file = 'evaluation.npz'


@evaluator_ing.capture
def build_evaluator(_log, model_path, out_file, output_dir):
    file_type = os.path.splitext(out_file)[1]
    out_path = os.path.join(output_dir, out_file)
    _log.info('loading data...')
    data = get_eval_data()
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
    return NPZEvaluator(model, dataloader, out_path)


@evaluator_ing.capture
def get_db_evaluator(model, dataloader, out_path):
    return DBEvaluator(model, dataloader, out_path)
