import os
import torch
from sacred import Ingredient

from schnetpack.evaluation import NPZEvaluator, DBEvaluator
from schnetpack.sacred.dataloader_ingredient import dataloader_ing,\
    build_eval_loader
from schnetpack.sacred.evaluation_data_ingredient import eval_data_ing,\
    get_eval_data


evaluator_ing = Ingredient('evaluator', ingredients=[dataloader_ing,
                                                     eval_data_ing])


@evaluator_ing.config
def config():
    """configuration of the evaluator ingredient"""
    name = 'to_npz'
    out_file = 'evaluation.npz'


@evaluator_ing.named_config
def overwrite_db():
    name = 'to_db'
    out_file = None


@evaluator_ing.capture
def build_evaluator(model_path, name, output_dir):
    data = get_eval_data()
    dataloader = build_eval_loader(data)
    model = torch.load(model_path)
    if name == 'to_npz':
        return get_npz_evaluator(model=model, dataloader=dataloader,
                                 output_dir=output_dir)
    elif name == 'to_db':
        return get_db_evaluator(model=model, dataloader=dataloader)
    else:
        raise NotImplementedError


@evaluator_ing.capture
def get_npz_evaluator(model, dataloader, output_dir, out_file):
    out_path = os.path.join(output_dir, out_file)
    return NPZEvaluator(model, dataloader, out_path)


@evaluator_ing.capture
def get_db_evaluator(model, dataloader):
    return DBEvaluator(model, dataloader)
