import os
import torch
from sacred import Ingredient

from schnetpack.evaluation import XYZEvaluator
from schnetpack.sacred.dataloader_ingredient import dataloader_ing,\
    build_eval_loader
from schnetpack.sacred.dataset_ingredients import dataset_ingredient,\
    get_dataset


evaluator_ing = Ingredient('evaluator', ingredients=[dataloader_ing,
                                                     dataset_ingredient])


@evaluator_ing.config
def config():
    """configuration of the evaluator ingredient"""
    name = 'to_npz'
    out_file = 'evaluation.npz'


@evaluator_ing.capture
def build_evaluator(model_path, name, output_dir):
    data = get_dataset()
    dataloader = build_eval_loader(data)
    model = torch.load(model_path)
    if name == 'to_npz':
        return get_npz_evaluator(model=model, dataloader=dataloader,
                                 output_dir=output_dir)
    else:
        raise NotImplementedError


@evaluator_ing.capture
def get_npz_evaluator(model, dataloader, output_dir, out_file):
    out_path = os.path.join(output_dir, out_file)
    return XYZEvaluator(model, dataloader, out_path)
