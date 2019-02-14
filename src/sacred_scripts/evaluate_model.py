import os
from sacred import Experiment
from schnetpack.sacred.evaluator_ingredient import evaluator_ing,\
    build_evaluator


eval_ex = Experiment('evaluation', ingredients=[evaluator_ing])


@eval_ex.config
def config():
    experiment_dir = './experiments'
    output_dir = os.path.join(experiment_dir, 'evaluation')
    model_path = os.path.join(experiment_dir, 'training/best_model')
    device = 'cpu'


@eval_ex.command
def evaluate(model_path, output_dir, device):
    evaluator = build_evaluator(model_path=model_path, output_dir=output_dir)
    evaluator.evaluate(device=device)


@eval_ex.automain
def main():
    print(eval_ex.config)
