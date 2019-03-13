import os
from sacred import Experiment
from schnetpack.sacred.evaluator_ingredient import evaluator_ing,\
    build_evaluator


eval_ex = Experiment('evaluation', ingredients=[evaluator_ing])


@eval_ex.config
def config():
    """
    Settings for the evaluation script.
    """
    out_path = './results.db'               # path to output file
    model_path = './training/best_model'    # path to trained model
    device = 'cpu'                          # device for evaluation


@eval_ex.command
def evaluate(_log, model_path, out_path, device):
    """
    Predict missing physical properties using a trained SchNet model for a
    given input file.

    Args:
        model_path (str): path to the trained model
        out_path (str): path to the output file
        device (str): train model on CPU/GPU
    """
    _log.info('build evaluator...')
    evaluator = build_evaluator(model_path=model_path, out_path=out_path)
    _log.info('evaluating...')
    evaluator.evaluate(device=device)


@eval_ex.automain
def main():
    print(eval_ex.config)
