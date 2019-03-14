import os
from sacred import Experiment
from schnetpack.sacred.evaluator_ingredient import evaluator_ing,\
    build_evaluator
import yaml

eval_ex = Experiment('evaluation', ingredients=[evaluator_ing])


@eval_ex.config
def config():
    """
    Settings for the evaluation script.
    """
    out_path = './results.db'               # path to output file
    model_dir = './training'    # path to trained model
    device = 'cpu'                          # device for evaluation


@eval_ex.capture
def save_config(_config, cfg_dir):
    """
    Save the evaluation configuration.

    Args:
        _config (dict): configuration of the experiment
        cfg_dir (str): path to the config directory

    """
    with open(os.path.join(cfg_dir, 'eval_config.yaml'), 'w') as f:
        yaml.dump(_config, f, default_flow_style=False)


@eval_ex.command
def evaluate(_log, model_dir, out_path, device):
    """
    Predict missing physical properties using a trained SchNet model for a
    given input file.

    Args:
        model_dir (str): dir to the trained model
        out_path (str): path to the output file
        device (str): train model on CPU/GPU
    """
    model_path = os.path.join(model_dir, 'best_model')
    save_config(cfg_dir=os.path.dirname(out_path))
    _log.info('build evaluator...')
    evaluator = build_evaluator(model_path=model_path, out_path=out_path)
    _log.info('evaluating...')
    evaluator.evaluate(device=device)


@eval_ex.automain
def main():
    evaluate()
