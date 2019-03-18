import os
import numpy as np
from sacred import Experiment
from ase.db import connect
from schnetpack.sacred.evaluator_ingredient import evaluator_ing,\
    build_evaluator
import yaml

eval_ex = Experiment('evaluation', ingredients=[evaluator_ing])


@eval_ex.config
def config():
    """
    Settings for the evaluation script.
    """
    in_path = './data/md17/ethanol.db'
    out_path = './results.db'               # path to output file
    model_dir = './training'    # path to trained model
    device = 'cpu'                          # device for evaluation
    on_split = None


@eval_ex.named_config
def test_set():
    on_split = 'test'


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


@eval_ex.capture
def build_subset_db(_log, subset, dbpath, dbsplitpath):
    _log.info('creating subset database')
    new_db = connect(dbsplitpath)
    old_db = connect(dbpath)
    for idx in subset:
        new_db.write(old_db.get_atoms(int(idx)))


@eval_ex.command
def evaluate(_log, model_dir, in_path, out_path, device, on_split):
    """
    Predict missing physical properties using a trained SchNet model for a
    given input file.

    Args:
        model_dir (str): dir to the trained model
        out_path (str): path to the output file
        device (str): train model on CPU/GPU
        on_split (str): name of subset in modeldir
    """
    model_path = os.path.join(model_dir, 'best_model')
    if on_split:
        split_path = np.load(os.path.join(model_dir, 'splits.npz'))
        subset = split_path[on_split]
        dbsplitpath = in_path[:-3] + '_' + on_split + '.db'
        build_subset_db(subset=subset, dbpath=in_path, dbsplitpath=dbsplitpath)
        in_path = dbsplitpath
    save_config(cfg_dir=os.path.dirname(out_path))
    _log.info('build evaluator...')
    evaluator = build_evaluator(model_path=model_path, in_path=in_path,
                                out_path=out_path)
    _log.info('evaluating...')
    evaluator.evaluate(device=device)


@eval_ex.automain
def main():
    evaluate()
