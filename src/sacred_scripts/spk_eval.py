import os
import numpy as np
from sacred import Experiment
from ase.db import connect
from schnetpack.sacred.evaluator_ingredient import evaluator_ing,\
    build_evaluator
from schnetpack.sacred.folder_ingredient import create_dirs, save_config,\
    folder_ing


eval_ex = Experiment('evaluation', ingredients=[evaluator_ing, folder_ing])


class EvaluationError(Exception):
    pass


@eval_ex.config
def config():
    """
    Settings for the evaluation script.
    """
    in_path = None                          # path to input file
    out_path = './results.db'               # path to output file
    model_dir = './training'                # path to trained model
    device = 'cpu'                          # device for evaluation
    on_split = None                         # use test/val/train split file


@eval_ex.named_config
def test_set():
    """
    Evaluate the test data.
    """
    on_split = 'test'


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
        in_path (str): path of the input file
        model_dir (str): dir to the trained model
        out_path (str): path to the output file
        device (str): train model on CPU/GPU
        on_split (str): name of subset in model_dir
    """
    if in_path is None:
        raise EvaluationError('Input file is not defined!')
    model_path = os.path.join(model_dir, 'best_model')
    if on_split:
        split_path = np.load(os.path.join(model_dir, 'splits.npz'))
        subset = split_path[on_split]
        dbsplitpath = in_path[:-3] + '_' + on_split + '.db'
        build_subset_db(subset=subset, dbpath=in_path, dbsplitpath=dbsplitpath)
        in_path = dbsplitpath
    _log.info('build evaluator...')
    evaluator = build_evaluator(model_path=model_path, in_path=in_path,
                                out_path=out_path)
    _log.info('evaluating...')
    evaluator.evaluate(device=device)


@eval_ex.automain
def main(_log, _config, out_path):
    out_dir = os.path.dirname(out_path)
    create_dirs(_log=_log, output_dir=out_dir)
    save_config(_config=_config, output_dir=out_dir)
    evaluate()
