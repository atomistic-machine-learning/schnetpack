import torch
import os
import yaml
import numpy as np
from shutil import rmtree
from sacred import Experiment
from model_ingredients import model_ingredient, build_model
from trainer_ingredients import train_ingredient, setup_trainer
from schnetpack.property_model import Properties
from schnetpack.data import AtomsData, AtomsLoader
from schnetpack.datasets.qm9 import QM9

ex = Experiment('qm9', ingredients=[model_ingredient, train_ingredient])


def is_extensive(prop):
    return prop == Properties.energy


@ex.config
def cfg():
    properties = {}
    loss_tradeoff = {}
    overwrite = True
    additional_outputs = []

    modeldir = None
    dbpath = None
    batch_size = 100
    num_train = 0.8
    num_val = 0.1
    num_workers = 2
    mean = None
    stddev = None
    device = 'cpu'


@ex.named_config
def debug_cfg():
    properties = {Properties.energy: QM9.U0}
    dbpath = './qm9a.db'
    modeldir = './model'


@ex.capture
def save_config(_config, modeldir):
    with open(os.path.join(modeldir, 'config.yaml'), 'w') as f:
        yaml.dump(_config, f, default_flow_style=False)


@ex.capture
def prepare_data(_seed, dbpath, properties,
                 batch_size, num_train, num_val, num_workers):
    # local seed
    np.random.seed(_seed)

    # load and split
    targets = [tgt for tgt in properties.values() if tgt is not None]
    data = AtomsData(dbpath, required_properties=targets, load_charge=False)

    if num_train < 1:
        num_train = int(num_train * len(data))
    if num_val < 1:
        num_val = int(num_val * len(data))

    train, val, test = data.create_splits(num_train, num_val)

    train_loader = AtomsLoader(train, batch_size, True, pin_memory=True,
                               num_workers=num_workers)
    val_loader = AtomsLoader(val, batch_size, False, pin_memory=True,
                             num_workers=num_workers)
    test_loader = AtomsLoader(test, batch_size, False, pin_memory=True,
                              num_workers=num_workers)

    atomrefs = {p: data.get_atomref(tgt)
                for p, tgt in properties.items()
                if tgt is not None}

    return train_loader, val_loader, test_loader, atomrefs


@ex.capture
def stats(train_loader, atomrefs, properties, mean, stddev, _config):
    props = [p for p, tgt in properties.items() if tgt is not None]
    targets = [properties[p] for p in props if
               p not in [Properties.polarizability, Properties.dipole_moment]]
    atomrefs = [atomrefs[p] for p in props if
                p not in [Properties.polarizability, Properties.dipole_moment]]
    extensive = [is_extensive(p) for p in props if
                 p not in [Properties.polarizability,
                           Properties.dipole_moment]]

    if len(targets) > 0:
        if mean is None or stddev is None:
            mean, stddev = train_loader.get_statistics(targets, extensive,
                                                       atomrefs)
            _config["mean"] = dict(
                zip(props, [m.detach().cpu().numpy().tolist() for m in mean]))
            _config["stddev"] = dict(
                zip(props, [m.detach().cpu().numpy().tolist() for m in stddev]))
    else:
        _config["mean"] = {}
        _config["stddev"] = {}
    return _config['mean'], _config['stddev']


@ex.capture
def create_modeldir(_log, modeldir, overwrite):
    _log.info("Create model directory")
    if modeldir is None:
        raise ValueError('Config `modeldir` has to be set!')

    if os.path.exists(modeldir) and not overwrite:
        raise ValueError(
            'Model directory already exists (set overwrite flag?):', modeldir)

    if os.path.exists(modeldir) and overwrite:
        rmtree(modeldir)

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)


@ex.capture
def build_loss(properties, loss_tradeoff):
    def loss_fn(batch, result):
        loss = 0.
        for p, tgt in properties.items():
            if tgt is not None:
                diff = batch[tgt] - result[p]
                diff = diff ** 2
                err_sq = torch.mean(diff)
                if p in loss_tradeoff.keys():
                    err_sq *= loss_tradeoff[p]
                loss += err_sq
        return loss
    return loss_fn


@ex.command
def train(_log, _config, modeldir, properties, additional_outputs, device):
    create_modeldir()
    save_config()

    _log.info("Load data")
    train_loader, val_loader, _, atomrefs = prepare_data()
    mean, stddev = stats(train_loader, atomrefs)

    _log.info("Build model")
    props = [p for p, tgt in properties.items() if tgt is not None]
    model = build_model(mean=mean, stddev=stddev, atomrefs=atomrefs,
                        properties=props,
                        additional_outputs=additional_outputs).to(device)

    _log.info("Setup training")
    loss_fn = build_loss()
    trainer = setup_trainer(model=model, loss_fn=loss_fn, modeldir=modeldir,
                            train_loader=train_loader, val_loader=val_loader,
                            properties=properties)
    _log.info("Training")
    trainer.train(device)


@ex.command
def evaluate():
    print("Evaluate")


@ex.automain
def main():
    print(ex.config)
