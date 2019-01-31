import os
from shutil import rmtree

import numpy as np
import torch
import yaml
from sacred import Experiment
from sacred.observers import MongoObserver
from schnetpack.sacred.dataset_ingredients import dataset_ingredient, \
    get_dataset, get_property_map
from schnetpack.sacred.model_ingredients import model_ingredient, build_model
from schnetpack.sacred.trainer_ingredients import train_ingredient, \
    setup_trainer

from schnetpack.data import AtomsLoader
from schnetpack.atomistic import Properties

ex = Experiment('experiment', ingredients=[model_ingredient, train_ingredient,
                                           dataset_ingredient])


def is_extensive(prop):
    return prop == Properties.energy


@ex.config
def cfg():
    """configuration configuration for training experiment"""

    loss_tradeoff = {}
    overwrite = True
    additional_outputs = []
    batch_size = 100
    num_train = 0.8
    num_val = 0.1
    num_workers = 2
    device = 'cpu'
    experiment_dir = './experiments'
    training_dir = os.path.join(experiment_dir, 'training')
    properties = ['energy', 'forces']
    element_wise = ['forces']
    mean = None
    stddev = None


@ex.named_config
def observe():
    """configuration for observing experiments"""

    mongo_url = 'mongodb://127.0.0.1:27017'
    mongo_db = 'test'
    ex.observers.append(MongoObserver.create(url=mongo_url,
                                             db_name=mongo_db))


@ex.capture
def save_config(_config, training_dir):
    """
    Save the configuration to the model directory.

    Args:
        _config (dict): configuration of the experiment
        training_dir (str): path to the training directory

    """
    with open(os.path.join(training_dir, 'config.yaml'), 'w') as f:
        yaml.dump(_config, f, default_flow_style=False)


@ex.capture
def prepare_data(_seed, property_map,
                 batch_size, num_train, num_val, num_workers):
    """
    Create the dataloaders for training.

    Args:
        _seed (int): seed for controlled randomness
        property_map (dict): mapping between model properties and dataset
            properties
        batch_size (int): batch size
        num_train (int): number of training samles
        num_val (int): number of validation samples
        num_workers (int): number of workers for the dataloaders

    Returns:
        schnetpack.data.Atomsloader objects for training, validation and
        testing and the atomic reference data
    """
    # local seed
    np.random.seed(_seed)

    # load and split
    data = get_dataset(dataset_properties=property_map.values())

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
                for p, tgt in property_map.items()
                if tgt is not None}

    return train_loader, val_loader, test_loader, atomrefs


@ex.capture
def stats(train_loader, atomrefs, property_map, mean, stddev, _config):
    """
    Calculate statistics of the input data.

    Args:
        train_loader (schnetpack.data.Atomsloader): loader for train data
        atomrefs (torch.Tensor): atomic reference data
        property_map (dict): mapping between the model properties and the
            dataset properties
        mean:
        stddev:
        _config (dict): configuration of the experiment

    Returns:
        mean and std for the configuration

    """
    props = [p for p, tgt in property_map.items() if tgt is not None]
    targets = [property_map[p] for p in props if
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
                zip(props,
                    [m.detach().cpu().numpy().tolist() for m in stddev]))
    else:
        _config["mean"] = {}
        _config["stddev"] = {}
    return _config['mean'], _config['stddev']


@ex.capture
def create_dirs(_log, training_dir, overwrite):
    """
    Create the directory for the experiment.

    Args:
        _log:
        experiment_dir (str): path to the experiment directory
        overwrite (bool): overwrites the model directory if True

    """
    _log.info("Create model directory")
    if training_dir is None:
        raise ValueError('Config `experiment_dir` has to be set!')

    if os.path.exists(training_dir) and not overwrite:
        raise ValueError(
            'Model directory already exists (set overwrite flag?):',
            training_dir)

    if os.path.exists(training_dir) and overwrite:
        rmtree(training_dir)

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)


@ex.capture
def build_loss(property_map, loss_tradeoff):
    """
    Build the loss function.

    Args:
        property_map (dict): mapping between the model properties and the
            dataset properties
        loss_tradeoff (dict): contains tradeoff factors for properties,
            if needed

    Returns:
        loss function

    """
    def loss_fn(batch, result):
        loss = 0.
        for p, tgt in property_map.items():
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
def train(_log, _config, training_dir, properties, additional_outputs, device,
          element_wise):
    """
    Build a trainer from the configuration and start the treining.

    Args:
        _log:
        _config (dict): configuration dictionary
        training_dir (str): path to the training directory
        properties (list): list of model properties
        additional_outputs (list): list of additional model properties that are
            not back-propagated
        device (str): choose device for calculations (CPU/GPU)

    """
    property_map = get_property_map(properties)
    create_dirs()
    save_config()

    _log.info("Load data")
    train_loader, val_loader, _, atomrefs = prepare_data(property_map=
                                                         property_map)
    mean, stddev = stats(train_loader, atomrefs, property_map)

    _log.info("Build model")
    model_properties = [p for p, tgt in property_map.items() if
                        tgt is not None]
    model = build_model(mean=mean, stddev=stddev, atomrefs=atomrefs,
                        model_properties=model_properties,
                        additional_outputs=additional_outputs).to(device)
    _log.info("Setup training")
    loss_fn = build_loss(property_map=property_map)
    trainer = setup_trainer(model=model, loss_fn=loss_fn,
                            training_dir=training_dir,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            property_map=property_map,
                            element_wise=element_wise)
    _log.info("Training")
    trainer.train(device)


@ex.command
def download():
    get_dataset()


@ex.command
def evaluate():
    print("Evaluate")


@ex.automain
def main():
    print(ex.config)
