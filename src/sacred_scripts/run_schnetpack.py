import os
from shutil import rmtree

import numpy as np
import torch
import yaml
from sacred import Experiment
from sacred.observers import MongoObserver
from schnetpack.atomistic import Structure
from schnetpack.sacred.dataset_ingredients import dataset_ingredient, \
    get_dataset, get_property_map
from schnetpack.sacred.model_ingredients import model_ingredient, build_model,\
    load_model
from schnetpack.sacred.trainer_ingredients import train_ingredient, \
    setup_trainer
from schnetpack.sacred.dataloader_ingredient import dataloader_ing, \
    build_dataloaders, build_eval_loader, stats



ex = Experiment('experiment', ingredients=[model_ingredient, train_ingredient,
                                           dataloader_ing, dataset_ingredient])


@ex.config
def cfg():
    """configuration configuration for training experiment"""

    loss_tradeoff = {}
    overwrite = True
    additional_outputs = []
    device = 'cpu'
    experiment_dir = './experiments'
    training_dir = os.path.join(experiment_dir, 'training')
    properties = ['energy', 'forces']
    element_wise = ['forces']
    mean = None
    stddev = None
    eval_file = None


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
    dataset = get_dataset(dataset_properties=property_map.values())
    train_loader, val_loader, _, atomrefs = \
        build_dataloaders(property_map=property_map, dataset=dataset)
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
    pass

@ex.command
def evaluate(device, properties, eval_file):
    assert eval_file is not None, 'Please define an evaluation file!'
    dataloader = build_eval_loader(property_map=get_property_map(properties))
    model = load_model()

    predicted = {}

    for batch in dataloader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for p in result.keys():
            value = result[p].cpu().detach().numpy()
            if p in predicted.keys():
                predicted[p].append(value)
            else:
                predicted[p] = [value]

        for p in [Structure.R, Structure.Z]:
            value = batch[p].cpu().detach().numpy()
            if p in predicted.keys():
                predicted[p].append(value)
            else:
                predicted[p] = [value]

    for p in predicted.keys():
        predicted[p] = np.vstack(predicted[p])

    np.savez(eval_file, **predicted)


@ex.automain
def main():
    print(ex.config)
