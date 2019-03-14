import os
from shutil import rmtree
import yaml
from sacred import Experiment
from schnetpack.sacred.dataset_ingredients import dataset_ingredient, \
    get_dataset, get_property_map
from schnetpack.sacred.model_ingredients import model_ingredient, build_model
from schnetpack.sacred.trainer_ingredients import train_ingredient, \
    setup_trainer
from schnetpack.sacred.dataloader_ingredient import dataloader_ing, \
    build_dataloaders, stats


ex = Experiment('experiment', ingredients=[model_ingredient, train_ingredient,
                                           dataloader_ing, dataset_ingredient])


@ex.config
def cfg():
    r"""
    configuration for training script
    """
    overwrite = False                   # overwrite model_dir if True
    additional_outputs = []             # additional model outputs
    device = 'cpu'                 # device that is used for training <cpu/cuda>
    model_dir = 'training'              # directory for training outputs
    properties = ['energy', 'forces']   # model properties


@ex.capture
def save_config(_config, model_dir):
    """
    Save the configuration to the model directory.

    Args:
        _config (dict): configuration of the experiment
        model_dir (str): path to the training directory

    """
    with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
        yaml.dump(_config, f, default_flow_style=False)


@ex.capture
def create_dirs(_log, model_dir, overwrite):
    """
    Create the directory for the experiment.

    Args:
        model_dir (str): path to the training directory
        overwrite (bool): overwrites the model directory if True
    """
    _log.info("Create model directory")
    if model_dir is None:
        raise ValueError('Config `model_dir` has to be set!')

    if os.path.exists(model_dir) and not overwrite:
        raise ValueError(
            'Model directory already exists (set overwrite flag?):',
            model_dir)

    if os.path.exists(model_dir) and overwrite:
        rmtree(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


@ex.command
def train(_log, _config, model_dir, properties, additional_outputs, device):
    """
    Build a trainer from the configuration and start the training.

    Args:
        _config (dict): configuration dictionary
        model_dir (str): path to the training directory
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
    trainer = setup_trainer(model=model, train_dir=model_dir,
                            train_loader=train_loader, val_loader=val_loader,
                            property_map=property_map)
    _log.info("Training")
    trainer.train(device)


@ex.automain
def main():
    train()
