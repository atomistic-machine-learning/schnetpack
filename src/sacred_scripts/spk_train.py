import os
from sacred import Experiment
import numpy as np
from schnetpack.sacred.dataset_ingredients import dataset_ingredient, \
    get_dataset, get_property_map
from schnetpack.sacred.model_ingredients import model_ingredient, build_model
from schnetpack.sacred.trainer_ingredients import train_ingredient, \
    setup_trainer
from schnetpack.sacred.dataloader_ingredient import dataloader_ing, \
    build_dataloaders, stats
from schnetpack.sacred.folder_ingredient import create_dirs, save_config,\
    folder_ing


ex = Experiment('experiment', ingredients=[model_ingredient, train_ingredient,
                                           dataloader_ing, dataset_ingredient,
                                           folder_ing])


@ex.config
def cfg():
    r"""
    configuration for training script
    """
    additional_outputs = []             # additional model outputs
    device = 'cpu'                 # device that is used for training <cpu/cuda>
    model_dir = 'training'              # directory for training outputs
    properties = ['energy', 'forces']   # model properties


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

    _log.info("Load data")
    dataset = get_dataset(dataset_properties=property_map.values())
    train_loader, val_loader, test_loader, atomrefs = \
        build_dataloaders(property_map=property_map, dataset=dataset)
    np.savez(os.path.join(model_dir, 'splits.npz'),
             train=train_loader.dataset.subset,
             val=val_loader.dataset.subset,
             test=test_loader.dataset.subset,
             atomrefs=atomrefs)
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
def main(_log, _config, model_dir):
    create_dirs(_log=_log, output_dir=model_dir)
    save_config(_config=_config, output_dir=model_dir)
    train()
