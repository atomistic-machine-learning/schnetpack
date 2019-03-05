from sacred import Ingredient
import numpy as np
from schnetpack.data import AtomsLoader
from schnetpack.atomistic import Properties
from schnetpack.data import train_test_split


dataloader_ing = Ingredient('dataloader')


def is_extensive(prop):
    return prop == Properties.energy


@dataloader_ing.config
def config():
    num_workers = 4
    num_train = 0.8
    num_val = 0.1
    batch_size = 128
    mean = None
    stddev = None


@dataloader_ing.capture
def build_dataloaders(_seed, num_train, num_val, batch_size, num_workers,
                      property_map, dataset):
    # local seed
    np.random.seed(_seed)

    if num_train < 1:
        num_train = int(num_train * len(dataset))
    if num_val < 1:
        num_val = int(num_val * len(dataset))

    train, val, test = train_test_split(dataset, num_train, num_val)

    train_loader = AtomsLoader(train, batch_size, True, pin_memory=True,
                               num_workers=num_workers)
    val_loader = AtomsLoader(val, batch_size, False, pin_memory=True,
                             num_workers=num_workers)
    test_loader = AtomsLoader(test, batch_size, False, pin_memory=True,
                              num_workers=num_workers)

    atomrefs = {p: dataset.get_atomref(tgt)
                for p, tgt in property_map.items()
                if tgt is not None}

    return train_loader, val_loader, test_loader, atomrefs


@dataloader_ing.capture
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


evaluation_loader_ing = Ingredient('dataloader')


@evaluation_loader_ing.config
def config():
    batch_size = 32
    num_workers = 4
    pin_memory = True


@evaluation_loader_ing.capture
def build_eval_loader(data, batch_size, num_workers, pin_memory):
    return AtomsLoader(data, batch_size, pin_memory=pin_memory,
                       num_workers=num_workers)
