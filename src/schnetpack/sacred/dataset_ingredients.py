import os
from sacred import Ingredient
from schnetpack.datasets import ANI1, ISO17, QM9, MD17, MaterialsProject
from schnetpack.data.parsing import extxyz_to_db
from schnetpack.data import AtomsData, AtomsDataError
from schnetpack.atomistic import Properties

dataset_ingredient = Ingredient("dataset")


@dataset_ingredient.config
def cfg():
    """Settings for training dataset"""
    dbpath = None           # path to ase.db
    dataset = 'CUSTOM'      # dataset name; use for pre-implemented datasets
    property_mapping = {}   # mapping from model properties to data properties


@dataset_ingredient.named_config
def qm9():
    """Default settings for QM9 dataset."""
    dbpath = './data/qm9.db'
    dataset = 'QM9'
    property_mapping = {Properties.energy: QM9.U0,
                        Properties.dipole_moment: QM9.mu,
                        Properties.iso_polarizability: QM9.alpha}

@dataset_ingredient.named_config
def iso17():
    """
    Default settings for ISO17 dataset.

    Adds:
        fold (str): name of the dataset inside the iso17 folder
    """
    dbpath = './data'
    dataset = 'ISO17'
    fold = 'reference'
    property_mapping = {Properties.energy: ISO17.E,
                        Properties.forces: ISO17.F}


@dataset_ingredient.named_config
def ani1():
    """
    Default settings for ANI1 dataset.

    Adds:
        num_heavy_atoms (int): maximum number of heavy atoms
    """
    dbpath = './data/ani1.db'
    dataset = 'ANI1'
    num_heavy_atoms = 2
    property_mapping = {Properties.energy: ANI1.energy}


@dataset_ingredient.named_config
def md17():
    """
    Default settings for MD17 dataset.

    Adds:
        molecule (str): name of the molecule that is included in MD17
    """
    dbpath = './data'
    dataset = 'MD17'
    molecule = 'aspirin'
    property_mapping = {Properties.energy: MD17.energy,
                        Properties.forces: MD17.forces}


@dataset_ingredient.named_config
def matproj():
    """
    Default settings for Materials Project dataset.

    Adds:
        cutoff (float):
        api_key (str): personal api key from https://materialsproject.org
    """
    dbpath = './data/matproj.db'
    dataset = 'MATPROJ'
    cutoff = 5.
    api_key = ''
    property_mapping = {Properties.energy_contributions:
                        MaterialsProject.EPerAtom}


@dataset_ingredient.capture
def get_property_map(properties, property_mapping, dbpath):
    """
    Provide property map from model properties to dataset properties.

    Args:
        properties (list): model properties
        property_mapping (dict): dict with all possible mappings
        dbpath (str): path to the local database

    Returns:
        dict: The mapping dictionary with model properties as keys and
        dataset properties as values.

    """
    if type(property_mapping) == str:
        property_mapping =\
            {key: value for key, value in
             [prop.split(':') for prop in property_mapping.split(',')]}
    property_map = {}
    for prop in properties:
        if prop in property_mapping.keys():
            property_map[prop] = property_mapping[prop]
        else:
            raise AtomsDataError('"{}" is not a valid property that is '
                                 'contained in the property_mapping for the '
                                 'database located ad {}.'.format(prop, dbpath))
    return property_map


@dataset_ingredient.capture
def get_ani1(dbpath, num_heavy_atoms, dataset_properties):
    """
    Args:
        dbpath (str): path to the local database
        num_heavy_atoms (int): max number of heavy atoms per molecule
        dataset_properties (list): properties of the dataset

    Returns:
        AtomsData object

    """
    return ANI1(dbpath, num_heavy_atoms=num_heavy_atoms,
                properties=dataset_properties)


@dataset_ingredient.capture
def get_matproj(dbpath, cutoff, api_key, dataset_properties):
    """
    Args:
        dbpath (str): path to the local database
        cutoff (float): cutoff radius
        api_key (str): personal api_key for materialsproject.org
        dataset_properties (list): properties of the dataset

    Returns:
        AtomsData object

    """
    return MaterialsProject(dbpath, cutoff, api_key,
                            properties=dataset_properties)


@dataset_ingredient.capture
def get_iso17(dbpath, fold, dataset_properties):
    """
    Args:
        dbpath (str): path to the local database directory
        fold (str): name of the local iso17 database
        dataset_properties (list): properties of the dataset

    Returns:
        AtomsData object

    """
    return ISO17(dbpath, fold, properties=dataset_properties)


@dataset_ingredient.capture
def get_md17(dbpath, molecule, dataset_properties):
    """
    Args:
        dbpath (str): path to the local database
        molecule (str): name of a molecule that is contained in the MD17 dataset
        dataset_properties (list): properties of the dataset

    Returns:
        AtomsData object

    """
    return MD17(dbpath, molecule=molecule, properties=dataset_properties)


@dataset_ingredient.capture
def get_dataset(_log, dbpath, dataset, dataset_properties=None):
    """
    Get a dataset from the configuration.

    Args:
        dbpath (str): path to the local database
        dataset (str): name of the dataset
        dataset_properties (list): properties of the dataset

    Returns:
        AtomsData object

    """
    dataset = dataset.upper()
    _log.info('Load {} dataset'.format(dataset))
    if dataset == 'QM9':
        return QM9(dbpath, properties=dataset_properties)
    elif dataset == 'ISO17':
        return get_iso17(dataset_properties=dataset_properties)
    elif dataset == 'ANI1':
        return get_ani1(dataset_properties=dataset_properties)
    elif dataset == 'MD17':
        return get_md17(dataset_properties=dataset_properties)
    elif dataset == 'MATPROJ':
        return get_matproj(dataset_properties=dataset_properties)
    elif dataset == 'CUSTOM':
        file, extension = os.path.splitext(dbpath)
        if extension == '.db':
            return AtomsData(dbpath, required_properties=dataset_properties)
        elif extension == '.xyz':
            extxyz_to_db(db_path=file + '.db', xyzpath=dbpath)
            return AtomsData(file+'.db', required_properties=dataset_properties)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
