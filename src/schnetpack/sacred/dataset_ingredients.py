from sacred import Ingredient
from schnetpack.datasets import ANI1, ISO17, QM9, MD17, MaterialsProject
from schnetpack.data import AtomsData, AtomsLoader, AtomsDataError
from schnetpack.atomistic import Properties

dataset_ingredient = Ingredient("dataset")


@dataset_ingredient.config
def cfg():
    dbpath = None
    dataset = 'CUSTOM'
    cutoff = None
    api_key = None
    molecule = None
    fold = None
    property_mapping = {}


@dataset_ingredient.named_config
def qm9():
    dbpath = './data/qm9.db'
    dataset = 'QM9'
    property_mapping = {Properties.energy: QM9.U0,
                        Properties.dipole_moment: QM9.mu,
                        Properties.iso_polarizability: QM9.alpha}


@dataset_ingredient.named_config
def iso17():
    dbpath = './data'
    dataset = 'ISO17'
    fold = 'reference'
    property_mapping = {Properties.energy: ISO17.E,
                        Properties.forces:ISO17.F}


@dataset_ingredient.named_config
def ani1():
    dbpath = './data/ani1.db'
    dataset = 'ANI1'
    num_heavy_atoms = 2
    property_mapping = {Properties.energy: ANI1.energy}


@dataset_ingredient.named_config
def md17():
    dbpath = './data'
    dataset = 'MD17'
    molecule = 'aspirin'
    property_mapping = {Properties.energy: MD17.energy,
                        Properties.forces: MD17.forces}


@dataset_ingredient.named_config
def matproj():
    dbpath = './data/matproj.db'
    dataset = 'MATPROJ'
    cutoff = 5.
    api_key = ''
    property_mapping = {Properties.energy_contributions:
                            MaterialsProject.EPerAtom}


@dataset_ingredient.capture
def get_property_map(properties, property_mapping, dbpath):
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
    return ANI1(dbpath, num_heavy_atoms=num_heavy_atoms,
                properties=dataset_properties)


@dataset_ingredient.capture
def get_dataset(dbpath, dataset, cutoff, api_key, molecule, fold,
                dataset_properties=None):
    dataset = dataset.upper()
    if dataset == 'QM9':
        return QM9(dbpath, properties=dataset_properties)
    elif dataset == 'ISO17':
        return ISO17(dbpath, fold, properties=dataset_properties)
    elif dataset == 'ANI1':
        return get_ani1(dataset_properties=dataset_properties)
    elif dataset == 'MD17':
        return MD17(dbpath, molecule=molecule, properties=dataset_properties)
    elif dataset == 'MATPROJ':
        if api_key is None:
            raise AtomsDataError('Materials Project requires a valid API key.')
        return MaterialsProject(dbpath, cutoff, api_key,
                                properties=dataset_properties)
    elif dataset == 'CUSTOM':
        return AtomsData(dbpath, required_properties=dataset_properties)
    else:
        raise NotImplementedError
