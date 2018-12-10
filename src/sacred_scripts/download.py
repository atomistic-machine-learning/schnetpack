from sacred import Experiment
from schnetpack.datasets import ANI1, ISO17, QM9, MD17, MaterialsProject

ex = Experiment("data")


@ex.config
def cfg():
    dbpath = None
    dataset = None
    folds = None
    cutoff = None
    api_key = None
    molecules = None


@ex.named_config
def qm9():
    dbpath = './qm9.db'
    dataset = 'QM9'


@ex.named_config
def iso17():
    dbpath = './'
    dataset = 'ISO17'
    folds = ISO17.existing_folds


@ex.named_config
def ani1():
    dbpath = './ani1.db'
    dataset = 'ANI1'


@ex.named_config
def md17():
    dbpath = './MD17/'
    dataset = 'MD17'
    molecules = MD17.datasets_dict.keys()


@ex.named_config
def matproj():
    dbpath = './matproj.db'
    dataset = 'MATPROJ'
    cutoff = 5.


@ex.capture
def download(dbpath, dataset, folds, cutoff, api_key, molecules):
    dataset = dataset.upper()
    if dataset == 'QM9':
        qm9 = QM9(dbpath)
    elif dataset == 'ISO17':
        for fold in folds:
            iso17 = ISO17(dbpath, fold)
    elif dataset == 'ANI1':
        ani1 = ANI1(dbpath)
    elif dataset == 'MD17':
        for molecule in molecules:
            md17 = MD17(dbpath + molecule + '.db', dataset=molecule)
    elif dataset == 'MATPROJ':
        matproj = MaterialsProject(dbpath, cutoff, api_key)
    else:
        raise NotImplementedError


@ex.automain
def main():
    download()
    print(ex.config)
