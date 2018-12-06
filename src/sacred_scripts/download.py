from sacred import Experiment
from schnetpack.datasets.qm9 import QM9


ex = Experiment("data")

@ex.config
def config():
    dbpath = './qm9.db'
    dataset = 'QM9'

@ex.capture
def download(dbpath, dataset):
    if dataset == 'QM9':
        qm9 = QM9(dbpath)


@ex.automain
def main():
    download()
    print(ex.config)
