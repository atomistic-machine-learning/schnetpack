import pytest
from src.schnetpack.data import AtomsLoader
from .dataset_fixtures import *
import schnetpack as spk


@pytest.fixture(scope="session")
def batch_size():
    return 4


@pytest.fixture(scope="session")
def shuffle():
    return True


@pytest.fixture(scope="session")
def dataloader(dataset, batch_size, shuffle):
    return AtomsLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="session")
def qm9_loader(qm9_dataset, batch_size, shuffle):
    return AtomsLoader(qm9_dataset, batch_size=batch_size, shuffle=shuffle)
