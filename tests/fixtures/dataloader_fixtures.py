import pytest
from src.schnetpack.data import AtomsLoader
from .dataset_fixtures import *


@pytest.fixture(scope="session")
def batch_size():
    return 4


@pytest.fixture(scope="session")
def shuffle():
    return True


@pytest.fixture(scope="session")
def dataloader(dataset, batch_size, shuffle):
    return AtomsLoader(dataset, batch_size=batch_size, shuffle=shuffle)
