import pytest
from src.schnetpack.data import AtomsLoader
from .dataset_fixtures import *
import schnetpack as spk


__all__ = [
    "batch_size",
    "shuffle",
    "dataloader",
    "qm9_loader",
    "qm9_train_loader",
    "qm9_val_loader",
    "qm9_test_loader",
]


@pytest.fixture(scope="session")
def batch_size():
    return 4


@pytest.fixture(scope="session")
def shuffle():
    return True


@pytest.fixture(scope="session")
def dataloader(dataset, batch_size, shuffle):
    return AtomsLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="module")
def qm9_loader(qm9_dataset, batch_size, shuffle):
    return AtomsLoader(qm9_dataset, batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="module")
def qm9_train_loader(qm9_splits, batch_size, shuffle):
    return AtomsLoader(qm9_splits[0], batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="module")
def qm9_val_loader(qm9_splits, batch_size, shuffle):
    return AtomsLoader(qm9_splits[1], batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="module")
def qm9_test_loader(qm9_splits, batch_size, shuffle):
    return AtomsLoader(qm9_splits[2], batch_size=batch_size, shuffle=shuffle)
