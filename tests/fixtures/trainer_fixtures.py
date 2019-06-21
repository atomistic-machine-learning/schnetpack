import pytest
import schnetpack as spk


@pytest.fixture(scope="session")
def loss_fn():
    pass


@pytest.fixture(scope="session")
def trainer(modeldir, qm9_model, loss_fn, optimizer):
    pass
