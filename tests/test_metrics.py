import pytest
import os
import numpy as np
import torch.nn as nn
from torch.optim.adam import Adam
import schnetpack as spk
from schnetpack.data import Structure
from schnetpack.metrics import *
from schnetpack.train.hooks import *
from schnetpack.datasets.qm9 import QM9
from schnetpack.atomistic import AtomisticModel, Atomwise
from schnetpack.representation.schnet import SchNet, SchNetInteraction


@pytest.fixture
def batchsize():
    return 4

@pytest.fixture
def data():
    return QM9(os.path.dirname(os.path.realpath(__file__)) + "/test_data/", properties=[QM9.U0], download=False)

@pytest.fixture
def filter_network(single_spatial_basis, n_filters):
    return nn.Linear(single_spatial_basis, n_filters)

@pytest.fixture
def mae():
    return MeanAbsoluteError(QM9.U0, 'y')

@pytest.fixture
def mse():
    return MeanSquaredError(QM9.U0, 'y')

@pytest.fixture
def reps():
    return SchNet()

@pytest.fixture
def output():
    return Atomwise()

@pytest.fixture
def model(reps, output):
    return AtomisticModel(reps, output)

class TestMetrics:

    def test_mae(self):
        pass

    def test_mse(self, mse, model, schnet_batch):
        result = model(schnet_batch)
        mse.add_batch(schnet_batch, result)
        error = mse.aggregate()

