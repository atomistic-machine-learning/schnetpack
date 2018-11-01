import pytest
import torch
import torch.nn as nn

from schnetpack.datasets import *
from schnetpack.data import Structure
from schnetpack.config_model import Hyperparameters
from schnetpack.representation.schnet import SchNet
import schnetpack.atomistic as atm
from tests.base_test import assert_params_changed


# Dummy classes
class Dummy(object):
    def __init__(self, dummy_a):
        self.dummy_a = dummy_a

class SubNetwork(Hyperparameters):

    def __init__(self, s1=1, s2=2, s3=3):
        Hyperparameters.__init__(self, locals())
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def __eq__(self, other):
        for attr in ['s1', 's2', 's3']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

subnetwork = SubNetwork()


class Network(Hyperparameters):

    def __init__(self, a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork):
        Hyperparameters.__init__(self, locals())
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dummy = dummy
        self.sub = sub

    def __eq__(self, other):
        for attr in ['a', 'b', 'c', 'd', 'dummy']:
            if getattr(self, attr) != getattr(other, attr):
                return False
            if self.sub != other.sub:
                return False
        return True

# Global variables
net = Network(c=13, d=14)
default_net = Network()
default_config = dict(a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork)
updated_config = dict(a=1, b=2, c=13, d=14, dummy=Dummy, sub=subnetwork)


# Testcases
class TestHyperparams(object):

    def test_init(self):
        assert net.a == 1
        assert net.b == 2
        assert net.c == 13
        assert net.d == 14

    def test_config_update(self):
        assert net.config == updated_config

    def test_empty_config_update(self):
        assert default_net.config == default_config

    def test_empty_config_init(self):
        for key in default_config.keys():
            assert getattr(default_net, key) == default_config[key]

    def test_dump_and_load(self):
        net.dump_config('config')
        net2 = Network().from_json('config')
        for attribute in ['a', 'b', 'c', 'd', 'config', 'default_config', 'dummy']:
            assert getattr(net, attribute) == getattr(net2, attribute)

    @classmethod
    def teardown_class(cls):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('config'):
            os.remove('config')

@pytest.fixture()
def schnet():
    return SchNet()

@pytest.fixture()
def mod_schnet():
    return SchNet(n_gaussians=999, n_filters=999)

@pytest.fixture
def batchsize():
    return 4


@pytest.fixture
def n_atom_basis():
    return 128


@pytest.fixture
def n_atoms():
    return 19


@pytest.fixture
def n_spatial_basis():
    return 25


@pytest.fixture
def single_spatial_basis():
    return 1


@pytest.fixture
def n_filters():
    return 128


@pytest.fixture
def atomic_env(batchsize, n_atoms, n_filters):
    return torch.rand((batchsize, n_atoms, n_filters))


@pytest.fixture
def atomic_numbers(batchsize, n_atoms):
    atoms = np.random.randint(1, 9, (1, n_atoms))
    return torch.LongTensor(np.repeat(atoms, batchsize, axis=0))


@pytest.fixture
def atomtypes(atomic_numbers):
    return set(atomic_numbers[0].data.numpy())


@pytest.fixture
def positions(batchsize, n_atoms):
    return torch.rand((batchsize, n_atoms, 3))


@pytest.fixture
def cell(batchsize):
    return torch.zeros((batchsize, 3, 3))


@pytest.fixture
def cell_offset(batchsize, n_atoms):
    return torch.zeros((batchsize, n_atoms, n_atoms - 1, 3))


@pytest.fixture
def neighbors(batchsize, n_atoms):
    neighbors = np.array([range(n_atoms)]*n_atoms)
    neighbors = neighbors[~np.eye(neighbors.shape[0], dtype=bool)].reshape(
        neighbors.shape[0], -1)[np.newaxis, :]
    return torch.LongTensor(np.repeat(neighbors, batchsize, axis=0))

@pytest.fixture
def atom_mask(atomic_numbers):
    return torch.zeros_like(atomic_numbers).float()

@pytest.fixture
def neighbor_mask(batchsize, n_atoms):
    return torch.ones((batchsize, n_atoms, n_atoms - 1))


@pytest.fixture
def schnet_batch(atomic_numbers, positions, cell, cell_offset, neighbors, neighbor_mask, atom_mask):
    inputs = {}
    inputs[Structure.Z] = atomic_numbers
    inputs[Structure.R] = positions
    inputs[Structure.cell] = cell
    inputs[Structure.cell_offset] = cell_offset
    inputs[Structure.neighbors] = neighbors
    inputs[Structure.neighbor_mask] = neighbor_mask
    inputs[Structure.atom_mask] = atom_mask
    return inputs


@pytest.fixture
def distances(batchsize, n_atoms):
    return torch.rand((batchsize, n_atoms, n_atoms - 1))


@pytest.fixture
def expanded_distances(batchsize, n_atoms, n_spatial_basis):
    return torch.rand((batchsize, n_atoms, n_atoms - 1, n_spatial_basis))


@pytest.fixture
def filter_network(single_spatial_basis, n_filters):
    return nn.Linear(single_spatial_basis, n_filters)

@pytest.fixture
def atomwise():
    return atm.Atomwise()

@pytest.fixture
def atomistic(schnet, atomwise):
    return atm.AtomisticModel(schnet, atomwise)


class TestSchnetHyperparams(object):

    def test_dump_and_load(self, schnet):
        schnet.dump_config('config')
        loaded_schnet = SchNet.from_json('config')
        assert schnet.config == loaded_schnet.config

    def test_dump_load_modified(self, mod_schnet):
        mod_schnet.dump_config('config')
        loaded_schnet = SchNet.from_json('config')
        assert mod_schnet.config == loaded_schnet.config

    def test_dump_load_train(self, mod_schnet, schnet_batch):
        mod_schnet.dump_config('config')
        schnet_from_json = SchNet.from_json('config')
        input_batch = [schnet_batch]
        assert_params_changed(schnet_from_json, input_batch, exclude=['distance_expansion',
                                                                      'interactions.0.cutoff_network',
                                                                      'interactions.0.cfconv.cutoff_network'])

    def teardown_method(self):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('config'):
            os.remove('config')
        if os.path.exists('before'):
            os.remove('before')


class TestAtomisticHyperparams(object):

    def test_dump_and_load(self, atomistic):
        atomistic.dump_config('config')
        model_from_json = atm.AtomisticModel.from_json('config')
        for key in atomistic.config.keys():
            assert atomistic.config[key].config == model_from_json.config[key].config

    def teardown_method(self):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('config'):
            os.remove('config')
        if os.path.exists('before'):
            os.remove('before')
