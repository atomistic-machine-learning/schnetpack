import os

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.modules import MSELoss
from torch.optim import Adam

from schnetpack import Properties
from schnetpack.nn.acsf import GaussianSmearing
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.base import Dense, GetItem, ScaleShift, Standardize, Aggregate
from schnetpack.nn.blocks import MLP, TiledMultiLayerNN, ElementalGate
from schnetpack.nn.cfconv import CFConv
from schnetpack.nn.cutoff import CosineCutoff, MollifierCutoff, HardCutoff
from schnetpack.nn.neighbors import NeighborElements
from schnetpack.representation.schnet import SchNet, SchNetInteraction


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
def n_interactions():
    return 1


@pytest.fixture
def atomic_env(batchsize, n_atoms, n_filters):
    return torch.rand((batchsize, n_atoms, n_filters))


@pytest.fixture
def atomic_numbers(batchsize, n_atoms):
    atoms = np.random.randint(1, 9, (1, n_atoms))
    return torch.LongTensor(np.repeat(atoms, batchsize, axis=0))


@pytest.fixture
def atom_mask(atomic_numbers):
    return torch.ones(atomic_numbers.shape)


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
    neighbors = np.array([range(n_atoms)] * n_atoms)
    neighbors = neighbors[~np.eye(neighbors.shape[0], dtype=bool)].reshape(
        neighbors.shape[0], -1
    )[np.newaxis, :]
    return torch.LongTensor(np.repeat(neighbors, batchsize, axis=0))


@pytest.fixture
def neighbor_mask(batchsize, n_atoms):
    return torch.ones((batchsize, n_atoms, n_atoms - 1))


@pytest.fixture
def schnet_batch(
    atomic_numbers, atom_mask, positions, cell, cell_offset, neighbors, neighbor_mask
):
    inputs = {}
    inputs[Properties.Z] = atomic_numbers
    inputs[Properties.R] = positions
    inputs[Properties.cell] = cell
    inputs[Properties.cell_offset] = cell_offset
    inputs[Properties.neighbors] = neighbors
    inputs[Properties.neighbor_mask] = neighbor_mask
    inputs[Properties.atom_mask] = atom_mask
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


def assert_params_changed(model, input, exclude=[]):
    """
    Check if all model-parameters are updated when training.

    Args:
        model (torch.nn.Module): model to test
        data (torch.utils.data.Dataset): input dataset
        exclude (list): layers that are not necessarily updated
    """
    # save state-dict
    torch.save(model.state_dict(), "before")
    # do one training step
    optimizer = Adam(model.parameters())
    loss_fn = MSELoss()
    pred = model(*input)
    loss = loss_fn(pred, torch.rand(pred.shape))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # check if all trainable parameters have changed
    after = model.state_dict()
    before = torch.load("before")
    for key in before.keys():
        if np.array([key.startswith(exclude_layer) for exclude_layer in exclude]).any():
            continue
        assert (
            before[key] != after[key]
        ).any(), "{} layer has not been updated!".format(key)


def assert_equal_shape(model, batch, out_shape):
    """
    Check if the model returns the desired output shape.

    Args:
        model (nn.Module): model that needs to be tested
        batch (list): input data
        out_shape (list): desired output shape
    """
    pred = model(*batch)
    assert list(pred.shape) == out_shape, "Model does not return expected shape!"


def test_parameter_update_schnet(schnet_batch, n_interactions):
    model = SchNet(n_interactions=n_interactions)
    schnet_batch = [schnet_batch]
    assert_params_changed(
        model,
        schnet_batch,
        exclude=[
            "distance_expansion",
            "interactions.0.cutoff_network",
            "interactions.0.cfconv.cutoff_network",
        ],
    )


def test_parameter_update_schnet_with_cutoff(
    schnet_batch, n_atom_basis, n_interactions
):
    model_cosine = SchNet(
        n_atom_basis, n_interactions=n_interactions, cutoff_network=CosineCutoff
    )
    model_mollifier = SchNet(
        n_atom_basis, n_interactions=n_interactions, cutoff_network=MollifierCutoff
    )
    schnet_batch = [schnet_batch]
    exclude = [
        "distance_expansion",
        "interactions.0.cutoff_network",
        "interactions.0.cfconv.cutoff_network",
    ]

    assert_params_changed(model_cosine, schnet_batch, exclude=exclude)
    assert_params_changed(model_mollifier, schnet_batch, exclude=exclude)


def test_gaussian_smearing_is_trainable(schnet_batch, n_interactions):
    model = SchNet(n_interactions=n_interactions, trainable_gaussians=True)
    schnet_batch = [schnet_batch]
    assert_params_changed(
        model,
        schnet_batch,
        exclude=[
            "interactions.0.cutoff_network",
            "interactions.0.cfconv.cutoff_network",
        ],
    )


def test_shape_schnet(schnet_batch, batchsize, n_atoms, n_atom_basis):
    schnet_batch = [schnet_batch]
    model = SchNet(n_atom_basis=n_atom_basis)

    assert_equal_shape(model, schnet_batch, [batchsize, n_atoms, n_atom_basis])


def test_shape_schnet_with_cutoff(schnet_batch, batchsize, n_atoms, n_atom_basis):
    schnet_batch = [schnet_batch]
    model_cosine = SchNet(n_atom_basis=n_atom_basis, cutoff_network=CosineCutoff)
    model_mollifier = SchNet(n_atom_basis=n_atom_basis, cutoff_network=MollifierCutoff)

    assert_equal_shape(model_cosine, schnet_batch, [batchsize, n_atoms, n_atom_basis])
    assert_equal_shape(
        model_mollifier, schnet_batch, [batchsize, n_atoms, n_atom_basis]
    )


def test_shape_schnetinteraction(
    batchsize,
    n_atoms,
    n_atom_basis,
    single_spatial_basis,
    n_filters,
    atomic_env,
    distances,
    neighbors,
    neighbor_mask,
):
    model = SchNetInteraction(n_atom_basis, single_spatial_basis, n_filters, 5.0)
    out_shape = [batchsize, n_atoms, n_filters]
    inputs = [atomic_env, distances, neighbors, neighbor_mask]
    assert_equal_shape(model, inputs, out_shape)


def test_shape_cfconv(
    batchsize,
    n_atom_basis,
    n_filters,
    filter_network,
    atomic_env,
    distances,
    neighbors,
    neighbor_mask,
    n_atoms,
):
    model = CFConv(n_atom_basis, n_filters, n_atom_basis, filter_network)
    out_shape = [batchsize, n_atoms, n_atom_basis]
    inputs = [atomic_env, distances, neighbors, neighbor_mask]
    assert_equal_shape(model, inputs, out_shape)


def test_gaussian_smearing(n_spatial_basis, distances):
    model = GaussianSmearing(n_gaussians=n_spatial_basis)
    out_shape = [*list(distances.shape), n_spatial_basis]
    inputs = [distances]
    assert_equal_shape(model, inputs, out_shape)


def test_shape_ssp():
    in_data = torch.rand(10)
    out_data = shifted_softplus(in_data)
    assert in_data.shape == out_data.shape


def test_shape_dense(expanded_distances):
    out_shape = [*list(expanded_distances.shape)[:-1], 10]
    model = Dense(expanded_distances.shape[-1], out_shape[-1])
    inputs = [expanded_distances]
    assert_equal_shape(model, inputs, out_shape)


def test_get_item(schnet_batch):
    model = GetItem(Properties.R)
    assert torch.all(torch.eq(model(schnet_batch), schnet_batch[Properties.R]))


def test_shape_scale_shift():
    mean = torch.rand(1)
    std = torch.rand(1)
    model = ScaleShift(mean, std)
    input_data = torch.rand((3, 4, 5))
    inputs = [input_data]
    assert_equal_shape(model, inputs, list(input_data.shape))


def test_shape_standardize():
    mean = torch.rand(1)
    std = torch.rand(1)
    model = Standardize(mean, std)
    input_data = torch.rand((3, 4, 5))
    inputs = [input_data]
    assert_equal_shape(model, inputs, list(input_data.shape))


def test_shape_aggregate():
    model = Aggregate(axis=1)
    input_data = torch.rand((3, 4, 5))
    inputs = [input_data]
    out_shape = [3, 5]
    assert_equal_shape(model, inputs, out_shape)


def test_shape_mlp():
    input_data = torch.rand((3, 4, 5))
    inputs = [input_data]
    out_shape = [3, 4, 10]
    model = MLP(input_data.shape[-1], out_shape[-1])
    assert_equal_shape(model, inputs, out_shape)


def test_shape_tiled_multilayer_network():
    input_data = torch.rand((3, 4, 5))
    inputs = [input_data]
    out = 10
    tiles = 3
    out_shape = [3, 4, out * tiles]
    model = TiledMultiLayerNN(input_data.shape[-1], out, tiles)
    assert_equal_shape(model, inputs, out_shape)


def test_shape_elemental_gate(batchsize, n_atoms, atomtypes, atomic_numbers):
    model = ElementalGate(atomtypes)
    input_data = atomic_numbers
    inputs = [input_data]
    out_shape = [batchsize, n_atoms, len(atomtypes)]
    assert_equal_shape(model, inputs, out_shape)


def test_shape_cutoffs(distances):
    cosine = CosineCutoff()
    mollifier = MollifierCutoff()
    hard = HardCutoff()
    inputs = [distances]
    out_shape = list(distances.shape)

    assert_equal_shape(cosine, inputs, out_shape)
    assert_equal_shape(mollifier, inputs, out_shape)
    assert_equal_shape(hard, inputs, out_shape)


def test_hard_cutoff_functionality(batchsize, n_atoms):
    cutoff = 20
    hard = HardCutoff(cutoff=cutoff)
    distances = (
        torch.arange(1, batchsize * n_atoms * (n_atoms - 1) + 1)
        .reshape(batchsize, n_atoms, n_atoms - 1)
        .float()
    )

    assert torch.sum(hard(distances) != 0) == cutoff


def x_test_shape_neighbor_elements(atomic_numbers, neighbors):
    # ToDo: change Docstring or squeeze()
    model = NeighborElements()
    inputs = [atomic_numbers.unsqueeze(-1), neighbors]
    out_shape = list(neighbors.shape)
    assert_equal_shape(model, inputs, out_shape)


def teardown_module():
    """
    Remove artifacts that have been created during testing.
    """
    if os.path.exists("before"):
        os.remove("before")
