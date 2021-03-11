import os
import torch
import schnetpack as spk


from tests.assertions import assert_output_shape_valid, assert_params_changed
from tests.fixtures import *


# test if parameters of model building blocks are updated
def test_parameter_update_schnet(
    schnet, schnet_batch, n_interactions, trainable_gaussians
):
    # define layers that are not updated
    exclude = [
        "interactions.{}.cutoff_network".format(i) for i in range(n_interactions)
    ] + [
        "interactions.{}.cfconv.cutoff_network".format(i) for i in range(n_interactions)
    ]
    # if not trainable gaussians, exclude distance expansion layer
    if not trainable_gaussians:
        exclude += ["distance_expansion"]

    assert_params_changed(
        schnet,
        schnet_batch,
        exclude=exclude,
    )


# def dont_test_parameter_update_schnet_with_cutoff(
#    schnet_batch, n_atom_basis, n_interactions
# ):
#    model_cosine = spk.SchNet(
#        n_atom_basis, n_interactions=n_interactions, cutoff_network=CosineCutoff
#    )
#    model_mollifier = spk.SchNet(
#        n_atom_basis, n_interactions=n_interactions, cutoff_network=MollifierCutoff
#    )
#    exclude = [
#        "distance_expansion",
#        "interactions.0.cutoff_network",
#        "interactions.0.cfconv.cutoff_network",
#    ]
#
#    assert_params_changed(model_cosine, schnet_batch, exclude=exclude)
#    assert_params_changed(model_mollifier, schnet_batch, exclude=exclude)


# test shapes of spk.representation
def test_shape_schnet(schnet, schnet_batch, schnet_output_shape):
    assert_output_shape_valid(schnet, [schnet_batch], schnet_output_shape)


def test_shape_schnetinteraction(
    schnet_interaction,
    random_atomic_env,
    r_ij,
    neighbors,
    neighbor_mask,
    f_ij,
    interaction_output_shape,
):
    inputs = [random_atomic_env, r_ij, neighbors, neighbor_mask, f_ij]
    assert_output_shape_valid(schnet_interaction, inputs, interaction_output_shape)


# test shapes of spk.nn
def test_shape_cfconv(
    cfconv_layer,
    random_atomic_env,
    r_ij,
    neighbors,
    neighbor_mask,
    f_ij,
    cfconv_output_shape,
):
    inputs = [random_atomic_env, r_ij, neighbors, neighbor_mask, f_ij]
    assert_output_shape_valid(cfconv_layer, inputs, cfconv_output_shape)


def test_gaussian_smearing(
    gaussion_smearing_layer, random_interatomic_distances, gaussian_smearing_shape
):
    assert_output_shape_valid(
        gaussion_smearing_layer, [random_interatomic_distances], gaussian_smearing_shape
    )


def test_shape_dense(dense_layer, random_float_input, random_shape, random_output_dim):
    out_shape = random_shape[:-1] + [random_output_dim]
    assert_output_shape_valid(dense_layer, [random_float_input], out_shape)


def test_shape_scale_shift(random_float_input, random_shape):
    mean = torch.rand(1)
    std = torch.rand(1)
    model = spk.nn.ScaleShift(mean, std)

    assert_output_shape_valid(model, [random_float_input], random_shape)


def test_shape_standardize(random_float_input, random_shape):
    mean = torch.rand(1)
    std = torch.rand(1)
    model = spk.nn.Standardize(mean, std)

    assert_output_shape_valid(model, [random_float_input], random_shape)


def test_shape_aggregate():
    model = spk.nn.Aggregate(axis=1)
    input_data = torch.rand((3, 4, 5))
    inputs = [input_data]
    out_shape = [3, 5]
    assert_output_shape_valid(model, inputs, out_shape)


def test_shape_mlp(mlp_layer, random_float_input, random_shape, random_output_dim):
    out_shape = random_shape[:-1] + [random_output_dim]
    assert_output_shape_valid(mlp_layer, [random_float_input], out_shape)


def test_shape_tiled_multilayer_network(
    tiled_mlp_layer, n_mlp_tiles, random_float_input, random_shape, random_output_dim
):
    out_shape = random_shape[:-1] + [random_output_dim * n_mlp_tiles]
    assert_output_shape_valid(tiled_mlp_layer, [random_float_input], out_shape)


def test_shape_elemental_gate(
    elemental_gate_layer,
    elements,
    random_int_input,
    random_shape,
):
    out_shape = random_shape + [len(elements)]
    assert_output_shape_valid(elemental_gate_layer, [random_int_input], out_shape)


def test_shape_cutoff(cutoff_layer, random_interatomic_distances):
    out_shape = list(random_interatomic_distances.shape)
    assert_output_shape_valid(cutoff_layer, [random_interatomic_distances], out_shape)


# functionality tests
def test_get_item(schnet_batch):
    for key, value in schnet_batch.items():
        get_item = spk.nn.GetItem(key)
        assert torch.all(torch.eq(get_item(schnet_batch), value))


def test_functionality_cutoff(cutoff_layer, cutoff, random_interatomic_distances):
    mask = random_interatomic_distances > cutoff
    cutoff_layer_mask = cutoff_layer(random_interatomic_distances)

    assert ((cutoff_layer_mask == 0.0) == mask).all()


def x_test_shape_neighbor_elements(atomic_numbers, neighbors):
    # ToDo: change Docstring or squeeze()
    model = spk.nn.NeighborElements()
    inputs = [atomic_numbers.unsqueeze(-1), neighbors]
    out_shape = list(neighbors.shape)
    assert_output_shape_valid(model, inputs, out_shape)


def teardown_module():
    """
    Remove artifacts that have been created during testing.
    """
    if os.path.exists("before"):
        os.remove("before")


def test_charge_correction(schnet_batch, n_atom_basis):
    """
    Test if charge correction yields the desired total charges.

    """
    model = spk.AtomisticModel(
        spk.SchNet(n_atom_basis),
        spk.atomistic.DipoleMoment(
            n_atom_basis, charge_correction="q", contributions="q"
        ),
    )
    q = torch.randint(0, 10, (schnet_batch["_positions"].shape[0], 1))
    schnet_batch.update(q=q)

    q_i = model(schnet_batch)["q"]

    assert torch.allclose(q.float(), q_i.sum(1), atol=1e-6)
