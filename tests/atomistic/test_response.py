import numpy as np
import torch
from ase.build import molecule

import schnetpack as spk
from schnetpack.data.loader import _atoms_collate_fn


def test_strain(environment_periodic):
    cutoff, props, neighbors = environment_periodic
    props.update(neighbors)
    batch = _atoms_collate_fn([props, props])
    strained_batch = spk.atomistic.Strain()(batch)
    assert np.allclose(
        batch[spk.properties.offsets].detach().numpy(),
        strained_batch[spk.properties.offsets].detach().numpy(),
    )


def _response_model(map_properties):
    torch.manual_seed(0)
    representation = spk.representation.PaiNN(
        n_atom_basis=8,
        n_interactions=1,
        radial_basis=spk.nn.GaussianRBF(n_rbf=5, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0),
    )
    return spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.Atomwise(n_in=8, output_key="energy"),
            spk.atomistic.Response(
                "energy", [spk.properties.forces], map_properties=map_properties
            ),
        ],
    )


def test_response_returns_renamed_properties():
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=spk.transform.ASENeighborList(cutoff=5.0)
    )
    inputs = converter(molecule("H2O"))

    renamed = _response_model({spk.properties.forces: "my_forces"})(dict(inputs))
    plain = _response_model(None)(dict(inputs))

    assert set(renamed) == {"energy", "my_forces"}
    torch.testing.assert_close(renamed["my_forces"], plain[spk.properties.forces])


def test_response_leaves_the_property_map_of_the_caller_alone():
    map_properties = {spk.properties.forces: "my_forces"}
    spk.atomistic.Response(
        "energy",
        [spk.properties.forces, spk.properties.hessian],
        map_properties=map_properties,
    )
    assert map_properties == {spk.properties.forces: "my_forces"}
