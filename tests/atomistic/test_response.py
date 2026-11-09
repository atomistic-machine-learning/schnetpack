import numpy as np
import pytest
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


@pytest.mark.parametrize("response_cls", [spk.atomistic.Forces, spk.atomistic.Response])
def test_stress_mixed_cell_handedness(response_cls):
    positions = torch.tensor(
        [[0.2, -0.1, 0.4], [1.0, 0.6, -0.2], [0.1, 0.2, 0.3], [1.2, 0.0, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.3, 5.0, 0.0], [0.2, 0.4, 6.0]],
            [[0.3, 5.0, 0.0], [4.0, 0.0, 0.0], [0.2, 0.4, 12.0]],
        ],
        dtype=torch.float64,
    )
    inputs = {
        spk.properties.R: positions,
        spk.properties.cell: cells,
        spk.properties.idx_m: torch.tensor([0, 0, 1, 1]),
        spk.properties.idx_i: torch.tensor([0, 2]),
        spk.properties.idx_j: torch.tensor([1, 3]),
        spk.properties.offsets: torch.zeros(2, 3, dtype=torch.float64),
    }
    inputs = spk.atomistic.Strain()(inputs)
    inputs = spk.atomistic.PairwiseDistances()(inputs)
    spring_constant = 1.7
    inputs[spk.properties.energy] = (
        0.5 * spring_constant * inputs[spk.properties.Rij].square().sum(dim=-1)
    )
    response = (
        response_cls(calc_stress=True)
        if response_cls is spk.atomistic.Forces
        else response_cls(
            spk.properties.energy, [spk.properties.forces, spk.properties.stress]
        )
    )
    result = response(inputs)
    # A harmonic pair has virial k r r^T, normalized by the physical volume.
    displacement = positions[torch.tensor([1, 3])] - positions[torch.tensor([0, 2])]
    volumes = torch.tensor([120.0, 240.0], dtype=torch.float64)
    expected_stress = (
        spring_constant
        * displacement[:, :, None]
        * displacement[:, None, :]
        / volumes[:, None, None]
    )
    expected_forces = torch.stack(
        (
            spring_constant * displacement,
            -spring_constant * displacement,
        ),
        dim=1,
    ).reshape(4, 3)
    torch.testing.assert_close(result[spk.properties.stress], expected_stress)
    torch.testing.assert_close(result[spk.properties.forces], expected_forces)
