import numpy as np
import pytest
import torch
from ase.build import bulk
from scipy.spatial.transform import Rotation

import schnetpack.nn as snn
from schnetpack import properties
from schnetpack.interfaces import AtomsConverter
from schnetpack.model import (
    AtomwiseTensor,
    PaiNN,
    PairwiseDistances,
    PairwiseTensor,
    SO3net,
)
from schnetpack.transform import MatScipyNeighborList

CUTOFF = 5.0
STRAIN = np.eye(3) + [[0.05, 0.03, 0.0], [0.03, -0.04, 0.02], [0.0, 0.02, 0.01]]


def strained(atoms):
    atoms.set_cell(atoms.cell @ STRAIN, scale_atoms=True)
    return atoms


def model(kind, trained=True, lmax=2):
    """An untrained representation with a tensor head, as one function of atoms."""
    torch.manual_seed(0)
    basis = {
        "radial_basis": snn.GaussianRBF(n_rbf=10, cutoff=CUTOFF),
        "cutoff_fn": snn.CosineCutoff(CUTOFF),
    }
    if kind == "atomwise":
        representation = SO3net(n_atom_basis=16, n_interactions=2, lmax=lmax, **basis)
        head = AtomwiseTensor(n_in=16)
        last = head.outnet[-1]
    else:
        representation = PaiNN(n_atom_basis=16, n_interactions=2, **basis)
        head = PairwiseTensor(n_in=16, **basis)
        last = head.center_net
    if trained:  # away from the zero start
        torch.nn.init.normal_(last.weight, std=0.1)
    converter = AtomsConverter(MatScipyNeighborList(cutoff=CUTOFF), dtype=torch.float64)
    representation, head = representation.double(), head.double()

    def predict(atoms, pad=False):
        batch = PairwiseDistances()(converter(atoms))
        if pad:  # a free slot as the cache pads it: no atoms
            n_atoms = batch[properties.n_atoms]
            batch[properties.n_atoms] = torch.cat([n_atoms, torch.tensor([0])])
        return head(representation(batch))

    return predict


KINDS = ["atomwise", "pairwise"]


@pytest.mark.parametrize("kind", KINDS)
def test_an_untrained_head_returns_zero(kind):
    y = model(kind, trained=False)(strained(bulk("Si", "diamond", a=5.43)))["y"]
    assert y.shape == (1, 3, 3) and y.abs().max() == 0


@pytest.mark.parametrize("kind", KINDS)
def test_the_tensor_is_symmetric_and_turns_with_the_crystal(kind):
    atoms = strained(bulk("ZnS", "zincblende", a=5.41))
    q = Rotation.random(random_state=1).as_matrix()
    turned = atoms.copy()
    turned.set_cell(atoms.cell @ q.T, scale_atoms=True)
    predict = model(kind)
    y, y_q = predict(atoms)["y"][0], predict(turned)["y"][0]
    q = torch.tensor(q)
    assert y.abs().max() > 1e-3
    torch.testing.assert_close(y, y.T)
    torch.testing.assert_close(y_q, q @ y @ q.T, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "atoms",
    [bulk("NaCl", "rocksalt", a=5.64), bulk("Cu", "fcc", a=3.6)],
    ids=["rocksalt", "fcc"],
)
def test_the_tensor_has_a_shear_where_the_vector_features_vanish(kind, atoms):
    # every atom sits on an inversion center, in the strained cell too
    out = model(kind)(strained(atoms))
    vectors = (
        out["multipole_representation"][:, 1:4]
        if kind == "atomwise"
        else out["vector_representation"]
    )
    assert vectors.abs().max() < 1e-12
    y = out["y"][0]
    shear = y - torch.eye(3, dtype=y.dtype) * y.diagonal().mean()
    assert shear.abs().max() > 1e-3


@pytest.mark.parametrize("kind", KINDS)
def test_the_tensor_is_the_same_for_a_supercell(kind):
    atoms = strained(bulk("Si", "diamond", a=5.43))
    predict = model(kind)
    torch.testing.assert_close(
        predict(atoms.repeat((2, 1, 1)))["y"], predict(atoms)["y"], atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("kind", KINDS)
def test_an_empty_structure_of_a_padded_batch_gets_zero(kind):
    atoms = strained(bulk("Si", "diamond", a=5.43))
    predict = model(kind)
    y = predict(atoms, pad=True)["y"]
    assert y.shape == (2, 3, 3) and y[1].abs().max() == 0
    torch.testing.assert_close(y[:1], predict(atoms)["y"])


def test_the_atomwise_tensor_needs_l2_features():
    with pytest.raises(ValueError, match="lmax >= 2"):
        model("atomwise", lmax=1)(bulk("Si", "diamond", a=5.43))
