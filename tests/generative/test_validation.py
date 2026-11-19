"""Metrics of generated molecules: validity by RDKit's bonds at charge 0 (no
radicals, one fragment), and uniqueness, novelty, diversity and distribution
distances over the valid molecules."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from schnetpack.generative.validation import (
    elongation,
    js_divergence,
    mean_tanimoto_similarity,
    molecule_smiles,
    novelty,
    posebusters_valid,
    uniqueness,
)

pytest.importorskip("rdkit")


def two_fragments():
    a, b = molecule("CH4"), molecule("H2O")
    b.translate([10.0, 0.0, 0.0])
    return a + b


def test_valid_molecules_give_their_canonical_smiles():
    assert molecule_smiles(molecule("CH3CH2OH")) == "CCO"
    assert molecule_smiles(molecule("H2O")) == "O"


@pytest.mark.parametrize(
    "atoms",
    [
        molecule("CH3"),  # a radical
        two_fragments(),
        Atoms("CH4", positions=[[0, 0, 0]] * 5),  # all atoms in one place
    ],
    ids=["radical", "two fragments", "collapsed"],
)
def test_invalid_molecules_give_none(atoms):
    assert molecule_smiles(atoms) is None


def test_set_metrics_count_the_valid_molecules_only():
    smiles = ["CCO", "CCO", "O", None, "C"]
    assert uniqueness(smiles) == 3 / 4
    assert novelty(smiles, known={"CCO"}) == 2 / 4
    assert np.isnan(uniqueness([None])) and np.isnan(novelty([None], set()))


def test_tanimoto_similarity_is_one_for_copies_and_below_for_others():
    assert mean_tanimoto_similarity(["CCO", "CCO", None]) == 1.0
    mixed = mean_tanimoto_similarity(["CCO", "CCO", "c1ccccc1"])
    assert 0.0 < mixed < 1.0
    assert np.isnan(mean_tanimoto_similarity(["CCO"]))


def test_elongation_is_the_largest_distance():
    water = molecule("H2O")
    assert elongation(water) == pytest.approx(water.get_distance(1, 2))
    assert elongation(Atoms("H")) == 0.0


def test_js_divergence_is_zero_for_one_distribution_and_one_for_disjoint_ones():
    x = np.random.default_rng(0).normal(size=10000)
    assert js_divergence(x, x) == 0.0
    assert js_divergence(x, x + 100.0) == pytest.approx(1.0)


def test_posebusters_accepts_a_built_molecule_and_refuses_a_collapsed_one():
    pytest.importorskip("posebusters")
    assert posebusters_valid(molecule("CH3CH2OH"))
    assert not posebusters_valid(Atoms("CH4", positions=[[0, 0, 0]] * 5))
