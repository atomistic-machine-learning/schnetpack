"""Metrics of generated crystals, after MatterGen: structural and compositional
validity, uniqueness and novelty by structure matching, the energy above the convex
hull of reference phases, and the displacement in a relaxation."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from schnetpack.generative.validation import (
    DisorderedMatcher,
    _structure,
    composition_valid,
    energy_above_hull,
    hull_reference,
    novel_crystals,
    relaxation_rmsd,
    structure_valid,
    unique_crystals,
)

pytest.importorskip("pymatgen")
pytest.importorskip("smact")


def rattled(atoms, stdev=0.05, seed=0):
    new = atoms.copy()
    new.rattle(stdev, seed=seed)
    return new


def layers(order):
    """Eight fcc sites in four (001) layers, each of one element of ``order``."""
    atoms = bulk("Cu", cubic=True, a=3.9) * (1, 1, 2)
    layer = np.round(atoms.positions[:, 2] / 1.95).astype(int)
    atoms.symbols = [order[k] for k in layer]
    return atoms


def test_a_built_crystal_is_structurally_valid():
    assert structure_valid(bulk("Si", cubic=True))
    assert structure_valid(bulk("Cu"))  # one atom: nothing to be close to


def across_the_boundary():
    atoms = bulk("Cu", "sc", a=3.0) * (2, 1, 1)
    atoms.positions[:, 0] = [0.1, 5.9]  # 5.8 Å apart, 0.2 Å over the boundary
    return atoms


def moved_together():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms.positions[1] = atoms.positions[0] + [0.3, 0.0, 0.0]
    return atoms


def flat_cell():
    atoms = bulk("Cu", "sc", a=3.0) * (2, 1, 1)
    atoms.set_cell([6.0, 3.0, 0.005])
    return atoms


def nan_positions():
    atoms = bulk("Si", cubic=True)
    atoms.positions[0, 0] = np.nan
    return atoms


@pytest.mark.parametrize(
    "atoms",
    [moved_together(), across_the_boundary(), flat_cell(), nan_positions()],
    ids=["two atoms 0.3 Å apart", "close over the boundary", "no volume", "nan"],
)
def test_a_broken_crystal_is_structurally_invalid(atoms):
    assert not structure_valid(atoms)


def test_structural_validity_takes_the_smallest_allowed_distance():
    atoms = moved_together()
    assert structure_valid(atoms, min_distance=0.2)


@pytest.mark.parametrize(
    "formula, valid",
    [
        ("NaCl", True),  # Na+ Cl-
        ("Na2Cl2", True),  # the reduced formula counts
        ("NaCl2", False),  # no neutral assignment
        ("Si4", True),  # one element
        ("CuZn", True),  # only metals
    ],
)
def test_composition_is_valid_if_a_neutral_assignment_exists(formula, valid):
    assert composition_valid(Atoms(formula)) == valid


def test_compositional_validity_depends_on_the_list_of_oxidation_states():
    # judged differently by SMACT 4's default list and SMACT 2's
    assert not composition_valid(Atoms("P2Pt"))
    assert composition_valid(Atoms("P2Pt"), oxidation_states="smact14")
    assert composition_valid(Atoms("TaSe3"))
    assert not composition_valid(Atoms("TaSe3"), oxidation_states="smact14")


def test_too_many_assignments_count_as_invalid():
    assert not composition_valid(Atoms("NaCl"), max_combinations=1)


def test_the_matcher_fits_a_crystal_to_its_distorted_copy_and_supercell():
    matcher = DisorderedMatcher()
    silicon = _structure(bulk("Si", cubic=True))
    assert matcher.fit(silicon, _structure(rattled(bulk("Si", cubic=True))))
    assert matcher.fit(silicon, _structure(bulk("Si", cubic=True) * (2, 1, 1)))
    assert not matcher.fit(silicon, _structure(bulk("Si", "fcc", a=3.9)))
    # another formula never fits
    assert not matcher.fit(
        _structure(bulk("NaCl", "rocksalt", a=5.64)),
        _structure(bulk("KCl", "rocksalt", a=5.64)),
    )


def test_the_matcher_takes_two_orderings_of_an_alloy_as_one_structure():
    matcher = DisorderedMatcher()
    # Cu and Au substitute each other (same radius, electronegativities 0.64 apart)
    one, two = layers(["Cu", "Au", "Cu", "Au"]), layers(["Cu", "Cu", "Au", "Au"])
    assert not matcher.ordered.fit(_structure(one), _structure(two))
    assert matcher.fit(_structure(one), _structure(two))
    # Na and Cl do not
    one, two = layers(["Na", "Cl", "Na", "Cl"]), layers(["Na", "Na", "Cl", "Cl"])
    assert not matcher.fit(_structure(one), _structure(two))


def test_unique_marks_the_first_of_each_set_of_duplicates():
    silicon, salt = bulk("Si", cubic=True), bulk("NaCl", "rocksalt", a=5.64)
    crystals = [silicon, rattled(silicon), salt, silicon * (2, 1, 1), rattled(salt)]
    assert unique_crystals(crystals).tolist() == [True, False, True, False, False]


def test_novel_marks_the_crystals_that_match_no_known_one():
    silicon, salt = bulk("Si", cubic=True), bulk("NaCl", "rocksalt", a=5.64)
    known = [rattled(silicon), bulk("KCl", "rocksalt", a=6.3)]
    assert novel_crystals([silicon, salt], known).tolist() == [False, True]
    assert novel_crystals([silicon, salt], []).tolist() == [True, True]


def test_the_matcher_is_an_argument():
    from pymatgen.analysis.structure_matcher import StructureMatcher

    one, two = layers(["Cu", "Au", "Cu", "Au"]), layers(["Cu", "Cu", "Au", "Au"])
    assert unique_crystals([one, two]).tolist() == [True, False]
    assert unique_crystals([one, two], matcher=StructureMatcher()).tolist() == [
        True,
        True,
    ]
    assert novel_crystals([one], [two], matcher=StructureMatcher()).tolist() == [True]


def test_hull_reference_keeps_the_lowest_energy_per_atom_of_each_formula():
    crystals = [Atoms("Na"), Atoms("Na2"), Atoms("Cl2"), Atoms("NaCl"), Atoms("Na2Cl2")]
    reference = hull_reference(crystals, [0.5, 0.0, 0.0, -2.0, -3.0])
    assert reference == {"Na1": 0.0, "Cl1": 0.0, "Na1Cl1": -1.0}


def test_energy_above_hull_is_measured_from_the_most_stable_mixture():
    reference = {"Na1": 0.0, "Cl1": 0.0, "Na1Cl1": -1.0}
    crystals = [
        Atoms("NaCl"),  # -0.8 per atom against the phase itself at -1.0
        Atoms("Na3Cl"),  # the hull is NaCl + 2 Na: -0.5 per atom
        Atoms("Na2Cl2"),  # below the hull
        Atoms("Na"),
        Atoms("NaK"),  # no K in the reference
        Atoms("NaCl"),  # no energy
    ]
    energies = [-1.6, -1.2, -4.4, 0.3, 0.0, np.nan]
    above = energy_above_hull(crystals, energies, reference)
    np.testing.assert_allclose(above[:4], [0.2, 0.2, -0.1, 0.3], atol=1e-12)
    assert np.isnan(above[4:]).all()


def test_relaxation_rmsd_is_the_displacement_of_the_atoms():
    relaxed = bulk("NaCl", "rocksalt", a=5.64)
    generated = rattled(relaxed, seed=3)
    moved = generated.positions - relaxed.positions
    moved -= moved.mean(axis=0)
    expected = np.sqrt((moved**2).sum(axis=1).mean())
    assert relaxation_rmsd(generated, relaxed) == pytest.approx(expected, rel=1e-6)
    assert relaxation_rmsd(relaxed, relaxed) == pytest.approx(0.0, abs=1e-6)


def test_relaxation_rmsd_is_capped_where_nothing_aligns():
    # the cell grows by 80 %, beyond the matcher's 50 %: half of (volume per
    # atom)^(1/3) of the mean cell, 0.5 * (3.0 + 5.4) / 2
    assert relaxation_rmsd(bulk("Cu", "sc", a=5.4), bulk("Cu", "sc", a=3.0)) == (
        pytest.approx(2.1)
    )


def test_relaxation_rmsd_misses_the_rattle_of_a_supercell():
    # MatterGen's measure: the primitive-cell reduction (0.25 Å) averages it away
    supercell = bulk("Cu", cubic=True) * (2, 2, 2)
    assert relaxation_rmsd(rattled(supercell), supercell) == pytest.approx(0.0)
