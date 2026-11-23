"""
Metrics of generated molecules. Per molecule:

- :func:`molecule_smiles`: the canonical SMILES of a valid molecule, else None.
  Validity is the fraction of molecules with one.
- :func:`posebusters_valid`: whether PoseBusters' ``mol`` checks pass.
- :func:`elongation`: the largest distance between two atoms.

Over a sample set, from the valid molecules' SMILES (None entries skipped):

- :func:`uniqueness`: the fraction of distinct SMILES.
- :func:`novelty`: the fraction not among known SMILES, e.g. the training
  molecules.
- :func:`mean_tanimoto_similarity`: the mean pairwise fingerprint similarity.

And :func:`js_divergence` between a property's distribution over the valid molecules
and a reference one, e.g. the elongation over the training molecules.

The definitions follow Hessmann et al., Generative Pseudo-Force Fields for Molecular
Generation, arXiv:2605.19050 (2026); :func:`posebusters_valid` runs the PoseBusters
checks (Buttenschoen et al., 2024).

Needs RDKit (``pip install schnetpack[rdkit]``); :func:`posebusters_valid` also
PoseBusters (``pip install posebusters``).

Metrics of generated crystals, after MatterGen (Zeni et al., Nature 639, 624, 2025;
its ``mattergen.evaluation``). Per crystal:

- :func:`structure_valid`: no two atoms too close and a cell with a volume.
- :func:`composition_valid`: a charge-neutral assignment of oxidation states that
  respects the electronegativities exists (SMACT).
- :func:`relaxation_rmsd`: the displacement of the atoms in a relaxation.

Over a sample set, as one bool or number per crystal:

- :func:`unique_crystals`: the first crystal of each set of duplicates.
- :func:`novel_crystals`: the crystals that match no known one, e.g. no training
  crystal.
- :func:`energy_above_hull`: the energy above the convex hull of reference phases
  (:func:`hull_reference`); a crystal is called stable up to 0.1 eV per atom, and
  S.U.N. if it is stable, unique and novel.

Two crystals are the same if :class:`DisorderedMatcher` fits them. The energies come
from a relaxation, by DFT or a force field, and must be on the footing of the
reference's.

Needs pymatgen and smact (``pip install schnetpack[crystals]``).
"""

import signal
from collections.abc import Iterable, Mapping, Sequence
from io import StringIO
from itertools import combinations, product

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write

__all__ = [
    "molecule_smiles",
    "posebusters_valid",
    "elongation",
    "uniqueness",
    "novelty",
    "mean_tanimoto_similarity",
    "js_divergence",
    "structure_valid",
    "composition_valid",
    "DisorderedMatcher",
    "unique_crystals",
    "novel_crystals",
    "hull_reference",
    "energy_above_hull",
    "relaxation_rmsd",
]


def _bonded_mol(atoms: Atoms):
    """The RDKit molecule of ``atoms`` with the bonds RDKit finds at charge 0 and
    without charged fragments; raises when it finds none."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    xyz = StringIO()
    write(xyz, atoms, format="xyz")
    mol = Chem.MolFromXYZBlock(xyz.getvalue())
    rdDetermineBonds.DetermineBonds(
        mol, charge=0, allowChargedFragments=False, embedChiral=True
    )
    return mol


def molecule_smiles(atoms: Atoms, timeout: int = 5) -> str | None:
    """The canonical SMILES of a valid molecule; None for an invalid one.

    A molecule is valid if RDKit finds its bonds (:func:`_bonded_mol`), no atom
    keeps a radical electron, and its SMILES is one fragment.

    Args:
        atoms: the molecule
        timeout: seconds before the bond search counts as failed (0: no limit)
    """
    from rdkit import Chem
    from rdkit.rdBase import BlockLogs

    def give_up(signum, frame):
        raise TimeoutError

    if timeout > 0:
        previous = signal.signal(signal.SIGALRM, give_up)
        signal.alarm(timeout)
    try:
        with BlockLogs():
            mol = _bonded_mol(atoms)
            if any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()):
                return None
            smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    except Exception:  # no bonds found, or the timeout: invalid
        return None
    finally:
        if timeout > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return smiles if smiles and "." not in smiles else None


def posebusters_valid(atoms: Atoms) -> bool:
    """Whether every PoseBusters ``mol`` check passes on the molecule with the bonds
    RDKit finds; False if it finds none."""
    from posebusters import PoseBusters
    from rdkit.rdBase import BlockLogs

    try:
        with BlockLogs():
            mol = _bonded_mol(atoms)
    except Exception:
        return False
    return bool(PoseBusters(config="mol").bust(mol).all(axis=1).iloc[0])


def elongation(atoms: Atoms) -> float:
    """The largest distance between two atoms of the molecule (0 for one atom)."""
    if len(atoms) < 2:
        return 0.0
    return float(atoms.get_all_distances().max())


def _valid(smiles: Iterable[str | None]) -> list[str]:
    return [s for s in smiles if s]


def uniqueness(smiles: Iterable[str | None]) -> float:
    """The fraction of distinct SMILES among the valid molecules (nan if none)."""
    valid = _valid(smiles)
    return len(set(valid)) / len(valid) if valid else float("nan")


def novelty(smiles: Iterable[str | None], known: set[str]) -> float:
    """The fraction of the valid molecules whose SMILES is not in ``known`` (nan if
    none)."""
    valid = _valid(smiles)
    return sum(s not in known for s in valid) / len(valid) if valid else float("nan")


def mean_tanimoto_similarity(
    smiles: Iterable[str | None], max_path: int = 5, n_bits: int = 2048
) -> float:
    """The mean Tanimoto similarity over all pairs of valid molecules, of RDKit's
    path fingerprints (bond paths up to ``max_path``, ``n_bits`` bits; the graph
    only, no stereochemistry). Low means diverse; nan for fewer than two.
    Quadratic in the number of molecules."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetRDKitFPGenerator(
        maxPath=max_path, fpSize=n_bits
    )
    mols = (Chem.MolFromSmiles(s) for s in _valid(smiles))
    fingerprints = [generator.GetFingerprint(m) for m in mols if m is not None]
    if len(fingerprints) < 2:
        return float("nan")
    total = sum(
        sum(DataStructs.BulkTanimotoSimilarity(f, fingerprints[i + 1 :]))
        for i, f in enumerate(fingerprints[:-1])
    )
    n = len(fingerprints)
    return total / (n * (n - 1) / 2)


def js_divergence(
    reference: np.ndarray, observed: np.ndarray, bins: int = 100
) -> float:
    """The Jensen-Shannon divergence (base 2, between 0 and 1) of two
    one-dimensional distributions, from histograms on ``bins`` edges evenly spaced
    over both samples' range."""
    from scipy.spatial.distance import jensenshannon

    edges = np.linspace(
        min(np.min(reference), np.min(observed)),
        max(np.max(reference), np.max(observed)),
        bins,
    )
    p, _ = np.histogram(reference, bins=edges, density=True)
    q, _ = np.histogram(observed, bins=edges, density=True)
    return float(jensenshannon(p, q, base=2) ** 2)


def _structure(atoms: Atoms):
    """The pymatgen structure of a crystal."""
    from pymatgen.core import Structure

    return Structure(
        atoms.cell.array, atoms.numbers, atoms.positions, coords_are_cartesian=True
    )


def _reduced_formula(atoms: Atoms) -> str:
    """The reduced formula with the elements in order of atomic number and every
    count written, e.g. ``O3Fe2`` or ``Si1``."""
    numbers, counts = np.unique(atoms.numbers, return_counts=True)
    counts = counts // np.gcd.reduce(counts)
    return "".join(f"{chemical_symbols[z]}{n}" for z, n in zip(numbers, counts))


def structure_valid(atoms: Atoms, min_distance: float = 0.5) -> bool:
    """Whether no two atoms of the crystal are closer than ``min_distance`` (in Å,
    over the periodic images) and the cell has a volume of at least 0.1 Å³. False
    for a crystal pymatgen cannot build (NaN, a flat cell, no element)."""
    if not (np.isfinite(atoms.positions).all() and np.isfinite(atoms.cell.array).all()):
        return False
    try:
        structure = _structure(atoms)
        if structure.volume < 0.1:
            return False
        distances = structure.distance_matrix
    except Exception:
        return False
    np.fill_diagonal(distances, np.inf)
    return bool(distances.min() >= min_distance)


def composition_valid(
    atoms: Atoms, oxidation_states: str = "icsd24", max_combinations: float = 1e7
) -> bool:
    """Whether the composition is chemically plausible by SMACT's rules, as CDVAE
    and MatterGen apply them: one element, or only metals, or some assignment of one
    oxidation state per element that is charge neutral at the reduced formula and
    gives no element a more positive state than a less electronegative one.

    Args:
        atoms: the crystal
        oxidation_states: SMACT's list of allowed states per element: ``"icsd24"``
            (SMACT 4's default), ``"smact14"`` (the default of SMACT 2, which
            CDVAE's code pins), ``"icsd16"``, ``"sp"`` or ``"wiki"``
        max_combinations: more assignments than this count as invalid untried, as
            in DiffCSP
    """
    import smact
    from smact.screening import pauling_test

    numbers, counts = np.unique(atoms.numbers, return_counts=True)
    counts = counts // np.gcd.reduce(counts)
    symbols = [chemical_symbols[z] for z in numbers]
    if len(symbols) == 1 and symbols[0] != "X":
        return True
    if all(symbol in smact.metals for symbol in symbols):
        return True
    try:
        elements = [smact.Element(symbol) for symbol in symbols]
    except NameError:  # not an element SMACT knows
        return False
    states = [
        getattr(element, f"oxidation_states_{oxidation_states}") or []
        for element in elements
    ]
    if np.prod([float(len(s)) for s in states]) > max_combinations:
        return False
    electronegativities = [element.pauling_eneg for element in elements]
    for assignment in product(*states):
        if np.dot(assignment, counts) != 0:
            continue
        try:
            if pauling_test(assignment, electronegativities):
                return True
        except TypeError:  # an element without electronegativity: taken as fine
            return True
    return False


class DisorderedMatcher:
    """Whether two ordered crystals are the same structure, by MatterGen's rule for
    uniqueness and novelty (its ``DisorderedStructureMatcher``, MIT license,
    Copyright (c) Microsoft Corporation).

    The reduced formulas must be equal. Then pymatgen's ``StructureMatcher`` is
    tried on the primitive cells scaled to one volume. If that fails, the elements
    of the first crystal that can substitute each other share their sites with
    partial occupancies, and the match is tried again with supercells: two
    orderings of one alloy count as one structure. Two elements substitute each
    other if their atomic radii differ by at most ``radius_tol`` of their mean and
    their electronegativities by at most ``electronegativity_tol`` (a
    Hume-Rothery rule).

    Any object with ``fit(structure, other) -> bool`` on pymatgen structures can
    take its place in :func:`unique_crystals` and :func:`novel_crystals`, e.g. a
    plain ``StructureMatcher``.

    Args:
        ltol: fractional tolerance of the cell lengths
        stol: tolerance of the site distances, in units of (volume per atom)^(1/3)
        angle_tol: tolerance of the cell angles in degrees
        radius_tol: largest relative difference of two substituting elements' radii
        electronegativity_tol: largest difference of their electronegativities
    """

    def __init__(
        self,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        radius_tol: float = 0.3,
        electronegativity_tol: float = 1.0,
    ):
        from pymatgen.analysis.structure_matcher import (
            OrderDisorderElementComparator,
            StructureMatcher,
        )

        self.ordered = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        self.disordered = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            attempt_supercell=True,
            allow_subset=True,
            comparator=OrderDisorderElementComparator(),
        )
        self.radius_tol = radius_tol
        self.electronegativity_tol = electronegativity_tol

    def fit(self, structure, other) -> bool:
        if structure.composition.reduced_formula != other.composition.reduced_formula:
            return False
        if self.ordered.fit(structure, other):
            return True
        merged = self._merged(structure)
        return merged is not None and bool(self.disordered.fit(merged, other))

    def _substitute(self, a, b) -> bool:
        if a.atomic_radius is None or b.atomic_radius is None:
            return False
        mean_radius = (a.atomic_radius + b.atomic_radius) / 2
        return bool(
            abs(a.atomic_radius - b.atomic_radius) / mean_radius <= self.radius_tol
            and abs(a.X - b.X) <= self.electronegativity_tol
        )

    def _merged(self, structure):
        """``structure`` with every group of substituting elements on shared sites,
        occupied by their shares of the composition; None if no two substitute."""
        groups: list[set] = []
        for a, b in combinations(list(structure.composition), 2):
            if self._substitute(a, b):
                joined = {a, b}.union(*(g for g in groups if g & {a, b}))
                groups = [g for g in groups if not g & joined] + [joined]
        if not groups:
            return None
        share = structure.composition.get_atomic_fraction
        merged = structure.copy()
        merged.replace_species(
            {
                element: {
                    other: share(other) / sum(share(e) for e in group)
                    for other in group
                }
                for group in groups
                for element in group
            }
        )
        return merged


def unique_crystals(crystals: Sequence[Atoms], matcher=None) -> np.ndarray:
    """True for the first crystal of each set of duplicates, False for its later
    copies; the mean is the uniqueness. Crystals are compared within one reduced
    formula with ``matcher`` (default: :class:`DisorderedMatcher`)."""
    matcher = matcher or DisorderedMatcher()
    kept: dict[str, list] = {}
    unique = np.zeros(len(crystals), dtype=bool)
    for i, atoms in enumerate(crystals):
        structure = _structure(atoms)
        seen = kept.setdefault(_reduced_formula(atoms), [])
        if not any(matcher.fit(structure, earlier) for earlier in seen):
            seen.append(structure)
            unique[i] = True
    return unique


def novel_crystals(
    crystals: Sequence[Atoms], known: Iterable[Atoms], matcher=None
) -> np.ndarray:
    """True for the crystals that match none of ``known``, e.g. the training and
    validation crystals; the mean is the novelty. Crystals are compared within one
    reduced formula with ``matcher`` (default: :class:`DisorderedMatcher`)."""
    matcher = matcher or DisorderedMatcher()
    formulas = [_reduced_formula(atoms) for atoms in crystals]
    wanted = set(formulas)
    known_by_formula: dict[str, list] = {}
    for atoms in known:
        formula = _reduced_formula(atoms)
        if formula in wanted:
            known_by_formula.setdefault(formula, []).append(_structure(atoms))
    return np.array(
        [
            not any(
                matcher.fit(_structure(atoms), other)
                for other in known_by_formula.get(formula, [])
            )
            for atoms, formula in zip(crystals, formulas)
        ],
        dtype=bool,
    )


def hull_reference(
    crystals: Iterable[Atoms], energies: Iterable[float]
) -> dict[str, float]:
    """The lowest energy per atom of every reduced formula among ``crystals`` with
    total ``energies``: the phases a convex hull is built from
    (:func:`energy_above_hull`). It must hold every element as a phase of its own."""
    lowest: dict[str, float] = {}
    for atoms, energy in zip(crystals, energies):
        formula = _reduced_formula(atoms)
        lowest[formula] = min(float(energy) / len(atoms), lowest.get(formula, np.inf))
    return lowest


def energy_above_hull(
    crystals: Sequence[Atoms], energies: Sequence[float], reference: Mapping[str, float]
) -> np.ndarray:
    """The energy per atom of each crystal above the convex hull of the reference
    phases of its chemical system, in the unit of the energies: how much energy the
    crystal would release by decomposing into the most stable mixture of reference
    phases. Negative for a crystal below the hull; nan without an energy or if
    ``reference`` lacks one of its elements.

    Args:
        crystals: the crystals, relaxed
        energies: their total energies, from the method the reference's are from
        reference: energy per atom by formula (:func:`hull_reference`)
    """
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition

    phases: dict[frozenset, list] = {}
    for formula, energy in reference.items():
        composition = Composition(formula)
        system = frozenset(element.symbol for element in composition)
        phases.setdefault(system, []).append(
            PDEntry(composition, energy * composition.num_atoms)
        )
    diagrams: dict[frozenset, PhaseDiagram] = {}
    above = np.full(len(crystals), np.nan)
    for i, (atoms, energy) in enumerate(zip(crystals, energies)):
        system = frozenset(atoms.get_chemical_symbols())
        if not np.isfinite(energy) or any(
            frozenset([element]) not in phases for element in system
        ):
            continue
        if system not in diagrams:
            diagrams[system] = PhaseDiagram(
                [
                    phase
                    for n in range(1, len(system) + 1)
                    for subsystem in combinations(sorted(system), n)
                    for phase in phases.get(frozenset(subsystem), [])
                ]
            )
        entry = PDEntry(Composition(atoms.get_chemical_formula()), float(energy))
        above[i] = diagrams[system].get_e_above_hull(entry, allow_negative=True)
    return above


def relaxation_rmsd(generated: Atoms, relaxed: Atoms) -> float:
    """The root mean square displacement of the atoms between a generated crystal
    and its relaxed structure in Å, as MatterGen measures it: the generated crystal
    is symmetrized and reduced to its primitive cell, aligned to the relaxed one by
    pymatgen's ``StructureMatcher`` (``ltol`` 0.5, ``stol`` 0.5, 10 degrees, no
    volume scaling), and the matcher's distance, which is in units of
    (volume per atom)^(1/3) of the mean cell, converted to Å. If nothing aligns
    within ``stol`` the value is that largest distance, 0.5 in those units; nan if
    pymatgen cannot symmetrize the generated crystal.

    Both crystals are reduced to primitive cells with pymatgen's tolerance of
    0.25 Å. So for a generated crystal that is, within that tolerance, a supercell
    of a smaller cell, the displacements that break the smaller cell's translations
    are not counted: a rattled supercell of a perfect crystal gives zero."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    stol = 0.5
    matcher = StructureMatcher(
        ltol=0.5, stol=stol, angle_tol=10, scale=False, attempt_supercell=True
    )
    after = _structure(relaxed)
    try:
        before = (
            SpacegroupAnalyzer(_structure(generated))
            .get_refined_structure()
            .get_primitive_structure()
            .get_reduced_structure(reduction_algo="LLL")
        )
        match = matcher.get_rms_dist(after, before)
    except Exception:
        return float("nan")
    mean_cell = Lattice.from_parameters(
        *(np.add(after.lattice.parameters, before.lattice.parameters) / 2)
    )
    length = (mean_cell.volume / len(after)) ** (1 / 3)
    return float((stol if match is None else match[0]) * length)
