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
"""

import signal
from collections.abc import Iterable
from io import StringIO

import numpy as np
from ase import Atoms
from ase.io import write

__all__ = [
    "molecule_smiles",
    "posebusters_valid",
    "elongation",
    "uniqueness",
    "novelty",
    "mean_tanimoto_similarity",
    "js_divergence",
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
