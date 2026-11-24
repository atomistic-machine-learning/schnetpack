"""
spk evaluate: metrics of generated structures
(:mod:`schnetpack.generative.validation`, ``configs/evaluate.yaml``). The results go
to ``eval.json`` next to each db, one entry per db (an entry of the same name is
replaced)::

    spk evaluate 'dbs=[<run>/samples/steps64_seed0.db,<run>/samples/steps32_seed0.db]'
    spk evaluate 'dbs=<run>/samples/*.db'

Patterns are expanded by the command; quote them so the shell leaves them alone.

Molecules (a db without a periodic cell): validity and uniqueness. Needs RDKit
(``pip install schnetpack[rdkit]``).

Crystals: structural and compositional validity, uniqueness, and the novelty against
the crystals of ``known``. With a ``calculator`` -- a
:class:`~schnetpack.dynamics.ForceCalculator` with a ``stress_key``, written as a
Hydra target, around any model that returns energy, forces and stress for a batch --
the structurally valid crystals are relaxed, cell and positions, ``batch_size`` at a
time (:class:`~schnetpack.dynamics.LBFGS` with ``relax_cell``); uniqueness and novelty
are then those of the relaxed crystals, which go to ``relaxed/<db name>.db`` with
their energies before and after. With a ``hull`` db of reference phases, whose
energies are from the calculator's level of theory, the energy above the convex hull,
the stable share and S.U.N. (stable, unique and novel) are added::

    spk evaluate dbs=samples.db known=mp20.db split=<run>/split.npz hull=reference.db \\
        'calculator={_target_: schnetpack.dynamics.ForceCalculator, model: <model folder>, stress_key: stress}'

(``configs/evaluate.yaml`` has the line with a neighbor list and a device.)

Every share is of all crystals of the db: one that is structurally invalid, before or
after its relaxation, counts as neither unique, novel nor stable. Needs pymatgen and
smact (``pip install schnetpack[crystals]``).
"""

import json
import os
import time
from glob import glob
from importlib.metadata import version
from itertools import tee

import hydra
import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from omegaconf import DictConfig

from schnetpack import properties
from schnetpack.generative.validation import (
    composition_valid,
    energy_above_hull,
    hull_reference,
    molecule_smiles,
    novel_crystals,
    relaxation_rmsd,
    structure_valid,
    unique_crystals,
    uniqueness,
)


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.2")
def main(config: DictConfig):
    if config.hull is not None and config.calculator is None:
        raise ValueError("hull= needs calculator=: the crystals' energies come from it")
    calculator = hydra.utils.instantiate(config.calculator)
    known = None if config.known is None else _known(config.known, config.split)
    reference = None
    if config.hull is not None:
        reference = _reference(config.hull, config.hull_energy)

    patterns = [config.dbs] if isinstance(config.dbs, str) else config.dbs
    for pattern in patterns:
        paths = sorted(glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no db matches {pattern}")
        for path in paths:
            start = time.perf_counter()
            with connect(path) as db:
                structures = [row.toatoms() for row in db.select()]
            if structures[0].pbc.any():
                result = _crystals(
                    path, structures, calculator, known, reference, config
                )
            else:
                result = _molecules(structures, config.timeout)
            result["seconds"] = time.perf_counter() - start

            out = os.path.join(os.path.dirname(os.path.abspath(path)), "eval.json")
            results = {}
            if os.path.exists(out):
                with open(out) as f:
                    results = json.load(f)
            results[os.path.splitext(os.path.basename(path))[0]] = result
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            shares = ", ".join(
                f"{key} {value:.4f}"
                for key, value in result.items()
                if isinstance(value, float) and key != "seconds"
            )
            print(f"{path}: {result['n']} structures, {shares}")


def _molecules(molecules: list[Atoms], timeout: int) -> dict:
    from rdkit import __version__ as rdkit_version

    smiles = [molecule_smiles(atoms, timeout) for atoms in molecules]
    valid = [s for s in smiles if s is not None]
    return {
        "n": len(smiles),
        "n_valid": len(valid),
        "n_unique": len(set(valid)),
        "validity": len(valid) / len(smiles),
        "uniqueness": uniqueness(smiles),
        "rdkit": rdkit_version,
    }


def _crystals(
    path: str,
    generated: list[Atoms],
    calculator,
    known: list[Atoms] | None,
    reference: dict[str, float] | None,
    config: DictConfig,
) -> dict:
    n = len(generated)
    structure = np.array([structure_valid(atoms) for atoms in generated])
    composition = np.array([composition_valid(atoms) for atoms in generated])
    result = {
        "n": n,
        "structure_validity": float(structure.mean()),
        "composition_validity": float(composition.mean()),
        "validity": float((structure & composition).mean()),
        "scored": "generated" if calculator is None else "relaxed",
    }

    # the crystals the set metrics see; None for one left out
    scored = [atoms if valid else None for atoms, valid in zip(generated, structure)]
    energies = np.full(n, np.nan)
    if calculator is not None:
        before, rmsd = np.full(n, np.nan), np.full(n, np.nan)
        converged = np.zeros(n, dtype=bool)
        valid = np.flatnonzero(structure)
        relaxed, e_generated, e_relaxed, done = _relax(
            [generated[i] for i in valid],
            calculator,
            config.fmax,
            config.steps,
            config.batch_size,
        )
        for j, i in enumerate(valid):
            if not structure_valid(relaxed[j]):  # also one the model sent to NaN
                scored[i] = None
                continue
            scored[i] = relaxed[j]
            before[i], energies[i], converged[i] = e_generated[j], e_relaxed[j], done[j]
            rmsd[i] = relaxation_rmsd(generated[i], relaxed[j])
        _write_relaxed(path, scored, before, energies, converged)
        n_atoms = np.array([len(atoms) for atoms in generated])
        relaxed = np.isfinite(energies)
        result.update(
            n_relaxed=int(relaxed.sum()),
            n_converged=int(converged.sum()),
            rmsd=float(np.mean(rmsd[relaxed])) if relaxed.any() else float("nan"),
            energy_change=(
                float(np.mean(((energies - before) / n_atoms)[relaxed]))
                if relaxed.any()
                else float("nan")
            ),
        )

    kept = [i for i, atoms in enumerate(scored) if atoms is not None]
    crystals = [scored[i] for i in kept]
    unique, novel = np.zeros(n, dtype=bool), None
    unique[kept] = unique_crystals(crystals)
    result["uniqueness"] = float(unique.mean())
    if known is not None:
        novel = np.zeros(n, dtype=bool)
        novel[kept] = novel_crystals(crystals, known)
        result["novelty"] = float(novel.mean())
    if reference is not None:
        above = np.full(n, np.nan)
        above[kept] = energy_above_hull(crystals, energies[kept], reference)
        stable = above <= config.stable_below
        on_hull = np.isfinite(above)
        result.update(
            energy_above_hull=(
                float(above[on_hull].mean()) if on_hull.any() else float("nan")
            ),
            stability=float(stable.mean()),
        )
        if novel is not None:
            result["sun"] = float((stable & unique & novel).mean())
    result.update(pymatgen=version("pymatgen"), smact=version("smact"))
    return result


def _relax(crystals: list[Atoms], calculator, fmax: float, steps: int, batch_size: int):
    """The crystals relaxed with ``calculator``, cell and positions, ``batch_size`` at
    a time; their energies before and after, in eV; and which are converged: at the
    relaxed crystal the largest force on an atom, and on the cell (a row of the
    virial over the number of atoms), is below ``fmax``."""
    from schnetpack.dynamics import LBFGS
    from schnetpack.interfaces.ase_interface import atoms_to_batch, batch_to_atoms

    relaxer = LBFGS(calculator, fmax=fmax, relax_cell=True)
    calculator = relaxer.calculator
    relaxed, before, after, converged = [], [], [], []
    for start in range(0, len(crystals), batch_size):
        batch = atoms_to_batch(
            crystals[start : start + batch_size], dtype=torch.float64
        )
        batch = calculator.prepare(batch)
        calculator.reset()  # the neighbor lists of the batch before
        before += calculator.energy(batch).flatten().tolist()
        batch = relaxer.run(batch, steps)
        after += calculator.energy(batch).flatten().tolist()
        n_atoms = batch[properties.n_atoms]
        volume = torch.linalg.det(batch[properties.cell].view(-1, 3, 3)).abs()
        on_cell = calculator.stress(batch) * (volume / n_atoms)[:, None, None]
        largest = on_cell.norm(dim=-1).amax(dim=-1)
        largest = largest.scatter_reduce(
            0, batch[properties.idx_m], calculator.forces(batch).norm(dim=-1), "amax"
        )
        converged += (largest < fmax).tolist()
        relaxed += batch_to_atoms(batch)
    return relaxed, before, after, converged


def _write_relaxed(path: str, scored, before, energies, converged):
    """The relaxed crystals to ``relaxed/<db name>.db`` beside ``path`` (a file of
    an earlier call is replaced): the energies before and after in ``data``, the
    row of the generated crystal as ``generated_id``."""
    folder = os.path.join(os.path.dirname(os.path.abspath(path)), "relaxed")
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, os.path.basename(path))
    if os.path.exists(out):
        os.remove(out)
    with connect(out) as db:
        for i, atoms in enumerate(scored):
            if atoms is not None:
                db.write(
                    atoms,
                    data={
                        "energy": np.array([energies[i]]),
                        "energy_generated": np.array([before[i]]),
                    },
                    generated_id=i + 1,
                    converged=bool(converged[i]),
                )


def _known(path: str, split: str | None) -> list[Atoms]:
    """The crystals of the db; with a split file, its train and val rows only."""
    keep = None
    if split is not None:
        indices = np.load(split)
        keep = set(indices["train_idx"].tolist()) | set(indices["val_idx"].tolist())
    with connect(path) as db:
        return [
            row.toatoms()
            for i, row in enumerate(db.select())
            if keep is None or i in keep
        ]


def _reference(path: str, energy: str) -> dict[str, float]:
    """The hull's phases from a db: the lowest energy per atom of each formula. The
    energy is read from the row's ``data`` (a SchNetPack db), else from the row."""
    with connect(path) as db:
        rows, again = tee(db.select())
        return hull_reference(
            (row.toatoms() for row in rows),
            (
                float(
                    np.ravel(row.data[energy] if energy in row.data else row[energy])[0]
                )
                for row in again
            ),
        )
