"""
spk evaluate: validity and uniqueness of generated molecules
(:mod:`schnetpack.generative.validation`, ``configs/evaluate.yaml``). The results go
to ``eval.json`` next to each db, one entry per db (an entry of the same name is
replaced)::

    spk evaluate 'dbs=[<run>/samples/steps64_seed0.db,<run>/samples/steps32_seed0.db]'
    spk evaluate 'dbs=<run>/samples/*.db'

Patterns are expanded by the command; quote them so the shell leaves them alone.
Needs RDKit (``pip install schnetpack[rdkit]``).
"""

import json
import os
import time
from glob import glob

import hydra
from ase.db import connect
from omegaconf import DictConfig

from schnetpack.generative.validation import molecule_smiles, uniqueness


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.2")
def main(config: DictConfig):
    from rdkit import __version__ as rdkit_version

    patterns = [config.dbs] if isinstance(config.dbs, str) else config.dbs
    for pattern in patterns:
        paths = sorted(glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no db matches {pattern}")
        for path in paths:
            _evaluate(path, config.timeout, rdkit_version)


def _evaluate(path: str, timeout: int, rdkit_version: str):
    start = time.perf_counter()
    with connect(path) as db:
        smiles = [molecule_smiles(row.toatoms(), timeout) for row in db.select()]
    valid = [s for s in smiles if s is not None]
    result = {
        "n": len(smiles),
        "n_valid": len(valid),
        "n_unique": len(set(valid)),
        "validity": len(valid) / len(smiles),
        "uniqueness": uniqueness(smiles),
        "rdkit": rdkit_version,
        "seconds": time.perf_counter() - start,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "eval.json")
    results = {}
    if os.path.exists(out):
        with open(out) as f:
            results = json.load(f)
    results[os.path.splitext(os.path.basename(path))[0]] = result
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"{path}: {result['n']} molecules, validity {result['validity']:.4f}, "
        f"uniqueness {result['uniqueness']:.4f}"
    )
