"""
spkevaluate: validity and uniqueness of generated molecules
(:mod:`schnetpack.generative.validation`). The results go to
``eval.json`` next to each db, one entry per db (an entry of the same name is
replaced)::

    spkevaluate <run>/samples/steps64_seed0.db <run>/samples/steps32_seed0.db

Needs RDKit (``pip install schnetpack[rdkit]``).
"""

import argparse
import json
import os
import time

from ase.db import connect

from schnetpack.generative.validation import molecule_smiles, uniqueness


def main():
    parser = argparse.ArgumentParser(
        description="Validity and uniqueness of generated molecules (RDKit)."
    )
    parser.add_argument("dbs", nargs="+", help="ASE dbs of molecules (spksample's)")
    parser.add_argument("--timeout", type=int, default=5, help="seconds per molecule")
    args = parser.parse_args()
    from rdkit import __version__ as rdkit_version

    for path in args.dbs:
        start = time.perf_counter()
        with connect(path) as db:
            smiles = [
                molecule_smiles(row.toatoms(), args.timeout) for row in db.select()
            ]
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


if __name__ == "__main__":
    main()
