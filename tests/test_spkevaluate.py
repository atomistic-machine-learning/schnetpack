"""spkevaluate writes the validity and uniqueness of each db to eval.json next to
it, one entry per db."""

import json
import sys

import pytest
from ase.build import molecule
from ase.db import connect

pytest.importorskip("rdkit")

from schnetpack.cli.evaluate import main  # noqa: E402


def test_eval_json_holds_one_entry_per_db(tmp_path, monkeypatch):
    path = tmp_path / "steps5_seed3.db"
    with connect(path) as db:
        for atoms in [molecule("CH3CH2OH"), molecule("CH3CH2OH"), molecule("H2O")]:
            db.write(atoms)
        db.write(molecule("CH3"))
    monkeypatch.setattr(sys, "argv", ["spkevaluate", str(path)])
    main()
    main()  # the entry of the same db is replaced
    ((name, result),) = json.loads((tmp_path / "eval.json").read_text()).items()
    assert name == "steps5_seed3"
    assert (result["n"], result["n_valid"], result["n_unique"]) == (4, 3, 2)
    assert result["validity"] == 0.75
    assert result["uniqueness"] == pytest.approx(2 / 3)
