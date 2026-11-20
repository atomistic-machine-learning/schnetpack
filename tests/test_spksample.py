"""spksample samples a generative spktrain run: the variables of the run's
generative: block, the compositions of its test split, one ASE db per number of
steps."""

import os
import shutil
import subprocess
import sys

import numpy as np
from ase.db import connect

from schnetpack import load_hyperparameters

TEST_QM9 = os.path.join(os.path.dirname(__file__), "testdata", "test_qm9.db")

# a small GPFF: the positions variable of the run, trained for 2 epochs
EXPERIMENT = """\
# @package _global_
defaults:
  - override /model: nnp
  - override /data: custom

seed: 3

generative:
  noising:
    _target_: schnetpack.generative.GaussianInterpolant
    schedule:
      _target_: schnetpack.generative.VELinear
      sigma_max: 30.0
      t_min: 0.0
    prior:
      _target_: schnetpack.generative.GaussianPrior
      std: 30.0
      centered: false
  positions:
    _target_: schnetpack.generative.Variable
    name: positions
    view: _positions
    per: atom
    noising: ${generative.noising}
    parametrization:
      _target_: schnetpack.generative.PseudoForceParametrization
    output_key: pseudo_force_pred

sample:
  grid:
    t_min: 3.3333333333333335e-4
    rho: 5.0

data:
  dataset:
    datapath: ${run.data_dir}/qm9.db
    load_properties: []
    transforms:
      - _target_: schnetpack.transform.SubtractCenterOfGeometry
      - _target_: schnetpack.transform.AllToAllNeighborList
  batch_transforms:
    - _target_: schnetpack.generative.Diffuse
      _convert_: all
      variables:
        - ${generative.positions}
    - _target_: schnetpack.transform.CastTo32
  batch_size: 5
  num_train: 10
  num_val: 5
  num_test: 5
  num_workers: 0

globals:
  lr: 1e-3
  cutoff: 150.0

model:
  representation:
    n_atom_basis: 16
    n_interactions: 1
    radial_basis:
      n_rbf: 8
  output_modules:
    - _target_: schnetpack.model.AtomwiseVector
      n_in: 16
      n_layers: 2
      output_key: pseudo_force_pred

task:
  outputs:
    - _target_: schnetpack.generative.VariableOutput
      variable: ${generative.positions}
"""


def spk(entry, *args, cwd):
    result = subprocess.run(
        [sys.executable, "-c", f"from schnetpack.cli.{entry} import main; main()"]
        + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_spksample_samples_the_runs_variables_on_its_test_compositions(tmp_path):
    os.makedirs(tmp_path / "data")
    shutil.copy(TEST_QM9, tmp_path / "data" / "qm9.db")
    os.makedirs(tmp_path / "configs" / "experiment")
    (tmp_path / "configs" / "experiment" / "gen.yaml").write_text(EXPERIMENT)
    spk(
        "train",
        "experiment=gen",
        f"run.data_dir={tmp_path / 'data'}",
        f"run.path={tmp_path / 'runs'}",
        "run.id=gen",
        "trainer.max_epochs=2",
        "trainer.accelerator=cpu",
        "print_config=False",
        cwd=tmp_path,
    )
    run = tmp_path / "runs" / "gen"
    spk(
        "sample",
        str(run),
        "sample.n_samples=7",
        "sample.batch_size=4",
        "sample.n_steps=[3,5]",
        cwd=tmp_path,
    )

    test_idx = np.load(run / "split.npz")["test_idx"].tolist()
    with connect(tmp_path / "data" / "qm9.db") as db:
        test_formulas = {db.get(id=i + 1).formula for i in test_idx}
    for n_steps in (3, 5):
        with connect(run / "samples" / f"steps{n_steps}_seed3.db") as db:
            rows = list(db.select())
            meta = db.metadata
        assert len(rows) == 7
        assert {row.formula for row in rows} <= test_formulas
        assert all(np.isfinite(row.positions).all() for row in rows)
        assert meta["n_steps"] == n_steps and meta["seed"] == 3
        assert meta["sample"]["grid"]["rho"] == 5.0  # the run's sample: block
        assert meta["variables"]["positions"]["output_key"] == "pseudo_force_pred"

    # the model folder carries the run's resolved generative: and sample: blocks
    blocks = load_hyperparameters(str(run / "best_model"))
    assert blocks["sample"] == {"grid": {"t_min": 3.3333333333333335e-4, "rho": 5.0}}
    noising = blocks["generative"]["positions"]["noising"]
    assert noising == blocks["generative"]["noising"]
    assert noising["schedule"]["sigma_max"] == 30.0
