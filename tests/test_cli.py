"""The spk commands, each run as `spk <command> key=value ...` in its own process, on
runs trained for two epochs on the test QM9 db."""

import json
import os
import re
import shutil
import subprocess
import sys

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import yaml
from ase.build import molecule
from ase.db import connect
from hydra import compose, initialize_config_dir

import schnetpack
from schnetpack import load_hyperparameters
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.transform import CastTo32, MatScipyNeighborList, SubtractCenterOfMass

CONFIG_DIR = os.path.join(os.path.dirname(schnetpack.__file__), "configs")
TEST_QM9 = os.path.join(os.path.dirname(__file__), "testdata", "test_qm9.db")

# a small GPFF: the positions variable of the run, trained for 2 epochs
GENERATIVE = """\
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


def spk(*args, cwd, check=True, pythonpath=None):
    """`spk <args>` in its own process; ``pythonpath`` is a folder it can import from."""
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-c", "from schnetpack.cli import main; main()", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if check:
        assert result.returncode == 0, result.stderr[-3000:]
    return result


def train(root, *overrides):
    """A spk train run on the test QM9 db, in ``root/runs/run``."""
    os.makedirs(root / "data", exist_ok=True)
    shutil.copy(TEST_QM9, root / "data" / "qm9.db")
    spk(
        "train",
        f"run.data_dir={root / 'data'}",
        f"run.path={root / 'runs'}",
        "run.id=run",
        "trainer.accelerator=cpu",
        "print_config=False",
        *overrides,
        cwd=root,
    )
    return root / "runs" / "run"


QM9 = (
    "experiment=qm9_atomwise",
    "data.num_train=10",
    "data.num_val=5",
    "data.num_test=5",
    "data.batch_size=5",
    "data.num_workers=0",
)


@pytest.fixture(scope="module")
def qm9_run(tmp_path_factory):
    return train(tmp_path_factory.mktemp("qm9"), *QM9, "trainer.max_epochs=2", "seed=0")


@pytest.fixture(scope="module")
def generative_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("generative")
    os.makedirs(root / "configs" / "experiment")
    (root / "configs" / "experiment" / "gen.yaml").write_text(GENERATIVE)
    return train(root, "experiment=gen", "trainer.max_epochs=2")


def test_spk_lists_its_commands(tmp_path):
    listing = spk(cwd=tmp_path).stdout
    assert "train, test, predict, sample, evaluate, convert, deploy" in listing
    assert spk("fit", cwd=tmp_path, check=False).returncode == 2


@pytest.mark.parametrize("trainer", ["default_trainer", "debug_trainer"])
def test_trainer_config_builds_a_trainer(trainer):
    with initialize_config_dir(CONFIG_DIR, version_base="1.2"):
        config = compose("train", overrides=[f"trainer={trainer}"])

    assert isinstance(hydra.utils.instantiate(config.trainer, logger=False), pl.Trainer)


def test_spk_train_keeps_the_random_seed_on_resume(tmp_path):
    run = train(tmp_path, *QM9, "trainer.max_epochs=1")
    seed = yaml.safe_load((run / "config.yaml").read_text())["seed"]
    assert isinstance(seed, int)

    train(tmp_path, *QM9, "trainer.max_epochs=2")
    logs = "".join(p.read_text() for p in run.glob("*.log"))
    logged = re.findall(r"Seed (?:randomly )?with <(\d+)>", logs)
    assert [int(s) for s in logged] == [seed, seed]

    # the resumed run loaded last.ckpt weights-only; the best model is a folder
    assert sorted(os.listdir(run / "best_model")) == [
        "hyperparameters.json",
        "model.safetensors",
    ]
    assert not (run / "best_model.task").exists()
    model = schnetpack.load_model(str(run / "best_model"))
    assert model.postprocessors[1].mean.shape == (1,)
    for checkpoint in (run / "checkpoints").glob("*.ckpt"):
        assert "hyper_parameters" not in torch.load(checkpoint, weights_only=True)


def test_spk_test_writes_the_test_metrics_spk_train_wrote(qm9_run, tmp_path):
    trained = json.loads((qm9_run / "test_metrics.json").read_text())
    # the data moved since training
    shutil.move(qm9_run.parent.parent / "data", tmp_path / "moved")

    spk(
        "test",
        f"run_dir={qm9_run}",
        f"data_dir={tmp_path / 'moved'}",
        "device=cpu",
        cwd=tmp_path,
    )
    tested = json.loads((qm9_run / "test_metrics.json").read_text())

    assert os.listdir(tmp_path) == ["moved"]  # no Hydra output folder
    assert trained["checkpoint"].startswith("checkpoints/epoch=")
    assert trained["n_test"] == 5
    units = {k: m["unit"] for k, m in trained["metrics"].items()}
    assert units == {
        "test_loss": None,
        "test_energy_U0_mae": "eV",
        "test_energy_U0_rmse": "eV",
    }
    for key in ("checkpoint", "epoch", "n_test"):
        assert tested[key] == trained[key]
    for name, metric in trained["metrics"].items():
        assert tested["metrics"][name]["unit"] == metric["unit"]
        assert tested["metrics"][name]["value"] == pytest.approx(
            metric["value"], rel=1e-6
        )


def test_spk_predict_runs_the_runs_best_model_on_every_structure(qm9_run, tmp_path):
    spk(
        "predict",
        f"model_dir={qm9_run}",
        f"db={TEST_QM9}",
        "out=predictions",
        "trainer.accelerator=cpu",
        cwd=tmp_path,
    )

    assert os.listdir(tmp_path) == ["predictions"]
    predicted = torch.load(tmp_path / "predictions" / "predictions.pt")
    model = schnetpack.load_model(str(qm9_run / "best_model"))
    dataset = ASEAtomsData(
        TEST_QM9,
        transforms=[
            SubtractCenterOfMass(),
            MatScipyNeighborList(cutoff=float(model.representation.cutoff)),
            CastTo32(),
        ],
    )
    batch = next(iter(AtomsLoader(dataset, batch_size=len(dataset))))
    expected = model(batch)["energy_U0"].detach()
    torch.testing.assert_close(predicted["energy_U0"], expected)


def test_spk_sample_samples_the_runs_variables_on_its_test_compositions(
    generative_run, tmp_path
):
    run = generative_run
    spk(
        "sample",
        f"model_dir={run}",
        "sample.n_samples=7",
        "sample.batch_size=4",
        "sample.n_steps=[3,5]",
        cwd=tmp_path,
    )

    assert os.listdir(tmp_path) == []
    test_idx = np.load(run / "split.npz")["test_idx"].tolist()
    with connect(TEST_QM9) as db:
        test_formulas = {db.get(id=i + 1).formula for i in test_idx}
    for n_steps in (3, 5):
        with connect(run / "samples" / f"steps{n_steps}_seed3.db") as db:
            rows = list(db.select())
            meta = db.metadata
        assert len(rows) == 7
        assert {row.formula for row in rows} <= test_formulas
        assert all(np.isfinite(row.positions).all() for row in rows)
        assert meta["n_steps"] == n_steps and meta["seed"] == 3
        assert meta["sample"]["grid"]["rho"] == 5.0  # the model's sample: block
        assert meta["variables"]["positions"]["output_key"] == "pseudo_force_pred"

    # the model folder carries the run's resolved generative: and sample: blocks
    blocks = load_hyperparameters(str(run / "best_model"))
    assert blocks["sample"] == {"grid": {"t_min": 3.3333333333333335e-4, "rho": 5.0}}
    noising = blocks["generative"]["positions"]["noising"]
    assert noising == blocks["generative"]["noising"]
    assert noising["schedule"]["sigma_max"] == 30.0


def test_spk_sample_takes_a_model_folder_and_the_compositions_of_a_db(
    generative_run, tmp_path
):
    model = generative_run / "best_model"
    failed = spk("sample", f"model_dir={model}", cwd=tmp_path, check=False)
    assert failed.returncode != 0 and "db=<ASE db>" in failed.stderr

    spk(
        "sample",
        f"model_dir={model}",
        f"db={TEST_QM9}",
        "out=samples",
        "sample.seed=4",
        "sample.n_samples=3",
        "sample.batch_size=3",
        "sample.n_steps=[2]",
        "sample.grid.rho=6.0",
        cwd=tmp_path,
    )

    with connect(TEST_QM9) as db:
        formulas = {row.formula for row in db.select()}
    with connect(tmp_path / "samples" / "steps2_seed4.db") as db:
        rows = list(db.select())
        grid = db.metadata["sample"]["grid"]
    assert len(rows) == 3 and {row.formula for row in rows} <= formulas
    # the command line over the model's sample: block over the command's defaults
    assert grid["rho"] == 6.0
    assert grid["t_min"] == 3.3333333333333335e-4
    assert grid["_target_"] == "schnetpack.dynamics.KarrasGrid"


def test_spk_evaluate_writes_one_entry_per_db_to_eval_json(tmp_path):
    pytest.importorskip("rdkit")
    for name in ("steps5_seed3", "steps7_seed3"):
        with connect(tmp_path / f"{name}.db") as db:
            for atoms in [molecule("CH3CH2OH"), molecule("CH3CH2OH"), molecule("H2O")]:
                db.write(atoms)
            db.write(molecule("CH3"))

    spk("evaluate", f"dbs={tmp_path}/*.db", cwd=tmp_path)
    spk("evaluate", f"dbs=[{tmp_path / 'steps5_seed3.db'}]", cwd=tmp_path)

    results = json.loads((tmp_path / "eval.json").read_text())
    assert sorted(results) == ["steps5_seed3", "steps7_seed3"]
    result = results["steps5_seed3"]
    assert (result["n"], result["n_valid"], result["n_unique"]) == (4, 3, 2)
    assert result["validity"] == 0.75
    assert result["uniqueness"] == pytest.approx(2 / 3)


def test_spk_evaluate_relaxes_crystals_with_the_calculator_it_is_given(tmp_path):
    pytest.importorskip("pymatgen")
    pytest.importorskip("smact")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    def alloy():  # Cu and Au in alternating layers of fcc
        atoms = bulk("Cu", cubic=True, a=3.9) * (1, 1, 2)
        atoms.symbols = ["Cu", "Au"] * 4
        return atoms

    copper = bulk("Cu", cubic=True, a=3.7)
    close = copper.copy()
    close.positions[1] = close.positions[0] + 0.2
    with connect(tmp_path / "samples.db") as db:
        for seed in (1, 2):  # one crystal twice
            atoms = copper.copy()
            atoms.rattle(0.05, seed=seed)
            db.write(atoms)
        db.write(alloy())
        db.write(close)  # structurally invalid
        db.write(bulk("Al", cubic=True, a=4.2))
    with connect(tmp_path / "known.db") as db:
        db.write(bulk("Cu", a=3.6))  # row 0, train
        db.write(alloy())  # row 1, test
        db.write(bulk("Al", a=4.05))  # row 2, val
    np.savez(tmp_path / "split.npz", train_idx=[0], val_idx=[2], test_idx=[1])
    with connect(tmp_path / "hull.db") as db:
        for atoms in (bulk("Cu", a=3.6), bulk("Au", a=4.08), bulk("Al", a=4.05)):
            atoms.calc = EMT()
            energy = atoms.get_potential_energy()
            atoms.calc = None
            db.write(atoms, data={"energy": np.array([energy])})

    spk("evaluate", "dbs=samples.db", cwd=tmp_path)
    result = json.loads((tmp_path / "eval.json").read_text())["samples"]
    assert result["scored"] == "generated"
    assert (result["structure_validity"], result["composition_validity"]) == (0.8, 1.0)
    assert result["uniqueness"] == 0.6  # copper once, the alloy, aluminium
    assert "novelty" not in result and "stability" not in result

    # ase's EMT as the model of a ForceCalculator; two crystals per batch
    emt = "tests.dynamics.relax.test_relaxer_vs_ase.EMTModel"
    spk(
        "evaluate",
        "dbs=samples.db",
        "known=known.db",
        "split=split.npz",
        "calculator={_target_: schnetpack.dynamics.ForceCalculator, "
        f"model: {{_target_: {emt}, stress: true}}, stress_key: stress}}",
        "batch_size=2",
        "hull=hull.db",
        cwd=tmp_path,
        pythonpath=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    result = json.loads((tmp_path / "eval.json").read_text())["samples"]
    assert result["scored"] == "relaxed"
    assert (result["n_relaxed"], result["n_converged"]) == (4, 4)
    assert result["energy_change"] < 0
    assert result["uniqueness"] == 0.6  # copper once, the alloy, aluminium
    assert result["novelty"] == 0.2  # the alloy: its known copy is a test crystal
    assert result["stability"] == 0.8
    assert result["sun"] == 0.2
    with connect(tmp_path / "relaxed" / "samples.db") as db:
        rows = list(db.select())
    assert [row.generated_id for row in rows] == [1, 2, 3, 5]
    assert all(row.converged for row in rows)
    assert all(row.data["energy"] < row.data["energy_generated"] for row in rows)
    # relaxed with EMT: its own lattice constant of copper, to the stop test
    assert rows[0].toatoms().cell.lengths() == pytest.approx([3.59] * 3, abs=0.01)

    assert (
        spk("evaluate", "dbs=samples.db", "hull=hull.db", cwd=tmp_path, check=False)
    ).returncode != 0  # the energies of a hull come from a calculator


def test_spk_convert_updates_the_metadata_of_an_old_db(tmp_path):
    with connect(tmp_path / "data.db") as db:
        db.write(molecule("H2O"), data={"dipole": np.zeros(3)})
        db.metadata = {"atomrefs": [[0.0], [-0.5]], "atref_labels": "energy"}

    spk(
        "convert",
        "db=data.db",
        "distunit=Ang",
        "propunit={energy:kcal/mol,dipole:Debye}",
        "expand_property_dims=[dipole]",
        cwd=tmp_path,
    )

    assert os.listdir(tmp_path) == ["data.db"]
    with connect(tmp_path / "data.db") as db:
        meta = db.metadata
        dipole = db.get(1).data["dipole"]
    assert meta["atomrefs"] == {"energy": [0.0, -0.5]}
    assert "atref_labels" not in meta
    assert meta["_distance_unit"] == "Ang"
    assert meta["_property_unit_dict"] == {"energy": "kcal/mol", "dipole": "Debye"}
    assert dipole.shape == (1, 3)
