import os
from ase.db import connect
from .fixtures import *

from tests.assertions import assert_valid_script
from tests.fixtures import *


def test_qm9_schnet(script_runner, modeldir, qm9_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="schnet",
        dataset="qm9",
        dbpath=qm9_path,
        property="energy_U0",
    )


def test_qm9_wacsf(script_runner, modeldir, qm9_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="wacsf",
        dataset="qm9",
        dbpath=qm9_path,
        property="energy_U0",
    )


def test_ani1_schnet(script_runner, modeldir, ani1_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="schnet",
        dataset="ani1",
        dbpath=ani1_path,
        property="energy",
    )


def test_ani1_wacsf(script_runner, modeldir, ani1_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="wacsf",
        dataset="ani1",
        dbpath=ani1_path,
        property="energy",
    )


def test_md1_schnet(script_runner, modeldir, ethanol_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="schnet",
        dataset="md17",
        dbpath=ethanol_path,
        property="energy",
    )


def test_md17_wacsf(script_runner, modeldir, ethanol_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="wacsf",
        dataset="md17",
        dbpath=ethanol_path,
        property="energy",
    )


def test_custom_schnet(script_runner, modeldir, ethanol_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="schnet",
        dataset="custom",
        dbpath=ethanol_path,
        property="energy",
        derivative="forces",
        negative_dr=True,
    )


def test_custom_wacsf(script_runner, modeldir, ethanol_path):
    assert_valid_script(
        script_runner,
        modeldir,
        representation="wacsf",
        dataset="custom",
        dbpath=ethanol_path,
        property="energy",
        derivative="forces",
        negative_dr=True,
        output_module="elemental_atomwise",
    )


def test_parsing_script(script_runner, xyz_path, testdir):

    # define settings
    dbpath = os.path.join(testdir, "database.db")
    atomic_properties = "Properties=species:S:1:pos:R:3:forces:R:3"
    molecular_properties = "energy"

    # run script
    ret = script_runner.run(
        "spk_parse.py",
        xyz_path,
        dbpath,
        "--atomic_properties",
        atomic_properties,
        "--molecular_properties",
        molecular_properties,
    )

    # check validity
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(dbpath))

    with connect(dbpath) as conn:
        assert len(conn) == 3
        atmsrow = conn.get(1)
        assert {"energy", "forces"} == set(list(atmsrow.data.keys()))


def test_spk_ase(script_runner, modeldir, sim_dir, molecule_path, ethanol_path):
    # train a model on md17
    ret = script_runner.run(
        "spk_run.py",
        "train",
        "schnet",
        "md17",
        ethanol_path,
        modeldir,
        "--split",
        "10",
        "5",
        "--max_epochs",
        "2",
    )
    assert ret.success, ret.stderr

    # test md simulation on model
    ret = script_runner.run(
        "spk_ase.py",
        molecule_path,
        os.path.join(modeldir, "best_model"),
        sim_dir,
        "--optimize",
        "2",
    )
    assert ret.success, ret.stderr
