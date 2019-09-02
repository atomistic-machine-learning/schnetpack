import os
import pytest
from ase.db import connect


@pytest.fixture
def max_epochs():
    return "4"


@pytest.fixture
def split():
    return "10", "5"


@pytest.fixture
def keep_n_checkpoints():
    return "2"


@pytest.fixture
def checkpoint_interval():
    return "1"


def assert_valid_script(
    script_runner,
    tmpdir_factory,
    dataset,
    dbpath,
    property,
    checkpoint_interval,
    keep_n_checkpoints,
    split,
    max_epochs,
    with_derivative=False,
    representation="schnet",
):
    # train model
    modeldir = tmpdir_factory.mktemp("{}_script_test".format(dataset)).strpath
    run_args = [
        "spk_run.py",
        "train",
        representation,
        dataset,
        dbpath,
        modeldir,
        "--max_epochs",
        max_epochs,
        "--split",
        *split,
        "--property",
        property,
        "--checkpoint_interval",
        checkpoint_interval,
        "--keep_n_checkpoints",
        keep_n_checkpoints,
    ]

    if dataset == "custom":
        run_args += ["--derivative", "forces"]

    # run training
    ret = script_runner.run(*run_args)
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "best_model"))

    # restore training
    ret = script_runner.run(*run_args)
    assert ret.success, ret.stderr
    assert os.path.exists(
        os.path.join(
            modeldir, "checkpoints", "checkpoint-{}.pth.tar".format(max_epochs)
        )
    )

    # run evaluation
    ret = script_runner.run("spk_run.py", "eval", dbpath, modeldir, "--overwrite")
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))

    # test on all sets
    ret = script_runner.run(
        "spk_run.py",
        "eval",
        dbpath,
        modeldir,
        "--split",
        "test",
        "train",
        "validation",
        "--overwrite",
    )
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))
    with open(os.path.join(modeldir, "evaluation.txt")) as f:
        lines = f.readlines()
        print(lines)
        n_evals = 6 if not with_derivative else 12
        assert len(lines[0].split(",")) == len(lines[1].split(",")) == n_evals
        assert len(lines) == 2


def test_qm9(
    script_runner,
    tmpdir_factory,
    checkpoint_interval,
    keep_n_checkpoints,
    split,
    max_epochs,
):
    dataset = "qm9"
    dbpath = "tests/data/test_qm9.db"
    property = "energy_U0"
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        dataset,
        dbpath,
        property,
        checkpoint_interval,
        keep_n_checkpoints,
        split,
        max_epochs,
    )


def test_ani1(
    script_runner,
    tmpdir_factory,
    checkpoint_interval,
    keep_n_checkpoints,
    split,
    max_epochs,
):
    dataset = "ani1"
    dbpath = "tests/data/test_ani1.db"
    property = "energy"
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        dataset,
        dbpath,
        property,
        checkpoint_interval,
        keep_n_checkpoints,
        split,
        max_epochs,
    )


# def test_matproj(script_runner, tmpdir_factory, checkpoint_interval,
# keep_n_checkpoints,
#             split, max_epochs):
#    dataset = "matproj"
#    dbpath = "tests/data/test_matproj.db"
#    property = "formation_energy_per_atom"
#    assert_valid_script(script_runner, tmpdir_factory, dataset, dbpath, property,
#                        checkpoint_interval, keep_n_checkpoints, split, max_epochs)
#


def test_md17(
    script_runner,
    tmpdir_factory,
    checkpoint_interval,
    keep_n_checkpoints,
    split,
    max_epochs,
):
    dataset = "md17"
    dbpath = "tests/data/test_ethanol.db"
    property = "energy"
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        dataset,
        dbpath,
        property,
        checkpoint_interval,
        keep_n_checkpoints,
        split,
        max_epochs,
        with_derivative=True,
    )


def test_custom(
    script_runner,
    tmpdir_factory,
    checkpoint_interval,
    keep_n_checkpoints,
    split,
    max_epochs,
):
    dataset = "custom"
    dbpath = "tests/data/test_ethanol.db"
    property = "energy"

    # test schnet
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        dataset,
        dbpath,
        property,
        checkpoint_interval,
        keep_n_checkpoints,
        split,
        max_epochs,
        with_derivative=True,
    )

    # test wacsf
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        dataset,
        dbpath,
        property,
        checkpoint_interval,
        keep_n_checkpoints,
        split,
        max_epochs,
        representation="wacsf",
        with_derivative=True,
    )


def test_parsing_script(script_runner, tmpdir_factory):

    # define settings
    file_path = "tests/data/ethanol_snip.xyz"
    dbpath = os.path.join(tmpdir_factory.mktemp("test_parsing").strpath, "database.db")
    atomic_properties = "Properties=species:S:1:pos:R:3:forces:R:3"
    molecular_properties = ["energy"]

    # run script
    ret = script_runner.run(
        "spk_parse.py",
        file_path,
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
