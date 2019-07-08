import os
import pytest


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


def assert_valid_script(script_runner, tmpdir_factory, dataset, dbpath, property,
                        checkpoint_interval, keep_n_checkpoints, split, max_epochs):
    # train model
    modeldir = tmpdir_factory.mktemp("{}_script_test".format(dataset)).strpath
    ret = script_runner.run(
        "schnetpack_x.py",
        "train",
        "schnet",
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
    )
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "best_model"))
    ret = script_runner.run(
        "schnetpack_x.py",
        "train",
        "schnet",
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
    )
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "checkpoints",
                                       "checkpoint-{}.pth.tar".format(max_epochs)))
    ret = script_runner.run(
        "schnetpack_x.py",
        "eval",
        dbpath,
        modeldir,
        "--split",
        "test"
    )
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))


def test_qm9(script_runner, tmpdir_factory, checkpoint_interval, keep_n_checkpoints,
             split, max_epochs):
    dataset = "qm9"
    dbpath = "tests/data/test_qm9.db"
    property = "energy_U0"
    assert_valid_script(script_runner, tmpdir_factory, dataset, dbpath, property,
                        checkpoint_interval, keep_n_checkpoints, split, max_epochs)


def test_ani1(script_runner, tmpdir_factory, checkpoint_interval, keep_n_checkpoints,
             split, max_epochs):
    dataset = "ani1"
    dbpath = "tests/data/test_ani1.db"
    property = "energy"
    assert_valid_script(script_runner, tmpdir_factory, dataset, dbpath, property,
                        checkpoint_interval, keep_n_checkpoints, split, max_epochs)


#def test_matproj(script_runner, tmpdir_factory, checkpoint_interval,
    # keep_n_checkpoints,
#             split, max_epochs):
#    dataset = "matproj"
#    dbpath = "tests/data/test_matproj.db"
#    property = "formation_energy_per_atom"
#    assert_valid_script(script_runner, tmpdir_factory, dataset, dbpath, property,
#                        checkpoint_interval, keep_n_checkpoints, split, max_epochs)
#

def test_md17(script_runner, tmpdir_factory, checkpoint_interval, keep_n_checkpoints,
             split, max_epochs):
    dataset = "md17"
    dbpath = "tests/data/test_ethanol.db"
    property = "energy"
    assert_valid_script(script_runner, tmpdir_factory, dataset, dbpath, property,
                        checkpoint_interval, keep_n_checkpoints, split, max_epochs)