import os
import pytest
import json
from ase.db import connect


def run_args_from_settings(settings):
    """
    Build valid list of run arguments for spk_run.py based on a settings dict.
    """
    # basic settings
    run_args = [
        settings["script"],
        settings["mode"],
        settings["representation"],
        settings["dataset"],
        settings["dbpath"],
        settings["modeldir"],
        "--split",
        *settings["split"],
        "--property",
        settings["property"],
        "--max_epochs",
        settings["max_epochs"],
        "--checkpoint_interval",
        settings["checkpoint_interval"],
        "--keep_n_checkpoints",
        settings["keep_n_checkpoints"],
    ]
    # optional settings
    if settings["derivative"] is not None:
        run_args += ["--derivative", settings["derivative"]]
        if settings["negative_dr"]:
            run_args += ["--negative_dr"]
    if settings["contributions"] is not None:
        run_args += ["--contributions", settings["contributions"]]
    if settings["output_module"] is not None:
        run_args += ["--output_module", settings["output_module"]]
    # string cast
    run_args = [str(arg) for arg in run_args]

    return run_args


def assert_valid_script(
    script_runner,
    tmpdir_factory,
    representation,
    dataset,
    dbpath,
    property,
    split=[10, 5],
    derivative=None,
    contributions=None,
    negative_dr=False,
    output_module=None,
    max_epochs=2,
    checkpoint_interval=1,
    keep_n_checkpoints=4,
):
    """
    Test spk_run.py with different settings.
    """

    # define settings
    modeldir = tmpdir_factory.mktemp("{}_script_test".format(dataset)).strpath
    settings = dict(
        script="spk_run.py",
        mode="train",
        representation=representation,
        dataset=dataset,
        dbpath=dbpath,
        modeldir=modeldir,
        max_epochs=max_epochs,
        split=split,
        property=property,
        checkpoint_interval=checkpoint_interval,
        keep_n_checkpoints=keep_n_checkpoints,
        derivative=derivative,
        negative_dr=negative_dr,
        contributions=contributions,
        output_module=output_module,
    )

    # get run arguments from settings dict
    run_args = run_args_from_settings(settings)

    # run training
    ret = script_runner.run(*run_args)
    assert ret.success, ret.stderr
    assert os.path.exists(os.path.join(modeldir, "best_model"))

    # continue training for one more epoch
    settings["max_epochs"] += 1
    ret = script_runner.run(*run_args)
    assert ret.success, ret.stderr
    assert os.path.exists(
        os.path.join(
            modeldir, "checkpoints", "checkpoint-{}.pth.tar".format(max_epochs)
        )
    )

    # train from json args
    # modify json
    json_path = os.path.join(modeldir, "args.json")
    with open(json_path, "r+") as f:
        data = json.load(f)
        data["max_epochs"] = 5
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    ret = script_runner.run("spk_run.py", "from_json", json_path)
    assert ret.success, ret.stderr
    assert os.path.exists(
        os.path.join(modeldir, "checkpoints", "checkpoint-{}.pth.tar".format(5))
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
        has_forces = True if derivative is not None or dataset == "md17" else False
        expected_eval_dim = 6 + int(has_forces) * 6
        assert len(lines[0].split(",")) == len(lines[1].split(",")) == expected_eval_dim
        assert len(lines) == 2


def test_qm9(script_runner, tmpdir_factory):
    # schnet test
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="schnet",
        dataset="qm9",
        dbpath="tests/data/test_qm9.db",
        property="energy_U0",
    )
    # wacsf test
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="wacsf",
        dataset="qm9",
        dbpath="tests/data/test_qm9.db",
        property="energy_U0",
    )


def test_ani1(script_runner, tmpdir_factory):
    # test schnet
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="schnet",
        dataset="ani1",
        dbpath="tests/data/test_ani1.db",
        property="energy",
    )
    # test wacsf
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="wacsf",
        dataset="ani1",
        dbpath="tests/data/test_ani1.db",
        property="energy",
    )


def test_md17(script_runner, tmpdir_factory):
    # test schnet
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="schnet",
        dataset="md17",
        dbpath="tests/data/test_ethanol.db",
        property="energy",
    )
    # test wacsf
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="wacsf",
        dataset="md17",
        dbpath="tests/data/test_ethanol.db",
        property="energy",
    )


def test_custom(script_runner, tmpdir_factory):
    # test schnet
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="schnet",
        dataset="custom",
        dbpath="tests/data/test_ethanol.db",
        property="energy",
        derivative="forces",
        negative_dr=True,
    )
    # test wacsf
    assert_valid_script(
        script_runner,
        tmpdir_factory,
        representation="wacsf",
        dataset="custom",
        dbpath="tests/data/test_ethanol.db",
        property="energy",
        derivative="forces",
        negative_dr=True,
        output_module="elemental_atomwise",
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


def test_spk_ase(script_runner, tmpdir_factory):
    # define dirs
    molecule_path = "tests/data/test_molecule.xyz"
    modeldir = tmpdir_factory.mktemp("modeldir").strpath
    simdir = tmpdir_factory.mktemp("simdir").strpath

    # train a model on md17
    ret = script_runner.run(
        "spk_run.py",
        "train",
        "schnet",
        "md17",
        "tests/data/test_ethanol.db",
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
        simdir,
        "--optimize",
        "2",
    )
    assert ret.success, ret.stderr
