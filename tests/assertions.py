import json
import os

import torch
import numpy as np
from numpy.testing import assert_array_equal
from ase.db import connect
from torch.optim import Adam
from torch.nn.modules import MSELoss

from tests.fixtures.script_fixtures import run_args_from_settings

__all__ = [
    "assert_equal_dict",
    "assert_atoms_equal",
    "assert_properties_equal",
    "assert_dataset_equal",
    "assert_database_equal",
    "assert_instance_has_props",
    "assert_params_changed",
    "assert_output_shape_valid",
]


def assert_equal_dict(dict1, dict2):
    """
    Check equality of key value pairs of two dicts.

    """
    # assert equal keys
    assert set(dict1.keys()) == set(dict2.keys())
    # assert equal values
    for key in dict1.keys():
        assert_array_equal(dict1[key], dict2[key])


def assert_atoms_equal(atms1, atms2):
    """
    Check equality of two ase.Atoms objects.

    """
    assert atms1 == atms2


def assert_properties_equal(pdict1, pdict2):
    """
    Check equality of key-value pairs of atomistic properties.

    """
    # extract properties
    clean_dict1, clean_dict2 = extract_properties(pdict1), extract_properties(pdict2)

    # validate equality of properties
    assert_equal_dict(clean_dict1, clean_dict2)


def assert_dataset_equal(ds1, ds2):
    """
    Validate if db1 and db2 contain the same data.

    """
    # check length
    assert len(ds1) == len(ds2), "Lenght of databases does not match."

    # test datapoints
    for idx in range(len(ds1)):
        atms1, data1 = ds1.get_properties(idx)
        atms2, data2 = ds2.get_properties(idx)

        assert_atoms_equal(atms1, atms2)
        assert_properties_equal(data1, data2)


def assert_database_equal(db_path1, db_path2):
    """
    Check equality of two ase database files.

    """
    conn1, conn2 = connect(db_path1), connect(db_path2)

    assert len(conn1) == len(conn2)

    for i in range(len(conn1)):
        atmsrw1, atmsrw2 = conn1.get(i), conn2.get(i)
        atms1, atms2 = atmsrw1.to_atoms(), atmsrw2.to_atoms()
        data1, data2 = atmsrw1.datd, atmsrw2.data

        assert_atoms_equal(atms1, atms2)
        assert_properties_equal(data1, data2)


def extract_properties(pdict):
    """
    Remove _keys from property dictionary.

    Args:
        pdict (dict): raw dictionary

    Returns:
        dict: clean property dictionary

    """
    clean_dict = dict()
    for key, val in pdict.items():
        if key.startswith("_"):
            continue
        clean_dict[key] = val

    return clean_dict


# script assertions
def assert_instance_has_props(instance, pdict, pnames):
    """
    Check if an instance has the desired properties.

    Args:
        instance (object): instance of something
        pdict (dict): property dictionary
        pnames (iterable): iterable with property names

    """
    for pname in pnames:
        assert getattr(instance, pname) == pdict[pname]


# model assertions
def assert_params_changed(model, batch, exclude=[]):
    """
    Check if all model-parameters are updated when training.

    Args:
        model (torch.nn.Module): model to test
        batch (torch.utils.data.Dataset): batch of input data
        exclude (list): layers that are not necessarily updated
    """
    # save state-dict
    torch.save(model.state_dict(), "before")
    # do one training step
    optimizer = Adam(model.parameters())
    loss_fn = MSELoss()
    pred = model(batch)
    loss = loss_fn(pred, torch.rand(pred.shape))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # check if all trainable parameters have changed
    after = model.state_dict()
    before = torch.load("before")
    for key in before.keys():
        if np.array([key.startswith(exclude_layer) for exclude_layer in exclude]).any():
            continue
        assert (
            before[key] != after[key]
        ).any(), "{} layer has not been updated!".format(key)


def assert_output_shape_valid(model, batch, out_shape):
    """
    Check if the model returns the desired output shape.

    Args:
        model (nn.Module): model that needs to be tested
        batch (list): input data
        out_shape (list): desired output shape
    """
    pred = model(*batch)
    assert list(pred.shape) == out_shape, "Model does not return expected shape!"


def assert_valid_script(
    script_runner,
    modeldir,
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
