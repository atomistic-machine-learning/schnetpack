import torch
import os
import numpy as np
import pytest
import schnetpack as spk
from argparse import Namespace
from numpy.testing import assert_almost_equal

from tests.fixtures import *


# scripts
def test_statistics(
    example_dataset,
    split_path,
    args,
    n_train_set,
    n_validation_set,
    batch_size,
):
    # test for statistics not in split file
    if os.path.exists(split_path):
        os.remove(split_path)
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    train, val, test = spk.data.train_test_split(
        example_dataset, n_train_set, n_validation_set, split_path
    )
    train_loader_tmp = spk.data.AtomsLoader(train, batch_size=batch_size)
    mean, stddev = spk.utils.get_statistics(
        split_path=split_path,
        train_loader=train_loader_tmp,
        args=args,
        atomref=None,
        divide_by_atoms=False,
    )
    energies = []
    for batch in train_loader_tmp:
        energies.append(batch["property1"])
    assert_almost_equal(torch.cat(energies).mean(), mean["property1"], 2)

    # test for statistics in split file
    split_file = np.load(split_path)
    saved_mean = split_file["mean"]
    mean, stddev = spk.utils.get_statistics(
        split_path=split_path,
        train_loader=train_loader_tmp,
        args=args,
        atomref=None,
        divide_by_atoms=False,
    )
    assert_almost_equal(saved_mean, mean["property1"])

    # test assertion on wrong split file
    with pytest.raises(Exception):
        spk.utils.get_statistics(
            split_path="I/do/not/exist.npz",
            train_loader=train_loader_tmp,
            args=args,
            atomref=None,
            divide_by_atoms=False,
        )


def test_get_loaders(example_dataset, args, split_path):
    train, val, test = spk.utils.get_loaders(args, example_dataset, split_path)
    assert train.dataset.__len__() == args.split[0]
    assert val.dataset.__len__() == args.split[1]


# setup
def test_setup_overwrite(modeldir):
    # write testfile
    test_file_path = os.path.join(modeldir, "test_file.txt")
    with open(test_file_path, "w") as test_file:
        test_file.write("in der schublade liegt noch ein schmalzbrot")

    args = Namespace(
        mode="train",
        modelpath=modeldir,
        overwrite=True,
        seed=20,
        dataset="qm9",
    )
    train_args = spk.utils.setup_run(args)

    # check if modeldir is empty
    assert not os.path.exists(test_file_path)

    args = Namespace(mode="eval", modelpath=modeldir, seed=20)
    assert train_args == spk.utils.setup_run(args)


# trainer
def test_trainer(train_schnet_args, train_loader, val_loader, schnet, modeldir):
    trainer = spk.utils.get_trainer(
        train_schnet_args, schnet, train_loader, val_loader, metrics=None
    )
    assert trainer._model == schnet
    hook_types = [type(hook) for hook in trainer.hooks]

    # test for logging hooks
    if train_schnet_args.logger == "csv":
        assert spk.train.hooks.CSVHook in hook_types
        assert spk.train.hooks.TensorboardHook not in hook_types
    else:
        assert spk.train.hooks.CSVHook not in hook_types
        assert spk.train.hooks.TensorboardHook in hook_types

    # test scheduling hooks
    assert spk.train.hooks.MaxEpochHook in hook_types
    assert spk.train.hooks.ReduceLROnPlateauHook in hook_types


def test_simple_loss():
    args = Namespace(property="prop")
    loss_fn = spk.utils.simple_loss_fn(args)
    loss = loss_fn(
        {"prop": torch.FloatTensor([100, 100])},
        {"prop": torch.FloatTensor([20, 20])},
    )
    assert loss == 80 ** 2


def test_tradeoff_loff():
    args = Namespace(property="prop", rho=0.0)
    property_names = dict(property=args.property, derivative="der")
    rho = dict(property=0.0, derivative=1.0)
    loss_fn = spk.utils.tradeoff_loss_fn(rho, property_names)
    loss = loss_fn(
        {
            "prop": torch.FloatTensor([100, 100]),
            "der": torch.FloatTensor([100, 100]),
        },
        {"prop": torch.FloatTensor([20, 20]), "der": torch.FloatTensor([40, 40])},
    )
    assert loss == 60 ** 2
    rho = dict(property=1.0, derivative=0.0)
    loss_fn = spk.utils.tradeoff_loss_fn(rho, property_names)
    loss = loss_fn(
        {
            "prop": torch.FloatTensor([100, 100]),
            "der": torch.FloatTensor([100, 100]),
        },
        {"prop": torch.FloatTensor([20, 20]), "der": torch.FloatTensor([40, 40])},
    )
    assert loss == 80 ** 2


# model
def test_schnet(train_schnet_args):
    # build representation
    repr = spk.utils.get_representation(train_schnet_args)

    # check representation type and n_interactions
    assert type(repr) == spk.SchNet
    assert len(repr.interactions) == train_schnet_args.interactions


# evaluation
def test_eval(train_schnet_args, train_loader, val_loader, test_loader, modeldir):
    mean = {train_schnet_args.property: None}
    model = spk.utils.get_model(
        train_schnet_args,
        train_loader=train_loader,
        mean=mean,
        stddev=mean,
        atomref=mean,
    )

    os.makedirs(modeldir, exist_ok=True)
    train_schnet_args.split = ["test"]
    spk.utils.evaluate(
        train_schnet_args,
        model,
        train_loader,
        val_loader,
        test_loader,
        "cpu",
        metrics=[
            spk.train.metrics.MeanAbsoluteError("property1", model_output="property1")
        ],
    )
    assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))
    train_schnet_args.split = ["train"]
    spk.utils.evaluate(
        train_schnet_args,
        model,
        train_loader,
        val_loader,
        test_loader,
        "cpu",
        metrics=[
            spk.train.metrics.MeanAbsoluteError("property1", model_output="property1")
        ],
    )
    train_schnet_args.split = ["validation"]
    spk.utils.evaluate(
        train_schnet_args,
        model,
        train_loader,
        val_loader,
        test_loader,
        "cpu",
        metrics=[
            spk.train.metrics.MeanAbsoluteError("property1", model_output="property1")
        ],
    )


def test_property_str():
    prop_str = "Properties=species:S:1"
    assert prop_str == spk.data.parse_property_string(prop_str)
    prop_str = "test:R:2"
    assert (
        spk.data.parse_property_string(prop_str)
        == "Properties=species:S:1:pos:R:3:test:R:2"
    )
