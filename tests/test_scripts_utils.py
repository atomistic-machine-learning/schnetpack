import pytest
import torch
import numpy as np
import os

import schnetpack.train.metrics
from schnetpack.utils import (
    get_loaders,
    get_statistics,
    setup_run,
    get_trainer,
    simple_loss_fn,
    tradeoff_loss_fn,
    get_representation,
    get_model,
    evaluate,
)
import schnetpack as spk
from numpy.testing import assert_almost_equal
from argparse import Namespace
from shutil import rmtree

# from scripts.schnetpack_parse import main

from .fixtures import *


class TestScripts:
    def test_statistics(self, qm9_dataset, split_path, args):
        # test for statistics not in split file
        if os.path.exists(split_path):
            os.remove(split_path)
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        train, val, test = spk.data.train_test_split(qm9_dataset, 10, 5, split_path)
        train_loader = spk.data.AtomsLoader(train, batch_size=5)
        mean, stddev = get_statistics(
            split_path=split_path,
            train_loader=train_loader,
            args=args,
            atomref=None,
            divide_by_atoms=False,
        )
        energies = []
        for batch in train_loader:
            energies.append(batch["energy_U0"])
        assert_almost_equal(torch.cat(energies).mean(), mean["energy_U0"], 2)

        # test for statistics in split file
        split_file = np.load(split_path)
        saved_mean = split_file["mean"]
        mean, stddev = get_statistics(
            split_path=split_path,
            train_loader=train_loader,
            args=args,
            atomref=None,
            divide_by_atoms=False,
        )
        assert_almost_equal(saved_mean, mean["energy_U0"])

        # test assertion on wrong split file
        with pytest.raises(Exception):
            get_statistics(
                split_path="I/do/not/exist.npz",
                train_loader=train_loader,
                args=args,
                atomref=None,
                divide_by_atoms=False,
            )

    def test_get_loaders(self, qm9_dataset, args, split_path):
        train, val, test = get_loaders(args, qm9_dataset, split_path)
        assert train.dataset.__len__() == args.split[0]
        assert val.dataset.__len__() == args.split[1]


class TestSetup:
    def test_setup_overwrite(self, modeldir):
        test_folder = os.path.join(modeldir, "testing")
        os.makedirs(test_folder)
        args = Namespace(
            mode="train", modelpath=modeldir, overwrite=True, seed=20, dataset="qm9"
        )
        train_args = setup_run(args)
        assert not os.path.exists(test_folder)
        args = Namespace(mode="eval", modelpath=modeldir, seed=20)
        assert train_args == setup_run(args)


class TestTrainer:
    def test_trainer(self, qm9_train_loader, qm9_val_loader, schnet, modeldir):
        args = Namespace(
            max_epochs=1,
            lr=0.01,
            lr_patience=10,
            lr_decay=0.5,
            lr_min=1e-6,
            logger="csv",
            modelpath=modeldir,
            log_every_n_epochs=2,
            max_steps=30,
            checkpoint_interval=1,
            keep_n_checkpoints=1,
            dataset="qm9",
        )
        trainer = get_trainer(
            args, schnet, qm9_train_loader, qm9_val_loader, metrics=None
        )
        assert trainer._model == schnet
        hook_types = [type(hook) for hook in trainer.hooks]
        assert schnetpack.train.hooks.CSVHook in hook_types
        assert schnetpack.train.hooks.TensorboardHook not in hook_types
        assert schnetpack.train.hooks.MaxEpochHook in hook_types
        assert schnetpack.train.hooks.ReduceLROnPlateauHook in hook_types

    def test_tensorboardhook(self, qm9_train_loader, qm9_val_loader, schnet, modeldir):
        # use TensorBoardHook
        if os.path.exists(os.path.join(modeldir, "checkpoints")):
            rmtree(os.path.join(modeldir, "checkpoints"))
        args = Namespace(
            max_epochs=1,
            lr=0.01,
            lr_patience=10,
            lr_decay=0.5,
            lr_min=1e-6,
            logger="tensorboard",
            modelpath=modeldir,
            log_every_n_epochs=2,
            max_steps=30,
            checkpoint_interval=1,
            keep_n_checkpoints=1,
            dataset="qm9",
        )
        trainer = get_trainer(
            args, schnet, qm9_train_loader, qm9_val_loader, metrics=None
        )
        assert schnetpack.train.hooks.TensorboardHook in [
            type(hook) for hook in trainer.hooks
        ]

    def test_simple_loss(self):
        args = Namespace(property="prop")
        loss_fn = simple_loss_fn(args)
        loss = loss_fn(
            {"prop": torch.FloatTensor([100, 100])},
            {"prop": torch.FloatTensor([20, 20])},
        )
        assert loss == 80 ** 2

    def test_tradeoff_loff(self):
        args = Namespace(property="prop", rho=0.0)
        property_names = dict(property=args.property, derivative="der")
        rho = dict(property=0.0, derivative=1.0)
        loss_fn = tradeoff_loss_fn(rho, property_names)
        loss = loss_fn(
            {
                "prop": torch.FloatTensor([100, 100]),
                "der": torch.FloatTensor([100, 100]),
            },
            {"prop": torch.FloatTensor([20, 20]), "der": torch.FloatTensor([40, 40])},
        )
        assert loss == 60 ** 2
        rho = dict(property=1.0, derivative=0.0)
        loss_fn = tradeoff_loss_fn(rho, property_names)
        loss = loss_fn(
            {
                "prop": torch.FloatTensor([100, 100]),
                "der": torch.FloatTensor([100, 100]),
            },
            {"prop": torch.FloatTensor([20, 20]), "der": torch.FloatTensor([40, 40])},
        )
        assert loss == 80 ** 2


class TestModel:
    def test_schnet(self):
        args = Namespace(
            model="schnet",
            cutoff_function="hard",
            features=100,
            n_filters=5,
            interactions=2,
            cutoff=4.0,
            num_gaussians=30,
        )
        repr = get_representation(args)
        assert type(repr) == spk.SchNet
        assert len(repr.interactions) == 2
        assert type(repr) != spk.representation.BehlerSFBlock
        args = Namespace(
            model="schnet",
            cutoff_function="cosine",
            features=100,
            n_filters=5,
            interactions=2,
            cutoff=4.0,
            num_gaussians=30,
        )
        repr = get_representation(args)
        assert type(repr) == spk.SchNet
        assert len(repr.interactions) == 2
        assert type(repr) != spk.representation.BehlerSFBlock
        args = Namespace(
            model="schnet",
            cutoff_function="mollifier",
            features=100,
            n_filters=5,
            interactions=2,
            cutoff=4.0,
            num_gaussians=30,
        )
        repr = get_representation(args)
        assert type(repr) == spk.SchNet
        assert len(repr.interactions) == 2
        assert type(repr) != spk.representation.BehlerSFBlock

    def test_wacsf(self, qm9_train_loader):
        args = Namespace(
            model="wacsf",
            cutoff_function="cosine",
            features=100,
            n_filters=5,
            interactions=3,
            cutoff=4.0,
            num_gaussians=30,
            behler=False,
            elements=["C"],
            radial=22,
            angular=5,
            zetas=[1],
            centered=True,
            crossterms=True,
            standardize=False,
            cuda=False,
        )
        repr = get_representation(args, qm9_train_loader)
        assert type(repr) != spk.SchNet
        assert type(repr) == spk.representation.BehlerSFBlock


class TestEvaluation:
    def test_eval(self, qm9_train_loader, qm9_val_loader, qm9_test_loader, modeldir):
        args = Namespace(
            mode="train",
            model="schnet",
            cutoff_function="hard",
            features=100,
            n_filters=5,
            interactions=2,
            cutoff=4.0,
            num_gaussians=30,
            modelpath=modeldir,
            split=["test"],
            property="energy_U0",
            dataset="qm9",
            parallel=False,
        )
        mean = {args.property: None}
        model = get_model(
            args, train_loader=qm9_train_loader, mean=mean, stddev=mean, atomref=mean
        )

        os.makedirs(modeldir, exist_ok=True)
        evaluate(
            args,
            model,
            qm9_train_loader,
            qm9_val_loader,
            qm9_test_loader,
            "cpu",
            metrics=[
                schnetpack.train.metrics.MeanAbsoluteError(
                    "energy_U0", model_output="energy_U0"
                )
            ],
        )
        assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))
        args.split = ["train"]
        evaluate(
            args,
            model,
            qm9_train_loader,
            qm9_val_loader,
            qm9_test_loader,
            "cpu",
            metrics=[
                schnetpack.train.metrics.MeanAbsoluteError(
                    "energy_U0", model_output="energy_U0"
                )
            ],
        )
        args.split = ["validation"]
        evaluate(
            args,
            model,
            qm9_train_loader,
            qm9_val_loader,
            qm9_test_loader,
            "cpu",
            metrics=[
                schnetpack.train.metrics.MeanAbsoluteError(
                    "energy_U0", model_output="energy_U0"
                )
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
