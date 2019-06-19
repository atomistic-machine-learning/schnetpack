import pytest
import torch
import numpy as np
import os
from .fixtures import *
from src.scripts.script_utils import (
    get_loaders,
    get_statistics,
    get_main_parser,
    add_subparsers,
    setup_run,
    get_trainer,
    simple_loss_fn,
    tradeoff_loff_fn,
    get_representation,
    get_model,
    evaluate,
)
import schnetpack as spk
from numpy.testing import assert_almost_equal
from argparse import Namespace
from shutil import rmtree


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
            per_atom=False,
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
            per_atom=False,
        )
        assert_almost_equal(saved_mean, mean["energy_U0"])

        # test assertion on wrong split file
        with pytest.raises(Exception):
            get_statistics(
                split_path="I/do/not/exist.npz",
                train_loader=train_loader,
                args=args,
                atomref=None,
                per_atom=False,
            )

    def test_get_loaders(self, qm9_dataset, args, split_path):
        train, val, test = get_loaders(args, qm9_dataset, split_path)
        assert train.dataset.__len__() == args.split[0]
        assert val.dataset.__len__() == args.split[1]


class TestParser:
    def test_main_parser(self):
        parser = get_main_parser()
        args = parser.parse_args([])
        assert type(args.batch_size) == int

        with pytest.raises(SystemExit):
            args = parser.parse_args(["--wrong"])

    def test_subparser(self):
        parser = get_main_parser()
        add_subparsers(parser)
        args = parser.parse_args(
            ["train", "schnet", "data/qm9.db", "model", "--split", "10000", "1000"]
        )
        assert args.mode == "train"
        assert args.model == "schnet"

        with pytest.raises(SystemExit):
            args = parser.parse_args([])


class TestSetup:
    def test_setup_overwrite(self, modeldir):
        test_folder = os.path.join(modeldir, "testing")
        os.makedirs(test_folder)
        args = Namespace(mode="train", modelpath=modeldir, overwrite=True, seed=20)
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
        )
        trainer = get_trainer(
            args, schnet, qm9_train_loader, qm9_val_loader, metrics=None, loss_fn=None
        )
        assert trainer._model == schnet
        hook_types = [type(hook) for hook in trainer.hooks]
        assert spk.train.CSVHook in hook_types
        assert spk.train.TensorboardHook not in hook_types
        assert spk.train.MaxEpochHook in hook_types
        assert spk.train.ReduceLROnPlateauHook in hook_types

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
        )
        trainer = get_trainer(
            args, schnet, qm9_train_loader, qm9_val_loader, metrics=None, loss_fn=None
        )
        assert spk.train.TensorboardHook in [type(hook) for hook in trainer.hooks]

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
        loss_fn = tradeoff_loff_fn(args, derivative="der")
        loss = loss_fn(
            {
                "prop": torch.FloatTensor([100, 100]),
                "der": torch.FloatTensor([100, 100]),
            },
            {"prop": torch.FloatTensor([20, 20]), "der": torch.FloatTensor([40, 40])},
        )
        assert loss == 60 ** 2
        args = Namespace(property="prop", rho=1.0)
        loss_fn = tradeoff_loff_fn(args, derivative="der")
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
            model="schnet",
            cutoff_function="hard",
            features=100,
            n_filters=5,
            interactions=2,
            cutoff=4.0,
            num_gaussians=30,
            modelpath=modeldir,
            split="test",
        )
        repr = get_representation(args)
        output_module = spk.Atomwise(args.features, property="energy_U0")
        model = get_model(repr, output_module)

        evaluate(
            args,
            model,
            qm9_train_loader,
            qm9_val_loader,
            qm9_test_loader,
            "cpu",
            metrics=[
                spk.metrics.MeanAbsoluteError("energy_U0", model_output="energy_U0")
            ],
        )
        assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))
        args.split = "train"
        evaluate(
            args,
            model,
            qm9_train_loader,
            qm9_val_loader,
            qm9_test_loader,
            "cpu",
            metrics=[
                spk.metrics.MeanAbsoluteError("energy_U0", model_output="energy_U0")
            ],
        )
        args.split = "val"
        evaluate(
            args,
            model,
            qm9_train_loader,
            qm9_val_loader,
            qm9_test_loader,
            "cpu",
            metrics=[
                spk.metrics.MeanAbsoluteError("energy_U0", model_output="energy_U0")
            ],
        )
