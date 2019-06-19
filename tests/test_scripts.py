import pytest
import torch
import numpy as np
from .fixtures import *
from src.scripts.script_utils import get_loaders, get_statistics, get_main_parser,\
    add_subparsers, setup_run
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
            split_path=split_path, train_loader=train_loader, args=args, atomref=None,
            per_atom=False
        )
        energies = []
        for batch in train_loader:
            energies.append(batch["energy_U0"])
        assert_almost_equal(torch.cat(energies).mean(), mean["energy_U0"], 2)

        # test for statistics in split file
        split_file = np.load(split_path)
        saved_mean = split_file["mean"]
        mean, stddev = get_statistics(
            split_path=split_path, train_loader=train_loader, args=args, atomref=None,
            per_atom=False
        )
        assert_almost_equal(saved_mean, mean["energy_U0"])

        # test assertion on wrong split file
        with pytest.raises(Exception):
            get_statistics(split_path="I/do/not/exist.npz", train_loader=train_loader,
                           args=args, atomref=None, per_atom=False)

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
        args = parser.parse_args(["train", "schnet", "data/qm9.db", "model",
                                  "--split", "10000", "1000"])
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
        rmtree(modeldir)