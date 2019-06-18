import pytest
import torch
import numpy as np
from .fixtures import *
from src.scripts.script_utils import get_loaders, get_statistics
import schnetpack as spk
from numpy.testing import assert_almost_equal


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
