import pytest
import os
from argparse import Namespace


__all__ = ["tmp_dir", "modeldir", "split_path", "args"]


@pytest.fixture(scope="module")
def tmp_dir(tmpdir_factory):
    modeldir = tmpdir_factory.mktemp("modeldir")
    return modeldir


@pytest.fixture(scope="module")
def modeldir(tmp_dir):
    return os.path.join(tmp_dir, "modeldir")


@pytest.fixture(scope="module")
def split_path(modeldir):
    return os.path.join(modeldir, "split.npz")


@pytest.fixture(scope="module")
def args(batch_size, qm9_split):
    return Namespace(property="energy_U0", batch_size=batch_size, split=qm9_split)
