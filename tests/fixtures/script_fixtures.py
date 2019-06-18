import pytest
import os
from argparse import Namespace


@pytest.fixture(scope="session")
def tmp_dir():
    return os.path.join("tmp", "testing")


@pytest.fixture(scope="session")
def modeldir(tmp_dir):
    return os.path.join(tmp_dir, "modeldir")


@pytest.fixture(scope="session")
def split_path(modeldir):
    return os.path.join(modeldir, "split.npz")


@pytest.fixture(scope="session")
def args(batch_size, qm9_split):
    return Namespace(property="energy_U0", batch_size=batch_size, split=qm9_split)
