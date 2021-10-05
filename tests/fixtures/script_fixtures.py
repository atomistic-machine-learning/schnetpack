import pytest
import os
from argparse import Namespace


__all__ = [
    "testdir",
    "modeldir",
    "sim_dir",
    "split_path",
    "args",
    "logger",
    "train_schnet_args",
    "train_wacsf_args",
    # utility functions
    "run_args_from_settings",
]


@pytest.fixture
def testdir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("tmp_testing"))


@pytest.fixture
def modeldir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("modeldir"))


@pytest.fixture
def sim_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("simdir"))


@pytest.fixture
def split_path(modeldir):
    return os.path.join(modeldir, "split.npz")


@pytest.fixture
def args(batch_size, n_train_set, n_validation_set):
    return Namespace(
        property="property1",
        batch_size=batch_size,
        split=[n_train_set, n_validation_set],
        mode="train",
        lr=0.01,
        cuda=False,
    )


@pytest.fixture(params=["csv", "tensorboard"])
def logger(request):
    return request.param


@pytest.fixture
def train_schnet_args(modeldir, ethanol_path, logger):
    return Namespace(
        mode="train",
        model="schnet",
        dataset="custom",
        datapath=ethanol_path,
        modelpath=modeldir,
        cuda=False,
        parallel=False,
        seed=None,
        overwrite=True,
        split_path=None,
        split=[1000, 100],
        max_epochs=5000,
        max_steps=None,
        lr=0.0001,
        lr_patience=25,
        lr_decay=0.8,
        lr_min=1e-06,
        logger=logger,
        log_every_n_epochs=1,
        n_epochs=1000,
        checkpoint_interval=1,
        keep_n_checkpoints=3,
        features=128,
        interactions=6,
        cutoff_function="cosine",
        num_gaussians=50,
        property="property1",
        cutoff=10.0,
        batch_size=100,
        environment_provider="simple",
        derivative=None,
        negative_dr=False,
        force=None,
        contributions=None,
        stress=None,
        aggregation_mode="sum",
        output_module="atomwise",
        normalize_filter=False,
        rho={},
    )


@pytest.fixture
def train_wacsf_args():
    return Namespace(
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


# utility functions
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
