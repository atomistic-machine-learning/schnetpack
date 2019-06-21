import pytest
import torch
from .data import *
from .model import *
import schnetpack as spk


__all__ = [
    "modeldir",
    "lr",
    "trainer",
    "keep_n_checkpoints",
    "checkpoint_interval",
    "metrics",
    "hooks",
]


@pytest.fixture(scope="session")
def modeldir(tmpdir_factory):
    return tmpdir_factory.mktemp("model")


@pytest.fixture(scope="session")
def lr():
    return 0.0001


@pytest.fixture(scope="session")
def keep_n_checkpoints():
    return 3


@pytest.fixture(scope="session")
def checkpoint_interval():
    return 2


@pytest.fixture(scope="session")
def metrics(properties):
    return [spk.metrics.MeanAbsoluteError(prop) for prop in properties]


@pytest.fixture(scope="session")
def hooks(metrics, modeldir):
    return []  # spk.train.CSVHook(modeldir, metrics)]


@pytest.fixture(scope="session")
def trainer(
    modeldir,
    atomistic_model,
    properties,
    lr,
    train_loader,
    val_loader,
    keep_n_checkpoints,
    checkpoint_interval,
    hooks,
):
    return spk.train.Trainer(
        model_path=modeldir,
        model=atomistic_model,
        loss_fn=spk.metrics.build_mse_loss(properties),
        optimizer=torch.optim.Adam(atomistic_model.parameters(), lr=lr),
        train_loader=train_loader,
        validation_loader=val_loader,
        keep_n_checkpoints=keep_n_checkpoints,
        checkpoint_interval=checkpoint_interval,
        validation_interval=1,
        hooks=hooks,
        loss_is_normalized=True,
    )
