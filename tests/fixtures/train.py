import pytest
import torch
import os
import schnetpack as spk


__all__ = [
    "lr",
    "trainer",
    "keep_n_checkpoints",
    "checkpoint_interval",
    "metrics",
    "hooks",
]


@pytest.fixture
def lr():
    return 0.0001


@pytest.fixture
def keep_n_checkpoints():
    return 3


@pytest.fixture
def checkpoint_interval():
    return 2


@pytest.fixture
def metrics(available_properties):
    metrics = []
    metrics += [spk.metrics.MeanAbsoluteError(prop) for prop in available_properties]
    metrics += [spk.metrics.AngleMAE(prop) for prop in available_properties]
    metrics += [spk.metrics.AngleMSE(prop) for prop in available_properties]
    metrics += [spk.metrics.AngleRMSE(prop) for prop in available_properties]
    # metrics += [spk.metrics.HeatmapMAE(prop) for prop in available_properties]
    metrics += [spk.metrics.LengthMAE(prop) for prop in available_properties]
    metrics += [spk.metrics.LengthMSE(prop) for prop in available_properties]
    metrics += [spk.metrics.LengthRMSE(prop) for prop in available_properties]
    metrics += [spk.metrics.MeanSquaredError(prop) for prop in available_properties]
    metrics += [spk.metrics.ModelBias(prop) for prop in available_properties]
    metrics += [spk.metrics.RootMeanSquaredError(prop) for prop in available_properties]
    # todo: fix
    # metrics += [spk.metrics.SumMAE(prop) for prop in properties]
    return metrics


@pytest.fixture
def hooks(metrics, modeldir):
    return [spk.hooks.CSVHook(os.path.join(modeldir, "csv_log"), metrics)]
    # todo: continue
    # spk.train.TensorboardHook(modeldir,
    #                          metrics)]


@pytest.fixture
def trainer(
    modeldir,
    atomistic_model,
    available_properties,
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
        loss_fn=spk.train.build_mse_loss(available_properties),
        optimizer=torch.optim.Adam(atomistic_model.parameters(), lr=lr),
        train_loader=train_loader,
        validation_loader=val_loader,
        keep_n_checkpoints=keep_n_checkpoints,
        checkpoint_interval=checkpoint_interval,
        validation_interval=1,
        hooks=hooks,
        loss_is_normalized=True,
    )
