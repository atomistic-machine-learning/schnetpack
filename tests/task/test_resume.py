import os

import pytest
import pytorch_lightning as pl
import torch
import torchmetrics

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.lightning import (
    AtomisticTask,
    AtomsDataModule,
    ExponentialMovingAverage,
    ModelCheckpoint,
)

QM9_PATH = os.path.join(os.path.dirname(__file__), "../testdata/test_qm9.db")
SEED = 42


def make_task():
    cutoff = 5.0
    representation = spk.model.SchNet(
        n_atom_basis=8,
        n_interactions=1,
        radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.Atomwise(n_in=8, output_key="energy_U0")],
    )
    output = spk.ModelOutput(
        name="energy_U0",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )
    return AtomisticTask(model=model, outputs=[output], optimizer_args={"lr": 1e-3})


def fit(run_dir, max_epochs, ckpt_path=None, ema=False, cached=False):
    """One spktrain-like process: seed, build everything, fit (or resume)."""
    pl.seed_everything(SEED, workers=True)
    # with cache=True, the training order comes from a generator of its own
    datamodule = AtomsDataModule(
        ASEAtomsData(
            QM9_PATH,
            load_properties=["energy_U0"],
            transforms=[trn.ASENeighborList(cutoff=5.0)],
        ),
        batch_size=2,
        num_train=12,
        num_val=4,
        split_file=str(run_dir / "split.npz"),
        batch_transforms=[trn.CastTo32()],
        cache=cached,
        cache_dir=str(run_dir / "cache"),
    )
    task = make_task()
    checkpoint = ModelCheckpoint(
        model_path=str(run_dir / "best_model"),
        dirpath=str(run_dir / "checkpoints"),
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint]
    if ema:
        callbacks.append(ExponentialMovingAverage(decay=0.9))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="cpu",
        deterministic=True,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(run_dir),
    )
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True)
    return task, checkpoint


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("ema", [False, True], ids=["plain", "ema"])
def test_resumed_training_matches_uninterrupted(tmp_path, ema, cached):
    """Stopping after 2 epochs and resuming from last.ckpt (loaded weights-only)
    in a new, re-seeded process must give the same weights as training 4 epochs
    in one go."""
    kwargs = dict(ema=ema, cached=cached)
    reference, _ = fit(tmp_path / "uninterrupted", max_epochs=4, **kwargs)

    _, first = fit(tmp_path / "resumed", max_epochs=2, **kwargs)
    resumed, _ = fit(
        tmp_path / "resumed", max_epochs=4, ckpt_path=first.last_model_path, **kwargs
    )

    for name, value in reference.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name


def test_checkpoint_loads_weights_only_into_a_task_built_from_code(tmp_path):
    """No pickled model or hyperparameters in the checkpoint: the task is built
    again and takes the checkpoint's state, with the RNG left alone."""
    trained, checkpoint = fit(tmp_path, max_epochs=1)
    state = torch.load(checkpoint.last_model_path, weights_only=True)
    assert "hyper_parameters" not in state

    task = make_task()
    task.load_state_dict(state["state_dict"], assign=True)
    torch.manual_seed(0)
    rng = torch.get_rng_state()
    task.on_load_checkpoint(state)  # without a trainer: does not touch the RNG
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in trained.state_dict().items():
        assert torch.equal(value, task.state_dict()[name]), name
