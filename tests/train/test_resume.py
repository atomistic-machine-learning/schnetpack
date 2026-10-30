import os

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import Subset

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.train import ModelCheckpoint
from schnetpack.utils import (
    load_task_from_checkpoint,
    trainer_fit_kwargs_for_checkpoint,
)

QM9_PATH = os.path.join(os.path.dirname(__file__), "../testdata/test_qm9.db")
SEED = 42


def make_task():
    cutoff = 5.0
    representation = spk.representation.SchNet(
        n_atom_basis=8,
        n_interactions=1,
        radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[spk.atomistic.Atomwise(n_in=8, output_key="energy_U0")],
    )
    output = spk.task.ModelOutput(
        name="energy_U0",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )
    return spk.task.AtomisticTask(
        model=model, outputs=[output], optimizer_args={"lr": 1e-3}
    )


def fit(run_dir, max_epochs, ckpt_path=None):
    """One spktrain-like process: seed, build everything, fit (or resume)."""
    pl.seed_everything(SEED, workers=True)
    dataset = ASEAtomsData(
        QM9_PATH,
        load_properties=["energy_U0"],
        transforms=[trn.ASENeighborList(cutoff=5.0), trn.CastTo32()],
    )
    train = AtomsLoader(Subset(dataset, range(12)), batch_size=2, shuffle=True)
    val = AtomsLoader(Subset(dataset, range(12, 16)), batch_size=2)
    task = make_task()
    checkpoint = ModelCheckpoint(
        model_path=str(run_dir / "best_model"),
        dirpath=str(run_dir / "checkpoints"),
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint],
        accelerator="cpu",
        deterministic=True,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(run_dir),
    )
    fit_kwargs = trainer_fit_kwargs_for_checkpoint(trainer) if ckpt_path else {}
    trainer.fit(task, train, val, ckpt_path=ckpt_path, **fit_kwargs)
    return task, checkpoint


def test_resumed_training_matches_uninterrupted(tmp_path):
    """Stopping after 2 epochs and resuming from last.ckpt in a new, re-seeded
    process must give the same weights as training 4 epochs in one go."""
    reference, _ = fit(tmp_path / "uninterrupted", max_epochs=4)

    _, first = fit(tmp_path / "resumed", max_epochs=2)
    resumed, _ = fit(
        tmp_path / "resumed", max_epochs=4, ckpt_path=first.last_model_path
    )

    for name, value in reference.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name


def test_loading_a_task_leaves_the_rng_alone(tmp_path):
    _, checkpoint = fit(tmp_path, max_epochs=1)

    torch.manual_seed(0)
    state = torch.get_rng_state()
    load_task_from_checkpoint(spk.task.AtomisticTask, checkpoint.last_model_path)
    assert torch.equal(torch.get_rng_state(), state)
