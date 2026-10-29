import math
import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from ase import Atoms

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.generative import (
    Diffuse,
    GaussianInterpolant,
    GaussianPrior,
    LogNormalSigmaTimes,
    PseudoForceParametrization,
    Variable,
    VariableOutput,
    VELinear,
)
from schnetpack.lightning import (
    DeviceAtomsDataModule,
    ExponentialMovingAverage,
    GraphStepTask,
    padded_loss,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SIGMA_MAX = 5.0
CUTOFF = 5 * SIGMA_MAX


def test_padded_loss_is_the_loss_of_the_real_rows():
    g = torch.Generator().manual_seed(0)
    mask_s = torch.tensor([1.0, 1.0, 1.0, 0.0])
    mask_a = torch.tensor([1.0] * 7 + [0.0] * 3)
    batch = {properties.structure_mask: mask_s, properties.atom_mask: mask_a}
    pred = {
        "energy": torch.randn(4, generator=g),
        "forces": torch.randn(10, 3, generator=g),
    }
    targets = {"E": torch.randn(4, generator=g), "F": torch.randn(10, 3, generator=g)}
    outputs = [
        spk.ModelOutput("energy", torch.nn.MSELoss(), 0.3, target_property="E"),
        spk.ModelOutput("forces", torch.nn.L1Loss(), 0.7, target_property="F"),
    ]
    loss = padded_loss(outputs, pred, targets, batch)
    expected = 0.3 * torch.nn.functional.mse_loss(
        pred["energy"][:3], targets["E"][:3]
    ) + 0.7 * torch.nn.functional.l1_loss(pred["forces"][:7], targets["F"][:7])
    torch.testing.assert_close(loss, expected)


def gpff_setup(tmp_path, n=48):
    db = ASEAtomsData.create(
        os.path.join(tmp_path, "mols.db"), distance_unit="Ang", property_unit_dict={}
    )
    rng = np.random.RandomState(0)
    atoms = []
    for i in range(n):
        k = 3 + i % 5
        atoms.append(Atoms(numbers=rng.randint(1, 10, k), positions=rng.randn(k, 3)))
    db.add_systems([{} for _ in atoms], atoms)
    split = os.path.join(tmp_path, "split.npz")
    np.savez(split, train_idx=np.arange(32), val_idx=np.arange(32, 48))
    noising = GaussianInterpolant(
        VELinear(sigma_max=SIGMA_MAX, t_min=0.0),
        prior=GaussianPrior(std=SIGMA_MAX, centered=False),
    )
    v = Variable(
        name="positions",
        view=properties.R,
        per="atom",
        noising=noising,
        parametrization=PseudoForceParametrization(),
        output_key="pseudo_force_pred",
        weight=lambda schedule, t, x_t: (1.0 / schedule.sigma(t) ** 2).clamp(max=1.0),
    )
    transforms = [
        trn.SubtractCenterOfGeometry(),
        Diffuse([v], t_sampler=LogNormalSigmaTimes(noising, truncate=True)),
        trn.CastTo32(),
    ]
    dm = DeviceAtomsDataModule(
        ASEAtomsData(db.datapath, load_properties=[]),
        batch_size=8,
        split_file=split,
        cutoff=CUTOFF,
        transforms=transforms,
        val_batch_size=16,
        cache_dir=os.path.join(tmp_path, "arrays"),
    )
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.PaiNN(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=40, cutoff=CUTOFF),
            cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
            norm_epsilon=1.0,
            banded_filter=True,
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.AtomwiseVector(n_in=16, output_key=v.output_key)],
    )
    return dm, model, v


def fit(dm, model, v, skip_loss_above, epochs=3):
    task = GraphStepTask(
        model=model,
        outputs=[VariableOutput(v)],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3, "fused": True},
        grad_clip=1.0,
        skip_loss_above=skip_loss_above,
    )
    ema = ExponentialMovingAverage(decay=0.9)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        inference_mode=False,
        log_every_n_steps=2,
        callbacks=[ema],
    )
    trainer.fit(task, datamodule=dm)
    return task, trainer, ema


@cuda
@pytest.mark.slow
def test_graph_step_trains(tmp_path):
    dm, model, v = gpff_setup(tmp_path)
    before = [p.detach().clone() for p in model.parameters()]
    task, trainer, ema = fit(dm, model, v, skip_loss_above=1e3)
    assert task.graph is not None and dm.train_batches_in_step
    assert trainer.global_step == 3 * 4
    assert math.isfinite(float(trainer.callback_metrics["val_loss"]))
    assert int(task.skipped_steps) == 0
    changed = [
        not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters())
    ]
    assert all(changed)


@cuda
@pytest.mark.slow
def test_skipped_steps_change_nothing(tmp_path):
    dm, model, v = gpff_setup(tmp_path)
    before = [p.detach().clone() for p in model.parameters()]
    task, trainer, ema = fit(dm, model, v, skip_loss_above=-1.0, epochs=1)
    assert int(task.skipped_steps) == 4
    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b.detach().cpu())  # Lightning moves the model back
    for a, b in zip(before, ema.ema.shadow_params):
        assert torch.equal(a, b.cpu())
