import copy
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from ase import Atoms

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.data.device import ArrayStats, DeviceAtomsData, DeviceAtomsLoader
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
    AtomisticTask,
    AtomsDataModule,
    ExponentialMovingAverage,
)
from schnetpack.objectives import (
    calculate_padded_loss,
    extract_targets,
    predict_without_postprocessing,
)
from schnetpack.train import EMA, GraphStep

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
    loss = calculate_padded_loss(outputs, pred, targets, batch)
    expected = 0.3 * torch.nn.functional.mse_loss(
        pred["energy"][:3], targets["E"][:3]
    ) + 0.7 * torch.nn.functional.l1_loss(pred["forces"][:7], targets["F"][:7])
    torch.testing.assert_close(loss, expected)


def gpff_setup(tmp_path, n=48, device="cuda", banded=True):
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
    np.savez(split, train_idx=np.arange(32), val_idx=np.arange(32, 48), test_idx=[])
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
    dm = AtomsDataModule(
        ASEAtomsData(db.datapath, load_properties=[], transforms=transforms),
        batch_size=8,
        num_train=32,
        num_val=16,
        split_file=split,
        val_batch_size=16,
        data_device=device,
        array_dir=os.path.join(tmp_path, "arrays"),
    )
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.PaiNN(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=40, cutoff=CUTOFF),
            cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
            norm_epsilon=1.0,
            banded_filter=banded,
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.AtomwiseVector(n_in=16, output_key=v.output_key)],
    )
    return dm, model, v


def fit(dm, model, v, skip_loss_above, epochs=3, cuda_graph=True):
    task = AtomisticTask(
        model=model,
        outputs=[VariableOutput(v)],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3, "fused": cuda_graph},
        cuda_graph=cuda_graph,
        grad_clip=1.0,
        skip_loss_above=skip_loss_above,
    )
    ema = ExponentialMovingAverage(decay=0.9)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if cuda_graph else "cpu",
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
    assert task._graph_step.graph is not None and dm.train_batches_in_step
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


def test_padded_batches_train_without_cuda_graph(tmp_path):
    """Data on the device without the graph: Lightning's own optimization on padded
    batches; skip_loss_above skips the steps (and the EMA updates) on the host."""
    dm, model, v = gpff_setup(tmp_path, device="cpu", banded=False)
    before = [p.detach().clone() for p in model.parameters()]
    task, trainer, ema = fit(
        dm, model, v, skip_loss_above=-1.0, epochs=1, cuda_graph=False
    )
    assert int(task.skipped_steps) == 4  # Lightning counts the steps it skipped
    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b.detach())
    for a, b in zip(before, ema.ema.shadow_params):
        assert torch.equal(a, b)
    assert math.isfinite(float(trainer.callback_metrics["val_loss"]))
    task, trainer, ema = fit(
        dm, model, v, skip_loss_above=1e3, epochs=1, cuda_graph=False
    )
    assert trainer.global_step == 4 and int(task.skipped_steps) == 0


def test_fast_training_imports_no_lightning():
    """A training loop of your own needs no Lightning for the fast route."""
    code = (
        "import sys\n"
        "from schnetpack.data.device import ArrayStats, DeviceAtomsData, "
        "DeviceAtomsLoader\n"
        "from schnetpack.model import CompiledPotential\n"
        "from schnetpack.train import EMA, GraphStep\n"
        "sys.exit(int('pytorch_lightning' in sys.modules or 'lightning' in "
        "sys.modules))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def plain_loop_setup(tmp_path, device):
    """Energies of small molecules on the device, as a loop of your own sets them
    up: arrays, statistics, transforms, batches and the model."""
    db = ASEAtomsData.create(
        os.path.join(tmp_path, "mols.db"),
        distance_unit="Ang",
        property_unit_dict={"energy": "eV"},
    )
    rng = np.random.RandomState(0)
    atoms = [
        Atoms(numbers=rng.randint(1, 10, 3 + i % 5), positions=rng.randn(3 + i % 5, 3))
        for i in range(24)
    ]
    db.add_systems([{"energy": np.array([rng.randn()])} for _ in atoms], atoms)
    dataset = ASEAtomsData(db.datapath, load_properties=["energy"])
    data = DeviceAtomsData.from_dataset(
        dataset,
        device,
        trn.MatScipyNeighborList(cutoff=3.0),
        os.path.join(tmp_path, "arrays"),
    )
    train_idx = np.arange(16)
    stats = ArrayStats(data, train_idx, dataset.atomrefs)
    transforms = [trn.RemoveOffsets("energy", remove_mean=True), trn.CastTo32()]
    for t in transforms:
        t.initialize(stats)
    train = DeviceAtomsLoader(data, train_idx, 4, transforms, batching="budget")
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.PaiNN(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=3.0),
            cutoff_fn=spk.nn.CosineCutoff(3.0),
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.Atomwise(n_in=16, output_key="energy")],
        postprocessors=[trn.CastTo64(), trn.AddOffsets("energy", add_mean=True)],
    )
    model.initialize_transforms(stats)
    model.to(device)
    return train, model, [spk.ModelOutput("energy", torch.nn.MSELoss())]


def padded_step(model, outputs, optimizer, batch, grad_clip=None):
    optimizer.zero_grad()
    pred = predict_without_postprocessing(model, batch)
    loss = calculate_padded_loss(outputs, pred, extract_targets(outputs, batch), batch)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.detach()


def test_padded_batches_in_a_loop_of_your_own(tmp_path):
    """The data on the device in a plain PyTorch loop: padded batches, the padded
    loss and the gated EMA, without Lightning."""
    train, model, outputs = plain_loop_setup(tmp_path, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model.parameters(), decay=0.9)
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(2):
        for batch in train:
            loss = padded_step(model, outputs, optimizer, batch)
            ema.update(gate=torch.tensor(1.0))
    assert math.isfinite(float(loss))
    assert any(
        not torch.equal(a, p.detach()) for a, p in zip(before, model.parameters())
    )
    assert any(not torch.equal(a, s) for a, s in zip(before, ema.shadow_params))


@cuda
@pytest.mark.slow
def test_graph_step_matches_eager_steps(tmp_path):
    """GraphStep in a loop of your own: the losses and parameters of eager steps on
    the same batches; a skipped step changes nothing."""
    train, model, outputs = plain_loop_setup(tmp_path, "cuda")
    eager = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    eager_optimizer = torch.optim.AdamW(eager.parameters(), lr=1e-3, fused=True)
    step = GraphStep(
        spk.model.CompiledPotential(model, dynamic=False),
        outputs,
        optimizer,
        train.build_batch,
        grad_clip=1.0,
        skip_loss_above=1e3,
    )
    sels = list(train.index_batches)
    for sel in sels:
        loss = step(sel)
        ref = padded_step(eager, outputs, eager_optimizer, train.build_batch(sel), 1.0)
        torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-6)
    for a, b in zip(model.parameters(), eager.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    assert float(step.keep) == 1.0 and int(step.skipped) == 0

    step.skip_loss_above = -1.0
    before = [p.detach().clone() for p in model.parameters()]
    step(sels[0])
    assert float(step.keep) == 0.0 and int(step.skipped) == 1
    for a, p in zip(before, model.parameters()):
        assert torch.equal(a, p.detach())
