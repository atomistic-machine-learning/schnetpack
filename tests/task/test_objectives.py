import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanAbsoluteError

from schnetpack.model.base import AtomisticModel
from schnetpack.objectives import ModelOutput, compute_loss
from schnetpack.task import AtomisticTask


class LinearModel(AtomisticModel):
    """Minimal AtomisticModel: predicts y = w * x."""

    def __init__(self):
        super().__init__(postprocessors=None, do_postprocessing=False)
        self.linear = nn.Linear(1, 1, bias=False)
        self.collect_derivatives()

    def forward(self, inputs):
        inputs["y"] = self.linear(inputs["x"])
        return inputs


def make_batch(n=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=generator)
    return {"x": x, "y_ref": 2.0 * x, "_idx": torch.arange(n)}


def make_output(loss_weight=1.0):
    return ModelOutput(
        name="y",
        loss_fn=nn.MSELoss(),
        loss_weight=loss_weight,
        metrics={"mae": MeanAbsoluteError()},
        target_property="y_ref",
    )


def test_compute_loss_weighted_composite():
    model = LinearModel()
    outputs = [make_output(loss_weight=0.5)]
    batch = make_batch()

    loss = compute_loss(outputs, model, batch)

    expected = 0.5 * nn.functional.mse_loss(batch["y"], batch["y_ref"])
    assert torch.isclose(loss, expected)


def test_compute_loss_when_model_overwrites_target_key():
    """SchNetPack models write predictions into the input dict. If the output
    name equals the target property (the common case, e.g. energy_U0), the
    targets must be extracted before the forward pass — otherwise the loss
    silently compares the prediction with itself and is always zero."""
    model = LinearModel()  # writes pred to inputs["y"]
    outputs = [ModelOutput(name="y", loss_fn=nn.MSELoss(), metrics={})]
    batch = make_batch()
    batch["y"] = batch.pop("y_ref")  # target under the same key as the output

    loss = compute_loss(outputs, model, batch)

    assert loss.item() > 0.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_plain_torch_training_loop_reduces_loss():
    """Model + outputs must be trainable in a hand-written loop, no Lightning."""
    torch.manual_seed(0)
    model = LinearModel()
    outputs = [make_output()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    batch = make_batch()

    initial = compute_loss(outputs, model, batch).item()
    for _ in range(100):
        loss = compute_loss(outputs, model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01 * initial
    assert torch.isclose(model.linear.weight.squeeze(), torch.tensor(2.0), atol=0.1)


def test_task_and_plain_loop_compute_identical_loss():
    model = LinearModel()
    outputs = [make_output(loss_weight=0.7)]
    task = AtomisticTask(model=model, outputs=outputs, optimizer_args={"lr": 1e-3})
    batch = make_batch()

    plain = compute_loss(outputs, model, dict(batch))
    from schnetpack.objectives import extract_targets

    targets = extract_targets(task.outputs, batch)
    pred = task.predict_without_postprocessing(dict(batch))
    pred, targets = task.apply_constraints(pred, targets)
    via_task = task.loss_fn(pred, targets)

    assert torch.isclose(plain, via_task)


def test_trainer_fast_dev_run(tmp_path):
    model = LinearModel()
    task = AtomisticTask(
        model=model, outputs=[make_output()], optimizer_args={"lr": 1e-3}
    )

    batch = make_batch()
    dataset = TensorDataset(batch["x"], batch["y_ref"], batch["_idx"])
    loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=lambda samples: {
            "x": torch.stack([s[0] for s in samples]),
            "y_ref": torch.stack([s[1] for s in samples]),
            "_idx": torch.stack([s[2] for s in samples]),
        },
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
