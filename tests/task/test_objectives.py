import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanAbsoluteError

from schnetpack.lightning import AtomisticTask
from schnetpack.model.base import AtomisticModel
from schnetpack.objectives import (
    AtomMask,
    ModelOutput,
    UnsupervisedModelOutput,
    calculate_loss,
    extract_targets,
)


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


def batch_loss(outputs, model, batch):
    """The loss of a hand-written training loop."""
    targets = extract_targets(outputs, batch)
    pred = model(batch)
    return calculate_loss(outputs, pred, targets)


def test_calculate_loss_weighted_composite():
    model = LinearModel()
    outputs = [make_output(loss_weight=0.5)]
    batch = make_batch()

    loss = batch_loss(outputs, model, batch)

    expected = 0.5 * nn.functional.mse_loss(batch["y"], batch["y_ref"])
    assert torch.isclose(loss, expected)


def test_targets_extracted_before_the_forward_pass_survive_overwrite():
    """SchNetPack models write predictions into the input dict. If the output
    name equals the target property (the common case, e.g. energy_U0), the
    targets must be extracted before the forward pass — otherwise the loss
    silently compares the prediction with itself and is always zero."""
    model = LinearModel()  # writes pred to inputs["y"]
    outputs = [ModelOutput(name="y", loss_fn=nn.MSELoss(), metrics={})]
    batch = make_batch()
    batch["y"] = batch.pop("y_ref")  # target under the same key as the output

    loss = batch_loss(outputs, model, batch)

    assert loss.item() > 0.0
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_plain_torch_training_loop_reduces_loss():
    """Model + outputs must be trainable in a hand-written loop, no Lightning."""
    torch.manual_seed(0)
    model = LinearModel()
    outputs = [make_output()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    batch = make_batch()

    initial = batch_loss(outputs, model, batch).item()
    for _ in range(100):
        loss = batch_loss(outputs, model, batch)
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

    plain = batch_loss(outputs, model, dict(batch))
    targets = extract_targets(task.outputs, batch)
    pred = task.predict_without_postprocessing(dict(batch))
    via_task = task.loss_fn(pred, targets)

    assert torch.isclose(plain, via_task)


def masked_output(masks, target_property="y_ref"):
    return ModelOutput(
        name="y",
        target_property=target_property,
        loss_fn=nn.MSELoss(),
        metrics={"mae": MeanAbsoluteError()},
        masks=masks,
    )


def test_a_mask_restricts_loss_and_metrics_to_the_selected_entries():
    model = LinearModel()
    batch = make_batch()
    keep = torch.arange(16) % 2 == 0
    batch["considered"] = keep
    output = masked_output([AtomMask("considered")])

    targets = extract_targets([output], batch)
    pred = model(batch)
    loss = output.calculate_loss(pred, targets)
    output.update_metrics(pred, targets, "train")

    y, y_ref = pred["y"][keep], batch["y_ref"][keep]
    assert torch.isclose(loss, nn.functional.mse_loss(y, y_ref))
    mae = output.metrics["train"]["mae"].compute()
    assert torch.isclose(mae, (y - y_ref).abs().mean())


def test_a_mask_on_one_output_leaves_another_on_the_same_prediction_whole():
    """Two terms on the same prediction, e.g. forces against labels and
    against the teacher: masking one must not shrink the other's prediction."""
    model = LinearModel()
    batch = make_batch()
    batch["y_other"] = torch.zeros(16, 1)
    keep = torch.arange(16) < 4
    batch["considered"] = keep
    outputs = [
        masked_output([AtomMask("considered")]),
        ModelOutput(
            name="y", target_property="y_other", loss_fn=nn.MSELoss(), metrics={}
        ),
    ]

    loss = batch_loss(outputs, model, batch)

    y = batch["y"]
    expected = nn.functional.mse_loss(
        y[keep], batch["y_ref"][keep]
    ) + nn.functional.mse_loss(y, batch["y_other"])
    assert torch.isclose(loss, expected)


def test_the_masks_of_one_output_combine():
    model = LinearModel()
    batch = make_batch()
    batch["first_half"] = torch.arange(16) < 8
    batch["even"] = torch.arange(16) % 2 == 0
    output = masked_output([AtomMask("first_half"), AtomMask("even")])

    loss = batch_loss([output], model, batch)

    keep = batch["first_half"] & batch["even"]
    expected = nn.functional.mse_loss(batch["y"][keep], batch["y_ref"][keep])
    assert torch.isclose(loss, expected)


def test_a_mask_key_missing_from_the_batch_names_mask_and_output():
    output = masked_output([AtomMask("considered")])

    with pytest.raises(KeyError, match="AtomMask of output 'y' reads 'considered'"):
        extract_targets([output], make_batch())


def test_a_mask_of_the_wrong_length_names_the_output():
    batch = make_batch()
    batch["considered"] = torch.ones(3, dtype=torch.bool)
    output = masked_output([AtomMask("considered")])

    with pytest.raises(ValueError, match="output 'y'"):
        batch_loss([output], LinearModel(), batch)


def test_an_unsupervised_output_takes_no_masks():
    with pytest.raises(ValueError, match="masks"):
        UnsupervisedModelOutput(
            name="y",
            loss_fn=nn.MSELoss(),
            metrics={},
            masks=[AtomMask("considered")],
        )


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
