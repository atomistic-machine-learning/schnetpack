import torch
import torchmetrics

from schnetpack.task import ModelOutput


def test_model_output_without_metrics():
    output = ModelOutput(name="energy", loss_fn=torch.nn.MSELoss())

    pred, target = {"energy": torch.ones(2)}, {"energy": torch.zeros(2)}
    assert output.calculate_loss(pred, target) == 1.0
    output.update_metrics(pred, target, "train")


def test_model_output_metrics_are_independent_per_subset():
    output = ModelOutput(
        name="energy", metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    assert output.train_metrics["MAE"] is not output.val_metrics["MAE"]
    assert output.val_metrics["MAE"] is not output.test_metrics["MAE"]
