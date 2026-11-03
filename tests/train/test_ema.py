import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from schnetpack.train import ExponentialMovingAverage


class LinearFit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return ((self.model(x) - y) ** 2).mean()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", ((self.model(x) - y) ** 2).mean())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class WeightProbe(pl.Callback):
    """Records the weights around every training batch and at every validation."""

    def __init__(self, ema):
        self.ema = ema
        self.train_steps = []
        self.validations = []

    @staticmethod
    def weights(pl_module):
        return torch.cat([p.detach().flatten() for p in pl_module.model.parameters()])

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_steps.append(("start", self.weights(pl_module)))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_steps.append(("end", self.weights(pl_module)))

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, *args):
        average = torch.cat([p.flatten() for p in self.ema.ema.shadow_params])
        self.validations.append((self.weights(pl_module), average))


@pytest.mark.parametrize(
    "trainer_kwargs",
    [{}, {"check_val_every_n_epoch": 2}, {"val_check_interval": 2}],
    ids=["validate_every_epoch", "validate_every_2nd_epoch", "validate_within_epoch"],
)
def test_training_continues_on_the_raw_weights(trainer_kwargs):
    """Every training batch must start from the weights the previous one ended
    with, however often validation swaps the EMA average in, and validation
    must see the average."""
    torch.manual_seed(0)
    x = torch.randn(8, 1)
    loader = DataLoader(TensorDataset(x, 3 * x + 5), batch_size=2)
    ema = ExponentialMovingAverage(decay=0.9)
    probe = WeightProbe(ema)
    trainer = pl.Trainer(
        max_epochs=4,
        accelerator="cpu",
        callbacks=[ema, probe],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        **trainer_kwargs,
    )
    trainer.fit(LinearFit(), loader, loader)

    steps = probe.train_steps
    for (kind, ended), (next_kind, started) in zip(steps[:-1], steps[1:]):
        if (kind, next_kind) == ("end", "start"):
            torch.testing.assert_close(started, ended, rtol=0, atol=0)

    assert probe.validations
    for weights, average in probe.validations:
        torch.testing.assert_close(weights, average, rtol=0, atol=0)
