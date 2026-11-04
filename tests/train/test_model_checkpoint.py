import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from schnetpack.train import ModelCheckpoint


class ConstantValLossModule(pl.LightningModule):
    """Trains a linear layer but always reports the same validation loss, so
    only the first epoch produces a new best checkpoint."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", torch.tensor(1.0))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def save_model(self, model_path, do_postprocessing=None):
        torch.save(self.layer, model_path)


@pytest.mark.parametrize("save_on_train_epoch_end", [False, True])
def test_last_checkpoint_holds_latest_epoch(tmp_path, save_on_train_epoch_end):
    """`last.ckpt` must hold the latest epoch even when the monitored metric
    stops improving, so that resuming from it does not repeat epochs."""
    data = TensorDataset(torch.randn(8, 2), torch.randn(8, 1))
    checkpoint = ModelCheckpoint(
        model_path=str(tmp_path / "best_model"),
        dirpath=str(tmp_path / "checkpoints"),
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=save_on_train_epoch_end,
    )
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[checkpoint],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(
        ConstantValLossModule(),
        train_dataloaders=DataLoader(data, batch_size=4),
        val_dataloaders=DataLoader(data, batch_size=4),
    )

    last = torch.load(checkpoint.last_model_path, weights_only=False)
    assert last["epoch"] == 2
    assert last["global_step"] == trainer.global_step

    # the best checkpoint stays at the only improving epoch
    best = torch.load(checkpoint.best_model_path, weights_only=False)
    assert best["epoch"] == 0
