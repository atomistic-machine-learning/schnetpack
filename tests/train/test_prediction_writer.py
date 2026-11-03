import pytorch_lightning as pl
import torch
from torch.utils.data import Subset

import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.train import PredictionWriter

from .test_resume import QM9_PATH, make_task


def test_epoch_predictions_hold_one_entry_per_molecule(tmp_path):
    """With one dataloader, Lightning passes the batch predictions as a flat list,
    which `spkpredict` (write_interval="epoch") has to write to predictions.pt."""
    dataset = ASEAtomsData(
        QM9_PATH,
        load_properties=["energy_U0"],
        transforms=[trn.ASENeighborList(cutoff=5.0), trn.CastTo32()],
    )
    loader = AtomsLoader(Subset(dataset, range(5)), batch_size=2)
    task = make_task()

    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[PredictionWriter(output_dir=str(tmp_path), write_interval="epoch")],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.predict(task, dataloaders=loader)

    predictions = torch.load(tmp_path / "predictions.pt")
    with torch.no_grad():
        expected = torch.cat([task(batch)["energy_U0"] for batch in loader])
    torch.testing.assert_close(predictions["energy_U0"], expected)
