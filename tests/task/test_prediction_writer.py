import os

import pytorch_lightning as pl
import torch

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.lightning import AtomisticTask, PredictionWriter

QM9_PATH = os.path.join(os.path.dirname(__file__), "../testdata/test_qm9.db")


def test_epoch_predictions_hold_one_entry_per_molecule(tmp_path):
    """With one dataloader, Lightning passes the batch predictions as a flat list,
    which `spkpredict` (write_interval="epoch") has to write to predictions.pt."""
    dataset = ASEAtomsData(
        QM9_PATH,
        load_properties=["energy_U0"],
        transforms=[trn.ASENeighborList(cutoff=5.0), trn.CastTo32()],
    )
    loader = AtomsLoader(dataset.subset(range(5)), batch_size=2)
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.SchNet(
            n_atom_basis=8,
            n_interactions=1,
            radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=5.0),
            cutoff_fn=spk.nn.CosineCutoff(5.0),
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.Atomwise(n_in=8, output_key="energy_U0")],
    )
    task = AtomisticTask(
        model=model,
        outputs=[spk.ModelOutput("energy_U0", torch.nn.MSELoss())],
        optimizer_args={"lr": 1e-3},
    )

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
    assert predictions["energy_U0"].shape == (5,)
    torch.testing.assert_close(predictions["energy_U0"], expected)
