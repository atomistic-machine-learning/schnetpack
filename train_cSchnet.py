import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.properties as props

from schnetpack.datasets import MergedAtomsDataModule, DATASET_REGISTRY


ENERGY_KEY = "energy"
FORCES_KEY = "forces"


# -------------------------
# 1) dataset_id conditioning 
# -------------------------
class ConditionOnDatasetID(nn.Module):
    """
    Adds dataset embedding into scalar_representation (atomwise features).

    dataset_id: (B,) or (B,1)
    idx_m:      (N_atoms,) map atom -> structure index
    """

    def __init__(self, base_representation: nn.Module, num_datasets: int, emb_dim: int, n_atom_basis: int):
        super().__init__()
        self.base_representation = base_representation
        self.dataset_emb = nn.Embedding(num_datasets, emb_dim)
        self.proj = nn.Linear(emb_dim, n_atom_basis)

    def forward(self, inputs: dict) -> dict:
        out = self.base_representation(inputs)

        dataset_id = out["dataset_id"]
        if dataset_id.dim() == 2 and dataset_id.size(-1) == 1:
            dataset_id = dataset_id.squeeze(-1)
        dataset_id = dataset_id.long()  # (B,)

        idx_m = out[props.idx_m].long()  # (N_atoms,)
        e_struct = self.dataset_emb(dataset_id)   # (B, emb_dim)
        e_atoms = e_struct[idx_m]                 # (N_atoms, emb_dim)

        h = out["scalar_representation"]          # (N_atoms, n_atom_basis)
        out["scalar_representation"] = h + self.proj(e_atoms)
        return out


# -------------------------
# 2) Lightning module (forces need grads in val/test)
# -------------------------
class EnergyForcesLit(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-4, w_energy=1.0, w_forces=1.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.w_energy = w_energy
        self.w_forces = w_forces
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _batch_with_pos_grad(self, batch: dict) -> dict:
        batch_in = {}
        for k, v in batch.items():
            if k == props.position:
                batch_in[k] = v.detach().clone().requires_grad_(True)
            else:
                batch_in[k] = v
        return batch_in

    def _shared_step(self, batch: dict, stage: str):
        batch_in = self._batch_with_pos_grad(batch)
        pred = self.model(batch_in)

        e_pred = pred[ENERGY_KEY].view(-1)
        e_true = batch[ENERGY_KEY].view(-1)

        f_pred = pred[FORCES_KEY]
        f_true = batch[FORCES_KEY]

        loss_e = F.l1_loss(e_pred, e_true)
        loss_f = F.l1_loss(f_pred, f_true)
        loss = self.w_energy * loss_e + self.w_forces * loss_f

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/mae_energy", loss_e, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/mae_forces", loss_f, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        was_training = self.model.training
        self.model.train()
        with torch.enable_grad():
            loss = self._shared_step(batch, "val")
        if not was_training:
            self.model.eval()
        return loss

    def test_step(self, batch, batch_idx):
        was_training = self.model.training
        self.model.train()
        with torch.enable_grad():
            loss = self._shared_step(batch, "test")
        if not was_training:
            self.model.eval()
        return loss


# -------------------------
# 3) Build SchNetPack model + postprocessors offsets
# -------------------------
def build_conditioned_schnet(
    cutoff: float,
    n_atom_basis: int,
    n_interactions: int,
    num_datasets: int,
    emb_dim: int,
    add_mean: bool,
    add_atomrefs: bool,
):
    radial_basis = spk.nn.GaussianRBF(n_rbf=50, cutoff=cutoff)
    cutoff_fn = spk.nn.CosineCutoff(cutoff=cutoff)

    base_rep = spk.representation.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    rep = ConditionOnDatasetID(
        base_representation=base_rep,
        num_datasets=num_datasets,
        emb_dim=emb_dim,
        n_atom_basis=n_atom_basis,
    )

    pred_energy = spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        output_key=ENERGY_KEY,
        aggregation_mode="sum",
    )
    pred_forces = spk.atomistic.Forces(
        energy_key=ENERGY_KEY,
        force_key=FORCES_KEY,
    )

    model = spk.model.NeuralNetworkPotential(
        representation=rep,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(ENERGY_KEY, add_mean=add_mean, add_atomrefs=add_atomrefs),
        ],
    )
    return model


# -------------------------
# 4) Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_atomrefs", type=int, default=0)   # 1/0
    parser.add_argument("--remove_mean", type=int, default=0)    # 1/0
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="cpu")  
    args = parser.parse_args()

    use_atomrefs = bool(args.use_atomrefs)
    remove_mean = bool(args.remove_mean)

    pl.seed_everything(42)

    dataset_root = "data_ethanol"
    os.makedirs(dataset_root, exist_ok=True)

    max_id = max(spec.dataset_id for spec in DATASET_REGISTRY.values())
    num_datasets = max_id + 1

    dm = MergedAtomsDataModule(
        dataset_names=["MD17", "rMD17"],
        proportions={"MD17": 0.5, "rMD17": 0.5},
        molecule="ethanol",
        dataset_root=dataset_root,
        total_size=10000,
        num_train=7000,
        num_val=2000,
        num_test=1000,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        seed=42,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.RemoveOffsets(ENERGY_KEY, remove_mean=remove_mean, remove_atomrefs=use_atomrefs),
            trn.CastTo32(),
        ],
        load_properties=[ENERGY_KEY, FORCES_KEY],
    )

    print("Preparing datasets...")
    dm.prepare_data()
    dm.setup()

    print("\nuse_atomrefs =", use_atomrefs, "| remove_mean =", remove_mean)
    print("dm.merged_atomrefs is None:", dm.merged_atomrefs is None)

    model = build_conditioned_schnet(
        cutoff=5.0,
        n_atom_basis=64,
        n_interactions=3,
        num_datasets=num_datasets,
        emb_dim=16,
        add_mean=remove_mean,
        add_atomrefs=use_atomrefs,
    )

    lit = EnergyForcesLit(model=model, lr=1e-4, w_energy=1.0, w_forces=1.0)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        default_root_dir="./runs_conditioned_model",
        log_every_n_steps=10,
        inference_mode=False,  
    )

    trainer.fit(lit, datamodule=dm)
    trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    main()
