"""Fixtures shared by the knowledge-distillation tests: small batches of molecules."""

import numpy as np
import pytest
import torch
from ase.build import molecule

import schnetpack as spk
from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.dynamics import BatchNeighborList
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.lightning import AtomsDataModule
from schnetpack.transform import AddOffsets, CastTo32, CastTo64, MatScipyNeighborList

CUTOFF = 5.0
#: the molecules of make_datamodule, in turn
MOLECULES = ("CH4", "H2O", "C2H6", "NH3")


def with_neighbors(batch, cutoff=CUTOFF):
    """``batch`` with the neighbor list a data pipeline would have given it."""
    neighbor_list = BatchNeighborList(
        MatScipyNeighborList(cutoff=cutoff), cutoff_skin=0.0
    )
    return {**batch, **neighbor_list.neighbors(batch)}


def molecules_batch(dtype=torch.float32):
    """Three rattled molecules of different sizes; the same batch on every call."""
    atoms = [molecule("CH4"), molecule("H2O"), molecule("C2H6")]
    for seed, structure in enumerate(atoms):
        structure.rattle(0.05, seed=seed)
    return with_neighbors(atoms_to_batch(atoms, dtype=dtype))


@pytest.fixture
def batch():
    return molecules_batch()


def make_datamodule(
    tmp_path,
    load_properties=(),
    energy=lambda atoms: 0.0,
    molecules=MOLECULES,
    transforms=None,
    **kwargs,
):
    """Twelve rattled molecules, ``molecules`` in turn, each with an energy label
    ``energy(atoms)`` in eV, loaded only if asked.

    The first eight are the training split, the next two validation, the last
    two test. The pipeline builds one neighbor list, 1 Å beyond the cutoff both
    models prune it to, unless ``transforms`` replaces it. ``kwargs`` go to the
    datamodule.
    """
    datapath = str(tmp_path / "structures.db")
    db = ASEAtomsData.create(
        datapath, distance_unit="Ang", property_unit_dict={"energy": "eV"}
    )
    atoms_list = []
    for seed in range(12):
        structure = molecule(molecules[seed % len(molecules)])
        structure.rattle(0.05, seed=seed)
        atoms_list.append(structure)
    db.add_systems(
        [{"energy": np.array([energy(atoms)])} for atoms in atoms_list], atoms_list
    )
    split_file = str(tmp_path / "split.npz")
    np.savez(
        split_file,
        train_idx=np.arange(8),
        val_idx=np.arange(8, 10),
        test_idx=np.arange(10, 12),
    )
    if transforms is None:
        transforms = [MatScipyNeighborList(cutoff=CUTOFF + 1.0), CastTo32()]
    dataset = ASEAtomsData(
        datapath, load_properties=list(load_properties), transforms=transforms
    )
    return AtomsDataModule(
        dataset,
        batch_size=4,
        num_train=8,
        num_val=2,
        num_test=2,
        split_file=split_file,
        num_workers=0,
        **kwargs,
    )


def make_nnp(
    seed=0,
    energy_mean=None,
    cast_to_64=True,
    n_atom_basis=16,
    filter_cutoff=None,
    energy_key=properties.energy,
    force_key=properties.forces,
    atomrefs=False,
):
    """A small SchNet potential.

    ``energy_mean`` (per atom) makes its ``AddOffsets`` add that offset; without
    it the offsets are left for ``initialize_transforms`` to fill in. With
    ``atomrefs`` it adds atomrefs as well, to be estimated there. TorchScript
    cannot script ``CastTo64`` in this version, hence ``cast_to_64``. With
    ``filter_cutoff`` it prunes a longer neighbor list to that cutoff, as a
    student sharing its list with a teacher of larger cutoff does. The weights
    depend on ``seed`` alone, not on the output keys.
    """
    torch.manual_seed(seed)
    representation = spk.model.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=CUTOFF),
        cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
    )
    property_mean = None if energy_mean is None else torch.tensor([energy_mean])
    postprocessors = [CastTo64()] if cast_to_64 else []
    postprocessors.append(
        AddOffsets(
            energy_key,
            add_mean=True,
            add_atomrefs=atomrefs,
            property_mean=property_mean,
            estimate_atomref=atomrefs,
        )
    )
    input_modules = [spk.model.PairwiseDistances()]
    if filter_cutoff is not None:
        input_modules.append(spk.model.FilterShortRange(filter_cutoff))
    return spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=input_modules,
        output_modules=[
            spk.model.Atomwise(n_in=n_atom_basis, output_key=energy_key),
            spk.model.Forces(energy_key=energy_key, force_key=force_key),
        ],
        postprocessors=postprocessors,
    )


@pytest.fixture
def teacher_model():
    return make_nnp(seed=1, energy_mean=-3.0)


@pytest.fixture
def teacher_path(tmp_path, teacher_model):
    path = tmp_path / "teacher.pt"
    torch.save(teacher_model, path)
    return str(path)
