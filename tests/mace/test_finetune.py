"""Fine-tuning MaceModel: trainable after loading, pickled with its core, its
E0s the ones the data pipeline removes, and through spktrain end to end."""

import copy
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.autograd import grad

from schnetpack import properties
from schnetpack.dynamics import BatchNeighborList, ForceCalculator
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.interfaces.mace_interface import export_mace
from schnetpack.model import MaceModel, load_mace
from schnetpack.model.mace import mace_atomrefs
from schnetpack.transform import CastTo32, MatScipyNeighborList, RemoveOffsets
from schnetpack.utils.compatibility import load_model

from ..distillation.conftest import make_datamodule, with_neighbors
from .fakes import R_MAX, FakeMace, rattled, reference

MOLECULES = ("CH4", "H2O", "C2H6", "CH3OH")
#: energy per H, C and O atom by which labels of "another level of theory"
#: differ from the stand-in
SHIFT = {1: 0.0, 6: 0.3, 8: -0.2}


def shifted_energy(atoms):
    energy = reference([atoms])[0][0].item()
    return energy + sum(SHIFT[z] for z in atoms.numbers)


def stand_in_forces(atoms):
    energies, positions = reference([atoms])
    return -grad(energies[0], positions[0])[0].numpy()


def removing(atomrefs):
    return [
        RemoveOffsets(properties.energy, remove_atomrefs=True, atomrefs=atomrefs),
        MatScipyNeighborList(cutoff=R_MAX),
        CastTo32(),
    ]


@pytest.fixture
def archive(tmp_path):
    """A trainable stand-in, saved frozen as MACE's calculator leaves it."""
    core = FakeMace(trainable=True).requires_grad_(False)
    return export_mace(core, str(tmp_path / "mace.pt2"))


def strength(model):
    """The stand-in's one parameter."""
    (parameter,) = model.core.parameters()
    return parameter


@pytest.fixture
def batch():
    return with_neighbors(atoms_to_batch(rattled(MOLECULES), dtype=torch.float32))


def test_load_mace_makes_a_frozen_archive_trainable(archive):
    parameters = list(load_mace(archive).parameters())

    assert parameters and all(p.requires_grad for p in parameters)


def test_pickles_with_its_core(archive, batch, tmp_path):
    model = MaceModel(load_mace(archive), dtype="float32")
    path = str(tmp_path / "best_model")

    torch.save(model, path)
    loaded = load_model(path)

    torch.testing.assert_close(
        loaded(dict(batch))[properties.energy], model(dict(batch))[properties.energy]
    )
    assert strength(loaded).dtype == torch.float32
    assert strength(loaded).requires_grad


def test_deep_copies_have_their_own_parameters(archive, batch):
    model = MaceModel(load_mace(archive), dtype="float32")

    copied = copy.deepcopy(model)

    assert strength(copied) is not strength(model)
    torch.testing.assert_close(
        copied(dict(batch))[properties.energy], model(dict(batch))[properties.energy]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_moves_to_the_gpu(archive, batch):
    model = MaceModel(load_mace(archive), dtype="float32")
    expected = model(dict(batch))[properties.energy]

    output = model.cuda()({key: value.cuda() for key, value in batch.items()})

    torch.testing.assert_close(output[properties.energy].cpu(), expected)


def test_an_unknown_e0_source_is_refused(archive):
    with pytest.raises(ValueError, match="unknown E0 source"):
        MaceModel(load_mace(archive), e0s="isolated")


def test_dataset_e0s_are_the_datasets_atomrefs(archive, tmp_path):
    atomrefs = np.zeros(9)
    atomrefs[[1, 6, 8]] = (-13.0, -1030.0, -2040.0)
    datamodule = make_datamodule(
        tmp_path,
        load_properties=[properties.energy],
        energy=shifted_energy,
        molecules=MOLECULES,
        transforms=removing(mace_atomrefs(archive, "dataset")),
        atomrefs={properties.energy: atomrefs.tolist()},
    )
    datamodule.setup()
    model = MaceModel(load_mace(archive), e0s="dataset")

    model.initialize_transforms(datamodule)

    offsets = model.postprocessors[1].atomref
    torch.testing.assert_close(offsets[:9].double(), torch.tensor(atomrefs))
    assert not offsets[9:].any()


def test_dataset_e0s_need_atomrefs_in_the_dataset(archive, tmp_path):
    datamodule = make_datamodule(
        tmp_path,
        load_properties=[properties.energy],
        molecules=MOLECULES,
        transforms=[MatScipyNeighborList(cutoff=R_MAX), CastTo32()],
    )
    datamodule.setup()
    model = MaceModel(load_mace(archive), e0s="dataset")

    with pytest.raises(RuntimeError, match="no atomrefs"):
        model.initialize_transforms(datamodule)


def test_the_fitted_correction_is_the_energy_per_element_the_labels_add(
    archive, tmp_path, monkeypatch
):
    datamodule = make_datamodule(
        tmp_path,
        load_properties=[properties.energy],
        energy=shifted_energy,
        molecules=MOLECULES,
        transforms=removing(mace_atomrefs(archive)),
    )
    datamodule.setup()
    model = MaceModel(load_mace(archive), dtype="float32", fit_e0_correction=True)

    model.initialize_transforms(datamodule)

    expected = torch.tensor([SHIFT[z] for z in (1, 6, 8)], dtype=torch.float64)
    torch.testing.assert_close(
        model.e0_correction[[1, 6, 8]], expected, atol=1e-5, rtol=0
    )

    # a resumed run reads the fit back instead of fitting again
    def refit(self, loader):
        raise AssertionError("fitted again")

    monkeypatch.setattr(MaceModel, "_fit_e0_correction", refit)
    resumed = MaceModel(load_mace(archive), dtype="float32", fit_e0_correction=True)
    resumed.initialize_transforms(datamodule)
    torch.testing.assert_close(resumed.e0_correction, model.e0_correction)


def test_the_correction_is_part_of_the_energy(archive, batch):
    model = MaceModel(load_mace(archive), dtype="float32")
    plain = model(dict(batch))[properties.energy]

    model.e0_correction[6] = 0.3
    corrected = model(dict(batch))[properties.energy]

    carbons = torch.zeros_like(plain).index_add(
        0, batch[properties.idx_m], (batch[properties.Z] == 6).double()
    )
    # the core runs in float32
    torch.testing.assert_close(corrected - plain, 0.3 * carbons, atol=1e-5, rtol=0)


def test_spktrain_fine_tunes_and_stores_a_loadable_model(archive, tmp_path):
    make_datamodule(
        tmp_path,
        energy=shifted_energy,
        forces=stand_in_forces,
        molecules=MOLECULES,
    )
    runs = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import schnetpack.cli; schnetpack.cli.train()",
            "experiment=mace_finetune",
            f"globals.mace_path={archive}",
            f"globals.cutoff={R_MAX}",
            "globals.fit_e0_correction=True",
            f"data.dataset.datapath={tmp_path / 'structures.db'}",
            "data.num_train=8",
            "data.num_val=2",
            "data.num_test=2",
            "data.batch_size=4",
            "data.num_workers=0",
            "trainer.max_epochs=2",
            "trainer.accelerator=cpu",
            f"run.path={runs}",
            "run.id=finetune",
            f"run.data_dir={tmp_path / 'data'}",
            "+seed=0",
            "print_config=False",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    model = load_model(str(runs / "finetune" / "best_model"))
    assert isinstance(model, MaceModel)
    assert model.do_postprocessing
    assert not torch.equal(strength(model).double(), FakeMace().strength), (
        "the weights did not change"
    )
    torch.testing.assert_close(
        model.e0_correction[[1, 6, 8]].float(),
        torch.tensor([SHIFT[z] for z in (1, 6, 8)]),
        atol=1e-4,
        rtol=0,
    )

    calculator = ForceCalculator(
        model,
        neighbor_list=BatchNeighborList(MatScipyNeighborList(cutoff=R_MAX)),
        dtype=torch.float64,
    )
    structures = rattled(MOLECULES)
    energy = calculator.energy(atoms_to_batch(structures, dtype=torch.float64))
    labels = torch.tensor(
        [shifted_energy(atoms) for atoms in structures], dtype=torch.float64
    )
    # absolute energies: the E0s are back, and two epochs move them little
    torch.testing.assert_close(energy, labels, atol=0.05, rtol=0)
