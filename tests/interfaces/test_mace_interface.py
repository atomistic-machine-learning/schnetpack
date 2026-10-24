"""The MACE interface: MACE-OFF exported with torch.export, and MACE models
with random weights built for a dataset. The golden test runs SchNetPack's
MaceTeacher on the exported archive against eager MACE; it and the built
models need mace-torch, and are skipped without."""

import importlib.util
import os
import subprocess
import sys

import ase.db
import numpy as np
import pytest
import torch

from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.dynamics import BatchNeighborList, ForceCalculator
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.interfaces.mace_interface import (
    CACHE_DIR,
    _avg_num_neighbors,
    _force_rms,
    build_mace,
    export_mace_off,
)
from schnetpack.model import MaceModel, load_mace
from schnetpack.train.teacher import MaceTeacher
from schnetpack.transform import MatScipyNeighborList
from schnetpack.utils.compatibility import load_model

from ..distillation.conftest import absolute_energy, make_datamodule, with_neighbors

NO_WEIGHTS_ONLY = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def test_exporting_without_mace_asks_for_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mace", None)

    with pytest.raises(ImportError, match=r"schnetpack\[mace\]"):
        export_mace_off()


def test_an_unknown_size_is_refused():
    with pytest.raises(ValueError, match="huge"):
        export_mace_off("huge")


def test_the_archive_lands_in_the_cache(mace_off_archive):
    assert os.path.dirname(mace_off_archive) == CACHE_DIR


def test_a_second_call_reuses_the_cache(mace_off_archive):
    written = os.path.getmtime(mace_off_archive)

    assert export_mace_off("medium") == mace_off_archive
    assert os.path.getmtime(mace_off_archive) == written


def test_exporting_leaves_torch_load_safe(mace_off_archive):
    """Importing mace turns off torch.load's weights_only for the process, on
    its first import only: hence a fresh interpreter."""
    env = {k: v for k, v in os.environ.items() if k != NO_WEIGHTS_ONLY}
    script = (
        "import os\n"
        "from schnetpack.interfaces.mace_interface import export_mace_off\n"
        "export_mace_off('medium')\n"
        f"assert {NO_WEIGHTS_ONLY!r} not in os.environ, 'weights_only is off'\n"
    )

    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_the_exported_archive_reproduces_mace(mace_off_archive, mace_off_reference):
    """The golden test: SchNetPack's batches, neighbor list and E0s through the
    archive give MACE's own energies and forces."""
    teacher = MaceTeacher(model_path=mace_off_archive, cutoff=5.0, dtype="float64")
    for name, atoms, energy, forces in mace_off_reference:
        batch = with_neighbors(atoms_to_batch([atoms], dtype=torch.float64))

        targets = teacher(batch)

        torch.testing.assert_close(
            absolute_energy(teacher, batch, targets),
            torch.tensor([float(energy)], dtype=torch.float64),
            rtol=0,
            atol=1e-6,
            msg=name,
        )
        torch.testing.assert_close(
            targets[properties.teacher_forces],
            torch.from_numpy(forces),
            rtol=0,
            atol=1e-6,
            msg=name,
        )


# MACE models with random weights, built for a dataset

MOLECULES = ("CH4", "H2O", "C2H6", "CH3OH")
#: energy per H, C and O atom of the labels
E0 = {1: -13.6, 6: -1029.0, 8: -2041.0}
#: a small architecture, to keep building and training fast
SMALL = {
    "num_channels": 8,
    "correlation": 2,
    "num_interactions": 1,
    "radial_MLP": (8,),
    "MLP_irreps": "4x0e",
}


def toy_energy(atoms):
    """The E0s plus a harmonic well about the center, in eV."""
    displacement = atoms.positions - atoms.positions.mean(0)
    return sum(E0[z] for z in atoms.numbers) + 0.5 * (displacement**2).sum()


def toy_forces(atoms):
    return -(atoms.positions - atoms.positions.mean(0))


@pytest.fixture(scope="module")
def toy_data(tmp_path_factory):
    """Twelve molecules with :func:`toy_energy` and :func:`toy_forces`, the
    first eight the training split, and the directory they are in."""
    if importlib.util.find_spec("mace") is None:
        pytest.skip("needs mace-torch: pip install 'schnetpack[mace]'")
    directory = tmp_path_factory.mktemp("toy")
    make_datamodule(
        directory,
        energy=toy_energy,
        forces=toy_forces,
        molecules=MOLECULES,
    )
    return ASEAtomsData(str(directory / "structures.db")), directory


@pytest.fixture(scope="module")
def built(toy_data):
    """A small MACE model with random weights, built on the training split."""
    dataset, directory = toy_data
    return build_mace(str(directory / "mace.pt2"), dataset, range(8), **SMALL)


def structures(directory, indices):
    with ase.db.connect(str(directory / "structures.db")) as db:
        return [db.get(index + 1).toatoms() for index in indices]


def scale(core):
    return core.state_dict()["module.model.scale_shift.scale"]


def test_building_without_mace_asks_for_the_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mace", None)

    with pytest.raises(ImportError, match=r"schnetpack\[mace\]"):
        build_mace(str(tmp_path / "mace.pt2"), dataset=None)


def test_a_built_model_has_no_e0s_of_its_own(built):
    core = load_mace(built)

    assert core.atomic_numbers.tolist() == [1, 6, 8]
    assert float(core.r_max) == 5.0
    assert not core.atomic_energies.any()
    assert not core.shift.any()


def test_the_scale_is_the_rms_of_the_training_forces(built, toy_data):
    dataset, directory = toy_data
    forces = np.concatenate(
        [toy_forces(atoms) for atoms in structures(directory, range(8))]
    )

    rms = _force_rms(dataset, range(8), properties.forces)

    assert rms == pytest.approx(np.sqrt(np.mean(forces**2)))
    assert scale(load_mace(built)).item() == pytest.approx(rms)


@pytest.mark.parametrize("r_max", [1.2, 5.0])
def test_the_neighbors_are_counted_as_mace_counts_them(toy_data, r_max):
    """Per atom with any neighbor within r_max."""
    dataset, directory = toy_data
    counts = []
    for atoms in structures(directory, range(8)):
        distances = atoms.get_all_distances()
        within = ((distances < r_max) & (distances > 0)).sum(1)
        counts.extend(within[within > 0])

    assert _avg_num_neighbors(dataset, range(8), r_max) == pytest.approx(
        np.mean(counts)
    )


def test_the_seed_fixes_the_weights(built, toy_data, tmp_path):
    dataset, _ = toy_data
    first = load_mace(built).state_dict()
    again = load_mace(
        build_mace(str(tmp_path / "again.pt2"), dataset, range(8), **SMALL)
    ).state_dict()
    other = load_mace(
        build_mace(str(tmp_path / "other.pt2"), dataset, range(8), seed=1, **SMALL)
    ).state_dict()

    for name, tensor in first.items():
        torch.testing.assert_close(again[name], tensor, msg=name)
    assert any(not torch.equal(other[name], tensor) for name, tensor in first.items())


def test_spktrain_trains_a_built_model_from_scratch(built, toy_data, tmp_path):
    _, directory = toy_data
    runs = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import schnetpack.cli; schnetpack.cli.train()",
            "experiment=mace_scratch",
            f"globals.mace_path={built}",
            f"data.dataset.datapath={directory / 'structures.db'}",
            f"data.split_file={directory / 'split.npz'}",
            "data.num_train=8",
            "data.num_val=2",
            "data.num_test=2",
            "data.batch_size=4",
            "data.num_workers=0",
            "trainer.max_epochs=2",
            "trainer.accelerator=cpu",
            f"run.path={runs}",
            "run.id=scratch",
            f"run.data_dir={tmp_path / 'data'}",
            "+seed=0",
            "print_config=False",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    model = load_model(str(runs / "scratch" / "best_model"))
    assert isinstance(model, MaceModel)
    trained = model.core.state_dict()
    assert any(
        not torch.equal(trained[name].double(), tensor.double())
        for name, tensor in load_mace(built).state_dict().items()
        if tensor.is_floating_point()
    ), "the weights did not change"

    calculator = ForceCalculator(
        model,
        neighbor_list=BatchNeighborList(MatScipyNeighborList(cutoff=5.0)),
        dtype=torch.float64,
    )
    toys = structures(directory, range(12))
    energy = calculator.energy(atoms_to_batch(toys, dtype=torch.float64))
    labels = torch.tensor([toy_energy(atoms) for atoms in toys], dtype=torch.float64)
    # absolute energies: the estimated E0s are back, to within what two epochs
    # from random weights leave of the harmonic well
    torch.testing.assert_close(energy, labels, atol=1.0, rtol=0)
