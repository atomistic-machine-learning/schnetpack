"""MaceModel on the stand-in: SchNetPack batches in, energies with the
E0s as offsets out, in a relaxation. Its match with MACE-OFF is the MACE
interface's golden test."""

import pytest
import torch
from torch.autograd import grad

from schnetpack import properties
from schnetpack.dynamics import LBFGS, BatchNeighborList, ForceCalculator
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.interfaces.mace_interface import export_mace
from schnetpack.model import MaceModel, load_mace
from schnetpack.transform import MatScipyNeighborList

from ..distillation.conftest import with_neighbors
from .fakes import (
    E0,
    R_MAX,
    FakeMace,
    RecordingMace,
    ethane_lattice,
    periodic,
    rattled,
    reference,
)

MOLECULES = ["CH4", "H2O", "C2H6"]


def e0_sums(atoms_list):
    table = dict(zip((1, 6, 8), E0))
    return torch.tensor(
        [sum(table[z] for z in atoms.numbers) for atoms in atoms_list],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("structures", [rattled(MOLECULES), periodic()])
def test_matches_the_stand_in(structures):
    batch = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))

    output = MaceModel(FakeMace())(batch)

    energies, positions = reference(structures)
    forces = [-grad(e, r)[0] for e, r in zip(energies, positions)]
    torch.testing.assert_close(
        output[properties.energy], torch.stack(energies).detach()
    )
    torch.testing.assert_close(output[properties.forces], torch.cat(forces))


def test_without_postprocessing_the_energy_lacks_the_e0s():
    structures = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))

    output = MaceModel(FakeMace(), do_postprocessing=False)(batch)

    energies, _ = reference(structures)
    torch.testing.assert_close(
        output[properties.energy],
        torch.stack(energies).detach() - e0_sums(structures),
    )


def test_a_float32_core_keeps_the_digits_of_the_e0s():
    """The E0s are read before the cast and added in float64."""
    structures = ethane_lattice()
    batch64 = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))
    batch32 = with_neighbors(atoms_to_batch(structures, dtype=torch.float32))

    energy = MaceModel(FakeMace(), dtype="float32")(batch32)[properties.energy]
    exact = MaceModel(FakeMace())(batch64)[properties.energy]

    assert energy.dtype == torch.float64
    assert abs(energy - exact).item() < 1e-3


def test_the_train_mode_is_maces_training_argument():
    batch = with_neighbors(atoms_to_batch(rattled(["CH4"]), dtype=torch.float64))
    model = MaceModel(RecordingMace(FakeMace()))

    model.train()(batch)
    assert model.core.training_argument is True
    model.eval()(batch)
    assert model.core.training_argument is False


def test_elements_the_core_does_not_cover_are_refused():
    batch = with_neighbors(atoms_to_batch(rattled(["CH4", "NH3"])))

    with pytest.raises(ValueError, match=r"\[7\]"):
        MaceModel(FakeMace())(batch)


def test_a_core_without_interaction_energy_is_refused():
    batch = with_neighbors(atoms_to_batch(rattled(["CH4"]), dtype=torch.float64))

    with pytest.raises(KeyError, match="interaction_energy"):
        MaceModel(FakeMace(interaction_energy=False))(batch)


def test_a_multi_head_core_warns_and_uses_its_first_head():
    structures = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))

    with pytest.warns(UserWarning, match="2 heads"):
        model = MaceModel(FakeMace(heads=2))

    energies, _ = reference(structures)
    torch.testing.assert_close(
        model(batch)[properties.energy], torch.stack(energies).detach()
    )


def test_drives_a_relaxation():
    calculator = ForceCalculator(
        MaceModel(FakeMace()),
        neighbor_list=BatchNeighborList(MatScipyNeighborList(cutoff=R_MAX)),
        dtype=torch.float64,
    )
    batch = atoms_to_batch(rattled(["H2O", "CH4"]), dtype=torch.float64)
    start = calculator.energy(dict(batch))

    relaxed = LBFGS(calculator, fmax=1e-3).run(dict(batch), 20)

    calculator.reset()
    assert (calculator.energy(relaxed) < start).all()


def test_load_mace_reads_an_archive(tmp_path):
    path = str(tmp_path / "mace.pt")
    export_mace(FakeMace(), path)
    batch = with_neighbors(atoms_to_batch(rattled(MOLECULES), dtype=torch.float64))

    model = MaceModel(load_mace(path))

    energies, _ = reference(rattled(MOLECULES))
    torch.testing.assert_close(
        model(batch)[properties.energy], torch.stack(energies).detach()
    )


def test_load_mace_reports_a_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="no MACE model"):
        load_mace(str(tmp_path / "missing.pt"))
