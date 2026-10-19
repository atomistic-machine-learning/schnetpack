"""MaceModel on the scripted stand-in: SchNetPack batches in, energies with the
E0s as offsets out, in a relaxation; and on MACE-OFF, if mace-torch is
installed."""

import pytest
import torch
from torch.autograd import grad

from schnetpack import properties
from schnetpack.dynamics import LBFGS, BatchNeighborList, ForceCalculator
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.model import MaceModel, load_mace
from schnetpack.transform import MatScipyNeighborList

from ..conftest import mace_off_structures
from ..distillation.conftest import with_neighbors
from ..distillation.test_mace_teacher import ethane_lattice, periodic, rattled
from .fakes import E0, R_MAX, FakeMace, RecordingMace, reference

MOLECULES = ["CH4", "H2O", "C2H6"]


def scripted(**kwargs):
    return torch.jit.script(FakeMace(**kwargs))


def e0_sums(atoms_list):
    table = dict(zip((1, 6, 8), E0))
    return torch.tensor(
        [sum(table[z] for z in atoms.numbers) for atoms in atoms_list],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("structures", [rattled(MOLECULES), periodic()])
def test_matches_the_stand_in(structures):
    batch = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))

    output = MaceModel(scripted())(batch)

    energies, positions = reference(structures)
    forces = [-grad(e, r)[0] for e, r in zip(energies, positions)]
    torch.testing.assert_close(
        output[properties.energy], torch.stack(energies).detach()
    )
    torch.testing.assert_close(output[properties.forces], torch.cat(forces))


def test_without_postprocessing_the_energy_lacks_the_e0s():
    structures = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))

    output = MaceModel(scripted(), do_postprocessing=False)(batch)

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

    energy = MaceModel(scripted(), dtype="float32")(batch32)[properties.energy]
    exact = MaceModel(scripted())(batch64)[properties.energy]

    assert energy.dtype == torch.float64
    assert abs(energy - exact).item() < 1e-3


def test_the_train_mode_is_maces_training_argument():
    batch = with_neighbors(atoms_to_batch(rattled(["CH4"]), dtype=torch.float64))
    model = MaceModel(RecordingMace(scripted()))

    model.train()(batch)
    assert model.core.training_argument is True
    model.eval()(batch)
    assert model.core.training_argument is False


def test_elements_the_core_does_not_cover_are_refused():
    batch = with_neighbors(atoms_to_batch(rattled(["CH4", "NH3"])))

    with pytest.raises(ValueError, match=r"\[7\]"):
        MaceModel(scripted())(batch)


def test_a_core_without_interaction_energy_is_refused():
    batch = with_neighbors(atoms_to_batch(rattled(["CH4"]), dtype=torch.float64))

    with pytest.raises(KeyError, match="interaction_energy"):
        MaceModel(scripted(interaction_energy=False))(batch)


def test_a_multi_head_core_warns_and_uses_its_first_head():
    structures = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(structures, dtype=torch.float64))

    with pytest.warns(UserWarning, match="2 heads"):
        model = MaceModel(scripted(heads=2))

    energies, _ = reference(structures)
    torch.testing.assert_close(
        model(batch)[properties.energy], torch.stack(energies).detach()
    )


def test_drives_a_relaxation():
    calculator = ForceCalculator(
        MaceModel(scripted()),
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
    torch.jit.save(scripted(), path)
    batch = with_neighbors(atoms_to_batch(rattled(MOLECULES), dtype=torch.float64))

    model = MaceModel(load_mace(path))

    energies, _ = reference(rattled(MOLECULES))
    torch.testing.assert_close(
        model(batch)[properties.energy], torch.stack(energies).detach()
    )


def test_load_mace_reports_a_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="no MACE model"):
        load_mace(str(tmp_path / "missing.pt"))


# ---------------------------------------------------------------- MACE-OFF


def test_reproduces_mace_off(mace_off_archive, mace_off_reference):
    model = MaceModel(load_mace(mace_off_archive), dtype="float64")
    for name, atoms, energy, forces in mace_off_reference:
        batch = with_neighbors(atoms_to_batch([atoms], dtype=torch.float64))

        output = model(batch)

        torch.testing.assert_close(
            output[properties.energy],
            torch.tensor([float(energy)], dtype=torch.float64),
            rtol=0,
            atol=1e-6,
            msg=name,
        )
        torch.testing.assert_close(
            output[properties.forces],
            torch.from_numpy(forces),
            rtol=0,
            atol=1e-6,
            msg=name,
        )


def test_relaxes_ethanol_with_mace_off(mace_off_archive):
    model = MaceModel(load_mace(mace_off_archive), dtype="float64")
    calculator = ForceCalculator(
        model,
        neighbor_list=BatchNeighborList(MatScipyNeighborList(cutoff=model.r_max)),
    )
    batch = atoms_to_batch([mace_off_structures()["ethanol"]], dtype=torch.float64)
    start = calculator.energy(dict(batch))

    relaxed = LBFGS(calculator, fmax=1e-2).run(dict(batch), 200)

    calculator.reset()
    assert (calculator.energy(relaxed) < start).all()
    assert calculator.forces(relaxed).norm(dim=-1).max() < 1e-2
