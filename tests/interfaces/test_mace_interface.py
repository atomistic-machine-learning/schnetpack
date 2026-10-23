"""The MACE interface: MACE-OFF compiled to TorchScript. The golden test runs
SchNetPack's MaceTeacher on the compiled archive against eager MACE; it needs
mace-torch, and is skipped without."""

import os
import subprocess
import sys

import pytest
import torch

from schnetpack import properties
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.interfaces.mace_interface import CACHE_DIR, compile_mace_off
from schnetpack.train.teacher import MaceTeacher

from ..distillation.conftest import absolute_energy, with_neighbors

NO_WEIGHTS_ONLY = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def test_compiling_without_mace_asks_for_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mace", None)

    with pytest.raises(ImportError, match=r"schnetpack\[mace\]"):
        compile_mace_off()


def test_an_unknown_size_is_refused():
    with pytest.raises(ValueError, match="huge"):
        compile_mace_off("huge")


def test_the_archive_lands_in_the_cache(mace_off_archive):
    assert os.path.dirname(mace_off_archive) == CACHE_DIR


def test_a_second_call_reuses_the_cache(mace_off_archive):
    written = os.path.getmtime(mace_off_archive)

    assert compile_mace_off("medium") == mace_off_archive
    assert os.path.getmtime(mace_off_archive) == written


def test_compiling_leaves_torch_load_safe(mace_off_archive):
    """Importing mace turns off torch.load's weights_only for the process, on
    its first import only: hence a fresh interpreter."""
    env = {k: v for k, v in os.environ.items() if k != NO_WEIGHTS_ONLY}
    script = (
        "import os\n"
        "from schnetpack.interfaces.mace_interface import compile_mace_off\n"
        "compile_mace_off('medium')\n"
        f"assert {NO_WEIGHTS_ONLY!r} not in os.environ, 'weights_only is off'\n"
    )

    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_the_compiled_archive_reproduces_mace(mace_off_archive, mace_off_reference):
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
