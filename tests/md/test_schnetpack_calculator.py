import os

import ase.io
import pytest
import torch

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.md import System
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.neighborlist_md import NeighborListMD

TESTDATA = os.path.join(os.path.dirname(__file__), "../testdata")


@pytest.fixture
def fresh_model_file(tmp_path):
    """A model with the postprocessors of the experiment configs."""
    cutoff = 5.0
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=spk.representation.PaiNN(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
        ),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.Atomwise(n_in=16, output_key="energy"),
            spk.atomistic.Forces(energy_key="energy", force_key="forces"),
        ],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("energy", add_mean=True, property_mean=torch.tensor(-1.0)),
        ],
    )
    path = str(tmp_path / "model")
    torch.save(model, path)
    return path


def forces(model_file, script_model):
    calculator = SchNetPackCalculator(
        model_file=model_file,
        force_key="forces",
        energy_unit="kcal/mol",
        position_unit="Angstrom",
        neighbor_list=NeighborListMD(
            cutoff=5.0, cutoff_shell=1.0, base_nbl=trn.ASENeighborList
        ),
        script_model=script_model,
    )
    system = System()
    system.load_molecules(
        [ase.io.read(os.path.join(TESTDATA, "md_ethanol.xyz"))], n_replicas=1
    )
    calculator.calculate(system)
    return system.forces


@pytest.mark.parametrize("model", ["saved", "fresh"])
def test_scripted_model_gives_the_same_forces(model, fresh_model_file):
    """Models with type casting postprocessors, as in the experiment configs, must
    survive `script_model=True`."""
    model_file = (
        os.path.join(TESTDATA, "md_ethanol.model")
        if model == "saved"
        else fresh_model_file
    )

    plain, scripted = forces(model_file, False), forces(model_file, True)

    assert plain.abs().max() > 0
    assert torch.allclose(plain, scripted)
