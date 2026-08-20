"""``BatchwiseEnsembleCalculator``: the ensemble mean drives the relaxation, and the
spread across the members is reported per structure.

The members are analytic springs of different stiffness, so every mean and every
standard deviation below is known exactly and no trained model is needed.
"""

import os
import shutil
from typing import Dict, List

import numpy as np
import pytest
import torch
from ase import Atoms
from torch import nn

from schnetpack import properties
from schnetpack.interfaces.ase_interface import (
    AbsoluteUncertainty,
    RelativeUncertainty,
    SpkEnsembleCalculator,
    atoms_to_batch,
)
from schnetpack.relax import (
    BatchwiseCalculatorError,
    BatchwiseEnsembleCalculator,
    BatchwiseLBFGS,
    NNEnsemble,
)

from .test_bw_optimizer_units import make_inputs


class HarmonicModel(nn.Module):
    """Every atom on a spring of stiffness ``k``, pulled towards the origin."""

    def __init__(self, spring_constant: float):
        super().__init__()
        self.spring_constant = spring_constant

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = inputs[properties.R]
        per_atom = 0.5 * self.spring_constant * positions.pow(2).sum(-1)
        counts = inputs[properties.n_atoms].tolist()
        return {
            "energy": torch.stack([c.sum() for c in torch.split(per_atom, counts)]),
            "forces": -self.spring_constant * positions,
        }


class NoForcesModel(nn.Module):
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"energy": torch.zeros(inputs[properties.n_atoms].shape[0])}


class PassThroughNeighborList:
    """Springs need no neighbourhoods, so the batch goes to the models untouched."""

    def update(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return inputs


def make_calculator(
    spring_constants: List[float] = [1.0, 2.0],
    uncertainty_fn=None,
    dtype: torch.dtype = torch.float32,
) -> BatchwiseEnsembleCalculator:
    return BatchwiseEnsembleCalculator(
        models=[HarmonicModel(k) for k in spring_constants],
        neighbor_list=PassThroughNeighborList(),
        uncertainty_fn=uncertainty_fn,
        dtype=dtype,
    )


# ------------------------------------------------------------------ the ensemble mean


def test_get_forces_returns_the_ensemble_mean():
    """The bug that made this class unusable: ``get_forces(inputs)`` used to raise."""
    inputs = make_inputs([4, 4])
    calculator = make_calculator([1.0, 2.0])

    forces = calculator.get_forces(inputs)

    # springs of 1 and 2 average to a spring of 1.5
    torch.testing.assert_close(forces, -1.5 * inputs[properties.R])


def test_get_potential_energy_returns_the_ensemble_mean():
    inputs = make_inputs([3, 3])
    calculator = make_calculator([1.0, 2.0])

    energy = calculator.get_potential_energy(inputs)

    expected = 0.75 * inputs[properties.R].pow(2).sum(-1).view(2, 3).sum(-1)
    assert energy.shape == (2,)
    torch.testing.assert_close(energy, expected)


def test_the_ensemble_model_keeps_a_model_axis():
    """``NNEnsemble`` is a model that returns one prediction per member."""
    inputs = make_inputs([5])
    ensemble = NNEnsemble(
        models=nn.ModuleList([HarmonicModel(1.0), HarmonicModel(2.0)]),
        properties=["energy", "forces"],
    )

    predictions = ensemble(inputs)

    assert predictions["forces"].shape == (2, 5, 3)
    assert predictions["energy"].shape == (2, 1)


# -------------------------------------------------------------------- the uncertainty


def test_uncertainty_is_reported_per_structure():
    inputs = make_inputs([4, 4])
    calculator = make_calculator([1.0, 2.0])

    uncertainty = calculator.get_uncertainty(inputs)

    assert uncertainty.shape == (2,)

    # the members differ by -1*R against -2*R, so the per-component spread is 0.5*|R|
    # and the per-atom uncertainty its norm; the structure's value is the mean of those
    positions = inputs[properties.R].view(2, 4, 3)
    expected = (0.5 * positions.norm(dim=-1)).mean(dim=-1)
    torch.testing.assert_close(uncertainty, expected)


def test_uncertainty_distinguishes_the_structures_of_a_batch():
    """The whole point of a per-structure value: one structure can be worse than another."""
    inputs = make_inputs([4, 4])
    positions = inputs[properties.R].view(2, 4, 3)
    positions[1] *= 10.0  # push the second structure far out

    uncertainty = make_calculator([1.0, 2.0]).get_uncertainty(inputs)

    assert uncertainty[1] > 5 * uncertainty[0]


def test_identical_members_are_certain():
    inputs = make_inputs([3, 3])
    uncertainty = make_calculator([1.5, 1.5]).get_uncertainty(inputs)
    torch.testing.assert_close(uncertainty, torch.zeros(2))


def test_a_single_member_is_certain_rather_than_nan():
    """``torch.std`` of one sample is ``nan`` unless the correction is turned off."""
    inputs = make_inputs([3, 3])
    uncertainty = make_calculator([1.0]).get_uncertainty(inputs)

    assert not torch.isnan(uncertainty).any()
    torch.testing.assert_close(uncertainty, torch.zeros(2))


def test_several_uncertainty_functions_are_reported_by_name():
    inputs = make_inputs([3, 3])
    calculator = make_calculator(
        [1.0, 2.0], uncertainty_fn=[AbsoluteUncertainty(), RelativeUncertainty()]
    )

    uncertainty = calculator.get_uncertainty(inputs)

    assert set(uncertainty) == {"AbsoluteUncertainty", "RelativeUncertainty"}
    for value in uncertainty.values():
        assert value.shape == (2,)
    # every member points the same way and differs only in magnitude, so the relative
    # spread is 0.5/1.5 whatever the structure looks like
    torch.testing.assert_close(
        uncertainty["RelativeUncertainty"], torch.full((2,), 1.0 / 3.0)
    )


def test_uncertainty_weights_are_honoured():
    inputs = make_inputs([3, 3])
    energy_only = make_calculator(
        [1.0, 2.0],
        uncertainty_fn=AbsoluteUncertainty(energy_weight=1.0, force_weight=0.0),
    ).get_uncertainty(inputs)

    # the energies are 0.5*k*sum(R^2), so their spread is 0.25*sum(R^2)
    expected = 0.25 * inputs[properties.R].pow(2).sum(-1).view(2, 3).sum(-1)
    torch.testing.assert_close(energy_only, expected)


# ------------------------------------------------- the same numbers as the ase version


def test_batch_of_one_matches_the_ase_ensemble_calculator():
    """Both calculators must reach the same uncertainty through the same functions."""

    class Converter:
        def __init__(self, **kwargs):
            pass

        def __call__(self, atoms):
            return atoms_to_batch([atoms])

    rng = np.random.default_rng(0)
    atoms = Atoms("H4", positions=rng.normal(size=(4, 3)))
    springs = [1.0, 2.0, 3.5]

    ase_calculator = SpkEnsembleCalculator(
        models=[HarmonicModel(k) for k in springs],
        neighbor_list=None,
        converter=Converter,
        uncertainty_fn=AbsoluteUncertainty(),
        energy_unit="eV",
        position_unit="Angstrom",
    )
    ase_calculator.calculate(atoms)

    batchwise = BatchwiseEnsembleCalculator(
        models=[HarmonicModel(k) for k in springs],
        neighbor_list=PassThroughNeighborList(),
        uncertainty_fn=AbsoluteUncertainty(),
    )
    batch_uncertainty = batchwise.get_uncertainty(atoms_to_batch([atoms]))

    assert batch_uncertainty.shape == (1,)
    assert batch_uncertainty.item() == pytest.approx(
        ase_calculator.results["uncertainty"], rel=1e-6
    )
    np.testing.assert_allclose(
        batchwise.get_forces(atoms_to_batch([atoms])).numpy(),
        ase_calculator.results["forces"],
        rtol=1e-6,
    )


# ------------------------------------------------------------------------- the basics


def test_a_missing_property_is_reported():
    calculator = BatchwiseEnsembleCalculator(
        models=[NoForcesModel(), NoForcesModel()],
        neighbor_list=PassThroughNeighborList(),
    )
    with pytest.raises(BatchwiseCalculatorError, match="forces"):
        calculator.get_forces(make_inputs([3]))


def test_the_ensemble_is_evaluated_once_per_step():
    """The cache works for the ensemble exactly as it does for a single model."""
    inputs = make_inputs([3, 3])
    calculator = make_calculator([1.0, 2.0])

    calls = []
    forward = calculator.model.forward
    calculator.model.forward = lambda batch: (calls.append(None), forward(batch))[1]

    calculator.get_forces(inputs)
    calculator.get_potential_energy(inputs)
    calculator.get_uncertainty(inputs)
    assert len(calls) == 1, "energy, forces and uncertainty come from one call"

    inputs[properties.R] += 0.1
    calculator.get_forces(inputs)
    assert len(calls) == 2, "moving the batch invalidates the cache"


def test_models_are_loaded_from_a_directory(tmp_path):
    """The layout schnetpack training leaves behind: ``<dir>/<run>/best_model``."""
    trained = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "md_ethanol.model"
    )
    for run in range(2):
        directory = tmp_path / f"run_{run}"
        directory.mkdir()
        shutil.copy(trained, directory / "best_model")

    calculator = BatchwiseEnsembleCalculator(
        models=str(tmp_path), neighbor_list=PassThroughNeighborList()
    )

    assert len(calculator.model.models) == 2
    # two copies of one model is an ensemble that agrees with itself perfectly
    assert calculator.model.properties == ["energy", "forces"]


def test_an_ensemble_relaxes_a_batch():
    """End to end: the mean drives the relaxation, and the batch reaches the minimum."""
    inputs = make_inputs([4, 4])
    calculator = make_calculator([1.0, 2.0])
    optimizer = BatchwiseLBFGS(calculator=calculator, inputs=inputs, logfile=None)

    assert optimizer.run(fmax=1e-4, steps=60)

    # every spring pulls towards the origin, and a relaxed batch sits there
    assert inputs[properties.R].abs().max() < 1e-4
    # a converged batch is one the ensemble agrees about, since the members only
    # disagree about the magnitude of a force that has gone to zero
    assert calculator.get_uncertainty(inputs).max() < 1e-4
