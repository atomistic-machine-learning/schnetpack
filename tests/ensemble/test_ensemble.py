import numbers

import numpy as np
import pytest
import torch
from ase import Atoms

from schnetpack.interfaces.ase_interface import SpkEnsembleCalculator
from schnetpack.transform import Transform
from schnetpack.uncertainty import AbsoluteUncertainty, RelativeUncertainty


# === Dummy Components ===
class OnesModel(torch.nn.Module):
    def forward(self, inputs):
        return {
            "energy": torch.tensor([1.0]),
            "forces": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            "stress": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        }


class TwosModel(torch.nn.Module):
    def forward(self, inputs):
        return {
            "energy": torch.tensor([2.0]),
            "forces": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            "stress": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        }


class BrokenModel(torch.nn.Module):
    def forward(self, inputs):
        return {"not_energy": torch.tensor([0.0])}


class NotAnUncertainty:
    def __call__(self, results):
        return 0.5


class DummyConverter:
    def __init__(self, **kwargs):
        pass

    def __call__(self, atoms):
        return {"dummy_input": torch.tensor([1.0])}


class DummyTransform(Transform):
    def __call__(self, inputs):
        return inputs


# === Fixtures ===
@pytest.fixture
def dummy_atoms():
    return Atoms("H2", positions=[[1, 1, 1], [1, 1, 1]])


def make_test_calc(models, uncertainty=None):
    return SpkEnsembleCalculator(
        models=models,
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        stress_key="stress",
        uncertainty=uncertainty,
    )


# === Functional Tests ===
def test_ensemble_average_energy(dummy_atoms):
    calc = make_test_calc(models=[OnesModel(), TwosModel()])
    calc.calculate(dummy_atoms)
    energy = calc.results["energy"]
    assert isinstance(energy, numbers.Number)
    assert energy > 0


def test_ensemble_average_forces(dummy_atoms):
    calc = make_test_calc(models=[OnesModel(), TwosModel()])
    calc.calculate(dummy_atoms)

    expected_value = 1.5 * calc.energy_conversion
    expected_forces = np.full((2, 3), expected_value)

    assert np.allclose(calc.results["forces"], expected_forces, rtol=1e-5)


def test_ensemble_stress_tensor(dummy_atoms):
    calc = make_test_calc(models=[OnesModel(), TwosModel()])
    calc.calculate(dummy_atoms)
    stress = calc.results["stress"]
    assert stress.shape == (3, 3)


# === Uncertainty Tests ===
def test_absolute_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty=AbsoluteUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert isinstance(calc.results["uncertainty"], numbers.Number)


def test_relative_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty=RelativeUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert isinstance(calc.results["uncertainty"], numbers.Number)


def test_relative_uncertainty_zero_for_identical(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), OnesModel()], uncertainty=RelativeUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert np.isclose(calc.results["uncertainty"], 0.0)


def test_zero_uncertainty_for_identical_models(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), OnesModel()], uncertainty=AbsoluteUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert np.isclose(calc.results["uncertainty"], 0.0)


def test_several_uncertainties_are_reported_under_their_keys(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty={
            "absolute": AbsoluteUncertainty(),
            "relative": RelativeUncertainty(),
        },
    )
    calc.calculate(dummy_atoms)
    assert "uncertainty" not in calc.results
    for key in ("absolute", "relative"):
        assert isinstance(calc.results[key], numbers.Number)
        assert calc.get_uncertainty(dummy_atoms, key) == calc.results[key]


def test_an_uncertainty_must_be_an_uncertainty():
    with pytest.raises(TypeError, match="not an Uncertainty"):
        make_test_calc(
            models=[OnesModel(), TwosModel()], uncertainty=NotAnUncertainty()
        )


# === Robustness Tests ===
def test_missing_property_raises(dummy_atoms):
    calc = SpkEnsembleCalculator(
        models=[BrokenModel()],
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        energy_key="energy",
    )
    with pytest.raises(Exception) as excinfo:
        calc.calculate(dummy_atoms)
    assert "energy" in str(excinfo.value)


# === Calculation Tests ===
def test_absolute_energy_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty=AbsoluteUncertainty(
            energy_weight=1.0, force_weight=0.0, stress_weight=0.0
        ),
    )

    calc.calculate(dummy_atoms)

    abs_energy = calc.results["uncertainty"]
    print(f"Absolute Energy Uncertainty: {abs_energy}")
    assert np.isclose(abs_energy, 0.0217, rtol=1e-3)


def test_absolute_force_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty=AbsoluteUncertainty(
            energy_weight=0.0, force_weight=1.0, stress_weight=0.0
        ),
    )

    calc.calculate(dummy_atoms)

    abs_force = calc.results["uncertainty"]
    print(f"Absolute Force Uncertainty: {abs_force}")
    assert np.isclose(abs_force, 0.0376, rtol=1e-2)


def test_absolute_stress_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty=AbsoluteUncertainty(
            energy_weight=0.0, force_weight=0.0, stress_weight=1.0
        ),
    )

    calc.calculate(dummy_atoms)

    abs_stress = calc.results["uncertainty"]
    print(f"Absolute Stress Uncertainty: {abs_stress}")
    assert np.isclose(abs_stress, 0.0376, rtol=1e-2)


def test_relative_energy_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty=RelativeUncertainty(
            energy_weight=1.0, force_weight=0.0, stress_weight=0.0
        ),
    )

    calc.calculate(dummy_atoms)

    rel_energy = calc.results["uncertainty"]
    print(f"Relative Energy Uncertainty: {rel_energy}")
    assert np.isclose(rel_energy, 0.333, rtol=1e-2)


def test_relative_force_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty=RelativeUncertainty(
            energy_weight=0.0, force_weight=1.0, stress_weight=0.0
        ),
    )

    calc.calculate(dummy_atoms)

    rel_force = calc.results["uncertainty"]
    print(f"Relative Force Uncertainty: {rel_force}")
    assert np.isclose(rel_force, 0.333, rtol=1e-2)


def test_relative_stress_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty=RelativeUncertainty(
            energy_weight=0.0, force_weight=0.0, stress_weight=1.0
        ),
    )

    calc.calculate(dummy_atoms)

    rel_stress = calc.results["uncertainty"]
    print(f"Relative Stress Uncertainty: {rel_stress}")
    assert np.isclose(rel_stress, 0.333, rtol=1e-2)


def test_absolute_uncertainty_calculation(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty=AbsoluteUncertainty()
    )

    calc.calculate(dummy_atoms)

    abs_unc = calc.results["uncertainty"]
    print(f"Absolute Uncertainty: {abs_unc}")
    assert np.isclose(abs_unc, 0.0376, rtol=1e-2)


def test_relative_uncertainty_calculation(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty=RelativeUncertainty()
    )

    calc.calculate(dummy_atoms)

    rel_unc = calc.results["uncertainty"]
    print(f"Relative Uncertainty: {rel_unc}")
    assert np.isclose(rel_unc, 0.333, rtol=1e-2)
