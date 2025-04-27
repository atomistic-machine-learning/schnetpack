import pytest
import torch
import numpy as np
from ase import Atoms
from schnetpack.interfaces.ase_interface import (
    SpkEnsembleCalculator,
    AbsoluteUncertainty,
    RelativeUncertainty,
)
from schnetpack.transform import Transform
from schnetpack.interfaces import AtomsConverter
import numbers


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


class InvalidUncertainty:
    def __call__(self, results):
        return "not a number"


class FakeUncertainty:
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


def make_test_calc(models, uncertainty_fn=None):
    return SpkEnsembleCalculator(
        models=models,
        neighbor_list=DummyTransform(),
        # converter=DummyConverter,
        energy_key="energy",
        force_key="forces",
        stress_key="stress",
        uncertainty_fn=uncertainty_fn,
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

    # Each atom's force should be averaged between OnesModel and TwosModel
    expected_value = 1.5 * calc.energy_conversion  # because (1+2)/2 = 1.5
    expected_forces = np.full((2, 3), expected_value)

    print("\nCalculated Forces:")
    print(calc.results["forces"])
    print("Expected Forces:")
    print(expected_forces)

    assert np.allclose(calc.results["forces"], expected_forces, rtol=1e-5)


def test_ensemble_stress_tensor(dummy_atoms):
    calc = make_test_calc(models=[OnesModel(), TwosModel()])
    calc.calculate(dummy_atoms)
    stress = calc.results["stress"]
    assert stress.shape == (3, 3)


# === Uncertainty Tests ===
def test_absolute_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty_fn=AbsoluteUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert isinstance(calc.results["uncertainty"], numbers.Number)


def test_relative_uncertainty(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty_fn=RelativeUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert isinstance(calc.results["uncertainty"], numbers.Number)


def test_relative_uncertainty_zero_for_identical(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), OnesModel()], uncertainty_fn=RelativeUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert np.isclose(calc.results["uncertainty"], 0.0)


def test_zero_uncertainty_for_identical_models(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), OnesModel()], uncertainty_fn=AbsoluteUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert np.isclose(calc.results["uncertainty"], 0.0)


def test_multiple_uncertainty_functions(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty_fn=[AbsoluteUncertainty(), FakeUncertainty()],
    )
    calc.calculate(dummy_atoms)
    result = calc.results["uncertainty"]
    assert isinstance(result, dict)
    assert "AbsoluteUncertainty" in result
    assert "FakeUncertainty" in result


def test_invalid_uncertainty_return(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()], uncertainty_fn=InvalidUncertainty()
    )
    calc.calculate(dummy_atoms)
    assert not isinstance(calc.results["uncertainty"], numbers.Number)


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


def test_uncertainty_keys_multiple_fns(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty_fn=[AbsoluteUncertainty(), RelativeUncertainty()],
    )
    calc.calculate(dummy_atoms)
    u = calc.results["uncertainty"]
    assert isinstance(u, dict)
    assert "AbsoluteUncertainty" in u
    assert "RelativeUncertainty" in u


def test_exact_predictions_and_uncertainties(dummy_atoms):
    calc = make_test_calc(
        models=[OnesModel(), TwosModel()],
        uncertainty_fn=[AbsoluteUncertainty(), RelativeUncertainty()],
    )

    calc.calculate(dummy_atoms)

    print("\n=== Predicted Properties ===")
    print(f"Energy: {calc.results['energy']}")
    print(f"Forces:\n{calc.results['forces']}")
    print(f"Stress:\n{calc.results['stress']}")

    # === Check predictions ===
    # expected_energy = (1.0 + 2.0) / 2.0
    # assert np.isclose(calc.results["energy"], 0.06)

    # expected_forces = np.array([
    #     [1.5, 1.5, 1.5],
    #     [1.5, 1.5, 1.5],
    # ])
    # assert np.allclose(calc.results["forces"], expected_forces)

    # expected_stress = np.full((3, 3), 1.5)
    # assert np.allclose(calc.results["stress"], expected_stress)

    # === Check uncertainties ===
    uncertainties = calc.results["uncertainty"]

    # AbsoluteUncertainty pieces
    energy_preds = [1.0, 2.0]
    force_preds = np.array(
        [
            [[1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2]],
        ]
    )
    stress_preds = np.array(
        [
            np.ones((3, 3)),
            2 * np.ones((3, 3)),
        ]
    )

    energy_std = np.std(energy_preds)
    force_std = np.std(force_preds, axis=0)
    per_atom_force_unc = np.linalg.norm(force_std, axis=1)
    force_unc = np.mean(per_atom_force_unc)

    stress_std = np.std(stress_preds, axis=0)
    per_plane_stress_unc = np.linalg.norm(stress_std, axis=1)
    stress_unc = np.mean(per_plane_stress_unc)

    expected_absolute_uncertainty = energy_std + force_unc + stress_unc

    print("\n=== Absolute Uncertainty Calculation ===")
    print(f"Energy std: {energy_std}")
    print(f"Force std (per component):\n{force_std}")
    print(f"Force uncertainty per atom: {per_atom_force_unc}")
    print(f"Force uncertainty mean: {force_unc}")
    print(f"Stress std (per component):\n{stress_std}")
    print(f"Stress uncertainty per plane: {per_plane_stress_unc}")
    print(f"Stress uncertainty mean: {stress_unc}")
    print(f"Total Absolute Uncertainty: {expected_absolute_uncertainty}")

    # RelativeUncertainty pieces
    mean_energy = np.mean(energy_preds)
    relative_energy_unc = energy_std / (abs(mean_energy) + 1e-8)

    mean_force = np.mean(force_preds, axis=0)
    mean_force_norm = np.linalg.norm(mean_force, axis=1)
    std_force_norm = np.linalg.norm(force_std, axis=1)
    relative_force_unc = np.mean(std_force_norm / (mean_force_norm + 1e-8))

    mean_stress = np.mean(stress_preds, axis=0)
    mean_stress_planes = np.linalg.norm(mean_stress, axis=1)
    std_stress_planes = np.linalg.norm(stress_std, axis=1)
    relative_stress_unc = np.mean(std_stress_planes / (mean_stress_planes + 1e-8))

    expected_relative_uncertainty = (
        relative_energy_unc + relative_force_unc + relative_stress_unc
    )

    print("\n=== Relative Uncertainty Calculation ===")
    print(f"Mean Energy: {mean_energy}")
    print(f"Relative Energy Uncertainty: {relative_energy_unc}")
    print(f"Mean Force (per atom):\n{mean_force}")
    print(f"Mean Force norms: {mean_force_norm}")
    print(f"Std Force norms: {std_force_norm}")
    print(f"Relative Force Uncertainty: {relative_force_unc}")
    print(f"Mean Stress:\n{mean_stress}")
    print(f"Mean Stress planes: {mean_stress_planes}")
    print(f"Std Stress planes: {std_stress_planes}")
    print(f"Relative Stress Uncertainty: {relative_stress_unc}")
    print(f"Total Relative Uncertainty: {expected_relative_uncertainty}")

    # === Final assertions ===
    assert np.isclose(
        uncertainties["AbsoluteUncertainty"], expected_absolute_uncertainty, rtol=1e-4
    )
    assert np.isclose(
        uncertainties["RelativeUncertainty"], expected_relative_uncertainty, rtol=1e-4
    )
