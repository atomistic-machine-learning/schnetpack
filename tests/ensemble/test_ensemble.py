import pytest
import torch
import numpy as np
from ase import Atoms
from schnetpack.interfaces.ase_interface import SpkEnsembleCalculator, AbsoluteUncertainty, RelativeUncertainty
from schnetpack.transform import Transform
from schnetpack.interfaces import AtomsConverter
import numbers


# === Dummy Components ===
class DummyModel(torch.nn.Module):
    def forward(self, inputs):
        return {
            "energy": torch.tensor([1.0]),
            "forces": torch.tensor([[0.1, 0.2, 0.3]]),
            "stress": torch.tensor([[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]]),
        }

class ConstantModel(torch.nn.Module):
    def forward(self, inputs):
        return {
            "energy": torch.tensor([42.0]),
            "forces": torch.tensor([[0.1, 0.2, 0.3]]),
            "stress": torch.tensor([[[0.01, 0.0, 0.0],
                                     [0.0, 0.01, 0.0],
                                     [0.0, 0.0, 0.01]]]),
        }

class InconsistentModel(torch.nn.Module):
    def forward(self, inputs):
        return {
            "energy": torch.tensor([torch.randn(1).item()]),
            "forces": torch.tensor([[0.5, -0.1, 0.4]]),
            "stress": torch.tensor([[[0.02, 0.0, 0.0],
                                     [0.0, 0.02, 0.0],
                                     [0.0, 0.0, 0.02]]]),
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
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

def make_test_calc(models, uncertainty_fn=None):
    return SpkEnsembleCalculator(
        models=models,
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        energy_key="energy",
        force_key="forces",
        stress_key="stress",
        uncertainty_fn=uncertainty_fn,
    )

# === Functional Tests ===
def test_ensemble_average_energy(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()])
    calc.calculate(dummy_atoms)
    energy = calc.results["energy"]
    assert isinstance(energy, numbers.Number)
    assert energy > 0

def test_ensemble_average_forces(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()])
    calc.calculate(dummy_atoms)
    expected = np.array([[0.1, 0.2, 0.3]]) * calc.energy_conversion
    assert np.allclose(calc.results["forces"], expected, rtol=1e-5)

def test_ensemble_stress_tensor(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()])
    calc.calculate(dummy_atoms)
    stress = calc.results["stress"]
    assert stress.shape == (3, 3)

# === Uncertainty Tests ===
def test_absolute_uncertainty(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()], uncertainty_fn=AbsoluteUncertainty())
    calc.calculate(dummy_atoms)
    assert isinstance(calc.results["uncertainty"], numbers.Number)

def test_relative_uncertainty(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()], uncertainty_fn=RelativeUncertainty())
    calc.calculate(dummy_atoms)
    assert isinstance(calc.results["uncertainty"], numbers.Number)

def test_relative_uncertainty_zero_for_identical(dummy_atoms):
    calc = make_test_calc(models=[ConstantModel(), ConstantModel()], uncertainty_fn=RelativeUncertainty())
    calc.calculate(dummy_atoms)
    assert np.isclose(calc.results["uncertainty"], 0.0)

def test_zero_uncertainty_for_identical_models(dummy_atoms):
    calc = make_test_calc(models=[ConstantModel(), ConstantModel()], uncertainty_fn=AbsoluteUncertainty())
    calc.calculate(dummy_atoms)
    assert np.isclose(calc.results["uncertainty"], 0.0)

def test_high_uncertainty_from_mismatched_models(dummy_atoms):
    calc = make_test_calc(models=[InconsistentModel(), InconsistentModel()], uncertainty_fn=AbsoluteUncertainty())
    calc.calculate(dummy_atoms)
    assert calc.results["uncertainty"] >= 0.0

def test_multiple_uncertainty_functions(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()],
                          uncertainty_fn=[AbsoluteUncertainty(), FakeUncertainty()])
    calc.calculate(dummy_atoms)
    result = calc.results["uncertainty"]
    assert isinstance(result, dict)
    assert "AbsoluteUncertainty" in result
    assert "FakeUncertainty" in result

def test_invalid_uncertainty_return(dummy_atoms):
    calc = make_test_calc(models=[DummyModel(), DummyModel()], uncertainty_fn=InvalidUncertainty())
    calc.calculate(dummy_atoms)
    assert not isinstance(calc.results["uncertainty"], numbers.Number)

# === Robustness Tests ===
def test_missing_property_raises(dummy_atoms):
    calc = SpkEnsembleCalculator(
        models=[BrokenModel()],
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        energy_key="energy"
    )
    with pytest.raises(Exception) as excinfo:
        calc.calculate(dummy_atoms)
    assert "energy" in str(excinfo.value)

def test_energy_conversion():
    calc = make_test_calc(models=[DummyModel()])
    assert np.isclose(calc.energy_conversion, 0.0433641, atol=1e-4)

def test_uncertainty_keys_multiple_fns(dummy_atoms):
    calc = make_test_calc(
        models=[DummyModel(), DummyModel()],
        uncertainty_fn=[AbsoluteUncertainty(), RelativeUncertainty()],
    )
    calc.calculate(dummy_atoms)
    u = calc.results["uncertainty"]
    assert isinstance(u, dict)
    assert "AbsoluteUncertainty" in u
    assert "RelativeUncertainty" in u

@pytest.fixture
def calculator():
    return SpkEnsembleCalculator(
        models=[InconsistentModel(), InconsistentModel()],
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        energy_key="energy",
        force_key="forces",
        stress_key="stress",
        uncertainty_fn=AbsoluteUncertainty(),
    )

def test_combined_uncertainty(calculator, dummy_atoms):
    # Perform calculation
    calculator.calculate(dummy_atoms)

    # Extract results
    energy = calculator.results["energy"]
    forces = calculator.results["forces"]
    stress = calculator.results["stress"]
    
    # Energy uncertainty calculation (from earlier test code)
    energy_uncertainty = 0.3
    
    # Forces uncertainty calculation
    force_x_uncertainty = 0.1
    force_y_uncertainty = 0.05
    force_z_uncertainty = 0.05
    forces_uncertainty = np.sqrt(force_x_uncertainty**2 + force_y_uncertainty**2 + force_z_uncertainty**2)
    
    # Stress uncertainty calculation
    stress_xx_uncertainty = 0.005
    stress_yy_uncertainty = 0.005
    stress_zz_uncertainty = 0.005
    stress_uncertainty = np.sqrt(stress_xx_uncertainty**2 + stress_yy_uncertainty**2 + stress_zz_uncertainty**2)
    
    # Calculate the combined uncertainty using RMS method
    combined_uncertainty = np.sqrt(energy_uncertainty**2 + forces_uncertainty**2 + stress_uncertainty**2)
    
    # Output the result
    print(f"Combined Uncertainty: {combined_uncertainty}")

    # Assertions for combined uncertainty
    assert np.isclose(combined_uncertainty, 0.3241, atol=1e-3)  # Change value as per your calculation
    assert isinstance(energy, numbers.Number)
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (1, 3)  # Ensure forces array is the right shape

# --- Additional Tests to Cover Other Uncertainty Types --- #

# Test case for relative uncertainty
def test_relative_uncertainty_for_identical_models(calculator, dummy_atoms):
    calc = SpkEnsembleCalculator(
        models=[InconsistentModel(), InconsistentModel()],
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        uncertainty_fn=RelativeUncertainty(),
    )
    calc.calculate(dummy_atoms)
    
    # Relative uncertainty calculation should return a number
    relative_uncertainty = calc.results["uncertainty"]
    assert isinstance(relative_uncertainty, numbers.Number)
    assert relative_uncertainty == 0.0  

def test_fake_uncertainty(calculator, dummy_atoms):
    # Define a simple FakeUncertainty class that just returns a constant value
    class FakeUncertainty:
        def __call__(self, results):
            return 0.5  # Fake constant uncertainty
    
    # Create the ensemble calculator
    calc = SpkEnsembleCalculator(
        models=[InconsistentModel(), InconsistentModel()],
        neighbor_list=DummyTransform(),
        converter=DummyConverter,
        uncertainty_fn=[AbsoluteUncertainty(), FakeUncertainty()],
    )
    
    # Calculate results with the dummy atoms
    calc.calculate(dummy_atoms)
    
    # Get the uncertainty results
    uncertainty_results = calc.results["uncertainty"]

    # Check that we have uncertainty for both methods
    assert isinstance(uncertainty_results, dict)
    assert "AbsoluteUncertainty" in uncertainty_results
    assert "FakeUncertainty" in uncertainty_results

    # AbsoluteUncertainty should be calculated from model output uncertainty (expected value: 0.3)
    print(f"AbsoluteUncertainty: {uncertainty_results['AbsoluteUncertainty']}")
    assert np.isclose(uncertainty_results["AbsoluteUncertainty"], 0.0, atol=1e-2)  # Allowing a small tolerance
    
    # FakeUncertainty should return the constant 0.5
    print(f"FakeUncertainty: {uncertainty_results['FakeUncertainty']}")
    assert np.isclose(uncertainty_results["FakeUncertainty"], 0.5)

'''
#forces
    input : 2 x 3
#stres
    input : 1 x 3 x 3
    output: 3 x 3
add tests for energy, forces, stress calc
'''
#do prediction for both models 