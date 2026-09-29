import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    Calculator,
    DirectDenoising,
    ForceFieldCalculator,
    Heun,
    Sampler,
)
from schnetpack.generative import (
    VE,
    VP,
    PseudoForceParametrization,
    VelocityParametrization,
)


class InPlaceNeighborList:
    """Writes into its input like spk's neighbor-list transforms; counts calls and resets."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def __call__(self, inputs):
        self.calls += 1
        inputs[properties.idx_i] = torch.arange(inputs[properties.R].shape[0])
        return inputs

    def reset(self):
        self.resets += 1


def zero_model(batch):
    assert properties.idx_i in batch
    return {"prediction": torch.zeros_like(batch[properties.R])}


def test_neighbor_list_runs_on_every_model_call_and_never_reaches_the_loop():
    # Heun evaluates twice per step; each call gets a fresh neighbor list,
    # built on a copy, so the loop's batch never holds a derived key.
    nbl = InPlaceNeighborList()
    calculator = Calculator(zero_model, neighbor_list=nbl)
    sampler = Sampler(calculator, VP(), VelocityParametrization(), Heun(), churn=0.0)
    batch = {properties.R: torch.randn(5, 3)}
    out = sampler.denoise(batch, 4)
    assert nbl.calls == 8
    assert nbl.resets == 1
    assert properties.idx_i not in out and properties.idx_i not in batch


def test_prepare_moves_floats_to_the_run_dtype_and_leaves_the_rest():
    calculator = Calculator(zero_model, dtype=torch.float64)
    batch = {
        properties.R: torch.zeros(2, 3),
        properties.Z: torch.tensor([1, 6]),
        "flag": 7,
    }
    out = calculator.prepare(batch)
    assert out[properties.R].dtype == torch.float64
    assert out[properties.Z].dtype == torch.int64
    assert out["flag"] == 7
    assert batch[properties.R].dtype == torch.float32


def test_driver_runs_in_the_calculators_dtype():
    seen = []

    def model(batch):
        seen.append(batch[properties.R].dtype)
        return {"prediction": torch.zeros_like(batch[properties.R])}

    sampler = DirectDenoising(
        Calculator(model, dtype=torch.float64),
        VE(0.01, 3.0),
        PseudoForceParametrization(),
    )
    out = sampler.denoise({properties.R: torch.randn(3, 3)}, 2)
    assert seen == [torch.float64, torch.float64]
    assert out[properties.R].dtype == torch.float64


def test_grad_policy_is_the_calculators():
    seen = []

    def model(batch):
        seen.append(torch.is_grad_enabled())
        return {"prediction": torch.zeros_like(batch[properties.R])}

    batch = {properties.R: torch.randn(2, 3)}
    Calculator(model)(batch)
    Calculator(model, enable_grad=True)(batch)
    assert seen == [False, True]


# ------------------------------------------------------------------ last-call cache


class CountingModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        return {"prediction": batch[properties.R] * 2.0}


def test_no_cache_by_default():
    model = CountingModel()
    calculator = Calculator(model)
    batch = {properties.R: torch.ones(2, 3)}
    calculator(batch)
    calculator(batch)
    assert model.calls == 2


def test_the_cache_answers_an_unchanged_batch():
    model = CountingModel()
    calculator = Calculator(model, cache_last=True)
    batch = {properties.R: torch.ones(2, 3)}
    first = calculator(batch)
    # a new dict holding the same tensors is the same batch
    second = calculator(dict(batch))
    assert model.calls == 1
    assert torch.equal(first["prediction"], second["prediction"])


def test_the_cache_notices_a_replaced_or_edited_tensor():
    model = CountingModel()
    calculator = Calculator(model, cache_last=True)
    batch = {properties.R: torch.ones(2, 3), properties.t: torch.zeros(2)}
    calculator(batch)

    calculator({**batch, properties.R: torch.ones(2, 3)})
    assert model.calls == 2, "a replaced tensor is a new batch, even if equal"

    batch = {properties.R: torch.ones(2, 3), properties.t: torch.zeros(2)}
    calculator(batch)
    batch[properties.t] += 1.0
    calculator(batch)
    assert model.calls == 4, "an in-place edit of any key is noticed"

    calculator({**batch, "extra": torch.zeros(1)})
    assert model.calls == 5, "an added key is a new batch"


def test_reset_clears_the_cache():
    model = CountingModel()
    calculator = Calculator(model, cache_last=True)
    batch = {properties.R: torch.ones(2, 3)}
    calculator(batch)
    calculator.reset()
    calculator(batch)
    assert model.calls == 2


def test_cached_outputs_are_detached():
    def model(batch):
        r = batch[properties.R].requires_grad_()
        return {"prediction": r * 2.0}

    calculator = Calculator(model, enable_grad=True, cache_last=True)
    out = calculator({properties.R: torch.ones(2, 3)})
    assert not out["prediction"].requires_grad


def test_the_drivers_positions_never_require_grad():
    """The model marks its input positions; that must not reach the caller's tensor."""

    def model(batch):
        r = batch[properties.R].requires_grad_()
        return {"prediction": r * 2.0}

    batch = {properties.R: torch.ones(2, 3)}
    Calculator(model, enable_grad=True)(batch)
    assert not batch[properties.R].requires_grad


def test_a_batch_neighbor_list_plugs_in_as_a_transform():
    from schnetpack.transform import BatchNeighborList

    assert callable(BatchNeighborList.__call__)
    assert BatchNeighborList.__call__ is not object.__call__


# ------------------------------------------------------------- force-field units

KCAL = 1 / 23.0605480121  # eV per kcal/mol


class RecordingForceField:
    """Unit-sized outputs of every kind, and a record of the inputs it was handed."""

    def __init__(self):
        self.inputs = None
        self.grad_enabled = None

    def __call__(self, batch):
        self.inputs = dict(batch)
        self.grad_enabled = torch.is_grad_enabled()
        n_structures = batch[properties.n_atoms].shape[0]
        return {
            "energy": torch.ones(n_structures),
            "forces": torch.ones_like(batch[properties.R]),
            "stress": torch.ones(n_structures, 3, 3),
            "dipole_moment": torch.ones(n_structures, 3),
        }


def angstrom_batch():
    return {
        properties.n_atoms: torch.tensor([2, 1]),
        properties.R: torch.randn(3, 3),
        properties.cell: torch.randn(2, 3, 3),
    }


def test_the_model_and_the_neighbor_list_see_the_models_units():
    seen = {}

    def neighbor_list(inputs):
        seen["positions"] = inputs[properties.R].clone()
        return inputs

    model = RecordingForceField()
    batch = angstrom_batch()
    before = {key: value.clone() for key, value in batch.items()}
    ForceFieldCalculator(model, neighbor_list=neighbor_list, position_unit="nm")(batch)

    torch.testing.assert_close(seen["positions"], batch[properties.R] / 10.0)
    torch.testing.assert_close(model.inputs[properties.R], batch[properties.R] / 10.0)
    torch.testing.assert_close(
        model.inputs[properties.cell], batch[properties.cell] / 10.0
    )
    for key, value in before.items():
        assert torch.equal(batch[key], value), key


def test_energy_forces_and_stress_come_back_in_ev_and_angstrom():
    calculator = ForceFieldCalculator(
        RecordingForceField(),
        energy_unit="kcal/mol",
        position_unit="nm",
        stress_key="stress",
    )
    out = calculator(angstrom_batch())

    torch.testing.assert_close(out["energy"], torch.full((2,), KCAL))
    torch.testing.assert_close(out["forces"], torch.full((3, 3), KCAL / 10.0))
    torch.testing.assert_close(out["stress"], torch.full((2, 3, 3), KCAL / 1000.0))
    # what is not a force-field quantity passes through as the model reported it
    torch.testing.assert_close(out["dipole_moment"], torch.ones(2, 3))


def test_stress_is_left_alone_without_a_stress_key():
    calculator = ForceFieldCalculator(RecordingForceField(), position_unit="nm")
    torch.testing.assert_close(
        calculator(angstrom_batch())["stress"], torch.ones(2, 3, 3)
    )


def test_a_model_in_ev_and_angstrom_gets_the_positions_as_they_are():
    model = RecordingForceField()
    batch = angstrom_batch()
    ForceFieldCalculator(model, enable_grad=False)(batch)
    assert model.inputs[properties.R] is batch[properties.R]
    assert model.inputs[properties.cell] is batch[properties.cell]


@pytest.mark.parametrize("missing", ["energy", "forces"])
def test_a_force_field_must_return_energy_and_forces(missing):
    def model(batch):
        outputs = RecordingForceField()(batch)
        del outputs[missing]
        return outputs

    with pytest.raises(KeyError, match=missing):
        ForceFieldCalculator(model)(angstrom_batch())


def test_a_force_field_runs_with_grad_by_default():
    model = RecordingForceField()
    ForceFieldCalculator(model)(angstrom_batch())
    assert model.grad_enabled


def test_the_cache_answers_an_unchanged_angstrom_batch():
    calls = []

    def model(batch):
        calls.append(1)
        return {
            "energy": torch.zeros(2),
            "forces": torch.zeros_like(batch[properties.R]),
        }

    calculator = ForceFieldCalculator(model, position_unit="nm", cache_last=True)
    batch = angstrom_batch()
    first = calculator(batch)
    second = calculator(batch)
    assert len(calls) == 1
    torch.testing.assert_close(first["forces"], second["forces"])


@pytest.mark.parametrize(
    "build",
    [
        lambda calc: Sampler(calc, VP(), VelocityParametrization(), Heun(), churn=0.0),
        lambda calc: DirectDenoising(calc, VE(0.01, 3.0), PseudoForceParametrization()),
    ],
    ids=["Sampler", "DirectDenoising"],
)
def test_generative_drivers_refuse_a_force_field_calculator(build):
    # it would convert the positions going in, but not the raw head coming out
    with pytest.raises(TypeError, match="ForceFieldCalculator"):
        build(ForceFieldCalculator(zero_model))


# ------------------------------------------------------------- integrator history


def test_one_step_integrators_carry_no_history():
    from schnetpack.dynamics import EulerMaruyama
    from schnetpack.generative import ReverseODE

    field = ReverseODE(lambda x, t: -x)
    x = torch.ones(3, 3)
    integrator = EulerMaruyama()
    state = integrator.init_state(field, x)
    x_new, state = integrator.step(field, x, torch.ones(3), torch.tensor(-0.1), state)
    assert state is None
    torch.testing.assert_close(x_new, x * 1.1)
