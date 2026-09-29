import torch

from schnetpack import properties
from schnetpack.dynamics import Calculator, DirectDenoising, Heun, Sampler
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
