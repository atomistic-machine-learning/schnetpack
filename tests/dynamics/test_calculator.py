import torch

from schnetpack import properties
from schnetpack.dynamics import Calculator, DirectDenoising, Heun, Sampler
from schnetpack.generative import (
    PseudoForceParametrization,
    VE,
    VP,
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
