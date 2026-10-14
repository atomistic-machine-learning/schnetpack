"""
The Sampler base: what every step rule inherits from it.
"""

import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    AncestralSampler,
    Euler,
    Heun,
    UniformGrid,
    VelocityCalculator,
)
from schnetpack.dynamics.guidance import Guidance
from schnetpack.dynamics.hooks import FreezeScaffold
from schnetpack.dynamics.sample import sampler as sample
from schnetpack.generative import (
    VP,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    Interpolant,
    PermutationCoupling,
    VELinear,
    VelocityParametrization,
)
from tests.dynamics.test_sampling import score_calculator


def model(batch):
    """A smooth toy head that reads both x and t."""
    x, t = batch[properties.R], batch[properties.t]
    return {"prediction": 0.3 * torch.tanh(x) * (1.0 + t[:, None])}


class Pull(Guidance):
    """A linear pull toward the origin, a guidance term that depends on t."""

    def forward(self, batch):
        return -batch[properties.R] * (1.0 + batch[properties.t][:, None])


def start(n_rows=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        properties.R: torch.randn(n_rows, 3, generator=generator, dtype=torch.float64)
    }


def test_sample_draws_from_the_forward_process_prior():
    process = GaussianInterpolant(VP())
    driver = sample.SDESampler(
        score_calculator(model, process, EpsParametrization()),
        process,
        integrator=Heun(),
    )
    assert driver.prior is process.prior
    template = {properties.R: torch.empty(5, 3, dtype=torch.float64)}
    out = driver.run(driver.prior.sample_from_batch(template), 4)
    assert out[properties.R].shape == (5, 3)
    assert torch.all(out[properties.t] == process.schedule.t_min)


def test_sample_refuses_a_calculator_without_a_score():
    process = GaussianInterpolant(VP())
    with pytest.raises(TypeError, match="ScoreCalculator"):
        sample.SDESampler(model, process, integrator=Heun())
    with pytest.raises(TypeError, match="ScoreCalculator"):
        sample.SDESampler(VelocityCalculator(model), process, integrator=Heun())
    with pytest.raises(TypeError, match="ScoreCalculator"):
        AncestralSampler(model, process)
    with pytest.raises(TypeError, match="ScoreCalculator"):
        AncestralSampler(VelocityCalculator(model), process)


def test_ode_sample_refuses_a_calculator_without_a_velocity():
    process = GaussianInterpolant(VP())
    with pytest.raises(TypeError, match="VelocityCalculator"):
        sample.ODESampler(model)
    with pytest.raises(TypeError, match="VelocityCalculator"):
        sample.ODESampler(score_calculator(model, process), integrator=Heun())


@pytest.mark.parametrize(
    "process",
    [Interpolant(FlowMatching()), Interpolant(VP(), coupling=PermutationCoupling())],
    ids=["plain-schedule", "coupled"],
)
def test_sample_refuses_a_non_gaussian_interpolant(process):
    calculator = score_calculator(model, process, VelocityParametrization())
    with pytest.raises(TypeError, match="GaussianInterpolant"):
        sample.SDESampler(calculator, process, eta2=0.0, integrator=Heun())
    with pytest.raises(TypeError, match="GaussianInterpolant"):
        sample.AncestralSampler(calculator, process)


def test_ode_sample_runs_along_its_grid():
    driver = sample.ODESampler(
        VelocityCalculator(lambda batch: {"velocity": -batch[properties.R]}),
        integrator=Heun(),
        grid=UniformGrid(0.5, 0.1),
    )
    out = driver.run(start(), 4)
    assert torch.allclose(out[properties.t], torch.full((6,), 0.1, dtype=torch.float64))


def test_freeze_scaffold_holds_the_reference_on_a_sampler():
    process, parametrization = GaussianInterpolant(VP()), EpsParametrization()
    batch = start()
    mask = torch.tensor([True, False, True, False, False, False])
    batch = {
        **batch,
        properties.fixed_atoms: mask,
        properties.R_reference: torch.zeros_like(batch[properties.R]),
    }
    driver = sample.SDESampler(
        score_calculator(model, process, parametrization),
        process,
        hooks=[FreezeScaffold()],
    )
    torch.manual_seed(0)
    x = driver.run(batch, 20)[properties.R]
    assert torch.equal(x[mask], torch.zeros_like(x[mask]))


@pytest.mark.parametrize(
    "process, parametrization, eta2",
    [
        (GaussianInterpolant(VP()), EpsParametrization(), 1.0),
        (GaussianInterpolant(VP()), EpsParametrization(), 0.0),
        (GaussianInterpolant(VELinear()), VelocityParametrization(), 0.0),
    ],
)
def test_euler_step_costs_exactly_one_model_evaluation(process, parametrization, eta2):
    # The step reads one score — a single model call, whatever the eta2.
    calls = []

    def counting(batch):
        calls.append(1)
        return model(batch)

    driver = sample.SDESampler(
        score_calculator(counting, process, parametrization), process, eta2=eta2
    )
    x = start()[properties.R]
    t = torch.full((x.shape[0],), 0.5, dtype=x.dtype)
    driver.step({}, x, t, torch.tensor(-0.1, dtype=x.dtype))
    assert len(calls) == 1


def counted(calls):
    def counting(batch):
        calls.append(1)
        return {
            **model(batch),
            "velocity": 0.3 * torch.tanh(batch[properties.R]),
        }

    return counting


@pytest.mark.parametrize(
    "integrator, n_calls",
    [(Euler(), 1), (Heun(), 2)],
    ids=["euler", "heun"],
)
def test_sde_step_costs_per_integrator(integrator, n_calls):
    calls = []
    process = GaussianInterpolant(VP())
    driver = sample.SDESampler(
        score_calculator(counted(calls), process, EpsParametrization()),
        process,
        eta2=0.0,
        integrator=integrator,
    )
    x = start()[properties.R]
    t = torch.full((x.shape[0],), 0.5, dtype=x.dtype)
    driver.step({}, x, t, torch.tensor(-0.1, dtype=x.dtype))
    assert len(calls) == n_calls


def test_ancestral_step_costs_exactly_one_model_evaluation():
    calls = []
    process = GaussianInterpolant(VP())
    driver = AncestralSampler(
        score_calculator(counted(calls), process, EpsParametrization()), process
    )
    x = start()[properties.R]
    t = torch.full((x.shape[0],), 0.5, dtype=x.dtype)
    driver.step({}, x, t, torch.tensor(-0.1, dtype=x.dtype))
    assert len(calls) == 1
    calls.clear()
    driver.run(start(), 3)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "integrator, n_calls", [(Euler(), 1), (Heun(), 2)], ids=["euler", "heun"]
)
def test_ode_step_costs_per_integrator(integrator, n_calls):
    calls = []
    driver = sample.ODESampler(
        VelocityCalculator(counted(calls)), integrator=integrator
    )
    x = start()[properties.R]
    t = torch.full((x.shape[0],), 0.5, dtype=x.dtype)
    driver.step({}, x, t, torch.tensor(-0.1, dtype=x.dtype))
    assert len(calls) == n_calls
    # and per step of a run
    calls.clear()
    driver.run(start(), 3)
    assert len(calls) == 3 * n_calls
