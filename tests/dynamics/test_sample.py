"""
The Sampler base: what every step rule inherits from it.
"""

import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import VelocityCalculator
from schnetpack.dynamics.guidance import Guidance
from schnetpack.dynamics.hooks import FreezeScaffold
from schnetpack.dynamics.sample import sampler as sample
from schnetpack.generative import (
    VP,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
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


def test_sample_draws_from_the_schedules_prior():
    schedule = VP()
    driver = sample.Heun(
        score_calculator(model, schedule, EpsParametrization()), schedule
    )
    assert driver.prior is not None
    template = {properties.R: torch.empty(5, 3, dtype=torch.float64)}
    out = driver.run(driver.prior.sample_from_batch(template), 4)
    assert out[properties.R].shape == (5, 3)
    assert torch.all(out[properties.t] == schedule.t_min)


def test_sample_refuses_a_calculator_without_a_score():
    with pytest.raises(TypeError, match="ScoreCalculator"):
        sample.Heun(model, VP())
    with pytest.raises(TypeError, match="ScoreCalculator"):
        sample.Heun(VelocityCalculator(model), VP())


def test_sample_refuses_a_plain_schedule():
    schedule = FlowMatching()
    calculator = score_calculator(model, schedule, VelocityParametrization())
    with pytest.raises(TypeError, match="chart"):
        sample.Heun(calculator, schedule, eta2=0.0)
    with pytest.raises(TypeError, match="chart"):
        sample.Ancestral(calculator, schedule)


def test_freeze_scaffold_holds_the_reference_on_a_sampler():
    process, parametrization = GaussianInterpolant(VP()), EpsParametrization()
    batch = start()
    mask = torch.tensor([True, False, True, False, False, False])
    batch = {
        **batch,
        properties.fixed_atoms: mask,
        properties.R_reference: torch.zeros_like(batch[properties.R]),
    }
    driver = sample.EulerMaruyama(
        score_calculator(model, process.schedule, parametrization),
        process.schedule,
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

    driver = sample.EulerMaruyama(
        score_calculator(counting, process.schedule, parametrization),
        process.schedule,
        eta2=eta2,
    )
    x = start()[properties.R]
    t = torch.full((x.shape[0],), 0.5, dtype=x.dtype)
    driver.step({}, x, t, torch.tensor(-0.1, dtype=x.dtype))
    assert len(calls) == 1
